# Backends and Extensions

`FlowFieldSpectra.jl` separates spectral algorithms from library dependencies using Julia 1.9+'s package extensions mechanism. By default, the core package is extremely lightweight and has no heavy external compiled dependencies (like FFTW or FINUFFT).

To activate optimized, high-performance backends, you simply import the corresponding package in your workspace.

Backends come in **two orthogonal axes** that compose freely — the public API takes one of each as
the `transform=` and `execution=` keywords. The tag types live in two shared packages:

- **Transform** (`SpectralBackends.AbstractSpectralBackend`) — *which spectral math*:
  `DirectSumSpectralBackend`, `FFTSpectralBackend`, `NUFFTSpectralBackend`, `FSHTSpectralBackend`,
  `NUFSHTSpectralBackend` (from [SpectralBackends.jl](https://github.com/jbphyswx/SpectralBackends.jl)).
  Covered by the matrix below.
- **Execution** (`ComputationalBackends.AbstractExecutionBackend`) — *where/how it runs*:
  `SerialBackend`, `ThreadedBackend`, `GPUBackend`, `DistributedBackend`, `MPIBackend`, `AutoBackend`
  (from [ComputationalBackends.jl](https://github.com/jbphyswx/ComputationalBackends.jl)). Covered in
  [Execution backends](@ref) below.

```julia
using SpectralBackends: SpectralBackends as SB
using ComputationalBackends: ComputationalBackends as CB

# FFT transform, run threaded:
calculate_spectrum(grid, fields, ms; transform = SB.FFTSpectralBackend(), execution = CB.ThreadedBackend())
# Fast NUFFT on a CUDA GPU (cuFINUFFT):
calculate_spectrum(grid, fields, ms; transform = FFS.FINUFFTBackend(), execution = CB.GPUBackend(CUDABackend()))
```

---

## Backend Selection Matrix (transform axis)

Choose your backend based on the grid structure (structured vs. scattered/non-uniform) and coordinate system (Cartesian vs. Spherical):

| Coordinate System | Grid Structure | Baseline Backend | Fast Backend | Required Library |
| :--- | :--- | :--- | :--- | :--- |
| **Cartesian** | Uniform / Regular | `DirectSumSpectralBackend()` | `FFTSpectralBackend()` | `using FFTW` |
| **Cartesian** | Scattered / Unstructured | `DirectSumSpectralBackend()` | `FINUFFTBackend()` | `using FINUFFT` |
| **Spherical** | Structured (Clenshaw-Curtis) | `DirectSumSpectralBackend()` | `FSHTSpectralBackend()` | `using FastSphericalHarmonics` |
| **Spherical** | Scattered / Unstructured | `DirectSumSpectralBackend()` | `NUFSHTSpectralBackend()` | `using NUFSHT` |

---

## Detailed Backend Profiles

### `DirectSumSpectralBackend`
- **Use Case**: Reference calculations, small grids, or zero-dependency runs.
- **Mathematical Method**: Direct Discrete Fourier Transform (DFT) summation or Spherical Harmonic Transform (SHT) direct integration via Legendre recurrence relations.
- **Complexity**: ``O(N \cdot M)``, where ``N`` is the number of grid nodes and ``M`` is the number of spectral modes.
- **Dependencies**: None.

### `FFTSpectralBackend`
- **Use Case**: Traditional uniform Cartesian grids (e.g. models on regular grids).
- **Mathematical Method**: Fast Fourier Transform (FFT) via `FFTW.jl`.
- **Complexity**: ``O(N \log N)``.
- **Dependencies**: Requires `using FFTW`.

### NUFFT — `FINUFFTBackend` / `NonuniformFFTsBackend`
- **Use Case**: Scattered spatial points in Cartesian coordinates (e.g. ship tracks, sensor arrays, float data).
- **Mathematical Method**: Non-Uniform Fast Fourier Transform (Type-1). `SpectralBackends.NUFFTSpectralBackend`
  is the abstract math; a *provider* is a library choice, so FlowFieldSpectra exposes two symmetric,
  concrete backends (neither is a default) — pass one as `transform=`:
  - **`FINUFFTBackend()`** — via `FINUFFT.jl` (`using FINUFFT`; GPU via cuFINUFFT on CUDA).
  - **`NonuniformFFTsBackend()`** — via `NonuniformFFTs.jl` (`using NonuniformFFTs`); a real-data fast path
    (real-to-complex FFT — ≈2× faster and half the memory when the field is real).
- **Complexity**: ``O(N \log N + M \log(1/\epsilon))``.
- **Dependencies**: `using FINUFFT` and/or `using NonuniformFFTs`. Both may be loaded at once.

### `FSHTSpectralBackend`
- **Use Case**: Regular spherical model grids (equiangular, Clenshaw-Curtis, etc.).
- **Mathematical Method**: Fast Spherical Harmonic Transform via `FastSphericalHarmonics.jl`.
- **Complexity**: ``O(L^3)`` or ``O(L^2 \log L)``, where ``L`` is the maximum degree (`lmax`).
- **Dependencies**: Requires `using FastSphericalHarmonics`.

### `NUFSHTSpectralBackend`
- **Use Case**: Unstructured grids on the sphere (e.g., geodesic grids, scattered ocean stations, planetary orbit tracking).
- **Mathematical Method**: Non-Uniform Fast Spherical Harmonic Transform via `NUFSHT.jl`.
- **Complexity**: ``O(M \log M + N \log(1/\epsilon))`` (where ``M`` is number of modes, ``N`` is number of points).
- **Dependencies**: Requires `using NUFSHT`.
- **Note on Coefficient Recovery**: Because scattered points are unstructured, the direct SHT projection (adjoint) is not the exact inverse. The backend supports an iterative Conjugate Gradient solver via `solve=true` to accurately reconstruct coefficients from scattered data.

---

## Execution backends

The execution axis chooses *where and how* the chosen transform runs; it is independent of the
transform axis. Defaults to `AutoBackend()`.

| Execution backend | Required library | Notes |
| :--- | :--- | :--- |
| `SerialBackend()` | none | Single-threaded CPU (always available). |
| `ThreadedBackend()` | `using OhMyThreads` | Multithreaded CPU: parallelises the direct-sum loop; sets the internal thread count of the FFTW (`FFTSpectralBackend`) and FINUFFT (`FINUFFTBackend`) plans (`NonuniformFFTsBackend` manages its own internal threading). `FSHTSpectralBackend`/`NUFSHTSpectralBackend` have no distinct threaded path (execution is a documented no-op there). |
| `GPUBackend(dev)` | `using KernelAbstractions` (+ vendor pkg) | GPU execution on the KernelAbstractions device `dev` (see the GPU table below). |
| `DistributedBackend(inner)` | `using Distributed` | Splits work across worker processes, each running `inner` locally. Parametric, e.g. `DistributedBackend(ThreadedBackend())`. Requires `addprocs` + `@everywhere using FlowFieldSpectra`. |
| `MPIBackend(inner)` | `using MPI` | Splits work across MPI ranks, each running `inner` locally; partials combined with `MPI.Allreduce!`. `MPIBackend(GPUBackend(dev))` targets a multi-GPU cluster. Launch under `mpiexec`. |
| `AutoBackend()` | — | Picks `ThreadedBackend` when OhMyThreads is loaded and `Threads.nthreads() > 1`, else `SerialBackend`. Never auto-selects GPU/Distributed/MPI (those need explicit context). |

Point-partitionable transforms (`DirectSumSpectralBackend`, the NUFFT backends, and
`NUFSHTSpectralBackend` in projection mode) distribute over the point/sample axis and sum the complex
coefficients; `FFTSpectralBackend`/`FSHTSpectralBackend` (which need the full grid on one process)
distribute over the batch/field axis instead.

The execution axis is **result-invariant**: it changes only where/how a transform runs, never the
answer. One FFT transform under `SerialBackend`, `ThreadedBackend`, and `GPUBackend(KA.CPU())`
returns identical coefficients (to machine ε):

![Execution-axis invariance](assets/execution_parity.png)

### GPU transforms on any KernelAbstractions device

`GPUBackend{B}` is parametric on the KernelAbstractions device object `B`. Data is staged to that
device (`KA.allocate` + `copyto!`), and the transform runs there:

- **`FFTSpectralBackend`** is device-generic through `AbstractFFTs.plan_fft`, which dispatches on the
  device *array type*: CUFFT on a `CUDABackend` (`CuArray`), rocFFT on a `ROCBackend` (`ROCArray`), and
  FFTW on `KA.CPU()` (plain `Array`). Requires `using KernelAbstractions` + the FFT provider for your
  device (`FFTW` for CPU arrays, `CUDA` for CUDA, `AMDGPU` for ROCm). This is *not* CUDA-specific.
- **`DirectSumSpectralBackend`** uses the portable KernelAbstractions direct-sum kernels on any KA device.
- **`FINUFFTBackend`** on a GPU uses cuFINUFFT — **CUDA-only** (FINUFFT.jl provides no portable GPU
  NUFFT). On a non-CUDA device it raises.
- **`NonuniformFFTsBackend`** on a GPU is **device-generic** through KernelAbstractions: it threads the
  execution backend into `NonuniformFFTs.PlanNUFFT(…; backend=…)` and reconstructs FFS's full centered
  spectrum with broadcasts / a KA kernel (real-input fast path included), so the same code runs on CUDA,
  ROCm, and `KA.CPU()`. This is the portable GPU scattered-Cartesian NUFFT; `DirectSumSpectralBackend`
  remains the ``O(N M)`` portable fallback.
- **Spherical** (`FSHTSpectralBackend`/`NUFSHTSpectralBackend`/`DirectSumSpectralBackend`) always uses
  the KA spherical direct-sum kernel — FastSphericalHarmonics and NUFSHT are CPU-only, so **there is no
  fast GPU spherical-harmonic transform**.

| Transform | Grid | `GPUBackend(CUDABackend())` | `GPUBackend(KA.CPU())` / other KA device |
| :--- | :--- | :--- | :--- |
| `FFTSpectralBackend` | uniform Cartesian | CUFFT (via AbstractFFTs) | FFTW on `KA.CPU()`, rocFFT on ROCm, … (via AbstractFFTs) |
| `FINUFFTBackend` | scattered Cartesian | **cuFINUFFT** — `using CUDA, FINUFFT` | errors (cuFINUFFT is CUDA-only) |
| `NonuniformFFTsBackend` | scattered Cartesian | NonuniformFFTs (device-generic, via `CUDA`) | NonuniformFFTs on any KA device (`KA.CPU()`, ROCm, …) |
| `DirectSumSpectralBackend` | any Cartesian | KA direct-sum kernel | KA direct-sum kernel |
| `DirectSumSpectralBackend`/`FSHTSpectralBackend`/`NUFSHTSpectralBackend` | any spherical | KA spherical direct-sum kernel | KA spherical direct-sum kernel |

The device-generic FFT, direct-sum, and NonuniformFFTs NUFFT paths are exercised on CI via
`GPUBackend(KA.CPU())`. The CUDA-specific paths (CUFFT on `CuArray`, cuFINUFFT) are validated on real
CUDA hardware via the package's `gpu/` project — CI has no GPU.
