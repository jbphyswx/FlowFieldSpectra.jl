module Types

export AbstractSpectralBackend, DirectSumBackend, FFTBackend, NUFFTBackend, SHTBackend, NUFSHTBackend
export AbstractExecutionBackend, SerialBackend, ThreadedBackend, GPUBackend, AutoBackend
export DistributedBackend, MPIBackend, local_backend, is_distributed, resolve_backend

# =============================================================================
# Two orthogonal backend axes that compose:
#   • AbstractSpectralBackend   — WHICH transform  (DirectSum / FFT / NUFFT / SHT / NUFSHT: the math)
#   • AbstractExecutionBackend  — HOW it is run     (serial / threaded / GPU / distributed / MPI)
# A computation picks one of each; the public API takes them as the `transform=` and `execution=`
# keywords. A transform's *internal* parallelism (FFTW/FINUFFT thread count, the GPU array) is
# selected by the execution backend; the execution backend also parallelises the direct-sum outer
# loop and distributes the point/segment reduction across processes.
#
# NOTE on naming: execution TYPES use the `…Backend` suffix specifically so they never collide with
# the packages loaded in the extensions (the stdlib `Distributed`, `MPI.jl`, …).
# =============================================================================

# -----------------------------------------------------------------------------
# Transform (spectral) axis
# -----------------------------------------------------------------------------

"""
    AbstractSpectralBackend

Abstract supertype for *transform* backends: which spectral math maps physical↔spectral
coefficients. Orthogonal to [`AbstractExecutionBackend`](@ref). Concrete subtypes dispatch the
`calculate_spectrum` interface to different mathematical methods and third-party libraries.
"""
abstract type AbstractSpectralBackend end

"""
    DirectSumBackend <: AbstractSpectralBackend

Slow, dependency-free reference transform that computes the Discrete Fourier Transform (DFT) or
Spherical Harmonic Transform (SHT) directly using ``O(N \\cdot M)`` direct summation.
This backend is fully self-contained and requires no external packages to be loaded. It is the
default transform.

# Details
- **Cartesian coordinates**: Computes the exact Discrete Fourier Transform (DFT) at the target frequencies.
- **Spherical coordinates**: Computes SHT coefficients using a direct projection onto the Spherical Harmonic basis, with associated Legendre polynomials computed via a type-stable recurrence relation.
- **Complexity**: ``O(N \\cdot M)`` where ``N`` is the number of spatial grid points and ``M`` is the number of spectral modes.
"""
struct DirectSumBackend <: AbstractSpectralBackend end

"""
    FFTBackend <: AbstractSpectralBackend

Fast Fourier Transform backend for uniform Cartesian grids.
Leverages `FFTW.jl` (via a package extension) to achieve optimal performance.

# Requirements
To use this backend, you must import `FFTW` in your script:
```julia
using FFTW
```

# Details
- **Grid requirements**: Grids must be uniform and rectilinear (Cartesian). Coordinates should represent grid axes rather than scattered point lists.
- **Complexity**: ``O(N \\log N)`` where ``N`` is the number of grid points.
"""
struct FFTBackend <: AbstractSpectralBackend end

"""
    NUFFTBackend <: AbstractSpectralBackend

Non-Uniform Fast Fourier Transform (NUFFT) backend for non-uniform / scattered Cartesian grids.
Leverages `FINUFFT.jl` (via a package extension). On a GPU execution backend
(`GPUBackend(CUDABackend())`) this routes to cuFINUFFT for a fast device NUFFT.

# Requirements
To use this backend, you must import `FINUFFT` in your script:
```julia
using FINUFFT
```

# Details
- **Grid requirements**: Scattered / non-uniform Cartesian coordinates.
- **Complexity**: ``O(N \\log N + M \\log(1/\\epsilon))`` where ``N`` is the number of points, ``M`` is the number of modes, and ``\\epsilon`` is the target accuracy.
- **Parameters**: Supports passing an accuracy parameter `eps` (defaults to `1e-8`).
"""
struct NUFFTBackend <: AbstractSpectralBackend end

"""
    SHTBackend <: AbstractSpectralBackend

Spherical Harmonic Transform backend for uniform / structured spherical grids.
Leverages `FastSphericalHarmonics.jl` (via a package extension) for high-performance SHT on equiangular and Clenshaw-Curtis grids.

# Requirements
To use this backend, you must import `FastSphericalHarmonics` in your script:
```julia
using FastSphericalHarmonics
```

# Details
- **Grid requirements**: Latitude/longitude grids structured specifically for Clenshaw-Curtis quadrature nodes.
- **Complexity**: ``O(L^3)`` or ``O(L^2 \\log L)`` where ``L`` is the maximum spherical degree (`lmax`).
"""
struct SHTBackend <: AbstractSpectralBackend end

"""
    NUFSHTBackend <: AbstractSpectralBackend

Non-Uniform Fast Spherical Harmonic Transform (NUFSHT) backend for unstructured/scattered spherical grids.
Leverages `NUFSHT.jl` (via a package extension).

# Requirements
To use this backend, you must import `NUFSHT` in your script:
```julia
using NUFSHT
```

# Details
- **Grid requirements**: Arbitrary scattered coordinates ``(\\theta, \\phi)`` on the sphere.
- **Complexity**: ``O(M \\log M + N \\log(1/\\epsilon))`` using Double Fourier Sphere (DFS) folding and NUFFT techniques.
- **Parameters**: Supports `solve::Bool` to trigger an iterative CG solver (conjugate gradient) for recovering the spectral coefficients from scattered grid measurements.
"""
struct NUFSHTBackend <: AbstractSpectralBackend end

# -----------------------------------------------------------------------------
# Execution (parallelism) axis
# -----------------------------------------------------------------------------

"""
    AbstractExecutionBackend

Abstract supertype for *execution* backends — the local compute backends
(`SerialBackend`, `ThreadedBackend`, `GPUBackend`) that say what one process computes on, and the
distribution wrappers (`DistributedBackend`, `MPIBackend`), parametric over an inner local backend,
that say how work is split across processes. Orthogonal to [`AbstractSpectralBackend`](@ref).
"""
abstract type AbstractExecutionBackend end

"Serial single-threaded CPU execution (always available, no extension needed)."
struct SerialBackend <: AbstractExecutionBackend end

"Multithreaded CPU execution over the direct-sum outer loop / internal library threads (requires `using OhMyThreads`)."
struct ThreadedBackend <: AbstractExecutionBackend end

"""
    GPUBackend{B}

GPU execution on the KernelAbstractions backend object `B` (e.g. `GPUBackend(CUDABackend())`,
`GPUBackend(KernelAbstractions.CPU())`). Requires `using KernelAbstractions` plus a vendor GPU
package. On a CUDA device, `FFTBackend`/`NUFFTBackend` route to CUFFT/cuFINUFFT (fast); on any KA
device `DirectSumBackend` (and every spherical transform) runs the portable KA direct-sum kernels.
"""
struct GPUBackend{B} <: AbstractExecutionBackend
    backend::B
end

"""
    DistributedBackend(inner = SerialBackend())

Multi-process execution via `Distributed`, each worker running `inner` locally. Requires
`using Distributed` and workers started with `addprocs()` + `@everywhere using FlowFieldSpectra`.
Parametric over the inner local backend, e.g. `DistributedBackend(ThreadedBackend())` for
multithreaded workers or `DistributedBackend(GPUBackend(dev))` for one GPU per worker.
"""
struct DistributedBackend{Inner<:AbstractExecutionBackend} <: AbstractExecutionBackend
    inner::Inner
end
DistributedBackend() = DistributedBackend(SerialBackend())

"""
    MPIBackend(inner = SerialBackend(); comm = nothing)

Multi-rank execution via `MPI.jl`, each rank running `inner` locally and partial coefficient
buffers combined in place with `MPI.Allreduce!` (every rank ends with the full result). Requires
`using MPI` and launching under `mpiexec` with `MPI.Init()` called. Not CPU-only:
`MPIBackend(GPUBackend(dev))` targets a multi-GPU cluster (one GPU per rank) and
`MPIBackend(ThreadedBackend())` is hybrid MPI+threads. `comm === nothing` ⇒ the MPI extension uses
`MPI.COMM_WORLD` (the core never references MPI).
"""
struct MPIBackend{Inner<:AbstractExecutionBackend, C} <: AbstractExecutionBackend
    inner::Inner
    comm::C
end
MPIBackend(inner::AbstractExecutionBackend = SerialBackend(); comm = nothing) = MPIBackend(inner, comm)

"""
    AutoBackend <: AbstractExecutionBackend

Select the best available *local* execution backend at call time: `ThreadedBackend` when the
OhMyThreads extension is loaded and `Threads.nthreads() > 1`, else `SerialBackend`. Never resolves
to GPU/Distributed/MPI — those require an explicit device/process context.
"""
struct AutoBackend <: AbstractExecutionBackend end

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

"""
    local_backend(backend) -> AbstractExecutionBackend

The per-process compute backend: the wrapped `inner` for a distribution wrapper, else the backend
itself.
"""
local_backend(b::AbstractExecutionBackend) = b
local_backend(b::DistributedBackend) = b.inner
local_backend(b::MPIBackend) = b.inner

"`is_distributed(backend)` — `true` if `backend` splits work across processes/ranks."
is_distributed(::AbstractExecutionBackend) = false
is_distributed(::DistributedBackend) = true
is_distributed(::MPIBackend) = true

"""
    resolve_backend(backend) -> AbstractExecutionBackend

Resolve `AutoBackend` to a concrete local backend instance; all other backends are returned as-is.
The `AutoBackend` method is defined in the parent `FlowFieldSpectra` module so it can detect a
loaded threading extension via `Base.get_extension`.
"""
resolve_backend(backend::AbstractExecutionBackend) = backend

end # module Types
