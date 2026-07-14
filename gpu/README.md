# GPU parity checks

This project holds the **real-device (CUDA)** GPU verification for FlowFieldSpectra.jl. It is
separate from the package test suite because CI has **no GPU**, so the paths that only run on an
`CuArray` — CUFFT (the CUDA case of the device-generic GPU FFT) and cuFINUFFT (scattered NUFFT,
CUDA-only) — can only be validated numerically on an actual NVIDIA CUDA device.

## What CI covers vs. what lives here

- **CI (`test/`)** — the entire two-axis backend, all CPU execution (Serial / Threaded), Distributed
  and MPI execution, and the **device-generic GPU paths exercised on `GPUBackend(KA.CPU())`**: the
  KernelAbstractions direct-sum kernels *and* the `AbstractFFTs`-based GPU FFT (which uses FFTW on the
  `KA.CPU()` host array, exactly as it would use CUFFT on a `CuArray`). CI also asserts that GPU
  NUFFT on a non-CUDA device raises a clear error (cuFINUFFT is CUDA-only — no silent fallback).
- **Here (`gpu/`)** — the CUDA-specific realizations: CUFFT on `CuArray`, cuFINUFFT, and the KA
  kernels on a real CUDA device, each checked against its CPU reference.

CUFFT-on-`CuArray` and cuFINUFFT numerical parity is verified **only** by running the script below on
real hardware; CI covers the same GPU-FFT and direct-sum code on `KA.CPU()`.

## Running

```bash
julia --project=gpu -e 'using Pkg; Pkg.instantiate()'
julia --project=gpu gpu/gpu_parity.jl
```

Requires a functional NVIDIA CUDA device (`CUDA.functional() == true`). On a machine without one the
script prints a notice and exits cleanly, so it is safe to invoke unconditionally.

`gpu_parity.jl` checks:

- CUFFT (uniform grid) vs CPU FFTW,
- cuFINUFFT (scattered grid) vs CPU FINUFFT,
- the KA Cartesian direct-sum kernel on the CUDA device vs the serial CPU direct sum,
- the KA spherical direct-sum kernel on the CUDA device vs the serial CPU direct sum (there is no
  fast GPU spherical-harmonic transform — FastSphericalHarmonics and NUFSHT are CPU-only, so every
  spherical × GPU combination uses this portable kernel).
