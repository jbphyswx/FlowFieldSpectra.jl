module Types

export AbstractSpectralBackend, DirectSumBackend, FFTBackend, NUFFTBackend, SHTBackend, NUFSHTBackend, ThreadedBackend, GPUBackend, AutoBackend

"""
    AbstractSpectralBackend

Abstract supertype for all spectral calculation backends in `FlowFieldSpectra.jl`.
Concrete subtypes dispatch the general `calculate_spectrum` interface to different mathematical methods and third-party libraries.
"""
abstract type AbstractSpectralBackend end

"""
    DirectSumBackend <: AbstractSpectralBackend

Slow, serial CPU fallback backend that computes the Discrete Fourier Transform (DFT) or Spherical Harmonic Transform (SHT) directly using ``O(N \\cdot M)`` direct summation.
This backend is fully self-contained and requires no external packages to be loaded.

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
Leverages `FINUFFT.jl` (via a package extension).

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

"""
    ThreadedBackend <: AbstractSpectralBackend

Multi-threaded CPU execution backend using `OhMyThreads.jl` for direct sum spectral calculations.
Requires the `OhMyThreads.jl` package to be loaded.
"""
struct ThreadedBackend <: AbstractSpectralBackend end

"""
    GPUBackend{B} <: AbstractSpectralBackend

GPU-accelerated execution backend using `KernelAbstractions.jl`.
Parameterized by the target GPU backend (e.g., `KernelAbstractions.CPU()`, `CUDA.CUDABackend()`).
Requires `KernelAbstractions.jl` to be loaded.
"""
struct GPUBackend{B} <: AbstractSpectralBackend
    backend::B
end

"""
    AutoBackend <: AbstractSpectralBackend

Automatic backend selection based on availability and runtime state.
"""
struct AutoBackend <: AbstractSpectralBackend end

end # module Types
