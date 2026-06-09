# Backends and Extensions

`FlowFieldSpectra.jl` separates spectral algorithms from library dependencies using Julia 1.9+'s package extensions mechanism. By default, the core package is extremely lightweight and has no heavy external compiled dependencies (like FFTW or FINUFFT). 

To activate optimized, high-performance backends, you simply import the corresponding package in your workspace.

---

## Backend Selection Matrix

Choose your backend based on the grid structure (structured vs. scattered/non-uniform) and coordinate system (Cartesian vs. Spherical):

| Coordinate System | Grid Structure | Baseline Backend | Fast Backend | Required Library |
| :--- | :--- | :--- | :--- | :--- |
| **Cartesian** | Uniform / Regular | `DirectSumBackend()` | `FFTBackend()` | `using FFTW` |
| **Cartesian** | Scattered / Unstructured | `DirectSumBackend()` | `NUFFTBackend()` | `using FINUFFT` |
| **Spherical** | Structured (Clenshaw-Curtis) | `DirectSumBackend()` | `SHTBackend()` | `using FastSphericalHarmonics` |
| **Spherical** | Scattered / Unstructured | `DirectSumBackend()` | `NUFSHTBackend()` | `using NUFSHT` |

---

## Detailed Backend Profiles

### `DirectSumBackend`
- **Use Case**: Reference calculations, small grids, or zero-dependency runs.
- **Mathematical Method**: Direct Discrete Fourier Transform (DFT) summation or Spherical Harmonic Transform (SHT) direct integration via Legendre recurrence relations.
- **Complexity**: ``O(N \cdot M)``, where ``N`` is the number of grid nodes and ``M`` is the number of spectral modes.
- **Dependencies**: None.

### `FFTBackend`
- **Use Case**: Traditional uniform Cartesian grids (e.g. models on regular grids).
- **Mathematical Method**: Fast Fourier Transform (FFT) via `FFTW.jl`.
- **Complexity**: ``O(N \log N)``.
- **Dependencies**: Requires `using FFTW`.

### `NUFFTBackend`
- **Use Case**: Scattered spatial points in Cartesian coordinates (e.g. ship tracks, sensor arrays, float data).
- **Mathematical Method**: Non-Uniform Fast Fourier Transform (Type 1 transform) via `FINUFFT.jl`.
- **Complexity**: ``O(N \log N + M \log(1/\epsilon))``.
- **Dependencies**: Requires `using FINUFFT`.

### `SHTBackend`
- **Use Case**: Regular spherical model grids (equiangular, Clenshaw-Curtis, etc.).
- **Mathematical Method**: Fast Spherical Harmonic Transform via `FastSphericalHarmonics.jl`.
- **Complexity**: ``O(L^3)`` or ``O(L^2 \log L)``, where ``L`` is the maximum degree (`lmax`).
- **Dependencies**: Requires `using FastSphericalHarmonics`.

### `NUFSHTBackend`
- **Use Case**: Unstructured grids on the sphere (e.g., geodesic grids, scattered ocean stations, planetary orbit tracking).
- **Mathematical Method**: Non-Uniform Fast Spherical Harmonic Transform via `NUFSHT.jl`.
- **Complexity**: ``O(M \log M + N \log(1/\epsilon))`` (where ``M`` is number of modes, ``N`` is number of points).
- **Dependencies**: Requires `using NUFSHT`.
- **Note on Coefficient Recovery**: Because scattered points are unstructured, the direct SHT projection (adjoint) is not the exact inverse. The backend supports an iterative Conjugate Gradient solver via `solve=true` to accurately reconstruct coefficients from scattered data.
