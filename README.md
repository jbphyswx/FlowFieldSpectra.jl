# FlowFieldSpectra.jl

*Unified, blazing-fast multi-dimensional spectral analysis and reductions for flow fields in Julia.*

[![Build Status](https://github.com/jbphyswx/FlowFieldSpectra.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/jbphyswx/FlowFieldSpectra.jl/actions/workflows/CI.yml)
[![Dev Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://jbphyswx.github.io/FlowFieldSpectra.jl/dev/)

`FlowFieldSpectra.jl` simplifies computing spectral coefficients, energy spectra, and spatial reductions (such as isotropic radial integration or dimension transects) across multi-dimensional grids. It supports **Cartesian** and **Spherical** coordinates on both **structured/uniform** and **unstructured/scattered** grids.

Instead of writing custom FFT grid shifting, scaling, non-uniform coordinate mapping, or Legendre recurrence code, `FlowFieldSpectra.jl` handles the complexity behind a single type-stable function call: `calculate_spectrum`.

---

## Core Features

- **Unified Interface**: Use `calculate_spectrum` for any combination of coordinates, grid types, and dimensions.
- **Cartesian & Spherical**: 
  - Cartesian: 1D, 2D, 3D, and arbitrary ND transforms.
  - Spherical: Spherical Harmonic Transforms (SHT) on the sphere.
- **Structured & Scattered Grids**: 
  - Uniform/rectilinear grids.
  - Non-uniform, scattered, or random coordinate point clouds.
- **Zero-Dependency Baseline**: Works out-of-the-box using naive direct-sum methods (`DirectSumBackend`) without compiling heavy C/C++ libraries.
- **Fast Path via Extensions**: Simply import your favorite backend package (`FFTW`, `FINUFFT`, `FastSphericalHarmonics`, `NUFSHT`) to automatically activate highly optimized, parallelized fast algorithms.
- **Reductions & Plotting**: Built-in methods for isotropic/radial spectrum binning, transects/slices, spherical degree spectra, and visualizations.

---

## Installation

Install `FlowFieldSpectra.jl` from the Julia package manager:

```julia
using Pkg
Pkg.add("FlowFieldSpectra")
```

---

## Extension Architecture: Unlocking Fast Paths

By default, the package runs slow-path direct sums (DFT/SHT, ``O(N \cdot M)`` complexity). To unlock ``O(N \log N)`` fast paths, simply load the corresponding package:

| Grid Type | Coordinate System | Required Library | Backend Type | Description |
|---|---|---|---|---|
| **Structured** | Cartesian (ND) | `using FFTW` | `FFTBackend()` | Fast Fourier Transform via FFTW |
| **Scattered** | Cartesian (ND) | `using FINUFFT` | `NUFFTBackend()` | Non-uniform Fast Fourier Transform |
| **Structured** | Spherical (2D) | `using FastSphericalHarmonics` | `SHTBackend()` | Fast SHT on Clenshaw-Curtis grids |
| **Scattered** | Spherical (2D) | `using NUFSHT` | `NUFSHTBackend()` | Non-uniform Fast SHT |
| **Any** | Visualization | `using CairoMakie` | (Plotting Functions) | Enclosing plotting extension |

---

## Quickstart: 2D Cartesian Flow Field

Below is a complete example showing how to compute and plot the 1D isotropic energy spectrum of a 2D Cartesian flow field.

```julia
using FlowFieldSpectra: FlowFieldSpectra as FFS
using FFTW: FFTW                          # activates the FFTBackend extension
using CairoMakie: CairoMakie as Mke       # activates the plotting extension

# 1. Define a 2D Cartesian Domain
L = 10.0
N = 64
dx = L / N
xs = range(0.0, stop=L-dx, length=N)
ys = range(0.0, stop=L-dx, length=N)

# Generate coordinate lists matching a grid
xv = vec([x for x in xs, y in ys])
yv = vec([y for x in xs, y in ys])

# 2. Synthesize a 2D flow field (u, v) with specific wave components
u = @. cos(2π * 2 * xv / L) + 0.5 * sin(2π * 5 * yv / L)
v = @. sin(2π * 2 * xv / L)

# 3. Build an explicit grid and compute spectral coefficients via FFTW.
#    The coordinate system is the grid type — there is no coordinate guessing.
grid = FFS.UniformCartesianGrid((xv, yv); domain_size=(L, L))
coeffs, ks = FFS.calculate_spectrum(FFS.FFTBackend(), grid, (u, v), (N, N))

# 4. Reduce 2D Coefficients to a 1D Isotropic (Radial) Spectrum
k_bins, E_k = FFS.isotropic_spectrum(ks, coeffs; num_bins=32)

# 5. Visualize the Energy Spectrum
fig = FFS.plot_spectrum(ks, coeffs; title="Radial Energy Spectrum")
Mke.save("energy_spectrum.png", fig)
```

Construct the grid that matches your data — `UniformCartesianGrid`, `ScatteredCartesianGrid`
(NUFFT), `StructuredSphericalGrid` (SHT), or `ScatteredSphericalGrid` (NUFSHT) — and dispatch is
exact. For repeated transforms on a fixed grid (e.g. each level/time of an `(x,y,z,t)` field),
build a plan once with `plan_spectrum` and reuse it via `calculate_spectrum!`.

---

## Example Figures

### Cartesian 2D Flow Field Spectral Analysis

![Cartesian Spectra](docs/src/assets/cartesian_spectra.png)

*Taylor-Green vortex: velocity vectors (top-left), 2D spectral energy density (top-right), and 1D isotropic vs transect spectra (bottom).*

### Spherical Harmonic Degree Spectrum

![Spherical Spectra](docs/src/assets/spherical_spectra.png)

*Left: scalar field on a Clenshaw-Curtis grid. Right: energy per spherical harmonic degree ℓ.*

### Backend Parity (DirectSum vs FFTW)

![Backend Parity](docs/src/assets/backend_parity.png)

*Coefficient magnitude comparison between DirectSum and FFT backends; difference is at machine-precision levels.*

---

## Core API Reference

### Spectral Calculation
- `calculate_spectrum(backend, coords, fields, ms; kwargs...)`: Computes complex coefficients and physical wavenumber grids.

### Reductions
- `isotropic_spectrum(ks, coeffs; num_bins)`: Integrate ND coefficients radially to get a 1D energy density spectrum.
- `transect_spectrum(ks, coeffs, dims)`: Integrate out specific dimensions (e.g. summing along y to get a zonal spectrum).
- `spherical_energy_spectrum(coeffs; lmax)`: Get the energy per spherical degree ``l``.

### Plotting (with `CairoMakie`)
- `plot_spectrum(ks, coeffs; title)`: Plot 1D/2D Cartesian or spherical energy spectra.
- `compare_spectra(spectra_list; labels)`: Compare multiple 1D spectra on the same axes.
- `compare_spectral_analysis(true_coeffs, approx_coeffs)`: Plot spectral errors and coefficient deviations.


### References