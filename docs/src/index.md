# FlowFieldSpectra.jl

`FlowFieldSpectra.jl` provides a unified, performance-oriented Julia interface for computing spectral coefficients and spatial energy reductions from flow fields across uniform or unstructured multi-dimensional grids.

---

## Installation

Install the package via Julia's package manager:

```julia
using Pkg
Pkg.add("FlowFieldSpectra")
```

---

## Unified Workflow Guide

A typical flow field spectral analysis consists of three main steps:

1. **Coordinate and Field Setup**: Define your grid coordinates (zonal, meridional, vertical, or spherical latitude/longitude) and spatial velocity field values.
2. **Spectral Transform**: Run `calculate_spectrum` with a chosen backend (e.g. `DirectSumBackend`, `FFTBackend`, `NUFFTBackend`, etc.).
3. **Spectral Reductions**: Convert high-dimensional Fourier coefficients to meaningful energy spectra (e.g., 1D isotropic / radial energy density, 1D slice/transect, or spherical degree energy spectra).

### Quickstart Tutorial: Cartesian 2D Flow Field

A complete example computing the isotropic energy spectrum of a 2D uniform flow field. You
construct an explicit **grid** (the coordinate system is the grid type — there is no guessing)
and pass it to `calculate_spectrum`.

```@example quickstart
using FlowFieldSpectra: FlowFieldSpectra as FFS
using FFTW: FFTW              # activates the FFTBackend extension
using CairoMakie: CairoMakie  # activates the plotting extension

# 1. Coordinate lists on a uniform grid
L = 2π
N = 32
dx = L / N
xs = range(0.0, stop = L - dx, length = N)
xv = vec([x for x in xs, y in xs])
yv = vec([y for x in xs, y in xs])

# 2. Synthesize zonal/meridional velocities with specific wavenumbers
u = @. cos(2 * xv) + 0.5 * sin(5 * yv)
v = @. sin(2 * xv)

# 3. Build the grid and compute Fourier coefficients (FFTBackend needs FFTW)
grid = FFS.UniformCartesianGrid((xv, yv); domain_size = (L, L))
coeffs, ks = FFS.calculate_spectrum(FFS.FFTBackend(), grid, (u, v), (N, N))

# 4. Radially integrate to a 1D isotropic energy spectrum
k_bins, E_k = FFS.isotropic_spectrum(ks, coeffs; num_bins = 16)

# 5. Plot
FFS.plot_spectrum(ks, coeffs; title = "Flow Field Energy Spectrum")
```
