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

Here is a complete, runnable example computing the isotropic energy spectrum of a 2D uniform flow field.

```julia
using FlowFieldSpectra
using FFTW        # Load FFTW to unlock fast FFTBackend()
using CairoMakie  # Load CairoMakie to unlock plot_spectrum()

# 1. Define coordinate grid axes
L = 2π
N = 32
dx = L / N
xs = range(0.0, stop=L-dx, length=N)
ys = range(0.0, stop=L-dx, length=N)

# Expand to matching flat coordinate lists
xv = vec([x for x in xs, y in ys])
yv = vec([y for x in xs, y in ys])

# 2. Synthesize u (zonal) and v (meridional) velocities
# Here we add wave components with specific wavenumbers
u = @. cos(2 * xv) + 0.5 * sin(5 * yv)
v = @. sin(2 * xv)

# 3. Calculate Fourier Coefficients
# FFTBackend() requires FFTW to be imported
coeffs, ks = calculate_spectrum(
    FFTBackend(),
    (xv, yv),
    (u, v),
    (N, N);
    domain_size=(L, L)
)

# 4. Integrate radially to compute a 1D isotropic energy spectrum
k_bins, E_k = isotropic_spectrum(ks, coeffs; num_bins=16)

# 5. Plot the result
fig = plot_spectrum(ks, coeffs; title="Flow Field Energy Spectrum")
save("spectrum_plot.png", fig)
```
