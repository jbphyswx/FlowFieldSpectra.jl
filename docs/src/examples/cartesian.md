# Cartesian spectra (FFT)

A uniform 2D Taylor–Green vortex, its 2D energy-density map, and the 1D isotropic and transect
reductions. The figures below are produced live when the documentation is built.

```@example cartesian
using FlowFieldSpectra: FlowFieldSpectra as FFS
using FFTW: FFTW                          # activates the FFTBackend extension
using CairoMakie: CairoMakie as Mke

L = 2π
N = 64
dx = L / N
xs = range(0.0, stop = L - dx, length = N)
xv = vec([x for x in xs, y in xs])
yv = vec([y for x in xs, y in xs])

# Taylor–Green vortex velocity field
u = @. cos(2 * xv) * sin(2 * yv)
v = @. -sin(2 * xv) * cos(2 * yv)

grid = FFS.UniformCartesianGrid((xv, yv); domain_size = (L, L))
coeffs, ks = FFS.calculate_spectrum(FFS.FFTBackend(), grid, (u, v), (N, N))
nothing # hide
```

## Isotropic (radial) energy spectrum

```@example cartesian
k_bins, E_k = FFS.isotropic_spectrum(ks, coeffs; num_bins = 32)

fig = Mke.Figure(size = (640, 420))
ax = Mke.Axis(fig[1, 1]; title = "Isotropic energy spectrum", xlabel = "k", ylabel = "E(k)",
    yscale = log10)
Mke.lines!(ax, k_bins, E_k .+ 1e-20; linewidth = 2)
fig
```

## Transect spectrum

Integrate out the second axis to get the 1D spectrum along the first:

```@example cartesian
k_red, E_red = FFS.transect_spectrum(ks, coeffs, (2,))

fig = Mke.Figure(size = (640, 420))
ax = Mke.Axis(fig[1, 1]; title = "Transect spectrum (integrated over kᵧ)", xlabel = "kₓ",
    ylabel = "E(kₓ)")
Mke.lines!(ax, k_red[1], E_red; linewidth = 2)
fig
```
