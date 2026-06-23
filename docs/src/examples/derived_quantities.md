# Derived-quantity spectra (vorticity, divergence, enstrophy)

`FlowFieldSpectra` forms the kinetic-energy spectrum from a velocity field; the same machinery,
via spectral differentiation (`ik^α`), gives the spectra of **vorticity**, **divergence**, and
**enstrophy**. Here we use a 2D incompressible flow `u = sin x cos y`, `v = -cos x sin y` (so
`∇·u = 0`) whose vorticity is `ω = 2 sin x sin y`.

```@example derived
using FlowFieldSpectra: FlowFieldSpectra as FFS
using FFTW: FFTW                          # activates the FFTBackend extension
using CairoMakie: CairoMakie as Mke

L = 2π
N = 32
dx = L / N
xs = range(0.0, stop = L - dx, length = N)
xv = vec([x for x in xs, y in xs])
yv = vec([y for x in xs, y in xs])
u = @. sin(xv) * cos(yv)
v = @. -cos(xv) * sin(yv)

grid = FFS.UniformCartesianGrid((xv, yv); domain_size = (L, L))
coeffs, ks = FFS.calculate_spectrum(FFS.FFTBackend(), grid, (u, v), (N, N))

# Divergence is ~0 for incompressible flow; vorticity → enstrophy spectrum.
divc = FFS.spectral_divergence(ks, coeffs)
vortc = FFS.spectral_vorticity(ks, coeffs)
@assert maximum(abs.(divc)) < 1e-12

k, E = FFS.isotropic_spectrum(ks, coeffs; num_bins = 8)   # energy spectrum
_, Z = FFS.isotropic_spectrum(ks, vortc; num_bins = 8)    # enstrophy spectrum

fig = Mke.Figure(size = (640, 420))
ax = Mke.Axis(fig[1, 1]; title = "Energy vs enstrophy spectra (Z = k² E on the active shell)",
    xlabel = "k", ylabel = "spectral density", yscale = log10)
Mke.scatter!(ax, k, E .+ 1e-20; label = "E(k) energy", markersize = 10)
Mke.scatter!(ax, k, Z .+ 1e-20; label = "Z(k) enstrophy", markersize = 10, marker = :diamond)
Mke.axislegend(ax; position = :rt)
fig
```
