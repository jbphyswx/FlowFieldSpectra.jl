# Derived-quantity spectra (vorticity, divergence, enstrophy)

`FlowFieldSpectra` forms the kinetic-energy spectrum from a velocity field; the same machinery,
via spectral differentiation (`ik^α`), gives the spectra of **vorticity**, **divergence**, and
**enstrophy**. Here we use a 2D incompressible flow `u = sin x cos y`, `v = -cos x sin y` (so
`∇·u = 0`) whose vorticity is `ω = 2 sin x sin y`.

```@example derived
using FlowFieldSpectra, FFTW, CairoMakie

L = 2π
N = 32
dx = L / N
xs = range(0.0, stop = L - dx, length = N)
xv = vec([x for x in xs, y in xs])
yv = vec([y for x in xs, y in xs])
u = @. sin(xv) * cos(yv)
v = @. -cos(xv) * sin(yv)

grid = UniformCartesianGrid((xv, yv); domain_size = (L, L))
coeffs, ks = calculate_spectrum(FFTBackend(), grid, (u, v), (N, N))

# Divergence is ~0 for incompressible flow; vorticity → enstrophy spectrum.
divc = spectral_divergence(ks, coeffs)
vortc = spectral_vorticity(ks, coeffs)
@assert maximum(abs.(divc)) < 1e-12

k, E = isotropic_spectrum(ks, coeffs; num_bins = 8)   # energy spectrum
_, Z = isotropic_spectrum(ks, vortc; num_bins = 8)    # enstrophy spectrum

fig = Figure(size = (640, 420))
ax = Axis(fig[1, 1]; title = "Energy vs enstrophy spectra (Z = k² E on the active shell)",
    xlabel = "k", ylabel = "spectral density", yscale = log10)
scatter!(ax, k, E .+ 1e-20; label = "E(k) energy", markersize = 10)
scatter!(ax, k, Z .+ 1e-20; label = "Z(k) enstrophy", markersize = 10, marker = :diamond)
axislegend(ax; position = :rt)
fig
```
