# Cross-spectra & flux by scale

The **co-spectrum** `Co_{fg}(k) = Re S_{fg}(k)` distributes a covariance such as the momentum
flux `⟨u'w'⟩` across scales — a staple of boundary-layer and turbulence analysis. Here two
correlated fields share a common large-scale mode plus independent small-scale structure.

```@example cross
using FlowFieldSpectra: FlowFieldSpectra as FFS
using FFTW: FFTW                          # activates the FFTBackend extension
using CairoMakie: CairoMakie as Mke

L = 2π
N = 64
dx = L / N
xs = range(0.0, stop = L - dx, length = N)
xv = vec([x for x in xs, y in xs])
yv = vec([y for x in xs, y in xs])

u = @. cos(2xv) + 0.5 * sin(6yv)             # shared mode at k≈2, own structure at k≈6
w = @. cos(2xv) - 0.4 * cos(9xv)             # shared mode at k≈2, own structure at k≈9

grid = FFS.UniformCartesianGrid((xv, yv); domain_size = (L, L))
cu, ks = FFS.calculate_spectrum(FFS.FFTBackend(), grid, (u,), (N, N))
cw, _ = FFS.calculate_spectrum(FFS.FFTBackend(), grid, (w,), (N, N))

k, Co = FFS.cospectrum(ks, cu, cw; num_bins = 24)

fig = Mke.Figure(size = (680, 430))
ax = Mke.Axis(fig[1, 1]; title = "Co-spectrum Co(k) — flux by scale", xlabel = "k", ylabel = "Co(k)")
Mke.lines!(ax, k, Co; linewidth = 2)
Mke.hlines!(ax, [0.0]; color = :gray, linestyle = :dash)
fig
```

The co-spectrum peaks at the shared scale (`k ≈ 2`) and is near zero where the two fields have
independent structure.
