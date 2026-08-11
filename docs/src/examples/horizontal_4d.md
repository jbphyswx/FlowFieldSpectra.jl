# Horizontal spectra of a 4D field on a fixed grid

The defining workflow: horizontal `(x, y)` spectra of a field `f(x, y, z, t)` sampled on a
**fixed, non-uniform** horizontal grid. Because the horizontal points never move, the FINUFFT
plan and point-sorting are built **once** and the whole `z × t` stack is transformed in a single
batched execution — then the plan is reused across the time loop. This is the fast path for
per-level / per-time spectra of large geophysical datasets.

```julia
using FlowFieldSpectra: FlowFieldSpectra as FFS
using FINUFFT: FINUFFT                    # activates the NUFFT extension
using Random: Random
using SpectralBackends: SpectralBackends as SB
using FlowGeometries: FlowGeometries as FG
Random.seed!(42)

L = 2π
N = 48
nz, nt = 3, 4
ms = (N, N)
npts = N * N

# Fixed non-uniform horizontal locations — an unstructured Cartesian grid.
xv = rand(npts) .* L
yv = rand(npts) .* L
hgrid = FG.Grids.UnstructuredGrid(FG.Geometry.CartesianGeometry{Float64}(), (xv, yv), ones(npts);
    periodic = (true, true), period = (L, L))

# f(x, y, z, t) as a real (Npts, nz, nt) tensor — the (z, t) batch is the trailing axes.
f = Array{Float64}(undef, npts, nz, nt)
kz = range(2, 6; length = nz)             # dominant horizontal wavenumber per level
ts = range(0, 1; length = nt)
for (it, t) in enumerate(ts), (iz, k0) in enumerate(kz)
    @. f[:, iz, it] = cos(k0 * xv + 2π * t) + 0.5 * sin((k0 + 1) * yv)
end

# ONE plan build for the fixed points; transform the whole (z, t) batch in a single exec.
plan = FFS.plan_spectrum(hgrid, Float64, ms; transform = FFS.FINUFFTBackend(), batch = (nz, nt), eps = 1e-9)
coeffs = zeros(ComplexF64, ms..., nz, nt)     # (N, N, nz, nt)
ks = FFS.calculate_spectrum!(coeffs, plan, f)

# ONE batch-preserving reduction → E(k, z, t); average over time → E(k, z).
nbins = 18
kbins, E = FFS.isotropic_spectrum(ks, coeffs; num_bins = nbins)   # (nbins, nz, nt)
Ekz = dropdims(sum(E; dims = 3); dims = 3) ./ nt                  # (nbins, nz)
```
