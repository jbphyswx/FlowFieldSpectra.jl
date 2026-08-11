using FlowFieldSpectra: FlowFieldSpectra as FFS
using FINUFFT: FINUFFT     # activates the NUFFT extension
using CairoMakie: CairoMakie as Mke
using Random: Random
using SpectralBackends: SpectralBackends as SB
using FlowGeometries: FlowGeometries as FG

"""
    run_horizontal_spectra_4d_example()

Horizontal spectra of a 4D field `f(x, y, z, t)` on a **fixed, non-uniform** horizontal grid — the
bread-and-butter geophysical workflow. The field is a real `(Npts, nz, nt)` tensor: the horizontal
points never move, so the FINUFFT plan is built **once** and the entire `(z, t)` batch transforms in a
single `ntrans` execution. Then ONE batch-preserving `isotropic_spectrum` call returns `E(k, z, t)` —
no per-slice reshape loop.
"""
function run_horizontal_spectra_4d_example()
    println("--- Running 4D Horizontal-Spectra (fixed nonuniform grid) Example ---")
    Random.seed!(42)

    L = 2π
    N = 48
    nz, nt = 3, 4
    ms = (N, N)
    npts = N * N

    # Fixed non-uniform horizontal sample locations.
    xv = rand(npts) .* L
    yv = rand(npts) .* L
    hgrid = FG.Grids.UnstructuredGrid(FG.Geometry.CartesianGeometry{Float64}(), (xv, yv), ones(npts);
        periodic = (true, true), period = (L, L))

    # f(x, y, z, t) as a real (Npts, nz, nt) tensor — the (z, t) batch is the trailing axes.
    f = Array{Float64}(undef, npts, nz, nt)
    kz = range(2, 6; length = nz)                     # dominant horizontal wavenumber per level
    ts = range(0, 1; length = nt)
    for (it, t) in enumerate(ts), (iz, k0) in enumerate(kz)
        @. f[:, iz, it] = cos(k0 * xv + 2π * t) + 0.5 * sin((k0 + 1) * yv)
    end

    # Build the plan ONCE for the fixed points; transform the whole (z, t) batch in one exec.
    plan = FFS.plan_spectrum(hgrid, Float64, ms; transform = FFS.FINUFFTBackend(), batch = (nz, nt), eps = 1e-9)
    coeffs = zeros(ComplexF64, ms..., nz, nt)         # (N, N, nz, nt)
    ks = FFS.calculate_spectrum!(coeffs, plan, f)

    # ONE reduction → E(k, z, t), batch dims preserved. Average over time → E(k, z).
    nbins = 18
    kbins, E = FFS.isotropic_spectrum(ks, coeffs; num_bins = nbins)   # (nbins, nz, nt)
    Ekz = dropdims(sum(E; dims = 3); dims = 3) ./ nt                  # (nbins, nz)

    fig = Mke.Figure(size = (1150, 470))
    Mke.Label(fig[0, 1:2], "Horizontal Spectra of f(x,y,z,t) on a Fixed Nonuniform Grid",
        fontsize = 18, font = :bold)

    ax1 = Mke.Axis(fig[1, 1]; title = "Fixed horizontal sample locations", xlabel = "x", ylabel = "y",
        aspect = Mke.DataAspect())
    Mke.scatter!(ax1, xv, yv; markersize = 3, color = :steelblue)

    ax2 = Mke.Axis(fig[1, 2]; title = "log₁₀ E(k, z): peak scale migrates with height",
        xlabel = "horizontal wavenumber k", ylabel = "z-level")
    hm = Mke.heatmap!(ax2, kbins, 1:nz, log10.(Ekz .+ 1e-12); colormap = :viridis)
    Mke.Colorbar(fig[1, 3], hm; label = "log₁₀ E(k)")
    ax2.yticks = 1:nz

    outpath = joinpath(@__DIR__, "horizontal_spectra_4d.png")
    Mke.save(outpath, fig)
    println("Transformed $(nz * nt) (z,t) slices with ONE plan build and ONE reduction. Saved: ", outpath)
    println("Example run successfully!")
    return fig
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_horizontal_spectra_4d_example()
end
