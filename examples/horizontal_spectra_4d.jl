using FlowFieldSpectra: FlowFieldSpectra as FFS
using FINUFFT: FINUFFT     # loaded to activate the NUFFTBackend extension
using CairoMakie: CairoMakie as Mke
using Random: Random

"""
    run_horizontal_spectra_4d_example()

Horizontal spectra of a 4D field `f(x, y, z, t)` sampled on a **fixed, non-uniform** horizontal
`(x, y)` grid — the bread-and-butter geophysical/turbulence workflow. Because the horizontal
points never move, the FINUFFT plan and point sorting are built **once** and the entire
`z × t` stack is transformed in a single batched execution, then reused across the time loop.
We recover `E(k, z, t)` and plot the spectrum at each level/time.
"""
function run_horizontal_spectra_4d_example()
    println("--- Running 4D Horizontal-Spectra (fixed nonuniform grid) Example ---")
    Random.seed!(42)

    L = 2π
    N = 48                         # horizontal points per axis (scattered)
    nz, nt = 3, 4                  # vertical levels, time steps
    ms = (N, N)

    # Fixed non-uniform horizontal sample locations.
    xv = rand(N * N) .* L
    yv = rand(N * N) .* L
    hgrid = FFS.ScatteredCartesianGrid((xv, yv); domain_size = (L, L))
    npts = length(xv)

    # Synthesize f(x,y,z,t): a horizontal wave whose dominant scale sharpens with height z
    # and drifts in time t. Packed as (npoints, nz*nt) — the batch (z,t) is the trailing axis.
    nb = nz * nt
    stack = Array{Float64}(undef, npts, nb)
    kz = range(2, 6; length = nz)               # dominant horizontal wavenumber per level
    for (it, t) in enumerate(range(0, 1; length = nt)), (iz, k0) in enumerate(kz)
        b = (it - 1) * nz + iz
        @. stack[:, b] = cos(k0 * xv + 2π * t) + 0.5 * sin((k0 + 1) * yv)
    end

    # Build the plan ONCE for the fixed points, transform the whole z*t stack in one exec.
    plan = FFS.plan_spectrum(FFS.NUFFTBackend(), hgrid, Float64, ms; n_transf = nb, eps = 1e-9)
    coeffs = zeros(ComplexF64, ms..., nb)
    ks = FFS.calculate_spectrum!(coeffs, plan, stack)

    # Reduce each (z,t) slice to an isotropic spectrum E(k) — gives E(k, z, t).
    nbins = 18
    E = Array{Float64}(undef, nbins, nb)
    kbins = nothing
    for b in 1:nb
        slice = reshape(view(coeffs, :, :, b), ms..., 1)
        kb, Ek = FFS.isotropic_spectrum(ks, slice; num_bins = nbins)
        kbins = kb
        E[:, b] .= Ek
    end

    # Average over time for a clean E(k, z); the dominant horizontal scale migrates with height.
    Ekz = reshape(E, nbins, nz, nt)
    Ekz_t = dropdims(sum(Ekz; dims = 3); dims = 3) ./ nt    # E(k, z)

    fig = Mke.Figure(size = (1150, 470))
    Mke.Label(fig[0, 1:2], "Horizontal Spectra of f(x,y,z,t) on a Fixed Nonuniform Grid",
        fontsize = 18, font = :bold)

    ax1 = Mke.Axis(fig[1, 1]; title = "Fixed horizontal sample locations", xlabel = "x", ylabel = "y",
        aspect = Mke.DataAspect())
    Mke.scatter!(ax1, xv, yv; markersize = 3, color = :steelblue)

    # Heatmap E(k) vs height: a bright ridge migrating to higher k shows the peak scale
    # sharpening with altitude — recovered for every level/time from ONE plan build.
    ax2 = Mke.Axis(fig[1, 2]; title = "log₁₀ E(k, z): peak scale migrates with height",
        xlabel = "horizontal wavenumber k", ylabel = "z-level")
    hm = Mke.heatmap!(ax2, kbins, 1:nz, log10.(Ekz_t' .+ 1e-12)'; colormap = :viridis)
    Mke.Colorbar(fig[1, 3], hm; label = "log₁₀ E(k)")
    ax2.yticks = 1:nz

    outpath = joinpath(@__DIR__, "horizontal_spectra_4d.png")
    Mke.save(outpath, fig)
    println("Transformed $(nb) (z,t) slices with ONE plan build. Saved figure: ", outpath)
    println("Example run successfully!")
    return fig
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_horizontal_spectra_4d_example()
end
