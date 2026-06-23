using FlowFieldSpectra: FlowFieldSpectra as FFS
using FINUFFT: FINUFFT     # loaded to activate the NUFFTBackend extension
using FFTW: FFTW           # loaded to activate the FFTBackend extension
using CairoMakie: CairoMakie as Mke
using Random: Random

"""
    run_nufft_coastline_example()

Recover the energy spectrum of a synthetic 2D field from a scattered, *masked* point cloud
using the non-uniform FFT (FINUFFT). This mirrors a common observational situation: data live on
jittered (non-gridded) locations and a chunk of the domain is missing — here an analytic
"coastline" cutout (a corner island plus a sinusoidal coast) removes the "land" points, leaving
only "ocean" samples. We then compare the NUFFT spectrum from the irregular ocean cloud against
the true spectrum computed on the full uniform grid by FFT.
"""
function run_nufft_coastline_example()
    println("--- Running NUFFT Jittered + Coastline-Cutout Example ---")
    Random.seed!(42)

    # 1. Synthetic field on a uniform reference grid: a few known Fourier modes.
    L = 2π
    N = 64
    dx = L / N
    xs = range(0.0, stop = L - dx, length = N)
    xv = vec([x for x in xs, y in xs])
    yv = vec([y for x in xs, y in xs])

    field(x, y) = cos(3x) + 0.6 * sin(5y) + 0.4 * cos(2x + 4y)
    f_grid = field.(xv, yv)

    # Reference spectrum: FFT on the full uniform grid.
    grid = FFS.UniformCartesianGrid((xv, yv); domain_size = (L, L))
    c_fft, k_fft = FFS.calculate_spectrum(FFS.FFTBackend(), grid, (f_grid,), (N, N))
    k_ref, E_ref = FFS.isotropic_spectrum(k_fft, c_fft; num_bins = 24)

    # 2. Jitter the sample locations off the grid (non-uniform sampling).
    xj = clamp.(xv .+ (rand(length(xv)) .- 0.5) .* (0.4 * dx), 0.0, L)
    yj = clamp.(yv .+ (rand(length(yv)) .- 0.5) .* (0.4 * dx), 0.0, L)

    # 3. Synthetic coastline mask: "land" = a corner island OR below a sinusoidal coast.
    is_land(x, y) =
        ((x - 0.0)^2 + (y - 0.0)^2 < (0.45L)^2) ||
        (y < 0.18L + 0.10L * sin(2π * x / L * 2))
    ocean = .!is_land.(xj, yj)

    xo = xj[ocean]
    yo = yj[ocean]
    fo = field.(xo, yo)
    println("Kept $(count(ocean)) / $(length(xj)) points after the coastline cutout.")

    # 4. NUFFT on the irregular ocean-only cloud → retrieved spectrum.
    ocean_grid = FFS.ScatteredCartesianGrid((xo, yo); domain_size = (L, L))
    c_nufft, k_nufft = FFS.calculate_spectrum(FFS.NUFFTBackend(), ocean_grid, (fo,), (N, N); eps = 1e-9)
    k_nu, E_nu = FFS.isotropic_spectrum(k_nufft, c_nufft; num_bins = 24)

    # 5. Figure: masked sample cloud + retrieved vs reference spectrum.
    fig = Mke.Figure(size = (1200, 500))
    Mke.Label(fig[0, 1:2], "NUFFT Spectrum from a Jittered, Coastline-Masked Cloud",
        fontsize = 18, font = :bold)

    ax1 = Mke.Axis(fig[1, 1]; title = "Ocean samples (land cut out)", xlabel = "x", ylabel = "y",
        aspect = Mke.DataAspect())
    Mke.scatter!(ax1, xo, yo; color = fo, colormap = :balance, markersize = 4)

    ax2 = Mke.Axis(fig[1, 2]; title = "Isotropic energy spectrum", xlabel = "k", ylabel = "E(k)",
        yscale = log10)
    Mke.lines!(ax2, k_ref, E_ref .+ 1e-20; color = :black, linewidth = 2, label = "Full grid (FFT)")
    Mke.scatter!(ax2, k_nu, E_nu .+ 1e-20; color = :crimson, markersize = 9,
        label = "Ocean cloud (NUFFT)")
    Mke.axislegend(ax2; position = :rt)

    outpath = joinpath(@__DIR__, "nufft_coastline.png")
    Mke.save(outpath, fig)
    println("Saved figure: ", outpath)
    println("Example run successfully!")
    return fig
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_nufft_coastline_example()
end
