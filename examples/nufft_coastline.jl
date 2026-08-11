using FlowFieldSpectra: FlowFieldSpectra as FFS
using FINUFFT: FINUFFT     # activates the NUFFT extension
using FFTW: FFTW           # activates the FFT extension
using CairoMakie: CairoMakie as Mke
using Random: Random
using SpectralBackends: SpectralBackends as SB
using FlowGeometries: FlowGeometries as FG

"""
    run_nufft_coastline_example()

Recover the energy spectrum of a 2D field from a scattered, *masked* point cloud (a synthetic
coastline cutout removes the "land" points) using the non-uniform FFT, and compare against the true
spectrum on the full uniform grid.
"""
function run_nufft_coastline_example()
    println("--- Running NUFFT Jittered + Coastline-Cutout Example ---")
    Random.seed!(42)

    L = 2π
    N = 64
    xs = range(0.0, L; length = N + 1)[1:N]
    ys = range(0.0, L; length = N + 1)[1:N]
    grid = FG.Grids.StructuredGrid(FG.Geometry.CartesianGeometry{Float64}(), xs, ys;
        periodic = (true, true), period = (L, L))
    field(x, y) = cos(3x) + 0.6 * sin(5y) + 0.4 * cos(2x + 4y)
    f_grid = [field(x, y) for x in xs, y in ys]         # (N, N) tensor

    c_fft, k_fft = FFS.calculate_spectrum(grid, f_grid, (N, N); transform = SB.FFTSpectralBackend())
    k_ref, E_ref = FFS.isotropic_spectrum(k_fft, c_fft; num_bins = 24)

    dx = L / N
    xv = vec([x for x in xs, y in ys])
    yv = vec([y for x in xs, y in ys])
    xj = clamp.(xv .+ (rand(length(xv)) .- 0.5) .* (0.4 * dx), 0.0, L)
    yj = clamp.(yv .+ (rand(length(yv)) .- 0.5) .* (0.4 * dx), 0.0, L)
    is_land(x, y) =
        ((x - 0.0)^2 + (y - 0.0)^2 < (0.45L)^2) || (y < 0.18L + 0.10L * sin(2π * x / L * 2))
    ocean = .!is_land.(xj, yj)
    xo = xj[ocean]
    yo = yj[ocean]
    fo = field.(xo, yo)                                 # (N_ocean,) scattered field
    println("Kept $(count(ocean)) / $(length(xj)) points after the coastline cutout.")

    ocean_grid = FG.Grids.UnstructuredGrid(FG.Geometry.CartesianGeometry{Float64}(), (xo, yo), ones(length(xo));
        periodic = (true, true), period = (L, L))
    c_nu, k_nu = FFS.calculate_spectrum(ocean_grid, fo, (N, N); transform = FFS.FINUFFTBackend(), eps = 1e-9)
    k_nu2, E_nu = FFS.isotropic_spectrum(k_nu, c_nu; num_bins = 24)

    fig = Mke.Figure(size = (1200, 500))
    Mke.Label(fig[0, 1:2], "NUFFT Spectrum from a Jittered, Coastline-Masked Cloud", fontsize = 18, font = :bold)
    ax1 = Mke.Axis(fig[1, 1]; title = "Ocean samples (land cut out)", xlabel = "x", ylabel = "y",
        aspect = Mke.DataAspect())
    Mke.scatter!(ax1, xo, yo; color = fo, colormap = :balance, markersize = 4)
    # symlog y-axis: linear through a small threshold near 0, logarithmic above it. The exact FFT's
    # off-peak roundoff-zeros (~1e-32) collapse into the linear region at ≈0 (not stretching the axis
    # over 30 empty decades), while the NUFFT's real ~2-decade leakage-floor→peak gap is shown honestly
    # in the log region. Nothing is clipped and nothing is compressed — all data is visible.
    ax2 = Mke.Axis(fig[1, 2]; title = "Isotropic energy spectrum (masked ⇒ broadband leakage floor)",
        xlabel = "k", ylabel = "E(k)", yscale = Mke.Makie.Symlog10(1.0e-4))
    Mke.lines!(ax2, k_ref, E_ref; color = :black, linewidth = 2, label = "Full grid (FFT)")
    Mke.scatter!(ax2, k_nu2, E_nu; color = :crimson, markersize = 9, label = "Ocean cloud (NUFFT)")
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
