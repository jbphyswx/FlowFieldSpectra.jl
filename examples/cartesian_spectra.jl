using FlowFieldSpectra: FlowFieldSpectra as FFS
using FFTW: FFTW           # activates the FFTBackend extension
using FINUFFT: FINUFFT     # activates the NUFFTBackend extension
using CairoMakie: CairoMakie as Mke
using Random: Random

function run_cartesian_example()
    println("--- Running Cartesian Grid Spectra Example ---")

    # 1. Uniform Cartesian grid — defined by its axes (ranges), fields are plain (Nx, Ny) tensors.
    L = 2π
    N = 64
    grid = FFS.UniformCartesianGrid(; domain = (L, L), n = (N, N))
    xs, ys = grid.axes

    # Taylor–Green vortex velocity field (u, v) as tensors — no flattening.
    u = [cos(2x) * sin(2y) for x in xs, y in ys]
    v = [-sin(2x) * cos(2y) for x in xs, y in ys]

    # 2. FFT spectrum. Pass (u, v) — a new trailing component axis; coeffs are (N, N, 2).
    println("Computing uniform Cartesian spectrum via FFTW...")
    c_fft, k_fft = FFS.calculate_spectrum(grid, (u, v), (N, N); transform = FFS.FFTBackend())
    # Kinetic-energy isotropic spectrum: fold the component axis (dim 3) into the energy.
    k_bins, E_k = FFS.isotropic_spectrum(k_fft, c_fft; num_bins = 32, dims = 3)

    # 3. NUFFT on jittered (scattered) samples of the same field.
    println("Computing non-uniform Cartesian spectrum via FINUFFT...")
    Random.seed!(42)
    dx = L / N
    xv = vec([x for x in xs, y in ys])
    yv = vec([y for x in xs, y in ys])
    xj = xv .+ (rand(N^2) .- 0.5) .* (0.2 * dx)
    yj = yv .+ (rand(N^2) .- 0.5) .* (0.2 * dx)
    us = @. cos(2 * xj) * sin(2 * yj)
    vs = @. -sin(2 * xj) * cos(2 * yj)
    gscat = FFS.ScatteredCartesianGrid((xj, yj); domain_size = (L, L))
    c_nu, k_nu = FFS.calculate_spectrum(gscat, (us, vs), (N, N); transform = FFS.NUFFTBackend())
    k_bins_s, E_k_s = FFS.isotropic_spectrum(k_nu, c_nu; num_bins = 32, dims = 3)

    # 4. Figure.
    fig = Mke.Figure(size = (1200, 800))
    Mke.Label(fig[0, 1:2], "Cartesian 2D Spectral Analysis", fontsize = 18, font = :bold)

    ax1 = Mke.Axis(fig[1, 1], title = "Taylor–Green Vortex Velocities", xlabel = "x", ylabel = "y",
        aspect = Mke.DataAspect())
    Mke.arrows!(ax1, xs[1:4:end], ys[1:4:end], u[1:4:end, 1:4:end], v[1:4:end, 1:4:end];
        lengthscale = 0.5, arrowcolor = :blue, linecolor = :blue)

    ax2 = Mke.Axis(fig[1, 2], title = "2D Energy Density log₁₀(½|C|²)", xlabel = "k_x", ylabel = "k_y",
        aspect = Mke.DataAspect())
    energy_2d = log10.(0.5 .* (abs2.(c_fft[:, :, 1]) .+ abs2.(c_fft[:, :, 2])) .+ 1e-15)
    emax = maximum(energy_2d)
    hm = Mke.heatmap!(ax2, k_fft[1], k_fft[2], energy_2d; colormap = :viridis, colorrange = (emax - 8, emax))
    Mke.Colorbar(fig[1, 3], hm)

    ax3 = Mke.Axis(fig[2, 1:2], title = "1D Isotropic Energy Spectrum", xlabel = "k (magnitude)",
        ylabel = "E(k)", yscale = log10)
    floor_y = 1e-8
    Mke.lines!(ax3, k_bins, max.(E_k, floor_y); label = "Uniform grid (FFT, exact)", color = :black, linewidth = 2)
    Mke.scatter!(ax3, k_bins_s, max.(E_k_s, floor_y); label = "Scattered grid (NUFFT)", color = :crimson, markersize = 8)
    Mke.ylims!(ax3, floor_y, 10 * maximum(E_k))
    Mke.axislegend(ax3)

    outpath = joinpath(@__DIR__, "cartesian_spectra.png")
    Mke.save(outpath, fig)
    println("Saved figure: ", outpath)
    println("Example run successfully!")
    return fig
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_cartesian_example()
end
