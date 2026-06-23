using FlowFieldSpectra
using FFTW
using FINUFFT
using CairoMakie
using Statistics
using Random

function run_cartesian_example()
    println("--- Running Cartesian Grid Spectra Example ---")
    
    # 1. Setup Cartesian Domain
    L = 2π
    N = 64
    dx = L / N
    xs = range(0.0, stop=L-dx, length=N)
    ys = range(0.0, stop=L-dx, length=N)
    
    # Grid coordinates
    xv = vec([x for x in xs, y in ys])
    yv = vec([y for x in xs, y in ys])
    
    # Taylor-Green vortex velocity field
    u = @. cos(2 * xv) * sin(2 * yv)
    v = @. -sin(2 * xv) * cos(2 * yv)
    
    # 2. Compute via FFTW
    println("Computing uniform Cartesian spectrum via FFTW...")
    grid = UniformCartesianGrid((xv, yv); domain_size=(L, L))
    c_fft, k_fft = calculate_spectrum(FFTBackend(), grid, (u, v), (N, N))
    
    # Isotropic 1D reduction
    k_bins, E_k = isotropic_spectrum(k_fft, c_fft; num_bins=32)
    println("Isotropic spectrum binned into ", length(k_bins), " bins.")
    
    # Transect reduction (along first dimension)
    k_red, E_red = transect_spectrum(k_fft, c_fft, (2,))
    
    # 3. Compute via FINUFFT (on scattered points)
    println("Computing non-uniform Cartesian spectrum via FINUFFT...")
    Random.seed!(42)
    N_scat = N^2
    # Add random jitter to coordinates
    xv_scat = xv .+ (rand(N_scat) .- 0.5) .* (0.2 * dx)
    yv_scat = yv .+ (rand(N_scat) .- 0.5) .* (0.2 * dx)
    
    u_scat = @. cos(2 * xv_scat) * sin(2 * yv_scat)
    v_scat = @. -sin(2 * xv_scat) * cos(2 * yv_scat)
    
    grid_scat = ScatteredCartesianGrid((xv_scat, yv_scat); domain_size=(L, L))
    c_nufft, k_nufft = calculate_spectrum(NUFFTBackend(), grid_scat, (u_scat, v_scat), (N, N))
    k_bins_scat, E_k_scat = isotropic_spectrum(k_nufft, c_nufft; num_bins=32)
    
    # 4. Generate Plot
    fig = Figure(size=(1200, 800))
    Label(fig[0, 1:2], "Cartesian 2D Spectral Analysis", fontsize=18, font=:bold)
    
    # Panel A: Spatial flow field
    ax1 = Axis(fig[1, 1], title="Taylor-Green Vortex Velocities", xlabel="x", ylabel="y", aspect=DataAspect())
    arrows!(ax1, xs[1:4:end], ys[1:4:end], 
            reshape(u, N, N)[1:4:end, 1:4:end], 
            reshape(v, N, N)[1:4:end, 1:4:end], 
            lengthscale=0.5, arrowcolor=:blue, linecolor=:blue)
    
    # Panel B: 2D Fourier energy density grid. The Taylor-Green vortex lives at a handful of
    # modes; clamp the colour range so they stand out against a clean background.
    ax2 = Axis(fig[1, 2], title="2D Energy Density log10(|C|²)", xlabel="k_x", ylabel="k_y", aspect=DataAspect())
    energy_2d = log10.(0.5 .* (abs2.(c_fft[:, :, 1]) .+ abs2.(c_fft[:, :, 2])) .+ 1e-15)
    kx_grid = k_fft[1]
    ky_grid = k_fft[2]
    emax = maximum(energy_2d)
    hm = heatmap!(ax2, kx_grid, ky_grid, energy_2d, colormap=:viridis, colorrange=(emax - 8, emax))
    Colorbar(fig[1, 3], hm)

    # Panel C: 1D radially-integrated spectrum. The uniform FFT is exact (its off-mode tail is at
    # machine precision); the scattered NUFFT recovers the same peak above a non-uniform-sampling
    # leakage floor. Clamp the y-range so both — not the 1e-30 FFT tail — are legible.
    ax3 = Axis(fig[2, 1:2], title="1D Isotropic Energy Spectrum", xlabel="k (magnitude)", ylabel="E(k)", yscale=log10)
    floor_y = 1e-8
    lines!(ax3, k_bins, max.(E_k, floor_y), label="Uniform grid (FFT, exact)", color=:black, linewidth=2)
    scatter!(ax3, k_bins_scat, max.(E_k_scat, floor_y), label="Scattered grid (NUFFT)", color=:crimson, markersize=8)
    ylims!(ax3, floor_y, 10 * maximum(E_k))
    axislegend(ax3)
    
    # Save Figure
    outpath = joinpath(@__DIR__, "cartesian_spectra.png")
    save(outpath, fig)
    println("Saved figure: ", outpath)
    println("Example run successfully!")
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_cartesian_example()
end
