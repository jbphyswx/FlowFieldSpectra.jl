"""
Generate static figure assets for FlowFieldSpectra.jl docs and README.md.

Run from the root directory:
    julia --project=docs/generate_assets docs/generate_assets/generate_assets.jl
"""

using FlowFieldSpectra
using FFTW
using FINUFFT
using FastSphericalHarmonics
using NUFSHT
using CairoMakie
using Statistics
using Random

const ASSETS_DIR = joinpath(@__DIR__, "..", "src", "assets")
mkpath(ASSETS_DIR)

# ─── Figure 1: Cartesian 2D Flow Field Spectrum analysis ──────────────────

function generate_cartesian_figure()
    L = 2π
    N = 64
    dx = L / N
    xs = range(0.0, stop=L-dx, length=N)
    ys = range(0.0, stop=L-dx, length=N)
    
    xv = vec([x for x in xs, y in ys])
    yv = vec([y for x in xs, y in ys])
    
    # Taylor-Green vortex velocities
    u = @. cos(2 * xv) * sin(2 * yv)
    v = @. -sin(2 * xv) * cos(2 * yv)
    
    # Transform using FFTW
    c_fft, k_fft = calculate_spectrum(FFTBackend(), (xv, yv), (u, v), (N, N); domain_size=(L, L))
    
    # 1D radial isotropic reduction
    k_bins, E_k = isotropic_spectrum(k_fft, c_fft; num_bins=32)
    
    # 1D transect reduction along y (integrating out y, leaving x)
    k_red, E_red = transect_spectrum(k_fft, c_fft, (2,))
    
    # Plot
    fig = Figure(size=(1200, 800), fontsize=14)
    Label(fig[0, 1:2], "Cartesian 2D Spectral Analysis — Taylor-Green Vortex", fontsize=18, font=:bold)
    
    # Panel A: Spatial flow field
    ax1 = Axis(fig[1, 1], title="A. Velocity Vectors (u, v)", xlabel="x", ylabel="y", aspect=DataAspect())
    arrows!(ax1, xs[1:4:end], ys[1:4:end], 
            reshape(u, N, N)[1:4:end, 1:4:end], 
            reshape(v, N, N)[1:4:end, 1:4:end], 
            lengthscale=0.5, arrowcolor=:blue, linecolor=:blue)
    
    # Panel B: 2D Fourier energy density grid
    ax2 = Axis(fig[1, 2], title="B. 2D Spectral Energy log10(|C|^2)", xlabel="k_x", ylabel="k_y", aspect=DataAspect())
    energy_2d = log10.(0.5 .* (abs2.(c_fft[:, :, 1]) .+ abs2.(c_fft[:, :, 2])) .+ 1e-15)
    hm = heatmap!(ax2, k_fft[1], k_fft[2], energy_2d, colormap=:viridis)
    Colorbar(fig[1, 3], hm)
    
    # Panel C: 1D Radially integrated & transect spectra
    ax3 = Axis(fig[2, 1:2], title="C. Reduced 1D Energy Spectra", xlabel="Wavenumber k", ylabel="Energy Density E(k)", yscale=log10)
    lines!(ax3, k_bins, E_k, label="Isotropic (Radial Shell Integration)", color=:red, linewidth=2)
    lines!(ax3, k_red[1], E_red, label="Transect (Integrated out y)", color=:blue, linewidth=2, linestyle=:dash)
    axislegend(ax3)
    
    outpath = joinpath(ASSETS_DIR, "cartesian_spectra.png")
    save(outpath, fig)
    println("Saved: $outpath")
end

# ─── Figure 2: Spherical Harmonic degree spectrum ───────────────────────

function generate_spherical_figure()
    lmax = 16
    Nθ = lmax + 1
    Nφ = 2 * lmax + 1
    
    # Clenshaw-Curtis grid
    pts = FastSphericalHarmonics.sph_points(Nθ)
    theta_grid = pts[1]
    phi_grid = pts[2]
    
    theta_nodes = vec([θ for θ in theta_grid, φ in phi_grid])
    phi_nodes = vec([φ for θ in theta_grid, φ in phi_grid])
    
    # Set specific modes: Y_2^1, Y_5^-3, and Y_8^4
    C_true = zeros(Nθ, Nφ)
    C_true[FastSphericalHarmonics.sph_mode(2, 1)] = 1.0
    C_true[FastSphericalHarmonics.sph_mode(5, -3)] = 0.8
    C_true[FastSphericalHarmonics.sph_mode(8, 4)] = 0.5
    
    f_val = vec(FastSphericalHarmonics.sph_evaluate(C_true))
    
    # Compute via SHTBackend
    c_sht, _ = calculate_spectrum(SHTBackend(), (theta_nodes, phi_nodes), (f_val,), (Nθ, Nφ))
    deg, E_l = spherical_energy_spectrum(c_sht)
    
    # Plot
    fig = Figure(size=(1200, 800), fontsize=14)
    Label(fig[0, 1:2], "Spherical Harmonic Transform & Degree Energy Spectrum", fontsize=18, font=:bold)
    
    # Panel A: Scalar field map
    ax1 = Axis(fig[1, 1], title="A. Scalar Field f(θ, φ) on CC Grid", xlabel="Longitude φ (rad)", ylabel="Colatitude θ (rad)")
    hm = heatmap!(ax1, phi_grid, theta_grid, reshape(f_val, Nθ, Nφ)', colormap=:balance)
    Colorbar(fig[1, 2], hm)
    
    # Panel B: Spherical Harmonic degree energy spectrum
    ax2 = Axis(fig[2, 1:2], title="B. Degree Energy Spectrum E(ℓ)", xlabel="Spherical Harmonic Degree ℓ", ylabel="Energy E(ℓ)")
    barplot!(ax2, deg, E_l, width=0.6, color=:darkred)
    xlims!(ax2, -0.5, lmax + 0.5)
    
    outpath = joinpath(ASSETS_DIR, "spherical_spectra.png")
    save(outpath, fig)
    println("Saved: $outpath")
end

# ─── Figure 3: Backend Parity & Error Comparison ──────────────────────────

function generate_parity_figure()
    L = 2π
    N = 32
    dx = L / N
    xs = range(0.0, stop=L-dx, length=N)
    ys = range(0.0, stop=L-dx, length=N)
    xv = vec([x for x in xs, y in ys])
    yv = vec([y for x in xs, y in ys])
    
    u = @. cos(3 * xv) * sin(3 * yv)
    v = @. -sin(3 * xv) * cos(3 * yv)
    
    # Compute via DirectSum
    c_direct, _ = calculate_spectrum(DirectSumBackend(), (xv, yv), (u, v), (N, N); domain_size=(L, L))
    
    # Compute via FFTW
    c_fft, _ = calculate_spectrum(FFTBackend(), (xv, yv), (u, v), (N, N); domain_size=(L, L))
    
    # Compute difference
    diff_u = abs.(c_direct[:, :, 1] .- c_fft[:, :, 1])
    diff_v = abs.(c_direct[:, :, 2] .- c_fft[:, :, 2])
    
    # Plot
    fig = Figure(size=(1200, 450), fontsize=14)
    Label(fig[0, 1:3], "Backend Parity: DirectSum vs FFTW Coefficients", fontsize=18, font=:bold)
    
    ax1 = Axis(fig[1, 1], title="DirectSum Backend u-Coeffs magnitude", aspect=DataAspect())
    hm1 = heatmap!(ax1, abs.(c_direct[:, :, 1]), colormap=:viridis)
    Colorbar(fig[1, 2], hm1)
    
    ax2 = Axis(fig[1, 3], title="FFT Backend u-Coeffs magnitude", aspect=DataAspect())
    hm2 = heatmap!(ax2, abs.(c_fft[:, :, 1]), colormap=:viridis)
    Colorbar(fig[1, 4], hm2)
    
    ax3 = Axis(fig[1, 5], title="Absolute Difference Log10", aspect=DataAspect())
    hm3 = heatmap!(ax3, log10.(diff_u .+ 1e-20), colormap=:inferno)
    Colorbar(fig[1, 6], hm3)
    
    outpath = joinpath(ASSETS_DIR, "backend_parity.png")
    save(outpath, fig)
    println("Saved: $outpath")
end

# ─── Execute ──────────────────────────────────────────────────────────────

println("Generating static figure assets...")
generate_cartesian_figure()
generate_spherical_figure()
generate_parity_figure()
println("Done!")
