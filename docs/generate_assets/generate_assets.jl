"""
Generate static figure assets for FlowFieldSpectra.jl docs and README.md.

Run from the root directory:
    julia --project=docs/generate_assets docs/generate_assets/generate_assets.jl
"""

using FlowFieldSpectra: FlowFieldSpectra as FFS
using FFTW: FFTW                                       # activates the FFTBackend extension
using FastSphericalHarmonics: FastSphericalHarmonics as FSH
using CairoMakie: CairoMakie as Mke

const ASSETS_DIR = joinpath(@__DIR__, "..", "src", "assets")
mkpath(ASSETS_DIR)

# ─── Figure 1: Cartesian 2D Flow Field Spectrum analysis ──────────────────

function generate_cartesian_figure()
    L = 2π
    N = 64
    dx = L / N
    xs = range(0.0, stop = L - dx, length = N)
    ys = range(0.0, stop = L - dx, length = N)

    xv = vec([x for x in xs, y in ys])
    yv = vec([y for x in xs, y in ys])

    # Taylor-Green vortex velocities
    u = @. cos(2 * xv) * sin(2 * yv)
    v = @. -sin(2 * xv) * cos(2 * yv)

    cart_grid = FFS.UniformCartesianGrid((xv, yv); domain_size = (L, L))
    c_fft, k_fft = FFS.calculate_spectrum(FFS.FFTBackend(), cart_grid, (u, v), (N, N))

    k_bins, E_k = FFS.isotropic_spectrum(k_fft, c_fft; num_bins = 32)
    k_red, E_red = FFS.transect_spectrum(k_fft, c_fft, (2,))

    fig = Mke.Figure(size = (1200, 800), fontsize = 14)
    Mke.Label(fig[0, 1:2], "Cartesian 2D Spectral Analysis — Taylor-Green Vortex",
        fontsize = 18, font = :bold)

    ax1 = Mke.Axis(fig[1, 1], title = "A. Velocity Vectors (u, v)", xlabel = "x", ylabel = "y",
        aspect = Mke.DataAspect())
    Mke.arrows!(ax1, xs[1:4:end], ys[1:4:end],
        reshape(u, N, N)[1:4:end, 1:4:end], reshape(v, N, N)[1:4:end, 1:4:end];
        lengthscale = 0.5, arrowcolor = :blue, linecolor = :blue)

    ax2 = Mke.Axis(fig[1, 2], title = "B. 2D Spectral Energy log10(|C|^2)", xlabel = "k_x",
        ylabel = "k_y", aspect = Mke.DataAspect())
    energy_2d = log10.(0.5 .* (abs2.(c_fft[:, :, 1]) .+ abs2.(c_fft[:, :, 2])) .+ 1e-15)
    emax = maximum(energy_2d)
    hm = Mke.heatmap!(ax2, k_fft[1], k_fft[2], energy_2d; colormap = :viridis,
        colorrange = (emax - 8, emax))
    Mke.Colorbar(fig[1, 3], hm)

    ax3 = Mke.Axis(fig[2, 1:2], title = "C. Reduced 1D Energy Spectra", xlabel = "Wavenumber k",
        ylabel = "Energy Density E(k)", yscale = log10)
    Mke.lines!(ax3, k_bins, E_k .+ 1e-20; label = "Isotropic (radial shell integration)",
        color = :red, linewidth = 2)
    Mke.lines!(ax3, k_red[1], E_red .+ 1e-20; label = "Transect (integrated out y)",
        color = :blue, linewidth = 2, linestyle = :dash)
    Mke.axislegend(ax3)

    outpath = joinpath(ASSETS_DIR, "cartesian_spectra.png")
    Mke.save(outpath, fig)
    println("Saved: $outpath")
end

# ─── Figure 2: Spherical Harmonic degree spectrum ───────────────────────

function generate_spherical_figure()
    lmax = 16
    Nθ = lmax + 1
    Nφ = 2 * lmax + 1

    pts = FSH.sph_points(Nθ)
    theta_grid = pts[1]
    phi_grid = pts[2]
    theta_nodes = vec([θ for θ in theta_grid, φ in phi_grid])
    phi_nodes = vec([φ for θ in theta_grid, φ in phi_grid])

    # Set specific modes: Y_2^1, Y_5^-3, and Y_8^4
    C_true = zeros(Nθ, Nφ)
    C_true[FSH.sph_mode(2, 1)] = 1.0
    C_true[FSH.sph_mode(5, -3)] = 0.8
    C_true[FSH.sph_mode(8, 4)] = 0.5
    f_val = vec(FSH.sph_evaluate(C_true))

    sht_grid = FFS.StructuredSphericalGrid(theta_nodes, phi_nodes)
    c_sht, _ = FFS.calculate_spectrum(FFS.SHTBackend(), sht_grid, (f_val,), (Nθ, Nφ))
    deg, E_l = FFS.spherical_energy_spectrum(c_sht)

    fig = Mke.Figure(size = (1200, 800), fontsize = 14)
    Mke.Label(fig[0, 1:2], "Spherical Harmonic Transform & Degree Energy Spectrum",
        fontsize = 18, font = :bold)

    ax1 = Mke.Axis(fig[1, 1], title = "A. Scalar Field f(θ, φ) on CC Grid",
        xlabel = "Longitude φ (rad)", ylabel = "Colatitude θ (rad)")
    hm = Mke.heatmap!(ax1, phi_grid, theta_grid, reshape(f_val, Nθ, Nφ)'; colormap = :balance)
    Mke.Colorbar(fig[1, 2], hm)

    ax2 = Mke.Axis(fig[2, 1:2], title = "B. Degree Energy Spectrum E(ℓ)",
        xlabel = "Spherical Harmonic Degree ℓ", ylabel = "Energy E(ℓ)")
    Mke.barplot!(ax2, deg, E_l; width = 0.6, color = :darkred)
    Mke.xlims!(ax2, -0.5, lmax + 0.5)

    outpath = joinpath(ASSETS_DIR, "spherical_spectra.png")
    Mke.save(outpath, fig)
    println("Saved: $outpath")
end

# ─── Figure 3: Backend Parity & Error Comparison ──────────────────────────

function generate_parity_figure()
    L = 2π
    N = 32
    dx = L / N
    xs = range(0.0, stop = L - dx, length = N)
    ys = range(0.0, stop = L - dx, length = N)
    xv = vec([x for x in xs, y in ys])
    yv = vec([y for x in xs, y in ys])

    u = @. cos(3 * xv) * sin(3 * yv)
    v = @. -sin(3 * xv) * cos(3 * yv)

    parity_grid = FFS.UniformCartesianGrid((xv, yv); domain_size = (L, L))
    c_direct, _ = FFS.calculate_spectrum(FFS.DirectSumBackend(), parity_grid, (u, v), (N, N))
    c_fft, _ = FFS.calculate_spectrum(FFS.FFTBackend(), parity_grid, (u, v), (N, N))

    diff_u = abs.(c_direct[:, :, 1] .- c_fft[:, :, 1])

    fig = Mke.Figure(size = (1200, 450), fontsize = 14)
    Mke.Label(fig[0, 1:3], "Backend Parity: DirectSum vs FFTW Coefficients",
        fontsize = 18, font = :bold)

    ax1 = Mke.Axis(fig[1, 1], title = "DirectSum u-coeffs magnitude", aspect = Mke.DataAspect())
    hm1 = Mke.heatmap!(ax1, abs.(c_direct[:, :, 1]); colormap = :viridis)
    Mke.Colorbar(fig[1, 2], hm1)

    ax2 = Mke.Axis(fig[1, 3], title = "FFT u-coeffs magnitude", aspect = Mke.DataAspect())
    hm2 = Mke.heatmap!(ax2, abs.(c_fft[:, :, 1]); colormap = :viridis)
    Mke.Colorbar(fig[1, 4], hm2)

    ax3 = Mke.Axis(fig[1, 5], title = "Absolute difference log10", aspect = Mke.DataAspect())
    hm3 = Mke.heatmap!(ax3, log10.(diff_u .+ 1e-20); colormap = :inferno)
    Mke.Colorbar(fig[1, 6], hm3)

    outpath = joinpath(ASSETS_DIR, "backend_parity.png")
    Mke.save(outpath, fig)
    println("Saved: $outpath")
end

# ─── Execute ──────────────────────────────────────────────────────────────

println("Generating static figure assets...")
generate_cartesian_figure()
generate_spherical_figure()
generate_parity_figure()
println("Done!")
