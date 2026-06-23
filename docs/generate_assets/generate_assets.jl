"""
Generate static figure assets for FlowFieldSpectra.jl docs and README.md.

Run from the root directory:
    julia --project=docs/generate_assets docs/generate_assets/generate_assets.jl
"""

using FlowFieldSpectra: FlowFieldSpectra as FFS
using FFTW: FFTW                                       # activates the FFTBackend extension
using FINUFFT: FINUFFT                                 # activates the NUFFTBackend extension
using FastSphericalHarmonics: FastSphericalHarmonics as FSH
using CairoMakie: CairoMakie as Mke
import Random

const ASSETS_DIR = joinpath(@__DIR__, "..", "src", "assets")
mkpath(ASSETS_DIR)

# ─── Synthetic broadband fields ────────────────────────────────────────────
# Discrete-mode test fields give spiky, delta-like spectra that are hard to read. For showcase
# figures we instead synthesize fields with a *broadband* power-law spectrum, so the recovered
# spectra are the recognizable straight lines (on log–log) that practitioners expect.

# Integer DFT frequencies (cycles) for an N-length axis, in FFTW order: 0,1,…,⌈N/2⌉-1,-⌊N/2⌋,…,-1.
_dftfreq(N) = [0:(cld(N, 2) - 1); -fld(N, 2):-1]

# Real scalar field on an N×N grid whose shell-integrated spectrum is E(k) ∝ k^{-slope}; `aniso`
# stretches kₓ to bias energy toward one direction (aniso = 1 is isotropic). Built by coloring
# white noise in Fourier space, so the field is smooth and statistically homogeneous.
function synthetic_field(N; slope = 5 / 3, aniso = 1.0, seed = 0)
    Random.seed!(seed)
    ŵ = FFTW.fft(randn(N, N))
    fr = _dftfreq(N)
    f̂ = similar(ŵ)
    @inbounds for j in 1:N, i in 1:N
        kk = sqrt((aniso * fr[i])^2 + (fr[j] / aniso)^2)
        f̂[i, j] = kk == 0 ? zero(eltype(ŵ)) : ŵ[i, j] * kk^(-(slope + 1) / 2)
    end
    return real(FFTW.ifft(f̂))
end

# Incompressible velocity (u, v) and its vorticity ω from a broadband streamfunction ψ:
# u = ∂ψ/∂y, v = -∂ψ/∂x, ω = -∇²ψ (spectral derivatives). With ψ chosen so the velocity energy
# spectrum is E(k) ∝ k^{-5/3}, the enstrophy spectrum is Z(k) = k² E(k) ∝ k^{+1/3}.
function synthetic_incompressible(N, L; seed = 0)
    ψ = synthetic_field(N; slope = 5 / 3 + 2, seed = seed)
    ψ̂ = FFTW.fft(ψ)
    fr = _dftfreq(N) .* (2π / L)
    û = similar(ψ̂)
    v̂ = similar(ψ̂)
    ω̂ = similar(ψ̂)
    @inbounds for j in 1:N, i in 1:N
        kx = fr[i]
        ky = fr[j]
        û[i, j] = im * ky * ψ̂[i, j]
        v̂[i, j] = -im * kx * ψ̂[i, j]
        ω̂[i, j] = (kx^2 + ky^2) * ψ̂[i, j]
    end
    return real(FFTW.ifft(û)), real(FFTW.ifft(v̂)), real(FFTW.ifft(ω̂))
end

# ─── Figure 1: Cartesian 2D Flow Field Spectrum analysis ──────────────────

function generate_cartesian_figure()
    L = 2π
    N = 128
    dx = L / N
    xs = range(0.0, stop = L - dx, length = N)
    xv = vec([x for x in xs, y in xs])
    yv = vec([y for x in xs, y in xs])

    # Broadband, incompressible "turbulence" with a k^{-5/3} energy cascade.
    u, v, _ = synthetic_incompressible(N, L; seed = 7)
    uv = (vec(u), vec(v))

    cart_grid = FFS.UniformCartesianGrid((xv, yv); domain_size = (L, L))
    c_fft, k_fft = FFS.calculate_spectrum(FFS.FFTBackend(), cart_grid, uv, (N, N))
    k_bins, E_k = FFS.isotropic_spectrum(k_fft, c_fft; num_bins = 40)

    fig = Mke.Figure(size = (1500, 460), fontsize = 15)
    Mke.Label(fig[0, 1:3], "Cartesian 2D spectra — synthetic turbulence with a k⁻⁵ᐟ³ cascade",
        fontsize = 19, font = :bold)

    ax1 = Mke.Axis(fig[1, 1]; title = "A. A velocity component (the flow)", xlabel = "x",
        ylabel = "y", aspect = Mke.DataAspect())
    umax = maximum(abs, u)
    hm1 = Mke.heatmap!(ax1, xs, xs, u; colormap = :balance, colorrange = (-umax, umax))
    Mke.Colorbar(fig[1, 2], hm1)

    ax2 = Mke.Axis(fig[1, 3]; title = "B. 2D spectral energy  log₁₀|C|²", xlabel = "kₓ",
        ylabel = "k_y", aspect = Mke.DataAspect())
    energy_2d = log10.(0.5 .* (abs2.(c_fft[:, :, 1]) .+ abs2.(c_fft[:, :, 2])) .+ 1e-30)
    emax = maximum(energy_2d)
    hm2 = Mke.heatmap!(ax2, k_fft[1], k_fft[2], energy_2d; colormap = :viridis,
        colorrange = (emax - 6, emax))
    Mke.Colorbar(fig[1, 4], hm2)

    ax3 = Mke.Axis(fig[1, 5]; title = "C. Isotropic energy spectrum E(k)", xlabel = "wavenumber k",
        ylabel = "E(k)", xscale = log10, yscale = log10)
    # Plot the resolved inertial range (skip k=0 bin and the dissipation tail near Nyquist).
    rng = 2:findlast(<=(0.6 * maximum(k_bins)), k_bins)
    Mke.lines!(ax3, k_bins[rng], E_k[rng]; color = :navy, linewidth = 3, label = "E(k)")
    # Kolmogorov k^{-5/3} guide, anchored to a mid-range point.
    mid = rng[length(rng) ÷ 2]
    guide = E_k[mid] .* (k_bins[rng] ./ k_bins[mid]) .^ (-5 / 3)
    Mke.lines!(ax3, k_bins[rng], guide; color = :red, linestyle = :dash, linewidth = 2,
        label = "k⁻⁵ᐟ³ (Kolmogorov)")
    Mke.axislegend(ax3; position = :lb)

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
    N = 48
    dx = L / N
    xs = range(0.0, stop = L - dx, length = N)
    xv = vec([x for x in xs, y in xs])
    yv = vec([y for x in xs, y in xs])

    # Broadband field so the coefficient maps show rich structure (not a few isolated dots).
    f = synthetic_field(N; slope = 5 / 3, seed = 2)
    parity_grid = FFS.UniformCartesianGrid((xv, yv); domain_size = (L, L))
    c_direct, ks = FFS.calculate_spectrum(FFS.DirectSumBackend(), parity_grid, (vec(f),), (N, N))
    c_fft, _ = FFS.calculate_spectrum(FFS.FFTBackend(), parity_grid, (vec(f),), (N, N))
    diff = abs.(c_direct[:, :, 1] .- c_fft[:, :, 1])

    fig = Mke.Figure(size = (1320, 430), fontsize = 15)
    Mke.Label(fig[0, 1:6], "Backend parity: every fast backend matches the direct-sum reference",
        fontsize = 19, font = :bold)

    ax1 = Mke.Axis(fig[1, 1]; title = "DirectSum  log₁₀|C|", xlabel = "kₓ", ylabel = "k_y",
        aspect = Mke.DataAspect())
    hm1 = Mke.heatmap!(ax1, ks[1], ks[2], log10.(abs.(c_direct[:, :, 1]) .+ 1e-30);
        colormap = :viridis)
    Mke.Colorbar(fig[1, 2], hm1)

    ax2 = Mke.Axis(fig[1, 3]; title = "FFTW  log₁₀|C|", xlabel = "kₓ", ylabel = "k_y",
        aspect = Mke.DataAspect())
    hm2 = Mke.heatmap!(ax2, ks[1], ks[2], log10.(abs.(c_fft[:, :, 1]) .+ 1e-30); colormap = :viridis)
    Mke.Colorbar(fig[1, 4], hm2)

    ax3 = Mke.Axis(fig[1, 5]; title = "|difference|  log₁₀  (≈ machine ε)", xlabel = "kₓ",
        ylabel = "k_y", aspect = Mke.DataAspect())
    hm3 = Mke.heatmap!(ax3, ks[1], ks[2], log10.(diff .+ 1e-30); colormap = :inferno,
        colorrange = (-18, -12))
    Mke.Colorbar(fig[1, 6], hm3)

    outpath = joinpath(ASSETS_DIR, "backend_parity.png")
    Mke.save(outpath, fig)
    println("Saved: $outpath")
end

# ─── Figure 4: Scattered NUFFT recovery with a coastline cutout ────────────

function generate_nufft_coastline_figure()
    Random.seed!(42)
    L = 2π
    N = 96
    dx = L / N
    xs = range(0.0, stop = L - dx, length = N)
    xv = vec([x for x in xs, y in xs])
    yv = vec([y for x in xs, y in xs])

    # Broadband field with a k^{-5/3} cascade on the full grid → reference spectrum (FFT).
    fld = synthetic_field(N; slope = 5 / 3, seed = 11)
    interp = let f = fld
        (x, y) -> f[clamp(round(Int, x / dx) + 1, 1, N), clamp(round(Int, y / dx) + 1, 1, N)]
    end
    grid = FFS.UniformCartesianGrid((xv, yv); domain_size = (L, L))
    c_ref, k_ref = FFS.calculate_spectrum(FFS.FFTBackend(), grid, (vec(fld),), (N, N))
    kr, E_ref = FFS.isotropic_spectrum(k_ref, c_ref; num_bins = 28)

    # Jittered (off-grid) sample cloud, then carve out a synthetic landmass: corner island OR
    # below a sinusoidal coast. Sample the field by nearest-grid lookup at the ocean points.
    xj = clamp.(xv .+ (rand(length(xv)) .- 0.5) .* (0.5dx), 0.0, L)
    yj = clamp.(yv .+ (rand(length(yv)) .- 0.5) .* (0.5dx), 0.0, L)
    is_land(x, y) =
        ((x - 0.0)^2 + (y - 0.0)^2 < (0.42L)^2) || (y < 0.16L + 0.10L * sin(4π * x / L))
    ocean = .!is_land.(xj, yj)
    xo, yo = xj[ocean], yj[ocean]
    fo = interp.(xo, yo)

    sgrid = FFS.ScatteredCartesianGrid((xo, yo); domain_size = (L, L))
    c_nu, k_nu = FFS.calculate_spectrum(FFS.NUFFTBackend(), sgrid, (fo,), (N, N); eps = 1e-9)
    knu, E_nu = FFS.isotropic_spectrum(k_nu, c_nu; num_bins = 28)

    rng = 2:findlast(<=(0.6 * maximum(kr)), kr)
    fig = Mke.Figure(size = (1250, 500), fontsize = 15)
    Mke.Label(fig[0, 1:2], "NUFFT recovery from a jittered, coastline-masked cloud",
        fontsize = 19, font = :bold)
    ax1 = Mke.Axis(fig[1, 1]; title = "A. Ocean samples, colored by field value (land cut out)",
        xlabel = "x", ylabel = "y", aspect = Mke.DataAspect())
    Mke.scatter!(ax1, xo, yo; color = fo, colormap = :balance, markersize = 4)
    ax2 = Mke.Axis(fig[1, 2]; title = "B. Recovered spectrum vs full-grid reference",
        xlabel = "wavenumber k", ylabel = "E(k)", xscale = log10, yscale = log10)
    Mke.lines!(ax2, kr[rng], E_ref[rng]; color = :black, linewidth = 3, label = "Full grid (FFT)")
    Mke.scatter!(ax2, knu[rng], E_nu[rng]; color = :crimson, markersize = 11,
        label = "Ocean cloud (NUFFT)")
    guide = E_ref[rng[length(rng) ÷ 2]] .* (kr[rng] ./ kr[rng[length(rng) ÷ 2]]) .^ (-5 / 3)
    Mke.lines!(ax2, kr[rng], guide; color = :gray, linestyle = :dash, label = "k⁻⁵ᐟ³")
    Mke.axislegend(ax2; position = :lb)
    outpath = joinpath(ASSETS_DIR, "nufft_coastline.png")
    Mke.save(outpath, fig)
    println("Saved: $outpath")
end

# ─── Figure 5: Anisotropy-resolved spectrum E(k, θ) ────────────────────────

function generate_anisotropy_figure()
    L = 2π
    N = 128
    dx = L / N
    xs = range(0.0, stop = L - dx, length = N)
    xv = vec([x for x in xs, y in xs])
    yv = vec([y for x in xs, y in xs])
    # Broadband field with a directional bias → elongated structures (horizontal "streaks"). A
    # fairly flat radial spectrum spreads energy across k so the preferred angle reads as a band.
    g = synthetic_field(N; slope = 0.6, aniso = 3.5, seed = 5)
    grid = FFS.UniformCartesianGrid((xv, yv); domain_size = (L, L))
    c, ks = FFS.calculate_spectrum(FFS.FFTBackend(), grid, (vec(g),), (N, N))
    k_bins, θ_bins, E = FFS.anisotropic_spectrum(ks, c; num_k_bins = 32, num_θ_bins = 36)

    fig = Mke.Figure(size = (1180, 460), fontsize = 15)
    Mke.Label(fig[0, 1:2], "Anisotropy: a directional field concentrates energy at a preferred angle",
        fontsize = 18, font = :bold)
    ax1 = Mke.Axis(fig[1, 1]; title = "A. Anisotropic field (elongated structures)", xlabel = "x",
        ylabel = "y", aspect = Mke.DataAspect())
    gmax = maximum(abs, g)
    hm1 = Mke.heatmap!(ax1, xs, xs, g; colormap = :balance, colorrange = (-gmax, gmax))
    Mke.Colorbar(fig[1, 2], hm1)
    ax2 = Mke.Axis(fig[1, 3]; title = "B. Anisotropy-resolved spectrum E(k, θ)",
        xlabel = "wavenumber k", ylabel = "angle θ [rad]")
    hm2 = Mke.heatmap!(ax2, k_bins, θ_bins, E; colormap = :viridis)
    Mke.Colorbar(fig[1, 4], hm2)
    outpath = joinpath(ASSETS_DIR, "anisotropy.png")
    Mke.save(outpath, fig)
    println("Saved: $outpath")
end

# ─── Figure 6: Cross-spectrum, coherence & phase (flux by scale) ───────────

function generate_cross_coherence_figure()
    Random.seed!(1)
    L = 2π
    N = 64
    dx = L / N
    xs = range(0.0, stop = L - dx, length = N)
    xv = vec([x for x in xs, y in xs])
    yv = vec([y for x in xs, y in xs])
    grid = FFS.UniformCartesianGrid((xv, yv); domain_size = (L, L))

    # Complex (rotary, e.g. u+iv) signal pair: a shared rotating mode at k≈2 with a fixed phase
    # lead ϕ of w over f, plus independent structure. Complex fields keep their spectral content at
    # +k only, so the cross-spectrum phase survives radial binning (unlike a real-field pair).
    nreal = 40
    ϕ = 0.7
    Cf = zeros(ComplexF64, N, N, nreal)
    Cg = zeros(ComplexF64, N, N, nreal)
    ks = nothing
    for r in 1:nreal
        a = 1.0 + 0.1 * randn()
        fr = @. a * exp(im * 2 * xv) + 0.5 * exp(im * (5 * xv) + im * 2π * rand())
        gr = @. a * exp(im * (2 * xv - ϕ)) + 0.5 * exp(im * (7 * yv) + im * 2π * rand())
        cfr, ksr = FFS.calculate_spectrum(FFS.FFTBackend(), grid, (fr,), (N, N))
        cgr, _ = FFS.calculate_spectrum(FFS.FFTBackend(), grid, (gr,), (N, N))
        Cf[:, :, r] .= cfr[:, :, 1]
        Cg[:, :, r] .= cgr[:, :, 1]
        ks = ksr
    end
    kw, Eu = FFS.welch_power_spectrum(ks, Cf; num_bins = 24)
    kc, γ², phase = FFS.coherence_spectrum(ks, Cf, Cg; num_bins = 24)
    # Phase is only meaningful where coherence is appreciable; mask the rest so it isn't misread.
    phase_plot = [γ²[i] > 0.3 ? phase[i] / π : NaN for i in eachindex(phase)]

    fig = Mke.Figure(size = (1100, 440), fontsize = 14)
    ax1 = Mke.Axis(fig[1, 1]; title = "A. Welch power spectrum E(k)", xlabel = "k", ylabel = "E(k)",
        yscale = log10)
    Mke.lines!(ax1, kw, Eu .+ 1e-20; linewidth = 2)
    ax2 = Mke.Axis(fig[1, 2]; title = "B. Coherence² (—) & phase/π (●)", xlabel = "k",
        ylabel = "γ² ,  phase/π")
    Mke.lines!(ax2, kc, γ²; linewidth = 2, label = "coherence²")
    Mke.scatter!(ax2, kc, phase_plot; color = :orange, label = "phase/π (where γ² > 0.3)")
    Mke.hlines!(ax2, [ϕ / π]; color = :gray, linestyle = :dash, label = "imposed ϕ/π")
    Mke.axislegend(ax2; position = :rc)
    outpath = joinpath(ASSETS_DIR, "cross_coherence.png")
    Mke.save(outpath, fig)
    println("Saved: $outpath")
end

# ─── Figure 7: Derived-quantity spectra (energy vs enstrophy) ──────────────

function generate_derived_figure()
    L = 2π
    N = 128
    u, v, _ = synthetic_incompressible(N, L; seed = 3)     # k^{-5/3} energy cascade
    dx = L / N
    xs = range(0.0, stop = L - dx, length = N)
    xv = vec([x for x in xs, y in xs])
    yv = vec([y for x in xs, y in xs])
    grid = FFS.UniformCartesianGrid((xv, yv); domain_size = (L, L))
    c, ks = FFS.calculate_spectrum(FFS.FFTBackend(), grid, (vec(u), vec(v)), (N, N))
    vort = FFS.spectral_vorticity(ks, c)                   # scalar vorticity coeffs (2D)
    k_bins, E_k = FFS.isotropic_spectrum(ks, c; num_bins = 40)
    _, Z_k = FFS.isotropic_spectrum(ks, vort; num_bins = 40)

    rng = 2:findlast(<=(0.6 * maximum(k_bins)), k_bins)
    fig = Mke.Figure(size = (760, 500), fontsize = 15)
    ax = Mke.Axis(fig[1, 1]; title = "Energy vs enstrophy spectra (derived by spectral curl)",
        xlabel = "wavenumber k", ylabel = "spectral density", xscale = log10, yscale = log10)
    Mke.lines!(ax, k_bins[rng], E_k[rng]; linewidth = 3, color = :navy,
        label = "E(k) — energy  (∝ k⁻⁵ᐟ³)")
    Mke.lines!(ax, k_bins[rng], Z_k[rng]; linewidth = 3, color = :darkorange,
        label = "Z(k) — enstrophy  (∝ k⁺¹ᐟ³)")
    # The identity Z(k) = k²E(k) holds exactly for an incompressible field — overlay to verify.
    Mke.scatter!(ax, k_bins[rng], (k_bins[rng] .^ 2) .* E_k[rng]; color = :black, markersize = 7,
        marker = :cross, label = "k² E(k)  (identity check)")
    Mke.axislegend(ax; position = :lt)
    outpath = joinpath(ASSETS_DIR, "derived_quantities.png")
    Mke.save(outpath, fig)
    println("Saved: $outpath")
end

# ─── Figure 8: Wavenumber–frequency spectrum E(k, ω) ───────────────────────

function generate_komega_figure()
    Random.seed!(0)
    Nx, Nt = 64, 64
    Lx, Lt = 2π, 2π
    dx, dt = Lx / Nx, Lt / Nt
    x = range(0.0, stop = Lx - dx, length = Nx)
    t = range(0.0, stop = Lt - dt, length = Nt)
    k0, ω0 = 6.0, 6.0
    xv = vec([xi for xi in x, _ in t])
    tv = vec([ti for _ in x, ti in t])
    f = @. cos(k0 * xv + ω0 * tv) + 0.4 * cos(2 * xv + 1.0 * tv + 0.5)
    grid = FFS.UniformCartesianGrid((xv, tv); domain_size = (Lx, Lt))
    coeffs, ks = FFS.calculate_spectrum(FFS.FFTBackend(), grid, (f,), (Nx, Nt))
    kx, kω = ks

    fig = Mke.Figure(size = (640, 480), fontsize = 14)
    ax = Mke.Axis(fig[1, 1]; title = "Wavenumber–frequency spectrum E(k, ω)", xlabel = "k",
        ylabel = "ω")
    hm = Mke.heatmap!(ax, kx, kω, abs2.(coeffs[:, :, 1]); colormap = :viridis)
    Mke.lines!(ax, kx, kx; color = :white, linestyle = :dash, label = "ω = k")
    Mke.Colorbar(fig[1, 2], hm)
    Mke.axislegend(ax)
    outpath = joinpath(ASSETS_DIR, "komega.png")
    Mke.save(outpath, fig)
    println("Saved: $outpath")
end

# ─── Figure 9: Irregular sampling (Lomb–Scargle) & multitaper ──────────────

function generate_estimation_figure()
    Random.seed!(7)
    # Lomb–Scargle on irregular samples.
    Nls = 250
    t = sort(rand(Nls) .* 10.0)
    f0 = 1.3
    y = @. sin(2π * f0 * t) + 0.3 * randn()
    freqs = range(0.1, stop = 4.0, length = 400)
    P = FFS.lomb_scargle(t, collect(y), collect(freqs))

    # Multitaper vs single periodogram on a broadband (red-noise) background plus a spectral tone.
    Nx = 256
    L = 2π
    dx = L / Nx
    x = range(0.0, stop = L - dx, length = Nx)
    fr = _dftfreq(Nx)
    bg = real(FFTW.ifft(FFTW.fft(randn(Nx)) .* [k == 0 ? 0.0 : abs(k)^(-1.0) for k in fr]))
    k0 = 20
    sig = @. 0.25 * cos(k0 * x) + bg               # tone buried in a k⁻² continuum
    K = 6
    V = FFS.dpss(Nx, 4.0, K)
    grid = FFS.UniformCartesianGrid((collect(x),); domain_size = (L,))
    C = zeros(ComplexF64, Nx, K)
    ks = nothing
    for k in 1:K
        c, ksk = FFS.calculate_spectrum(FFS.FFTBackend(), grid, (V[:, k] .* sig,), (Nx,))
        C[:, k] .= c[:, 1]
        ks = ksk
    end
    kb, Emt = FFS.welch_power_spectrum(ks, C; num_bins = 48)
    # Compare against a single taper computed the same way, so only the variance differs (a raw
    # periodogram would carry a different normalization and sit at a different level).
    kb1, Esingle = FFS.welch_power_spectrum(ks, C[:, 1:1]; num_bins = 48)

    fig = Mke.Figure(size = (1180, 460), fontsize = 15)
    ax1 = Mke.Axis(fig[1, 1]; title = "A. Lomb–Scargle: spectrum from irregular samples",
        xlabel = "frequency", ylabel = "power")
    Mke.lines!(ax1, freqs, P; linewidth = 2, color = :navy)
    Mke.vlines!(ax1, [f0]; color = :red, linestyle = :dash, label = "true f₀ = $f0")
    Mke.axislegend(ax1)
    ax2 = Mke.Axis(fig[1, 2]; title = "B. Multitaper variance reduction", xlabel = "wavenumber k",
        ylabel = "E(k)", yscale = log10)
    Mke.lines!(ax2, kb1, Esingle .+ 1e-12; color = (:gray, 0.7), label = "single taper (noisy)")
    Mke.lines!(ax2, kb, Emt .+ 1e-12; linewidth = 2.5, color = :navy, label = "multitaper (K=$K)")
    Mke.vlines!(ax2, [Float64(k0)]; color = :red, linestyle = :dash, label = "tone at k = $k0")
    Mke.axislegend(ax2; position = :lb)
    outpath = joinpath(ASSETS_DIR, "irregular_estimation.png")
    Mke.save(outpath, fig)
    println("Saved: $outpath")
end

# ─── Execute ──────────────────────────────────────────────────────────────

println("Generating static figure assets...")
generate_cartesian_figure()
generate_spherical_figure()
generate_parity_figure()
generate_nufft_coastline_figure()
generate_anisotropy_figure()
generate_cross_coherence_figure()
generate_derived_figure()
generate_komega_figure()
generate_estimation_figure()
println("Done!")
