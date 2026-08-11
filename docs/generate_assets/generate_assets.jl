"""
Generate static figure assets for FlowFieldSpectra.jl docs and README.md.

Run from the root directory:
    julia --project=docs/generate_assets docs/generate_assets/generate_assets.jl
"""

using FlowFieldSpectra: FlowFieldSpectra as FFS
using FFTW: FFTW                                       # activates the FFT extension
using FINUFFT: FINUFFT                                 # activates the NUFFT extension
using FastSphericalHarmonics: FastSphericalHarmonics as FSH
using KernelAbstractions: KernelAbstractions as KA     # activates GPUBackend (+ device-generic GPU FFT)
using OhMyThreads: OhMyThreads                         # activates ThreadedBackend
using CairoMakie: CairoMakie as Mke
import Random
using ComputationalBackends: ComputationalBackends as CB
using SpectralBackends: SpectralBackends as SB
using FlowGeometries: FlowGeometries as FG

const ASSETS_DIR = joinpath(@__DIR__, "..", "src", "assets")
mkpath(ASSETS_DIR)

const CART = FG.Geometry.CartesianGeometry{Float64}()
# Uniform periodic Cartesian grid over the given axes (FlowGeometries StructuredGrid).
ucg(axes...; L) = FG.Grids.StructuredGrid(CART, axes...;
    periodic = ntuple(_ -> true, length(axes)), period = ntuple(_ -> L, length(axes)))
# Scattered periodic Cartesian grid (unused per-node measure = ones).
scg(coords::Tuple; L) = FG.Grids.UnstructuredGrid(CART, coords, ones(length(coords[1]));
    periodic = ntuple(_ -> true, length(coords)), period = ntuple(_ -> L, length(coords)))
_uaxis(L, N) = range(0.0, L; length = N + 1)[1:N]

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
    xs = _uaxis(L, N)

    # Broadband, incompressible "turbulence" with a k^{-5/3} energy cascade.
    u, v, _ = synthetic_incompressible(N, L; seed = 7)

    cart_grid = ucg(xs, xs; L = L)
    c_fft, k_fft = FFS.calculate_spectrum(cart_grid, (u, v), (N, N); transform = SB.FFTSpectralBackend())
    # Kinetic-energy isotropic spectrum: fold the component axis (dim 3) into the energy.
    k_bins, E_k = FFS.isotropic_spectrum(k_fft, c_fft; num_bins = 40, dims = 3)

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
    rng = 2:findlast(<=(0.6 * maximum(k_bins)), k_bins)
    Mke.lines!(ax3, k_bins[rng], E_k[rng]; color = :navy, linewidth = 3, label = "E(k)")
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

    # Clenshaw–Curtis grid (FlowGeometries) — the grid FastSphericalHarmonics transforms on.
    sht_grid = FG.Connectivity.structured_grid(Float64, FG.SphericalSampling.ClenshawCurtisSampling(), Nθ)
    λ = collect(FG.Grids.coordinates(sht_grid, 1))                # longitude (nlon)
    θ_colat = (π / 2) .- collect(FG.Grids.coordinates(sht_grid, 2))  # colatitude (nlat)

    C_true = zeros(ComplexF64, Nθ, Nφ)
    C_true[FFS.sph_mode_index(2, 1)] = 1.0
    C_true[FFS.sph_mode_index(5, -3)] = 0.8
    C_true[FFS.sph_mode_index(8, 4)] = 0.5
    f_val = real(FFS.synthesize(sht_grid, C_true, (Nθ, Nφ)))      # (nlon, nlat)

    c_sht, _ = FFS.calculate_spectrum(sht_grid, f_val, (Nθ, Nφ); transform = SB.FSHTSpectralBackend())
    deg, E_l = FFS.spherical_energy_spectrum(c_sht)

    fig = Mke.Figure(size = (1200, 800), fontsize = 14)
    Mke.Label(fig[0, 1:2], "Spherical Harmonic Transform & Degree Energy Spectrum",
        fontsize = 18, font = :bold)

    ax1 = Mke.Axis(fig[1, 1], title = "A. Scalar Field f(θ, φ) on CC Grid",
        xlabel = "Longitude φ (rad)", ylabel = "Colatitude θ (rad)")
    hm = Mke.heatmap!(ax1, λ, θ_colat, f_val; colormap = :balance)
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
    xs = _uaxis(L, N)

    f = synthetic_field(N; slope = 5 / 3, seed = 2)
    parity_grid = ucg(xs, xs; L = L)
    c_direct, ks = FFS.calculate_spectrum(parity_grid, f, (N, N); transform = SB.DirectSumSpectralBackend())
    c_fft, _ = FFS.calculate_spectrum(parity_grid, f, (N, N); transform = SB.FFTSpectralBackend())
    diff = abs.(c_direct .- c_fft)

    fig = Mke.Figure(size = (1320, 430), fontsize = 15)
    Mke.Label(fig[0, 1:6], "Backend parity: every fast backend matches the direct-sum reference",
        fontsize = 19, font = :bold)

    ax1 = Mke.Axis(fig[1, 1]; title = "DirectSum  log₁₀|C|", xlabel = "kₓ", ylabel = "k_y",
        aspect = Mke.DataAspect())
    hm1 = Mke.heatmap!(ax1, ks[1], ks[2], log10.(abs.(c_direct) .+ 1e-30); colormap = :viridis)
    Mke.Colorbar(fig[1, 2], hm1)

    ax2 = Mke.Axis(fig[1, 3]; title = "FFTW  log₁₀|C|", xlabel = "kₓ", ylabel = "k_y",
        aspect = Mke.DataAspect())
    hm2 = Mke.heatmap!(ax2, ks[1], ks[2], log10.(abs.(c_fft) .+ 1e-30); colormap = :viridis)
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
    xs = _uaxis(L, N)
    xv = vec([x for x in xs, y in xs])
    yv = vec([y for x in xs, y in xs])

    fld = synthetic_field(N; slope = 5 / 3, seed = 11)
    interp = let f = fld
        (x, y) -> f[clamp(round(Int, x / dx) + 1, 1, N), clamp(round(Int, y / dx) + 1, 1, N)]
    end
    grid = ucg(xs, xs; L = L)
    c_ref, k_ref = FFS.calculate_spectrum(grid, fld, (N, N); transform = SB.FFTSpectralBackend())
    kr, E_ref = FFS.isotropic_spectrum(k_ref, c_ref; num_bins = 28)

    xj = clamp.(xv .+ (rand(length(xv)) .- 0.5) .* (0.5dx), 0.0, L)
    yj = clamp.(yv .+ (rand(length(yv)) .- 0.5) .* (0.5dx), 0.0, L)
    is_land(x, y) =
        ((x - 0.0)^2 + (y - 0.0)^2 < (0.42L)^2) || (y < 0.16L + 0.10L * sin(4π * x / L))
    ocean = .!is_land.(xj, yj)
    xo, yo = xj[ocean], yj[ocean]
    fo = interp.(xo, yo)

    sgrid = scg((xo, yo); L = L)
    c_nu, k_nu = FFS.calculate_spectrum(sgrid, fo, (N, N); transform = FFS.FINUFFTBackend(), eps = 1e-9)
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
    xs = _uaxis(L, N)
    g = synthetic_field(N; slope = 0.6, aniso = 3.5, seed = 5)
    grid = ucg(xs, xs; L = L)
    c, ks = FFS.calculate_spectrum(grid, g, (N, N); transform = SB.FFTSpectralBackend())
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
    xs = _uaxis(L, N)
    xv = vec([x for x in xs, y in xs])
    yv = vec([y for x in xs, y in xs])
    grid = ucg(xs, xs; L = L)

    nreal = 40
    ϕ = 0.7
    Cf = zeros(ComplexF64, N, N, nreal)
    Cg = zeros(ComplexF64, N, N, nreal)
    ks = nothing
    for r in 1:nreal
        a = 1.0 + 0.1 * randn()
        fr = reshape((@. a * exp(im * 2 * xv) + 0.5 * exp(im * (5 * xv) + im * 2π * rand())), N, N)
        gr = reshape((@. a * exp(im * (2 * xv - ϕ)) + 0.5 * exp(im * (7 * yv) + im * 2π * rand())), N, N)
        cfr, ksr = FFS.calculate_spectrum(grid, fr, (N, N); transform = SB.FFTSpectralBackend())
        cgr, _ = FFS.calculate_spectrum(grid, gr, (N, N); transform = SB.FFTSpectralBackend())
        Cf[:, :, r] .= cfr
        Cg[:, :, r] .= cgr
        ks = ksr
    end
    kw, Eu = FFS.welch_power_spectrum(ks, Cf; num_bins = 24)
    kc, γ², phase = FFS.coherence_spectrum(ks, Cf, Cg; num_bins = 24)
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
    xs = _uaxis(L, N)
    grid = ucg(xs, xs; L = L)
    c, ks = FFS.calculate_spectrum(grid, (u, v), (N, N); transform = SB.FFTSpectralBackend())
    vort = FFS.spectral_vorticity(ks, c)                   # scalar vorticity coeffs (2D)
    k_bins, E_k = FFS.isotropic_spectrum(ks, c; num_bins = 40, dims = 3)
    _, Z_k = FFS.isotropic_spectrum(ks, vort; num_bins = 40, dims = 3)

    rng = 2:findlast(<=(0.6 * maximum(k_bins)), k_bins)
    fig = Mke.Figure(size = (760, 500), fontsize = 15)
    ax = Mke.Axis(fig[1, 1]; title = "Energy vs enstrophy spectra (derived by spectral curl)",
        xlabel = "wavenumber k", ylabel = "spectral density", xscale = log10, yscale = log10)
    Mke.lines!(ax, k_bins[rng], E_k[rng]; linewidth = 3, color = :navy,
        label = "E(k) — energy  (∝ k⁻⁵ᐟ³)")
    Mke.lines!(ax, k_bins[rng], Z_k[rng]; linewidth = 3, color = :darkorange,
        label = "Z(k) — enstrophy  (∝ k⁺¹ᐟ³)")
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
    x = _uaxis(Lx, Nx)
    t = _uaxis(Lt, Nt)
    k0, ω0 = 6.0, 6.0
    f = [cos(k0 * xi + ω0 * ti) + 0.4 * cos(2 * xi + 1.0 * ti + 0.5) for xi in x, ti in t]  # (Nx, Nt)
    grid = FG.Grids.StructuredGrid(CART, x, t; periodic = (true, true), period = (Lx, Lt))
    coeffs, ks = FFS.calculate_spectrum(grid, f, (Nx, Nt); transform = SB.FFTSpectralBackend())
    kx, kω = ks

    fig = Mke.Figure(size = (640, 480), fontsize = 14)
    ax = Mke.Axis(fig[1, 1]; title = "Wavenumber–frequency spectrum E(k, ω)", xlabel = "k",
        ylabel = "ω")
    hm = Mke.heatmap!(ax, kx, kω, abs2.(coeffs); colormap = :viridis)
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
    Nls = 250
    t = sort(rand(Nls) .* 10.0)
    f0 = 1.3
    y = @. sin(2π * f0 * t) + 0.3 * randn()
    freqs = range(0.1, stop = 4.0, length = 400)
    P = FFS.lomb_scargle(t, collect(y), collect(freqs))

    Nx = 256
    L = 2π
    x = _uaxis(L, Nx)
    fr = _dftfreq(Nx)
    bg = real(FFTW.ifft(FFTW.fft(randn(Nx)) .* [k == 0 ? 0.0 : abs(k)^(-1.0) for k in fr]))
    k0 = 20
    sig = @. 0.25 * cos(k0 * x) + bg               # tone buried in a k⁻² continuum
    K = 6
    V = FFS.dpss(Nx, 4.0, K)
    grid = FG.Grids.StructuredGrid(CART, x; periodic = (true,), period = (L,))
    C = zeros(ComplexF64, Nx, K)
    ks = nothing
    for k in 1:K
        c, ksk = FFS.calculate_spectrum(grid, V[:, k] .* sig, (Nx,); transform = SB.FFTSpectralBackend())
        C[:, k] .= c
        ks = ksk
    end
    kb, Emt = FFS.welch_power_spectrum(ks, C; num_bins = 48)
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

# ─── Figure: execution-axis invariance (two orthogonal backend axes) ────────

# The `execution=` axis says only WHERE/HOW a transform runs; it never changes the result. Here one
# FFT transform is run under three execution backends — Serial, Threaded (OhMyThreads), and
# GPUBackend(KA.CPU()) (the device-generic GPU FFT, which uses FFTW on the KA.CPU() host array
# exactly as it would use CUFFT on a CUDA CuArray) — and the recovered spectra coincide to machine ε.
function generate_execution_figure()
    L = 2π
    N = 96
    xs = _uaxis(L, N)
    f = synthetic_field(N; slope = 5 / 3, seed = 5)
    grid = ucg(xs, xs; L = L)
    ms = (N, N)

    execs = [
        ("Serial", CB.SerialBackend()),
        ("Threaded", CB.ThreadedBackend()),
        ("GPU (KA.CPU())", CB.GPUBackend(KA.CPU())),
    ]
    results = [FFS.calculate_spectrum(grid, f, ms; transform = SB.FFTSpectralBackend(), execution = e)
               for (_, e) in execs]
    ks = results[1][2]
    cref = results[1][1]
    specs = [FFS.isotropic_spectrum(ks, r[1]; num_bins = 40) for r in results]
    maxdiff = [maximum(abs.(r[1] .- cref)) for r in results]

    fig = Mke.Figure(size = (1360, 470), fontsize = 15)
    Mke.Label(fig[0, 1:3],
        "Execution axis is result-invariant — one FFT transform, three execution backends",
        fontsize = 19, font = :bold)

    kb1 = specs[1][1]
    rng = 2:findlast(<=(0.6 * maximum(kb1)), kb1)
    ax1 = Mke.Axis(fig[1, 1]; title = "Isotropic E(k) — the three curves coincide",
        xlabel = "k", ylabel = "E(k)", xscale = log10, yscale = log10)
    styles = [(:solid, 5), (:dash, 3), (:dot, 2)]
    for (i, (name, _)) in enumerate(execs)
        kb, Ek = specs[i]
        Mke.lines!(ax1, kb[rng], Ek[rng] .+ 1e-30; linestyle = styles[i][1],
            linewidth = styles[i][2], label = name)
    end
    Mke.axislegend(ax1; position = :lb)

    ax2 = Mke.Axis(fig[1, 2]; title = "2D spectral energy  log₁₀|C(kₓ, k_y)|²",
        xlabel = "kₓ", ylabel = "k_y", aspect = Mke.DataAspect())
    hm = Mke.heatmap!(ax2, ks[1], ks[2], log10.(abs2.(cref) .+ 1e-30); colormap = :viridis)
    Mke.Colorbar(fig[1, 3], hm)

    diffstr = join(("$(n): $(round(d, sigdigits = 2))" for ((n, _), d) in zip(execs, maxdiff)), ",   ")
    Mke.Label(fig[2, 1:3], "Max |coefficient difference| vs. Serial —  " * diffstr *
        "   (all below machine ε · max|C| ≈ $(round(eps(Float64) * maximum(abs.(cref)), sigdigits = 2)))";
        fontsize = 12, color = :gray35)
    Mke.colgap!(fig.layout, 24)

    outpath = joinpath(ASSETS_DIR, "execution_parity.png")
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
generate_execution_figure()
println("Done!")
