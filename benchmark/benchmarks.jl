using BenchmarkTools: BenchmarkTools, BenchmarkGroup, @benchmarkable, tune!, run
using FlowFieldSpectra: FlowFieldSpectra as FFS
using FFTW: FFTW
using FINUFFT: FINUFFT
using FastSphericalHarmonics: FastSphericalHarmonics as FSH
using NUFSHT: NUFSHT
using Random: Random

const SUITE = BenchmarkGroup()

# =============================================================================
# Cartesian uniform grid (tensor field): DirectSum vs FFTW.
# =============================================================================
SUITE["cartesian_uniform"] = BenchmarkGroup()
for N in [64, 128, 256]
    SUITE["cartesian_uniform"]["N=$N"] = BenchmarkGroup()
    L = 10.0
    g = FFS.UniformCartesianGrid(; domain = (L, L), n = (N, N))
    xs, ys = g.axes
    u = [cos(2π * 2 * x / L + 2π * y / L) + 0.5 * sin(2π * (-3) * x / L + 2π * 2 * y / L) for x in xs, y in ys]
    v = [sin(2π * 2 * x / L + 2π * y / L) for x in xs, y in ys]
    uv = cat(u, v; dims = 3)
    SUITE["cartesian_uniform"]["N=$N"]["fftw"] =
        @benchmarkable FFS.calculate_spectrum($g, $uv, ($N, $N); transform = FFS.FFTBackend())
    N <= 128 && (SUITE["cartesian_uniform"]["N=$N"]["direct_sum"] =
        @benchmarkable FFS.calculate_spectrum($g, $uv, ($N, $N); transform = FFS.DirectSumBackend()))
end

# =============================================================================
# Cartesian scattered grid: DirectSum vs FINUFFT.
# =============================================================================
SUITE["cartesian_scattered"] = BenchmarkGroup()
for N in [1000, 10000, 100000]
    SUITE["cartesian_scattered"]["N=$N"] = BenchmarkGroup()
    Random.seed!(42)
    L = 10.0
    ms = (64, 64)
    xv = rand(N) .* L
    yv = rand(N) .* L
    kx, ky = 2π * 1 / L, 2π * (-1) / L
    f = @. cos(kx * xv + ky * yv)
    g = FFS.ScatteredCartesianGrid((xv, yv); domain_size = (L, L))
    SUITE["cartesian_scattered"]["N=$N"]["finufft"] =
        @benchmarkable FFS.calculate_spectrum($g, $f, $ms; transform = FFS.NUFFTBackend(), eps = 1e-9)
    N <= 10000 && (SUITE["cartesian_scattered"]["N=$N"]["direct_sum"] =
        @benchmarkable FFS.calculate_spectrum($g, $f, $ms; transform = FFS.DirectSumBackend()))
end

# =============================================================================
# Spherical structured grid: DirectSum vs FastSphericalHarmonics.
# =============================================================================
SUITE["spherical_structured"] = BenchmarkGroup()
for lmax in [16, 32, 64]
    SUITE["spherical_structured"]["lmax=$lmax"] = BenchmarkGroup()
    Nθ = lmax + 1
    Nφ = 2 * lmax + 1
    pts = FSH.sph_points(Nθ)
    g = FFS.StructuredSphericalGrid(pts[1], pts[2])
    C_true = zeros(Nθ, Nφ)
    C_true[FSH.sph_mode(2, 1)] = 1.0
    C_true[FSH.sph_mode(3, -2)] = 0.5
    f = FSH.sph_evaluate(C_true)
    SUITE["spherical_structured"]["lmax=$lmax"]["fast_sht"] =
        @benchmarkable FFS.calculate_spectrum($g, $f, ($Nθ, $Nφ); transform = FFS.SHTBackend())
    lmax <= 32 && (SUITE["spherical_structured"]["lmax=$lmax"]["direct_sum"] =
        @benchmarkable FFS.calculate_spectrum($g, $f, ($Nθ, $Nφ); transform = FFS.DirectSumBackend()))
end

# =============================================================================
# Spherical scattered grid: NUFSHT (adjoint + CG solve).
# =============================================================================
SUITE["spherical_scattered"] = BenchmarkGroup()
for lmax in [8, 16, 32]
    SUITE["spherical_scattered"]["lmax=$lmax"] = BenchmarkGroup()
    Nθ = lmax + 1
    Nφ = 2 * lmax + 1
    N_pts = 4 * Nθ^2
    Random.seed!(42)
    ga = π * (3 - sqrt(5))
    zf = [1 - 2 * (i + 0.5) / N_pts for i in 0:(N_pts-1)]
    θ = acos.(clamp.(zf, -1.0, 1.0))
    φ = mod.(ga .* (0:(N_pts-1)), 2π)
    C_true = zeros(Nθ, Nφ)
    C_true[FSH.sph_mode(2, 0)] = 1.0
    plan = NUFSHT.make_plan(θ, φ, lmax)
    f = zeros(N_pts)
    NUFSHT.nusht_type2!(f, C_true, plan)
    g = FFS.ScatteredSphericalGrid(θ, φ)
    SUITE["spherical_scattered"]["lmax=$lmax"]["nufsht_adjoint"] =
        @benchmarkable FFS.calculate_spectrum($g, $f, ($Nθ, $Nφ); transform = FFS.NUFSHTBackend(), solve = false, tol = 1e-8)
    lmax <= 16 && (SUITE["spherical_scattered"]["lmax=$lmax"]["nufsht_cg"] =
        @benchmarkable FFS.calculate_spectrum($g, $f, ($Nθ, $Nφ); transform = FFS.NUFSHTBackend(), solve = true, rtol = 1e-6, maxiter = 500))
end

# =============================================================================
# Reductions.
# =============================================================================
SUITE["reductions"] = BenchmarkGroup()
let
    L = 10.0
    N = 128
    ms = (N, N)
    g = FFS.UniformCartesianGrid(; domain = (L, L), n = ms)
    xs, ys = g.axes
    u = [cos(2π * 2 * x / L + 2π * y / L) for x in xs, y in ys]
    v = [sin(2π * 2 * x / L + 2π * y / L) for x in xs, y in ys]
    coeffs, ks = FFS.calculate_spectrum(g, cat(u, v; dims = 3), ms; transform = FFS.FFTBackend())
    SUITE["reductions"]["isotropic_spectrum"] = @benchmarkable FFS.isotropic_spectrum($ks, $coeffs; num_bins = 32, dims = 3)
    SUITE["reductions"]["transect_spectrum"] = @benchmarkable FFS.transect_spectrum($ks, $coeffs, (1,))

    lmax = 16
    Nθ = lmax + 1
    Nφ = 2 * lmax + 1
    pts = FSH.sph_points(Nθ)
    gs = FFS.StructuredSphericalGrid(pts[1], pts[2])
    Ct = zeros(Nθ, Nφ)
    Ct[FSH.sph_mode(2, 1)] = 1.0
    fsph = FSH.sph_evaluate(Ct)
    c_sph, _ = FFS.calculate_spectrum(gs, fsph, (Nθ, Nφ); transform = FFS.SHTBackend())
    SUITE["reductions"]["spherical_energy_spectrum"] = @benchmarkable FFS.spherical_energy_spectrum($c_sph; lmax = $lmax)
end

if abspath(PROGRAM_FILE) == @__FILE__
    println("Running FlowFieldSpectra.jl benchmarks...")
    tune!(SUITE)
    results = run(SUITE; verbose = true)
    println("\n" * "="^60 * "\nBENCHMARK RESULTS\n" * "="^60)
    display(results)
    BenchmarkTools.save("benchmark_results.json", results)
end
