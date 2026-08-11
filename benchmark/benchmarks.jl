using BenchmarkTools: BenchmarkTools, BenchmarkGroup, @benchmarkable, tune!, run
using FlowFieldSpectra: FlowFieldSpectra as FFS
using FFTW: FFTW
using FINUFFT: FINUFFT
using FastSphericalHarmonics: FastSphericalHarmonics as FSH
using NUFSHT: NUFSHT
using Random: Random
using SpectralBackends: SpectralBackends as SB
using FlowGeometries: FlowGeometries as FG

const SUITE = BenchmarkGroup()
const CARTGEOM = FG.Geometry.CartesianGeometry{Float64}()
_uaxis(L, N) = range(0.0, L; length = N + 1)[1:N]

# =============================================================================
# Cartesian uniform grid (tensor field): DirectSum vs FFTW.
# =============================================================================
SUITE["cartesian_uniform"] = BenchmarkGroup()
for N in [64, 128, 256]
    SUITE["cartesian_uniform"]["N=$N"] = BenchmarkGroup()
    L = 10.0
    xs = _uaxis(L, N)
    g = FG.Grids.StructuredGrid(CARTGEOM, xs, xs; periodic = (true, true), period = (L, L))
    u = [cos(2π * 2 * x / L + 2π * y / L) + 0.5 * sin(2π * (-3) * x / L + 2π * 2 * y / L) for x in xs, y in xs]
    v = [sin(2π * 2 * x / L + 2π * y / L) for x in xs, y in xs]
    uv = cat(u, v; dims = 3)
    SUITE["cartesian_uniform"]["N=$N"]["fftw"] =
        @benchmarkable FFS.calculate_spectrum($g, $uv, ($N, $N); transform = SB.FFTSpectralBackend())
    N <= 128 && (SUITE["cartesian_uniform"]["N=$N"]["direct_sum"] =
        @benchmarkable FFS.calculate_spectrum($g, $uv, ($N, $N); transform = SB.DirectSumSpectralBackend()))
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
    g = FG.Grids.UnstructuredGrid(CARTGEOM, (xv, yv), ones(N); periodic = (true, true), period = (L, L))
    SUITE["cartesian_scattered"]["N=$N"]["finufft"] =
        @benchmarkable FFS.calculate_spectrum($g, $f, $ms; transform = FFS.FINUFFTBackend(), eps = 1e-9)
    N <= 10000 && (SUITE["cartesian_scattered"]["N=$N"]["direct_sum"] =
        @benchmarkable FFS.calculate_spectrum($g, $f, $ms; transform = SB.DirectSumSpectralBackend()))
end

# =============================================================================
# Spherical structured grid: DirectSum vs FastSphericalHarmonics (Clenshaw–Curtis grid).
# =============================================================================
SUITE["spherical_structured"] = BenchmarkGroup()
for lmax in [16, 32, 64]
    SUITE["spherical_structured"]["lmax=$lmax"] = BenchmarkGroup()
    Nθ = lmax + 1
    Nφ = 2 * lmax + 1
    g = FG.Connectivity.structured_grid(Float64, FG.SphericalSampling.ClenshawCurtisSampling(), Nθ)
    C = zeros(ComplexF64, Nθ, Nφ)
    C[FFS.sph_mode_index(2, 1)] = 1.0
    C[FFS.sph_mode_index(3, -2)] = 0.5
    f = real(FFS.synthesize(g, C, (Nθ, Nφ)))
    SUITE["spherical_structured"]["lmax=$lmax"]["fast_sht"] =
        @benchmarkable FFS.calculate_spectrum($g, $f, ($Nθ, $Nφ); transform = SB.FSHTSpectralBackend())
    lmax <= 32 && (SUITE["spherical_structured"]["lmax=$lmax"]["direct_sum"] =
        @benchmarkable FFS.calculate_spectrum($g, $f, ($Nθ, $Nφ); transform = SB.DirectSumSpectralBackend()))
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
    plan = NUFSHT.make_plan(Float64, θ, φ, lmax)
    f = zeros(N_pts)
    NUFSHT.nusht_type2!(f, C_true, plan)
    g = FG.Grids.UnstructuredGrid(FG.Geometry.SphericalGeometry(1.0), (φ, π / 2 .- θ), ones(N_pts))
    SUITE["spherical_scattered"]["lmax=$lmax"]["nufsht_adjoint"] =
        @benchmarkable FFS.calculate_spectrum($g, $f, ($Nθ, $Nφ); transform = SB.NUFSHTSpectralBackend(), solve = false, tol = 1e-8)
    lmax <= 16 && (SUITE["spherical_scattered"]["lmax=$lmax"]["nufsht_cg"] =
        @benchmarkable FFS.calculate_spectrum($g, $f, ($Nθ, $Nφ); transform = SB.NUFSHTSpectralBackend(), solve = true, rtol = 1e-6, maxiter = 500))
end

# =============================================================================
# Reductions.
# =============================================================================
SUITE["reductions"] = BenchmarkGroup()
let
    L = 10.0
    N = 128
    ms = (N, N)
    xs = _uaxis(L, N)
    g = FG.Grids.StructuredGrid(CARTGEOM, xs, xs; periodic = (true, true), period = (L, L))
    u = [cos(2π * 2 * x / L + 2π * y / L) for x in xs, y in xs]
    v = [sin(2π * 2 * x / L + 2π * y / L) for x in xs, y in xs]
    coeffs, ks = FFS.calculate_spectrum(g, cat(u, v; dims = 3), ms; transform = SB.FFTSpectralBackend())
    SUITE["reductions"]["isotropic_spectrum"] = @benchmarkable FFS.isotropic_spectrum($ks, $coeffs; num_bins = 32, dims = 3)
    SUITE["reductions"]["transect_spectrum"] = @benchmarkable FFS.transect_spectrum($ks, $coeffs, (1,))

    lmax = 16
    Nθ = lmax + 1
    Nφ = 2 * lmax + 1
    gs = FG.Connectivity.structured_grid(Float64, FG.SphericalSampling.ClenshawCurtisSampling(), Nθ)
    Ct = zeros(ComplexF64, Nθ, Nφ)
    Ct[FFS.sph_mode_index(2, 1)] = 1.0
    fsph = real(FFS.synthesize(gs, Ct, (Nθ, Nφ)))
    c_sph, _ = FFS.calculate_spectrum(gs, fsph, (Nθ, Nφ); transform = SB.FSHTSpectralBackend())
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
