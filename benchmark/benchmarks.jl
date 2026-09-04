using BenchmarkTools: BenchmarkTools, BenchmarkGroup, @benchmarkable, tune!, run
using FlowFieldSpectra: FlowFieldSpectra as FFS
using FFTW: FFTW
using FINUFFT: FINUFFT
using NonuniformFFTs: NonuniformFFTs
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

# =============================================================================
# Reusable plans: `plan_spectrum` once, then `calculate_spectrum!` per field. The one-shot beside each
# plan is what the plan is measured against, since the difference is the per-call setup the plan holds —
# FFTW's plan, a NUFFT's point sorting, a spherical grid's nodes and quadrature. `BenchmarkTools` records
# the allocations alongside the time, which is the quantity a time loop actually feels.
#
# CAVEAT, observed: a group's TIMING can differ several-fold between a whole-suite run and that group run
# alone — the NUFSHT plan measured 5.5 ms in a multi-group run against 0.32 ms alone and 0.14 ms outside
# the suite entirely, while its MEMORY was identical in all three. FFTW's thread count is process-global
# and every group here holds live FFTW/FINUFFT plans; which of those carries across was not established.
# Re-run a single group before believing a surprising time from it.
# =============================================================================
SUITE["plans"] = BenchmarkGroup()

# Uniform Cartesian: FFTW plan.
for N in [128, 256, 512]
    SUITE["plans"]["fft_N=$N"] = BenchmarkGroup()
    xs = _uaxis(2π, N)
    g = FG.Grids.StructuredGrid(CARTGEOM, xs, xs; periodic = (true, true), period = (2π, 2π))
    Random.seed!(1)
    f = randn(N, N)
    ms = (N, N)
    p = FFS.plan_spectrum(g, Float64, ms; transform = SB.FFTSpectralBackend())
    buf = zeros(ComplexF64, FFS.Packing.packed_size(ms, Val(true))...)
    SUITE["plans"]["fft_N=$N"]["one_shot"] =
        @benchmarkable FFS.calculate_spectrum($g, $f, $ms; transform = SB.FFTSpectralBackend())
    SUITE["plans"]["fft_N=$N"]["plan_exec"] = @benchmarkable FFS.calculate_spectrum!($buf, $p, $f)
end

# Uniform in one direction, stretched in the other: the hybrid FFT/NUFFT composite.
for N in [64, 128, 256]
    SUITE["plans"]["hybrid_N=$N"] = BenchmarkGroup()
    L = 2π
    uni = _uaxis(L, N)
    str = [L * (i - 1) / N + 0.04 * L * sinpi(2 * (i - 1) / N) for i in 1:N]
    g = FG.Grids.StructuredGrid(CARTGEOM, uni, str; periodic = (true, true), period = (L, L))
    Random.seed!(2)
    f = randn(N, N)
    ms = (N, N)
    p = FFS.plan_spectrum(g, Float64, ms; transform = SB.FFTSpectralBackend())
    buf = zeros(ComplexF64, FFS.Packing.packed_size(ms, Val(true))...)
    SUITE["plans"]["hybrid_N=$N"]["one_shot"] =
        @benchmarkable FFS.calculate_spectrum($g, $f, $ms; transform = SB.FFTSpectralBackend())
    SUITE["plans"]["hybrid_N=$N"]["plan_exec"] = @benchmarkable FFS.calculate_spectrum!($buf, $p, $f)
end

# Scattered Cartesian: the NUFFT guru plan.
for M in [10_000, 100_000]
    SUITE["plans"]["nufft_M=$M"] = BenchmarkGroup()
    L = 2π
    ms = (64, 64)
    Random.seed!(3)
    xv = rand(M) .* L
    yv = rand(M) .* L
    g = FG.Grids.UnstructuredGrid(CARTGEOM, (xv, yv), fill(L^2 / M, M);
        periodic = (true, true), period = (L, L))
    f = randn(M)
    p = FFS.plan_spectrum(g, Float64, ms; transform = FFS.NonuniformFFTsBackend(), eps = 1e-9)
    buf = zeros(ComplexF64, FFS.Packing.packed_size(ms, Val(true))...)
    SUITE["plans"]["nufft_M=$M"]["one_shot"] = @benchmarkable FFS.calculate_spectrum($g, $f, $ms;
        transform = FFS.NonuniformFFTsBackend(), eps = 1e-9)
    SUITE["plans"]["nufft_M=$M"]["plan_exec"] = @benchmarkable FFS.calculate_spectrum!($buf, $p, $f)
end

# Cartesian direct sum: the plan holds the per-axis DFT matrices, the contraction's working arrays and
# the Nyquist-twin storage. A stretched grid takes the factorized tensor path and carries a twin; a
# scattered cloud takes the direct sum.
for N in [32, 64, 128]
    SUITE["plans"]["directsum_N=$N"] = BenchmarkGroup()
    L = 2π
    str = [L * (i - 1) / N + 0.03 * L * sinpi(2 * (i - 1) / N) for i in 1:N]
    g = FG.Grids.StructuredGrid(CARTGEOM, str, str; periodic = (true, true), period = (L, L))
    Random.seed!(6)
    f = randn(N, N)
    ms = (N, N)
    p = FFS.plan_spectrum(g, Float64, ms; transform = SB.DirectSumSpectralBackend())
    buf = zeros(ComplexF64, FFS.Packing.packed_size(ms, Val(true))...)
    SUITE["plans"]["directsum_N=$N"]["one_shot"] =
        @benchmarkable FFS.calculate_spectrum($g, $f, $ms; transform = SB.DirectSumSpectralBackend())
    SUITE["plans"]["directsum_N=$N"]["plan_exec"] = @benchmarkable FFS.calculate_spectrum!($buf, $p, $f)
end

for M in [2_000, 8_000]
    SUITE["plans"]["directsum_cloud_M=$M"] = BenchmarkGroup()
    L = 2π
    ms = (32, 32)
    Random.seed!(7)
    xv = rand(M) .* L
    yv = rand(M) .* L
    g = FG.Grids.UnstructuredGrid(CARTGEOM, (xv, yv), fill(L^2 / M, M);
        periodic = (true, true), period = (L, L))
    f = randn(M)
    p = FFS.plan_spectrum(g, Float64, ms; transform = SB.DirectSumSpectralBackend())
    buf = zeros(ComplexF64, FFS.Packing.packed_size(ms, Val(true))...)
    SUITE["plans"]["directsum_cloud_M=$M"]["one_shot"] =
        @benchmarkable FFS.calculate_spectrum($g, $f, $ms; transform = SB.DirectSumSpectralBackend())
    SUITE["plans"]["directsum_cloud_M=$M"]["plan_exec"] =
        @benchmarkable FFS.calculate_spectrum!($buf, $p, $f)
end

# Spherical direct sum: the plan holds the nodes, the ring table, the quadrature and the Legendre tables.
for lmax in [32, 64, 96]
    SUITE["plans"]["sph_directsum_lmax=$lmax"] = BenchmarkGroup()
    ms = (lmax + 1, 2 * lmax + 1)
    g = FG.Connectivity.structured_grid(Float64, FG.SphericalSampling.GaussLegendreSampling(), lmax + 1)
    Random.seed!(4)
    f = randn(size(g)...)
    p = FFS.plan_spectrum(g, Float64, ms; transform = SB.DirectSumSpectralBackend())
    buf = zeros(Float64, ms...)                      # a real field's spherical coefficients are real
    SUITE["plans"]["sph_directsum_lmax=$lmax"]["one_shot"] =
        @benchmarkable FFS.calculate_spectrum($g, $f, $ms; transform = SB.DirectSumSpectralBackend())
    SUITE["plans"]["sph_directsum_lmax=$lmax"]["plan_exec"] =
        @benchmarkable FFS.calculate_spectrum!($buf, $p, $f)
end

# NUFSHT's `batch_chunk`: how many batch slices one execution carries. The optimum is a property of the
# NUFFT engine NUFSHT resolves and of the core count — one engine parallelizes over points and one over
# transforms — so this sweep is what a caller reads before setting it, and the default (0, the whole
# batch) is what every other row is measured against.
for lmax in [16, 32]
    SUITE["plans"]["nufsht_chunk_lmax=$lmax"] = BenchmarkGroup()
    ms = (lmax + 1, 2 * lmax + 1)
    N_pts = 8 * (lmax + 1)^2
    ga = π * (3 - sqrt(5))
    λ = [mod(ga * i, 2π) for i in 1:N_pts]
    φ = asin.([-1 + 2 * (i - 0.5) / N_pts for i in 1:N_pts])
    g = FG.Grids.UnstructuredGrid(FG.Geometry.SphericalGeometry(1.0), (λ, φ), fill(4π / N_pts, N_pts))
    Random.seed!(8)
    B = 16
    f = randn(N_pts, B)
    for bc in [0, 1, 2, 4, 8, 16]
        p = FFS.plan_spectrum(g, Float64, ms; transform = SB.NUFSHTSpectralBackend(),
            batch = (B,), batch_chunk = bc)
        buf = zeros(Float64, ms..., B)
        SUITE["plans"]["nufsht_chunk_lmax=$lmax"]["chunk=$bc"] =
            @benchmarkable FFS.calculate_spectrum!($buf, $p, $f)
    end
end

# Scattered spherical: the NUFSHT plan (points preset once).
for lmax in [16, 32]
    SUITE["plans"]["nufsht_lmax=$lmax"] = BenchmarkGroup()
    ms = (lmax + 1, 2 * lmax + 1)
    N_pts = 8 * (lmax + 1)^2
    ga = π * (3 - sqrt(5))
    λ = [mod(ga * i, 2π) for i in 1:N_pts]
    φ = asin.([-1 + 2 * (i - 0.5) / N_pts for i in 1:N_pts])
    g = FG.Grids.UnstructuredGrid(FG.Geometry.SphericalGeometry(1.0), (λ, φ), fill(4π / N_pts, N_pts))
    Random.seed!(5)
    f = randn(N_pts)
    p = FFS.plan_spectrum(g, Float64, ms; transform = SB.NUFSHTSpectralBackend())
    buf = zeros(Float64, ms...)
    SUITE["plans"]["nufsht_lmax=$lmax"]["one_shot"] =
        @benchmarkable FFS.calculate_spectrum($g, $f, $ms; transform = SB.NUFSHTSpectralBackend())
    SUITE["plans"]["nufsht_lmax=$lmax"]["plan_exec"] = @benchmarkable FFS.calculate_spectrum!($buf, $p, $f)
end

if abspath(PROGRAM_FILE) == @__FILE__
    println("Running FlowFieldSpectra.jl benchmarks...")
    tune!(SUITE)
    results = run(SUITE; verbose = true)
    println("\n" * "="^60 * "\nBENCHMARK RESULTS\n" * "="^60)
    display(results)
    BenchmarkTools.save("benchmark_results.json", results)
end
