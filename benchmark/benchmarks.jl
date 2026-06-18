using BenchmarkTools: BenchmarkTools, BenchmarkGroup, @benchmarkable, tune!, run
using FlowFieldSpectra: FlowFieldSpectra as FFS
using FFTW: FFTW
using FINUFFT: FINUFFT
using FastSphericalHarmonics: FastSphericalHarmonics as FSH
using NUFSHT: NUFSHT
using Random: Random
using Statistics: Statistics

const SUITE = BenchmarkGroup()

# =============================================================================
# Cartesian Uniform Grid Benchmarks
# =============================================================================
SUITE["cartesian_uniform"] = BenchmarkGroup()

for N in [64, 128, 256]
    SUITE["cartesian_uniform"]["N=$N"] = BenchmarkGroup()

    # Setup
    L = 10.0
    ms = (N, N)
    dx = L / N
    dy = L / N
    xs = range(0.0, stop = L - dx, length = N)
    ys = range(0.0, stop = L - dy, length = N)
    xv = vec([x for x in xs, y in ys])
    yv = vec([y for x in xs, y in ys])

    kx1, ky1 = 2π * 2 / L, 2π * 1 / L
    kx2, ky2 = 2π * (-3) / L, 2π * 2 / L
    u = @. cos(kx1 * xv + ky1 * yv) + 0.5 * sin(kx2 * xv + ky2 * yv)
    v = @. sin(kx1 * xv + ky1 * yv)

    # DirectSum backend
    SUITE["cartesian_uniform"]["N=$N"]["direct_sum"] =
        @benchmarkable FFS.calculate_spectrum(
            FFS.DirectSumBackend(), ($xv, $yv), ($u, $v), $ms;
            domain_size = ($L, $L)
        )

    # FFTW backend
    SUITE["cartesian_uniform"]["N=$N"]["fftw"] =
        @benchmarkable FFS.calculate_spectrum(
            FFS.FFTBackend(), ($xv, $yv), ($u, $v), $ms;
            domain_size = ($L, $L)
        )
end

# =============================================================================
# Cartesian Non-Uniform (Scattered) Benchmarks
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
    u = @. cos(kx * xv + ky * yv)
    v = @. sin(kx * xv + ky * yv)

    # DirectSum backend (very slow for large N)
    if N <= 10000
        SUITE["cartesian_scattered"]["N=$N"]["direct_sum"] =
            @benchmarkable FFS.calculate_spectrum(
                FFS.DirectSumBackend(), ($xv, $yv), ($u, $v), $ms;
                domain_size = ($L, $L)
            )
    end

    # FINUFFT backend
    SUITE["cartesian_scattered"]["N=$N"]["finufft"] =
        @benchmarkable FFS.calculate_spectrum(
            FFS.NUFFTBackend(), ($xv, $yv), ($u, $v), $ms;
            domain_size = ($L, $L), eps = 1e-9
        )
end

# =============================================================================
# Spherical Structured Grid Benchmarks
# =============================================================================
SUITE["spherical_structured"] = BenchmarkGroup()

for lmax in [16, 32, 64]
    SUITE["spherical_structured"]["lmax=$lmax"] = BenchmarkGroup()

    Nθ = lmax + 1
    Nφ = 2 * lmax + 1

    # Clenshaw-Curtis grid points
    pts = FSH.sph_points(Nθ)
    theta_nodes = vec([θ for θ in pts[1], φ in pts[2]])
    phi_nodes = vec([φ for θ in pts[1], φ in pts[2]])

    # Test field
    C_true = zeros(Nθ, Nφ)
    C_true[FSH.sph_mode(2, 1)] = 1.0
    C_true[FSH.sph_mode(3, -2)] = 0.5
    f_val = vec(FSH.sph_evaluate(C_true))

    # DirectSum backend
    if lmax <= 32
        SUITE["spherical_structured"]["lmax=$lmax"]["direct_sum"] =
            @benchmarkable FFS.calculate_spectrum(
                FFS.DirectSumBackend(), ($theta_nodes, $phi_nodes), ($f_val,), ($Nθ, $Nφ)
            )
    end

    # FastSphericalHarmonics backend
    SUITE["spherical_structured"]["lmax=$lmax"]["fast_sht"] =
        @benchmarkable FFS.calculate_spectrum(
            FFS.SHTBackend(), ($theta_nodes, $phi_nodes), ($f_val,), ($Nθ, $Nφ)
        )
end

# =============================================================================
# Spherical Unstructured (Scattered) Benchmarks
# =============================================================================
SUITE["spherical_scattered"] = BenchmarkGroup()

for lmax in [8, 16, 32]
    SUITE["spherical_scattered"]["lmax=$lmax"] = BenchmarkGroup()

    Nθ = lmax + 1
    Nφ = 2 * lmax + 1
    N_modes = Nθ^2
    N_pts = 4 * N_modes

    Random.seed!(42)
    φ_base = (2π / N_pts) .* (0:N_pts-1)
    θ_base = acos.(clamp.(2 .* ((0:N_pts-1) .+ 0.5) ./ N_pts .- 1, -1.0, 1.0))
    theta_nodes = θ_base .+ (rand(N_pts) .- 0.5) .* (0.4 * π / sqrt(N_pts))
    phi_nodes = mod.(φ_base .+ (rand(N_pts) .- 0.5) .* (0.4 * 2π / sqrt(N_pts)), 2π)
    theta_nodes = clamp.(theta_nodes, 1e-10, π - 1e-10)

    C_true = zeros(Nθ, Nφ)
    C_true[FSH.sph_mode(2, 0)] = 1.0

    plan = NUFSHT.make_plan(theta_nodes, phi_nodes, lmax)
    f_val = zeros(N_pts)
    NUFSHT.nusht_type2!(f_val, C_true, plan)

    # DirectSum backend (very slow for scattered)
    if lmax <= 16
        w = fill(4π / N_pts, N_pts)
        SUITE["spherical_scattered"]["lmax=$lmax"]["direct_sum"] =
            @benchmarkable FFS.calculate_spectrum(
                FFS.DirectSumBackend(), ($theta_nodes, $phi_nodes), ($f_val,), ($Nθ, $Nφ);
                weights = $w
            )
    end

    # NUFSHT backend (adjoint)
    SUITE["spherical_scattered"]["lmax=$lmax"]["nufsht_adjoint"] =
        @benchmarkable FFS.calculate_spectrum(
            FFS.NUFSHTBackend(), ($theta_nodes, $phi_nodes), ($f_val,), ($Nθ, $Nφ);
            solve = false, tol = 1e-8
        )

    # NUFSHT backend (CG solve) - more expensive
    if lmax <= 16
        SUITE["spherical_scattered"]["lmax=$lmax"]["nufsht_cg"] =
            @benchmarkable FFS.calculate_spectrum(
                FFS.NUFSHTBackend(), ($theta_nodes, $phi_nodes), ($f_val,), ($Nθ, $Nφ);
                solve = true, rtol = 1e-6, maxiter = 500
            )
    end
end

# =============================================================================
# Spectral Reductions Benchmarks
# =============================================================================
SUITE["reductions"] = BenchmarkGroup()

# Setup for reductions
L = 10.0
N = 128
ms = (N, N)
dx = L / N
dy = L / N
xs = range(0.0, stop = L - dx, length = N)
ys = range(0.0, stop = L - dy, length = N)
xv = vec([x for x in xs, y in ys])
yv = vec([y for x in xs, y in ys])

kx1, ky1 = 2π * 2 / L, 2π * 1 / L
u = @. cos(kx1 * xv + ky1 * yv)
v = @. sin(kx1 * xv + ky1 * yv)

coeffs, ks = FFS.calculate_spectrum(FFS.FFTBackend(), (xv, yv), (u, v), ms; domain_size=(L, L))

SUITE["reductions"]["isotropic_spectrum"] =
    @benchmarkable FFS.isotropic_spectrum($ks, $coeffs; num_bins=32)

SUITE["reductions"]["transect_spectrum"] =
    @benchmarkable FFS.transect_spectrum($ks, $coeffs, (1,))

# Spherical energy spectrum reduction
lmax = 16
Nθ = lmax + 1
Nφ = 2 * lmax + 1
pts = FSH.sph_points(Nθ)
theta_nodes = vec([θ for θ in pts[1], φ in pts[2]])
phi_nodes = vec([φ for θ in pts[1], φ in pts[2]])
C_true = zeros(Nθ, Nφ)
C_true[FSH.sph_mode(2, 1)] = 1.0
f_val = vec(FSH.sph_evaluate(C_true))
c_sph, _ = FFS.calculate_spectrum(FFS.SHTBackend(), (theta_nodes, phi_nodes), (f_val,), (Nθ, Nφ))

SUITE["reductions"]["spherical_energy_spectrum"] =
    @benchmarkable FFS.spherical_energy_spectrum($c_sph; lmax=$lmax)

# =============================================================================
# Main execution
# =============================================================================
if abspath(PROGRAM_FILE) == @__FILE__
    println("Running FlowFieldSpectra.jl benchmarks...")
    println("="^60)

    # Tune benchmarks
    println("Tuning benchmarks...")
    tune!(SUITE)

    # Run benchmarks
    println("Running benchmarks...")
    results = run(SUITE; verbose = true)

    # Display results
    println("\n" * "="^60)
    println("BENCHMARK RESULTS")
    println("="^60)
    display(results)

    # Save results
    println("\nSaving results to benchmark_results.json...")
    BenchmarkTools.save("benchmark_results.json", results)
end
