# Dispatch audit: every (transform × execution × grid) either transforms or says why it cannot.
#
# The contract is that no combination reaches a `MethodError`. A `MethodError` names Julia internals and
# tells a caller nothing; an `ArgumentError` names the grid, the transform and the fix. So this walks the
# matrix and asserts on the KIND of outcome: coefficients of the documented shape, or a raise a caller
# can act on.
#
# Every extension is loaded here, so each raise is a statement about the grid itself.

using Test: Test
using FlowFieldSpectra: FlowFieldSpectra as FFS
using FlowGeometries: FlowGeometries as FG
using SpectralBackends: SpectralBackends as SB
using ComputationalBackends: ComputationalBackends as CB
using FFTW: FFTW
using NonuniformFFTs: NonuniformFFTs
using FINUFFT: FINUFFT
using FastSphericalHarmonics: FastSphericalHarmonics as FSH
using NUFSHT: NUFSHT
using OhMyThreads: OhMyThreads
using KernelAbstractions: KernelAbstractions as KA
using Random: Random
using Logging: Logging

# `:ok` when it transformed, `:raised` for a raise naming the problem, `:methoderror` for one that does
# not. Auto's fallback warning is silenced so the matrix output stays readable.
function dm_outcome(g, f, ms, transform, exec)
    return Logging.with_logger(Logging.NullLogger()) do
        try
            c, _ = FFS.calculate_spectrum(g, f, ms; transform = transform, execution = exec)
            (c isa AbstractArray && all(isfinite, c)) ? :ok : :nonfinite
        catch err
            err isa MethodError && return :methoderror
            (err isa ArgumentError || err isa DimensionMismatch) && return :raised
            rethrow()
        end
    end
end

const DM_CART = FG.Geometry.CartesianGeometry{Float64}()
const DM_SPH = FG.Geometry.SphericalGeometry(1.0)
const DM_SPHEROID = FG.Geometry.SpheroidGeometry(1.0, inv(298.257223563))

dm_unif(L, N) = range(0.0, L; length = N + 1)[1:N]
dm_stretch(L, N) = [L * (i - 1) / N + 0.04 * L * sinpi(2 * (i - 1) / N) for i in 1:N]

# (name, grid, field, ms) over every architecture × geometry the package dispatches on.
function dm_cases()
    Random.seed!(6060)
    L = 2π
    N = 8
    lmax = 3
    sms = (lmax + 1, 2 * lmax + 1)
    cases = Any[]

    # ---- Cartesian ----
    gu = FG.Grids.StructuredGrid(DM_CART, dm_unif(L, N), dm_unif(L, N);
        periodic = (true, true), period = (L, L))
    push!(cases, ("cartesian uniform", gu, randn(N, N), (N, N)))
    gm = FG.Grids.StructuredGrid(DM_CART, dm_unif(L, N), dm_stretch(L, N);
        periodic = (true, true), period = (L, L))
    push!(cases, ("cartesian mixed", gm, randn(N, N), (N, N)))
    gs = FG.Grids.StructuredGrid(DM_CART, dm_stretch(L, N), dm_stretch(L, N);
        periodic = (true, true), period = (L, L))
    push!(cases, ("cartesian stretched", gs, randn(N, N), (N, N)))
    M = 40
    gc = FG.Grids.UnstructuredGrid(DM_CART, (L .* rand(M), L .* rand(M)), fill(L^2 / M, M);
        periodic = (true, true), period = (L, L))
    push!(cases, ("cartesian cloud", gc, randn(M), (N, N)))
    X = [L * (i - 1) / N + 0.1 * L / N * sinpi(2 * (j - 1) / N) for i in 1:N, j in 1:N]
    Y = [L * (j - 1) / N + 0.1 * L / N * cospi(2 * (i - 1) / N) for i in 1:N, j in 1:N]
    gv = FG.Grids.CurvilinearGrid(DM_CART, (X, Y), nothing, fill(1.0, N, N), trues(N, N),
        (FG.Grids.Periodic(), FG.Grids.Periodic()), (L, L))
    push!(cases, ("cartesian curvilinear", gv, randn(N, N), (N, N)))

    # ---- Spherical ----
    ggl = FG.Connectivity.structured_grid(Float64, FG.SphericalSampling.GaussLegendreSampling(), lmax + 1)
    push!(cases, ("sphere gauss-legendre", ggl, randn(size(ggl)...), sms))
    Θ, Φ = FSH.sph_points(lmax + 1)
    gcc = FG.Grids.StructuredGrid(DM_SPH, collect(Φ), Float64(π) / 2 .- collect(Θ);
        periodic = (true, false), period = (2π, 0.0))
    push!(cases, ("sphere clenshaw-curtis", gcc, randn(size(gcc)...), sms))
    ghp = FG.Grids.HEALPixGrid(DM_SPH, 2)
    push!(cases, ("sphere healpix", ghp, randn(length(ghp)), sms))
    grg = FG.Grids.RingGrid(DM_SPH, FG.SphericalSampling.OctahedralGaussianSampling(2))
    push!(cases, ("sphere ring", grg, randn(length(grg)), sms))
    gcs = FG.Grids.CubedSphereGrid(DM_SPH, 4)
    push!(cases, ("sphere cubed", gcs, randn(length(gcs)), sms))
    gic = FG.Grids.IcosahedralGrid(DM_SPH, 2)
    push!(cases, ("sphere icosahedral", gic, randn(length(gic)), sms))
    Ns = 120
    ga = π * (3 - sqrt(5))
    λs = [mod(ga * i, 2π) for i in 1:Ns]
    φs = asin.([-1 + 2 * (i - 0.5) / Ns for i in 1:Ns])
    gsc = FG.Grids.UnstructuredGrid(DM_SPH, (λs, φs), fill(4π / Ns, Ns))
    push!(cases, ("sphere cloud", gsc, randn(Ns), sms))

    # ---- Spheroid ----
    nlon, nlat = 2 * lmax + 2, lmax + 2
    gsd = FG.Grids.StructuredGrid(DM_SPHEROID,
        [2π * (i - 1) / nlon for i in 1:nlon], [π * (j - 0.5) / nlat - π / 2 for j in 1:nlat];
        periodic = (true, false), period = (2π, 0.0))
    push!(cases, ("spheroid surface", gsd, randn(nlon, nlat), sms))
    gsdc = FG.Grids.UnstructuredGrid(DM_SPHEROID, (λs, φs), fill(4π / Ns, Ns))
    push!(cases, ("spheroid cloud", gsdc, randn(Ns), sms))

    return cases
end

const DM_TRANSFORMS = (
    ("auto", SB.AutoSpectralBackend()),
    ("directsum", SB.DirectSumSpectralBackend()),
    ("fft", SB.FFTSpectralBackend()),
    ("nonuniformffts", FFS.NonuniformFFTsBackend()),
    ("finufft", FFS.FINUFFTBackend()),
    ("fsht", SB.FSHTSpectralBackend()),
    ("nufsht", SB.NUFSHTSpectralBackend()),
)

Test.@testset "No (transform × execution × grid) reaches a MethodError" begin
    cases = dm_cases()
    for exec in (CB.SerialBackend(), CB.ThreadedBackend(), CB.GPUBackend(KA.CPU()))
        for (gname, g, f, ms) in cases, (tname, t) in DM_TRANSFORMS
            out = dm_outcome(g, f, ms, t, exec)
            Test.@test out !== :methoderror
            Test.@test out !== :nonfinite
        end
    end
end

Test.@testset "Auto transforms every grid the package dispatches on" begin
    # Auto's job is to find a working transform, so with every extension loaded it succeeds on each grid;
    # a raise here marks a geometry/architecture no backend covers.
    for (gname, g, f, ms) in dm_cases()
        Test.@test dm_outcome(g, f, ms, SB.AutoSpectralBackend(), CB.SerialBackend()) === :ok
    end
end

Test.@testset "A direction's Fourier length comes from its periodicity flag" begin
    # `period`'s own contract is that it is meaningful only where `isperiodic` holds, and a node cloud or
    # a curvilinear grid stores whatever `period` it was given even on a direction declared
    # non-periodic. So the FLAG decides the Fourier length. A `StructuredGrid` normalizes such an entry
    # to zero, so this shows up only on the pointwise architectures.
    L = 2π
    N = 8
    xs = collect(dm_unif(L, N))
    cloud = FG.Grids.UnstructuredGrid(DM_CART, (xs, xs), fill(1.0, N);
        periodic = (true, false), period = (L, 5.0))
    Test.@test FG.Grids.period(cloud, 2) == 5.0        # the grid carries it
    Test.@test !FG.Grids.isperiodic(cloud, 2)          # and declares the direction non-periodic
    offs, ranges = FFS.Grids.axis_geometry(Float64, cloud, 2)
    Test.@test ranges[1] ≈ L                           # the periodic axis uses its wrap length
    Test.@test ranges[2] == 1.0                        # the non-periodic one is raw per-sample

    # The wavenumber scaling follows the same flag: `2π/L` on the periodic axis, `2π` on the other.
    ks = FFS.Grids.physical_wavenumbers(cloud, (N, N), Val(true))
    Test.@test isapprox(ks[1][2] - ks[1][1], 2π / L; rtol = 1e-12)
    Test.@test isapprox(ks[2][2] - ks[2][1], 2π; rtol = 1e-12)

    # `offsets` is each direction's smallest coordinate, read through `bounds`.
    Test.@test offs[1] ≈ minimum(xs)
    Test.@test offs[2] ≈ minimum(xs)

    # A structured grid zeroes the entry, so both readings agree there.
    gs = FG.Grids.StructuredGrid(DM_CART, dm_unif(L, N), dm_unif(L, N);
        periodic = (true, false), period = (L, 5.0))
    Test.@test FG.Grids.period(gs, 2) == 0.0
    _, rs = FFS.Grids.axis_geometry(Float64, gs, 2)
    Test.@test rs[2] == 1.0

    # `bounds` orders a direction's extremes, so a descending axis still reports its smallest coordinate.
    gd = FG.Grids.StructuredGrid(DM_CART, range(L, 0.0; length = N), dm_unif(L, N);
        periodic = (false, true), period = (0.0, L))
    od, _ = FFS.Grids.axis_geometry(Float64, gd, 2)
    Test.@test od[1] ≈ 0.0
end

Test.@testset "The catch-all names the grid and the transform" begin
    # A spheroid grid reaching FastSphericalHarmonics is a mistake, and the message states which grid and
    # which transform, so the raise comes from the hub as an `ArgumentError`.
    lmax = 3
    sms = (lmax + 1, 2 * lmax + 1)
    nlon, nlat = 2 * lmax + 1, lmax + 1
    g = FG.Grids.StructuredGrid(DM_SPHEROID,
        [2π * (i - 1) / nlon for i in 1:nlon], [π * (j - 0.5) / nlat - π / 2 for j in 1:nlat];
        periodic = (true, false), period = (2π, 0.0))
    f = randn(nlon, nlat)
    err = try
        FFS.calculate_spectrum(g, f, sms; transform = SB.FSHTSpectralBackend(),
            execution = CB.SerialBackend())
        nothing
    catch e
        e
    end
    Test.@test err isa ArgumentError
    Test.@test occursin("SpheroidGeometry", err.msg)
    # `FSHTSpectralBackend` is an alias, so the message carries the concrete type's own name.
    Test.@test occursin("SphericalHarmonics", err.msg)

    # An FFT on a spherical grid likewise.
    gsph = FG.Connectivity.structured_grid(Float64, FG.SphericalSampling.GaussLegendreSampling(), lmax + 1)
    Test.@test_throws ArgumentError FFS.calculate_spectrum(gsph, randn(size(gsph)...), sms;
        transform = SB.FFTSpectralBackend(), execution = CB.SerialBackend())
end
