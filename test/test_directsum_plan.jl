# The direct sum through its reusable plans. On SerialBackend the one-shot runs the same setup and
# contraction once, so a reused plan reproduces it bit for bit on every field it is handed; under
# ThreadedBackend the plans split the sum across tasks and agree with it to round-off.

using Test: Test
using Random: Random
using FlowFieldSpectra: FlowFieldSpectra as FFS
using FlowTransformBindings: FlowTransformBindings as FTB
using FlowGeometries: FlowGeometries as FG
using SpectralBackends: SpectralBackends as SB
using ComputationalBackends: ComputationalBackends as CB
using OhMyThreads: OhMyThreads

const DP_DS = SB.DirectSumSpectralBackend()
const DP_SER = CB.SerialBackend()
dp_cart() = FG.Geometry.CartesianGeometry{Float64}()
dp_str(L, N) = collect(range(0.0, L; length = N + 1)[1:N]) .+ (0.03 * L / N) .* sinpi.(2 .* (0:(N - 1)) ./ N)

Test.@testset "Cartesian direct-sum plan reuses its setup" begin
    Random.seed!(62)
    L = 2π
    N = 8
    ms = (N, N)
    gs = FG.Grids.StructuredGrid(dp_cart(), dp_str(L, N), dp_str(L, N); topology = (FG.Grids.Periodic(), FG.Grids.Periodic()), period = (L, L))
    gc = FG.Grids.UnstructuredGrid(dp_cart(), (L .* rand(40), L .* rand(40)), fill(L^2 / 40, 40);
        topology = (FG.Grids.Periodic(), FG.Grids.Periodic()), period = (L, L))

    for (g, n) in ((gs, (N, N)), (gc, (40,)))
        p = FFS.plan_spectrum(g, Float64, ms; transform = DP_DS, execution = DP_SER)
        buf = FFS.allocate_coefficients(p)
        for _ in 1:2
            f = randn(n...)
            keep = copy(f)
            ks = FFS.calculate_spectrum!(buf, p, f)
            c1, k1 = FFS.calculate_spectrum(g, f, ms; transform = DP_DS, execution = DP_SER)
            Test.@test buf == c1
            Test.@test map(collect, ks) == map(collect, k1)
            Test.@test f == keep
        end
        # A field of another batch length is refused.
        Test.@test_throws DimensionMismatch FFS.calculate_spectrum!(buf, p, randn(2 * prod(n)))
    end
end

Test.@testset "A uniform grid transformed below its own length widens its even axes" begin
    # Below the grid's length an even axis's `±ms_d/2` are two modes on its samples, so a real field's
    # packed half covers `ms_d + 1` modes there. It unpacks to the complex field's spectrum over that odd
    # box, and synthesizes as the complex series over it does.
    Random.seed!(64)
    L = 2π
    N = 12
    ax = range(0.0, L; length = N + 1)[1:N]
    g = FG.Grids.StructuredGrid(dp_cart(), ax, ax; topology = (FG.Grids.Periodic(), FG.Grids.Periodic()), period = (L, L))
    f = randn(N, N)
    for (ms, box) in (((8, 8), (9, 9)), ((7, 8), (7, 9)), ((8, 6), (9, 7)))
        c, ks = FFS.calculate_spectrum(g, f, ms; transform = DP_DS, execution = DP_SER)
        Test.@test FTB.full_size(ks) == box
        cf, _ = FFS.calculate_spectrum(g, complex.(f), box; transform = DP_DS, execution = DP_SER)
        Test.@test maximum(abs, FTB.unpacked(c, box) .- cf) < 1e-13
        sf = FFS.synthesize(ComplexF64, g, cf, box; transform = DP_DS, execution = DP_SER)
        Test.@test maximum(abs, FFS.synthesize(g, c, ms; transform = DP_DS, execution = DP_SER) .- real.(sf)) <
            1e-12
    end
end

dp_rel(a, b) = maximum(abs, a .- b) / max(maximum(abs, b), eps())

Test.@testset "Direct-sum plans under ThreadedBackend" begin
    # A point set's modes, or a sphere's latitudes, rings or points, are split across tasks; each reused
    # plan equals the serial one-shot to round-off.
    Random.seed!(63)
    thr = CB.ThreadedBackend()
    cases = ((Float64, ()), (Float64, (3,)), (ComplexF64, ()))
    L = 2π
    N = 8
    ms = (N, N)
    gs = FG.Grids.StructuredGrid(dp_cart(), dp_str(L, N), dp_str(L, N); topology = (FG.Grids.Periodic(), FG.Grids.Periodic()), period = (L, L))
    gc = FG.Grids.UnstructuredGrid(dp_cart(), (L .* rand(40), L .* rand(40)), fill(L^2 / 40, 40);
        topology = (FG.Grids.Periodic(), FG.Grids.Periodic()), period = (L, L))
    for (g, n) in ((gs, (N, N)), (gc, (40,))), (T, batch) in cases
        p = FFS.plan_spectrum(g, T, ms; transform = DP_DS, execution = thr, batch = batch)
        buf = FFS.allocate_coefficients(p)
        for _ in 1:2
            f = randn(T, n..., batch...)
            ks = FFS.calculate_spectrum!(buf, p, f)
            c1, k1 = FFS.calculate_spectrum(g, f, ms; transform = DP_DS, execution = DP_SER)
            Test.@test dp_rel(buf, c1) < 1e-13
            Test.@test map(collect, ks) == map(collect, k1)
        end
    end

    lmax = 5
    msph = (lmax + 1, 2 * lmax + 1)
    sph = FG.Geometry.SphericalGeometry(1.0)
    Ns = 80
    ga = π * (3 - sqrt(5))
    grids = (
        FG.Connectivity.structured_grid(Float64, FG.SphericalSampling.GaussLegendreSampling(), lmax + 1),
        FG.Grids.HEALPixGrid(sph, 4),
        FG.Grids.UnstructuredGrid(sph, ([mod(ga * i, 2π) for i in 1:Ns],
            asin.([-1 + 2 * (i - 0.5) / Ns for i in 1:Ns])), fill(4π / Ns, Ns)),
    )
    for g in grids, (T, batch) in cases
        p = FFS.plan_spectrum(g, T, msph; transform = DP_DS, execution = thr, batch = batch)
        buf = FFS.allocate_coefficients(p)
        for _ in 1:2
            f = randn(T, size(g)..., batch...)
            FFS.calculate_spectrum!(buf, p, f)
            c1, _ = FFS.calculate_spectrum(g, f, msph; transform = DP_DS, execution = DP_SER)
            Test.@test dp_rel(buf, c1) < 1e-13
        end
    end
end
