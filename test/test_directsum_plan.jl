# The Cartesian direct sum through its reusable plan.
#
# `DirectSumCartesianPlan` holds the per-axis dense DFT matrices, the working arrays the contraction walks
# through, the Nyquist-twin storage and the quadrature-scaled field buffer. The one-shot composes the same
# `cart_setup`/`cart_run!` pair, so the two agree BIT FOR BIT and these testsets assert `== 0.0`.
#
# The execution is also asserted to allocate EXACTLY zero, including on the grids that carry a twin: a
# twin is returned output, and the plan holds its slices.

using Test: Test
using Random: Random
using FlowFieldSpectra: FlowFieldSpectra as FFS
using FlowGeometries: FlowGeometries as FG
using SpectralBackends: SpectralBackends as SB
using ComputationalBackends: ComputationalBackends as CB
using OhMyThreads: OhMyThreads

const DP_DS = SB.DirectSumSpectralBackend()
const DP_SER = CB.SerialBackend()
dp_rel(a, b) = maximum(abs, a .- b) / max(maximum(abs, b), eps())
dp_cart() = FG.Geometry.CartesianGeometry{Float64}()
dp_uni(L, N) = range(0.0, L; length = N + 1)[1:N]
dp_str(L, N) = collect(dp_uni(L, N)) .+ (0.03 * L / N) .* sinpi.(2 .* (0:(N - 1)) ./ N)
dp_struct(L, axs) = FG.Grids.StructuredGrid(dp_cart(), axs...;
    periodic = ntuple(_ -> true, length(axs)), period = ntuple(_ -> L, length(axs)))
dp_exec(c, p, f) = @allocated FFS.calculate_spectrum!(c, p, f)

# `(plan result, one-shot result, plan twin, one-shot twin, plan)` for one grid and field.
function dp_pair(g, f, ms; batch = (), exec = DP_SER)
    T = eltype(f) <: Real ? Float64 : ComplexF64
    c1, k1 = FFS.calculate_spectrum(g, f, ms; transform = DP_DS, execution = exec)
    p = FFS.plan_spectrum(g, T, ms; transform = DP_DS, execution = exec, batch = batch)
    c2 = similar(c1)
    k2 = FFS.calculate_spectrum!(c2, p, f)
    return c2, c1, FFS.Packing.axis_twin(k2[1]), FFS.Packing.axis_twin(k1[1]), p
end

# Every nonempty twin slice, plan against one-shot.
function dp_twin_agrees(t2, t1)
    (t1 === nothing || t2 === nothing) && return t1 === t2 === nothing
    for i in eachindex(t1.slices)
        isempty(t1.slices[i]) && continue
        dp_rel(t2.slices[i], t1.slices[i]) == 0.0 || return false
    end
    return true
end

Test.@testset "Cartesian direct-sum plan equals its one-shot bit for bit" begin
    Random.seed!(61)
    L = 2π
    # Even and odd extents on each axis: an even axis `d ≥ 2` requires a twin, and an odd one empties
    # that mask's slice.
    for (N1, N2) in ((8, 8), (9, 8), (8, 9), (9, 9))
        f = randn(N1, N2)
        for axs in ((dp_uni(L, N1), dp_uni(L, N2)), (dp_str(L, N1), dp_str(L, N2)))
            g = dp_struct(L, axs)
            c2, c1, t2, t1, _ = dp_pair(g, f, (N1, N2))
            Test.@test size(c2) == size(c1)
            Test.@test dp_rel(c2, c1) == 0.0
            Test.@test dp_twin_agrees(t2, t1)
        end
    end

    N = 8
    gs = dp_struct(L, (dp_str(L, N), dp_str(L, N)))
    # Complex field, a batch axis, and fewer requested modes than the grid has points.
    for (f, ms, batch) in ((randn(ComplexF64, N, N), (N, N), ()),
                           (randn(N, N, 3), (N, N), (3,)),
                           (randn(N, N), (6, 6), ()))
        c2, c1, t2, t1, _ = dp_pair(gs, f, ms; batch = batch)
        Test.@test dp_rel(c2, c1) == 0.0
        Test.@test dp_twin_agrees(t2, t1)
    end

    # 1-D and 3-D: the contraction chain's ends.
    g1 = dp_struct(L, (dp_str(L, 12),))
    c2, c1, _, _, _ = dp_pair(g1, randn(12), (12,))
    Test.@test dp_rel(c2, c1) == 0.0
    g3 = dp_struct(L, (dp_str(L, 6), dp_str(L, 6), dp_str(L, 6)))
    for (f, batch) in ((randn(6, 6, 6), ()), (randn(6, 6, 6, 2), (2,)))
        c2, c1, t2, t1, _ = dp_pair(g3, f, (6, 6, 6); batch = batch)
        Test.@test dp_rel(c2, c1) == 0.0
        Test.@test dp_twin_agrees(t2, t1)
    end

    # A point cloud takes the direct sum, and a curvilinear grid reads as one.
    M = 50
    gc = FG.Grids.UnstructuredGrid(dp_cart(), (L .* rand(M), L .* rand(M)), fill(L^2 / M, M);
        periodic = (true, true), period = (L, L))
    for (f, batch) in ((randn(M), ()), (randn(M, 2), (2,)), (randn(ComplexF64, M), ()))
        c2, c1, t2, t1, _ = dp_pair(gc, f, (N, N); batch = batch)
        Test.@test dp_rel(c2, c1) == 0.0
        Test.@test dp_twin_agrees(t2, t1)
    end
    n = 6
    X = [L * (i - 1) / n + 0.1 * L / n * sinpi(2 * (j - 1) / n) for i in 1:n, j in 1:n]
    Y = [L * (j - 1) / n + 0.1 * L / n * cospi(2 * (i - 1) / n) for i in 1:n, j in 1:n]
    meas = [1.0 + 0.3 * sinpi(2 * (i + j) / n) for i in 1:n, j in 1:n]
    gv = FG.Grids.CurvilinearGrid(dp_cart(), (X, Y), nothing, meas, trues(n, n),
        (FG.Grids.Periodic(), FG.Grids.Periodic()), (L, L))
    c2, c1, t2, t1, _ = dp_pair(gv, randn(n, n), (n, n))
    Test.@test dp_rel(c2, c1) == 0.0
    Test.@test dp_twin_agrees(t2, t1)

    # The threaded execution builds and runs its own plan to the same values.
    c2, c1, t2, t1, _ = dp_pair(gs, randn(N, N), (N, N); exec = CB.ThreadedBackend())
    Test.@test dp_rel(c2, c1) == 0.0
    Test.@test dp_twin_agrees(t2, t1)
end

Test.@testset "Cartesian direct-sum plan reuses its setup" begin
    Random.seed!(62)
    L = 2π
    N = 8
    ms = (N, N)
    gs = dp_struct(L, (dp_str(L, N), dp_str(L, N)))
    gc = FG.Grids.UnstructuredGrid(dp_cart(), (L .* rand(40), L .* rand(40)), fill(L^2 / 40, 40);
        periodic = (true, true), period = (L, L))

    for (g, f) in ((gs, randn(N, N)), (gc, randn(40)))
        p = FFS.plan_spectrum(g, Float64, ms; transform = DP_DS, execution = DP_SER)
        pms = FFS.Packing.packed_size(ms, Val(true))
        buf = zeros(ComplexF64, pms...)
        ks = FFS.calculate_spectrum!(buf, p, f)
        Test.@test FFS.Packing.axis_twin(ks[1]) !== nothing   # both grids need a twin, so the check is live
        first = copy(buf)
        first_twin = copy(FFS.Packing.axis_twin(ks[1]).slices[1])

        # A second execution of the same field reproduces it exactly, twin included.
        FFS.calculate_spectrum!(buf, p, f)
        Test.@test buf == first
        Test.@test FFS.Packing.axis_twin(ks[1]).slices[1] == first_twin

        # A different field moves both, and matches that field's own one-shot.
        f2 = randn(size(f)...)
        FFS.calculate_spectrum!(buf, p, f2)
        Test.@test buf != first
        c1, _ = FFS.calculate_spectrum(g, f2, ms; transform = DP_DS, execution = DP_SER)
        Test.@test dp_rel(buf, c1) == 0.0

        # The caller's field is never written to.
        f3 = randn(size(f)...)
        keep = copy(f3)
        FFS.calculate_spectrum!(buf, p, f3)
        Test.@test f3 == keep

        # Steady state allocates nothing, twin storage included.
        dp_exec(buf, p, f)
        Test.@test dp_exec(buf, p, f) == 0

        # A field of the wrong batch length is rejected.
        Test.@test_throws DimensionMismatch FFS.calculate_spectrum!(buf, p, vcat(vec(f), vec(f)))
    end

    # A batched plan is fixed to its batch shape.
    pb = FFS.plan_spectrum(gs, Float64, ms; transform = DP_DS, execution = DP_SER, batch = (2,))
    bufb = zeros(ComplexF64, FFS.Packing.packed_size(ms, Val(true))..., 2)
    FFS.calculate_spectrum!(bufb, pb, randn(N, N, 2))
    Test.@test dp_exec(bufb, pb, randn(N, N, 2)) == 0
    Test.@test_throws DimensionMismatch FFS.calculate_spectrum!(bufb, pb, randn(N, N))

    # A spherical grid gets the spherical plan.
    ggl = FG.Connectivity.structured_grid(Float64, FG.SphericalSampling.GaussLegendreSampling(), 5)
    Test.@test FFS.plan_spectrum(ggl, Float64, (5, 9); transform = DP_DS, execution = DP_SER) isa
        FFS.DirectSumSphericalPlan
end
