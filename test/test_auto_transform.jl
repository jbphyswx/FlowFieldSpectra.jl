# `AutoSpectralBackend` — the default `transform` — resolves to the fastest transform a grid admits among
# the loaded extensions.
#
# Two things are gated here. FIRST, the choice: `FFS._resolve_transform` is asserted directly, so a
# routing change shows up as a routing failure. SECOND, the equivalence that licenses the choice: Auto's
# coefficients must match the direct sum's on the same grid, since a caller who does not name a transform
# still gets the same answer.
#
# Every extension is loaded in this suite, so the preconditions are the grid's alone.

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
using KernelAbstractions: KernelAbstractions as KA
using Random: Random

const AT_DS = SB.DirectSumSpectralBackend()
const AT_SER = CB.SerialBackend()
at_rel(a, b) = maximum(abs, a .- b) / max(maximum(abs, b), eps())
at_auto(g, ms, exec = AT_SER) = FFS._resolve_transform(SB.AutoSpectralBackend(), g, ms, exec)

# A uniform axis is an `AbstractRange` by type; a collected copy of one is stretched.
at_unif(L, N) = range(0.0, L; length = N + 1)[1:N]
at_stretch(L, N) = [L * (i - 1) / N + 0.03 * L * sinpi(2 * (i - 1) / N) for i in 1:N]

Test.@testset "Auto picks the fast Cartesian transform" begin
    L = 2π
    N = 8
    cart = FG.Geometry.CartesianGeometry{Float64}()

    # Uniform in every direction, asked for its own length: the FFT.
    gu = FG.Grids.StructuredGrid(cart, at_unif(L, N), at_unif(L, N);
        periodic = (true, true), period = (L, L))
    Test.@test at_auto(gu, (N, N)) isa SB.AbstractFFTSpectralBackend

    # Fewer modes than the grid has points on a uniform axis: an FFT cannot serve that, so Auto takes a
    # NUFFT, which can.
    Test.@test at_auto(gu, (4, N)) isa SB.AbstractNUFFTSpectralBackend

    # Uniform in axis 1, stretched in axis 2: the hybrid composite, which the FFT tag selects.
    gh = FG.Grids.StructuredGrid(cart, at_unif(L, N), at_stretch(L, N);
        periodic = (true, true), period = (L, L))
    Test.@test at_auto(gh, (N, N)) isa SB.AbstractFFTSpectralBackend

    # Axis 1 stretched: the hybrid needs axis 1 uniform, so Auto takes the separable NUFFT.
    gs = FG.Grids.StructuredGrid(cart, at_stretch(L, N), at_unif(L, N);
        periodic = (true, true), period = (L, L))
    Test.@test at_auto(gs, (N, N)) isa SB.AbstractNUFFTSpectralBackend

    # Every axis stretched, and a node cloud: the NUFFT.
    ga = FG.Grids.StructuredGrid(cart, at_stretch(L, N), at_stretch(L, N);
        periodic = (true, true), period = (L, L))
    Test.@test at_auto(ga, (N, N)) isa SB.AbstractNUFFTSpectralBackend
    Random.seed!(77)
    cloud = FG.Grids.UnstructuredGrid(cart, (L .* rand(50), L .* rand(50)), fill(L^2 / 50, 50);
        periodic = (true, true), period = (L, L))
    Test.@test at_auto(cloud, (N, N)) isa SB.AbstractNUFFTSpectralBackend

    # A curvilinear grid reads as a point cloud, so it takes the NUFFT too.
    X = [L * (i - 1) / N + 0.1 * L / N * sinpi(2 * (j - 1) / N) for i in 1:N, j in 1:N]
    Y = [L * (j - 1) / N + 0.1 * L / N * cospi(2 * (i - 1) / N) for i in 1:N, j in 1:N]
    gc = FG.Grids.CurvilinearGrid(cart, (X, Y), nothing, fill(1.0, N, N), trues(N, N),
        (FG.Grids.Periodic(), FG.Grids.Periodic()), (L, L))
    Test.@test at_auto(gc, (N, N)) isa SB.AbstractNUFFTSpectralBackend
end

Test.@testset "Auto agrees with the direct sum" begin
    Random.seed!(78)
    L = 2π
    N = 8
    ms = (N, N)
    cart = FG.Geometry.CartesianGeometry{Float64}()
    f = randn(N, N)

    for (name, g) in (
            ("uniform", FG.Grids.StructuredGrid(cart, at_unif(L, N), at_unif(L, N);
                periodic = (true, true), period = (L, L))),
            ("hybrid", FG.Grids.StructuredGrid(cart, at_unif(L, N), at_stretch(L, N);
                periodic = (true, true), period = (L, L))),
            ("stretched", FG.Grids.StructuredGrid(cart, at_stretch(L, N), at_stretch(L, N);
                periodic = (true, true), period = (L, L))))
        ca, _ = FFS.calculate_spectrum(g, f, ms; execution = AT_SER)          # default transform
        cd, _ = FFS.calculate_spectrum(g, f, ms; transform = AT_DS, execution = AT_SER)
        Test.@test size(ca) == size(cd)
        Test.@test at_rel(ca, cd) < 1e-7
    end
end

Test.@testset "Auto picks the fast spherical transform" begin
    lmax = 4
    ms = (lmax + 1, 2 * lmax + 1)

    # FastTransforms' own Clenshaw–Curtis grid at the matching size: the FSHT analysis applies.
    Θ, Φ = FSH.sph_points(lmax + 1)
    gcc = FG.Grids.StructuredGrid(FG.Geometry.SphericalGeometry(1.0), collect(Φ),
        Float64(π) / 2 .- collect(Θ); periodic = (true, false), period = (2π, 0.0))
    Test.@test FFS._sht_applicable(gcc, ms)
    Test.@test at_auto(gcc, ms) isa SB.AbstractFSHTSpectralBackend
    # At a size the grid does not match, the analysis does not apply.
    Test.@test !FFS._sht_applicable(gcc, (lmax + 2, 2 * lmax + 3))

    # A Gauss–Legendre grid has different nodes, so the FSHT analysis declines and NUFSHT takes it.
    ggl = FG.Connectivity.structured_grid(Float64, FG.SphericalSampling.GaussLegendreSampling(), lmax + 1)
    Test.@test !FFS._sht_applicable(ggl, ms)
    Test.@test at_auto(ggl, ms) isa SB.AbstractNUFSHTSpectralBackend

    # A ring layout and a point cloud likewise: every spherical direct-sum path costs O(L³).
    Test.@test at_auto(FG.Grids.HEALPixGrid(2), ms) isa SB.AbstractNUFSHTSpectralBackend
    N = 200
    ga = π * (3 - sqrt(5))
    λ = [mod(ga * i, 2π) for i in 1:N]
    φ = asin.([-1 + 2 * (i - 0.5) / N for i in 1:N])
    gsc = FG.Grids.UnstructuredGrid(FG.Geometry.SphericalGeometry(1.0), (λ, φ), fill(4π / N, N))
    Test.@test at_auto(gsc, ms) isa SB.AbstractNUFSHTSpectralBackend

    # And the coefficients agree with the direct sum on each of them.
    for g in (ggl, gsc)
        f = [1.0 + 0.4 * sin(2 * a) for a in FFS.Grids._sph_points(g)[2]]
        fr = reshape(f, size(g)...)
        ca, _ = FFS.calculate_spectrum(g, fr, ms; execution = AT_SER)
        cd, _ = FFS.calculate_spectrum(g, fr, ms; transform = AT_DS, execution = AT_SER)
        Test.@test at_rel(ca, cd) < 1e-7
    end
end

Test.@testset "Auto on the in-place and plan entries" begin
    L = 2π
    N = 8
    cart = FG.Geometry.CartesianGeometry{Float64}()
    g = FG.Grids.StructuredGrid(cart, at_unif(L, N), at_unif(L, N);
        periodic = (true, true), period = (L, L))
    f = randn(N, N)

    # The grid form of `calculate_spectrum!` is the direct sum's, so Auto names it there.
    buf = zeros(ComplexF64, FFS.Packing.packed_size((N, N), Val(true))...)
    ks = FFS.calculate_spectrum!(buf, g, f, (N, N); execution = AT_SER)
    cd, _ = FFS.calculate_spectrum(g, f, (N, N); transform = AT_DS, execution = AT_SER)
    Test.@test at_rel(buf, cd) < 1e-12

    # `plan_spectrum` resolves Auto to a library plan, and the direct sum has one of its own.
    p = FFS.plan_spectrum(g, Float64, (N, N); execution = AT_SER)
    Test.@test p isa FFS.AbstractSpectralPlan
    pbuf = zeros(ComplexF64, FFS.Packing.packed_size((N, N), Val(true))...)
    FFS.calculate_spectrum!(pbuf, p, f)
    Test.@test at_rel(pbuf, cd) < 1e-10
    pd = FFS.plan_spectrum(g, Float64, (N, N); transform = AT_DS, execution = AT_SER)
    Test.@test pd isa FFS.DirectSumCartesianPlan
    dbuf = zeros(ComplexF64, FFS.Packing.packed_size((N, N), Val(true))...)
    FFS.calculate_spectrum!(dbuf, pd, f)
    Test.@test at_rel(dbuf, cd) == 0.0
end
