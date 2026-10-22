# `AutoSpectralBackend`, the default `transform`, resolves to the fastest transform a grid admits among the
# loaded extensions, and its coefficients match the direct sum's on the same grid to the requested
# tolerance. Every extension is loaded in this suite.

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

Test.@testset "Auto picks the fast Cartesian transform and agrees with the direct sum" begin
    Random.seed!(78)
    L = 2π
    N = 8
    cart = FG.Geometry.CartesianGeometry{Float64}()
    gu = FG.Grids.StructuredGrid(cart, at_unif(L, N), at_unif(L, N); periodic = (true, true), period = (L, L))
    gh = FG.Grids.StructuredGrid(cart, at_unif(L, N), at_stretch(L, N); periodic = (true, true), period = (L, L))
    gs = FG.Grids.StructuredGrid(cart, at_stretch(L, N), at_unif(L, N); periodic = (true, true), period = (L, L))
    ga = FG.Grids.StructuredGrid(cart, at_stretch(L, N), at_stretch(L, N); periodic = (true, true), period = (L, L))
    cloud = FG.Grids.UnstructuredGrid(cart, (L .* rand(50), L .* rand(50)), fill(L^2 / 50, 50);
        periodic = (true, true), period = (L, L))
    X = [L * (i - 1) / N + 0.1 * L / N * sinpi(2 * (j - 1) / N) for i in 1:N, j in 1:N]
    Y = [L * (j - 1) / N + 0.1 * L / N * cospi(2 * (i - 1) / N) for i in 1:N, j in 1:N]
    gc = FG.Grids.CurvilinearGrid(cart, X, Y, trues(N, N); measure = fill(1.0, N, N),
        periodic = (true, true), period = (L, L))

    # Uniform and asked for its own length: the FFT. Uniform in axis 1 and stretched in axis 2: the
    # hybrid composite, which the FFT tag selects. Axis 1 stretched, every axis stretched, a node cloud and
    # a curvilinear grid: a NUFFT. Fewer modes than a uniform axis has points: a NUFFT too.
    for (g, ms, tag) in ((gu, (N, N), SB.AbstractFFTSpectralBackend), (gh, (N, N), SB.AbstractFFTSpectralBackend),
                         (gs, (N, N), SB.AbstractNUFFTSpectralBackend), (ga, (N, N), SB.AbstractNUFFTSpectralBackend),
                         (cloud, (N, N), SB.AbstractNUFFTSpectralBackend), (gc, (N, N), SB.AbstractNUFFTSpectralBackend),
                         (gu, (4, N), SB.AbstractNUFFTSpectralBackend))
        Test.@test at_auto(g, ms) isa tag
        f = randn(size(g)...)
        ca, _ = FFS.calculate_spectrum(g, f, ms; execution = AT_SER, eps = 1e-12)       # default transform
        cd, _ = FFS.calculate_spectrum(g, f, ms; transform = AT_DS, execution = AT_SER)
        Test.@test at_rel(ca, cd) < 1e-10
    end
end

Test.@testset "Auto picks the fast spherical transform and agrees with the direct sum" begin
    lmax = 4
    ms = (lmax + 1, 2 * lmax + 1)

    # FastTransforms' own Clenshaw–Curtis grid at the matching size: the FSHT analysis. A Gauss–Legendre
    # grid, a ring layout and a point cloud: NUFSHT.
    Θ, Φ = FSH.sph_points(lmax + 1)
    gcc = FG.Grids.StructuredGrid(FG.Geometry.SphericalGeometry(1.0), collect(Φ),
        Float64(π) / 2 .- collect(Θ); periodic = (true, false), period = (2π, 0.0))
    Test.@test at_auto(gcc, ms) isa SB.AbstractFSHTSpectralBackend
    Test.@test at_auto(gcc, (lmax + 2, 2 * lmax + 3)) isa SB.AbstractNUFSHTSpectralBackend
    ggl = FG.Connectivity.structured_grid(Float64, FG.SphericalSampling.GaussLegendreSampling(), lmax + 1)
    N = 200
    ga = π * (3 - sqrt(5))
    gsc = FG.Grids.UnstructuredGrid(FG.Geometry.SphericalGeometry(1.0),
        ([mod(ga * i, 2π) for i in 1:N], asin.([-1 + 2 * (i - 0.5) / N for i in 1:N])), fill(4π / N, N))
    for g in (ggl, FG.Grids.HEALPixGrid(2), gsc)
        Test.@test at_auto(g, ms) isa SB.AbstractNUFSHTSpectralBackend
    end
    for g in (ggl, gsc)
        f = randn(size(g)...)
        ca, _ = FFS.calculate_spectrum(g, f, ms; execution = AT_SER, tol = 1e-12)
        cd, _ = FFS.calculate_spectrum(g, f, ms; transform = AT_DS, execution = AT_SER)
        Test.@test at_rel(ca, cd) < 1e-10
    end
end

Test.@testset "Auto on the in-place and plan entries" begin
    Random.seed!(79)
    L = 2π
    N = 8
    cart = FG.Geometry.CartesianGeometry{Float64}()
    g = FG.Grids.StructuredGrid(cart, at_unif(L, N), at_unif(L, N); periodic = (true, true), period = (L, L))
    gh = FG.Grids.StructuredGrid(cart, at_unif(L, N), at_stretch(L, N); periodic = (true, true), period = (L, L))
    M = 60
    cloud = FG.Grids.UnstructuredGrid(cart, (L .* rand(M), L .* rand(M)), fill(L^2 / M, M);
        periodic = (true, true), period = (L, L))
    f = randn(N, N)
    fcl = randn(M)
    cd, _ = FFS.calculate_spectrum(g, f, (N, N); transform = AT_DS, execution = AT_SER)
    cdh, _ = FFS.calculate_spectrum(gh, f, (N, N); transform = AT_DS, execution = AT_SER)
    cdc, _ = FFS.calculate_spectrum(cloud, fcl, (N, N); transform = AT_DS, execution = AT_SER)

    # The grid form of `calculate_spectrum!` runs Auto, and every library transform, through a plan built
    # for the call.
    for (gg, ff, ref, t) in ((g, f, cd, SB.AutoSpectralBackend()), (cloud, fcl, cdc, SB.AutoSpectralBackend()),
                             (gh, f, cdh, SB.FFTSpectralBackend()))
        b = zero(ref)
        FFS.calculate_spectrum!(b, gg, ff, (N, N); transform = t, execution = AT_SER, eps = 1e-12)
        Test.@test at_rel(b, ref) < 1e-10
    end
    # A buffer holding as many values in another shape is refused.
    Test.@test_throws DimensionMismatch FFS.calculate_spectrum!(zeros(ComplexF64, (N ÷ 2 + 1) * N), g, f,
        (N, N); transform = AT_DS, execution = AT_SER)
end
