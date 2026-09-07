# The inverse has the same pair the forward does: `plan_synthesis` once, `synthesize!` per coefficient set.
#
# The reference every plan is held to is the allocating `synthesize` on the same grid and coefficients.
# The plan holds what the GRID fixes — the backward FFTW plan, a NUFFT's point sorting, a spherical
# grid's nodes and Legendre tables. The Nyquist twin a packed inverse needs is a functional of the
# COEFFICIENTS, so it arrives with them on `ks`.
#
# Allocation is gated in `test_allocs.jl`, whose `_zero_profile` measures a plan execution in the shape
# that reads a true count here.

using Test: Test
using Random: Random
using FFTW: FFTW
using FINUFFT: FINUFFT
using NonuniformFFTs: NonuniformFFTs
using FastSphericalHarmonics: FastSphericalHarmonics
using NUFSHT: NUFSHT
using KernelAbstractions: KernelAbstractions as KA
using FlowFieldSpectra: FlowFieldSpectra as FFS
using FlowGeometries: FlowGeometries as FG
using SpectralBackends: SpectralBackends as SB
using ComputationalBackends: ComputationalBackends as CB

const SP_SER = CB.SerialBackend()
sp_cart() = FG.Geometry.CartesianGeometry{Float64}()
sp_uni(L, N) = range(0.0, L; length = N + 1)[1:N]
sp_str(L, N) = collect(sp_uni(L, N)) .+ (0.03 * L / N) .* sinpi.(2 .* (0:(N - 1)) ./ N)

# The plan against the allocating `synthesize`, plus reuse and the caller's coefficients surviving.
function sp_agrees(g, f, ms, T; batch = (), kw...)
    c, ks = FFS.calculate_spectrum(g, f, ms; kw...)
    ref = FFS.synthesize(g, c, ms; ks = ks, real_output = T <: Real, kw...)
    p = FFS.plan_synthesis(g, T, ms; batch = batch, kw...)

    Test.@test FFS.field_size(p) == size(ref)
    Test.@test FFS.field_type(p) === eltype(ref)

    out = FFS.allocate_field(p)
    Test.@test size(out) == size(ref)
    Test.@test eltype(out) === eltype(ref)
    Test.@test all(iszero, out)

    keep = copy(c)
    FFS.synthesize!(out, p, c; ks = ks)
    Test.@test isapprox(out, ref; rtol = 1e-8)
    Test.@test c == keep                                  # the caller's coefficients are never written to

    # A second execution reproduces it, so nothing in the plan carried state between calls.
    first = copy(out)
    FFS.synthesize!(out, p, c; ks = ks)
    Test.@test out == first
    return p
end

Test.@testset "Cartesian synthesis plans" begin
    Random.seed!(81)
    L = 2π
    N = 8
    gu = FG.Grids.StructuredGrid(sp_cart(), sp_uni(L, N), sp_uni(L, N);
        periodic = (true, true), period = (L, L))
    gs = FG.Grids.StructuredGrid(sp_cart(), sp_str(L, N), sp_str(L, N);
        periodic = (true, true), period = (L, L))
    M = 50
    gc = FG.Grids.UnstructuredGrid(sp_cart(), (L .* rand(M), L .* rand(M)), fill(L^2 / M, M);
        periodic = (true, true), period = (L, L))

    Test.@testset "FFTW" begin
        for (f, batch, T) in ((randn(N, N), (), Float64), (randn(N, N, 3), (3,), Float64),
                              (randn(ComplexF64, N, N), (), ComplexF64))
            sp_agrees(gu, f, (N, N), T; batch = batch, transform = SB.FFTSpectralBackend())
        end
    end

    # The direct sum owns its whole execution, so it is held to exactly zero on every grid kind.
    Test.@testset "direct sum" begin
        for (g, f, batch, T) in ((gu, randn(N, N), (), Float64), (gs, randn(N, N), (), Float64),
                                 (gs, randn(N, N, 2), (2,), Float64), (gc, randn(M), (), Float64),
                                 (gs, randn(ComplexF64, N, N), (), ComplexF64))
            sp_agrees(g, f, (N, N), T; batch = batch,
                transform = SB.DirectSumSpectralBackend(), execution = SP_SER)
        end
    end

    Test.@testset "FINUFFT" begin
        for (f, batch, T) in ((randn(M), (), Float64), (randn(M, 2), (2,), Float64),
                              (randn(ComplexF64, M), (), ComplexF64))
            sp_agrees(gc, f, (N, N), T; batch = batch, transform = FFS.FINUFFTBackend())
        end
    end

    # NonuniformFFTs' `exec_type2!` allocates inside a threaded region, so this is a bound.
    Test.@testset "NonuniformFFTs" begin
        for (g, f, batch, T) in ((gc, randn(M), (), Float64), (gc, randn(M, 3), (3,), Float64),
                                 (gs, randn(N, N), (), Float64),
                                 (gc, randn(ComplexF64, M), (), ComplexF64))
            sp_agrees(g, f, (N, N), T; batch = batch, transform = FFS.NonuniformFFTsBackend())
        end
    end
end

Test.@testset "Spherical synthesis plans" begin
    Random.seed!(82)
    lmax = 4
    ms = (lmax + 1, 2 * lmax + 1)
    ggl = FG.Connectivity.structured_grid(Float64, FG.SphericalSampling.GaussLegendreSampling(), lmax + 1)
    gcc = FG.Connectivity.structured_grid(Float64, FG.SphericalSampling.ClenshawCurtisSampling(), lmax + 1)
    Ns = 80
    ga = π * (3 - sqrt(5))
    λ = [mod(ga * i, 2π) for i in 1:Ns]
    φ = asin.([-1 + 2 * (i - 0.5) / Ns for i in 1:Ns])
    gsc = FG.Grids.UnstructuredGrid(FG.Geometry.SphericalGeometry(1.0), (λ, φ), fill(4π / Ns, Ns))

    Test.@testset "direct sum" begin
        for (f, batch, T) in ((randn(size(ggl)...), (), Float64),
                              (randn(size(ggl)..., 2), (2,), Float64))
            sp_agrees(ggl, f, ms, T; batch = batch,
                transform = SB.DirectSumSpectralBackend(), execution = SP_SER)
        end
    end

    # `FSH.sph_evaluate!` allocates per call upstream, once per batch slice and per component.
    Test.@testset "FastSphericalHarmonics" begin
        for (f, batch, T) in ((randn(size(gcc)...), (), Float64),
                              (randn(size(gcc)..., 2), (2,), Float64))
            sp_agrees(gcc, f, ms, T; batch = batch, transform = SB.FSHTSpectralBackend())
        end
    end

    Test.@testset "NUFSHT" begin
        for (f, batch, T) in ((randn(Ns), (), Float64), (randn(Ns, 2), (2,), Float64))
            sp_agrees(gsc, f, ms, T; batch = batch, transform = SB.NUFSHTSpectralBackend())
        end
    end
end

Test.@testset "Device synthesis plans" begin
    Random.seed!(85)
    dev = CB.GPUBackend(KA.CPU())
    L = 2π
    N = 8
    gu = FG.Grids.StructuredGrid(sp_cart(), sp_uni(L, N), sp_uni(L, N);
        periodic = (true, true), period = (L, L))
    lmax = 4
    ms = (lmax + 1, 2 * lmax + 1)
    Ns = 80
    ga = π * (3 - sqrt(5))
    λ = [mod(ga * i, 2π) for i in 1:Ns]
    φ = asin.([-1 + 2 * (i - 0.5) / Ns for i in 1:Ns])
    gsc = FG.Grids.UnstructuredGrid(FG.Geometry.SphericalGeometry(1.0), (λ, φ), fill(4π / Ns, Ns))

    for (f, batch, T) in ((randn(N, N), (), Float64), (randn(N, N, 2), (2,), Float64),
                          (randn(ComplexF64, N, N), (), ComplexF64))
        sp_agrees(gu, f, (N, N), T; batch = batch,
            transform = SB.FFTSpectralBackend(), execution = dev)
    end
    for (f, batch, T) in ((randn(Ns), (), Float64), (randn(Ns, 2), (2,), Float64))
        sp_agrees(gsc, f, ms, T; batch = batch,
            transform = SB.NUFSHTSpectralBackend(), execution = dev)
    end
end

Test.@testset "A synthesis plan is fixed to its shape" begin
    Random.seed!(83)
    L = 2π
    N = 8
    g = FG.Grids.StructuredGrid(sp_cart(), sp_uni(L, N), sp_uni(L, N);
        periodic = (true, true), period = (L, L))
    p = FFS.plan_synthesis(g, Float64, (N, N); transform = SB.FFTSpectralBackend())
    out = FFS.allocate_field(p)
    c, _ = FFS.calculate_spectrum(g, randn(N, N), (N, N); transform = SB.FFTSpectralBackend())

    Test.@test_throws DimensionMismatch FFS.synthesize!(zeros(Float64, N + 1, N), p, c)
    Test.@test_throws DimensionMismatch FFS.synthesize!(out, p, zeros(ComplexF64, N, N))

    # A round trip through both plans returns the field the forward consumed.
    f = randn(N, N)
    fp = FFS.plan_spectrum(g, Float64, (N, N); transform = SB.FFTSpectralBackend())
    coeffs = FFS.allocate_coefficients(fp)
    ks = FFS.calculate_spectrum!(coeffs, fp, f)
    FFS.synthesize!(out, p, coeffs; ks = ks)
    Test.@test isapprox(out, f; rtol = 1e-12)
end
