# Device genericity on a device array that throws on host scalar indexing.
#
# `GPUBackend(KA.CPU())` allocates plain `Array`s, so a path that walks a "device" array element by
# element on the host still returns the right answer there. A `JLArray` throws on scalar `getindex`, so
# the same path fails loudly here. The first testset asserts the guard is armed, so the rest cannot pass
# vacuously.
#
# These are the DirectSum device kernels. JLArrays implements no `AbstractFFTs` methods, so the GPU FFT,
# NUFFT and NUFSHT device paths, which delegate their transform to a library that needs a device FFT,
# keep their `KA.CPU()` coverage in the other topic files.

using Test: Test
using Random: Random
using JLArrays: JLArrays
using GPUArraysCore: GPUArraysCore
using KernelAbstractions: KernelAbstractions as KA
using FlowFieldSpectra: FlowFieldSpectra as FFS
using FlowGeometries: FlowGeometries as FG
using SpectralBackends: SpectralBackends as SB
using ComputationalBackends: ComputationalBackends as CB

# A KernelAbstractions backend whose arrays are `JLArray`s. `synchronize` is declared by KA with no
# default, so every backend must supply one and JLArrays supplies none; this file's backend does. Each
# method below takes `NoScalarBackend`, which this file owns; the kernel-call method is KA's documented
# extension point for a backend, and it hands the launch to JLArrays.
struct NoScalarBackend <: KA.GPU end
KA.allocate(::NoScalarBackend, ::Type{T}, dims::Tuple) where {T} = JLArrays.JLArray{T}(undef, dims)
KA.synchronize(::NoScalarBackend) = nothing
# `@kernel` picks its kernel variant by `isgpu(backend)`, and the launch below runs the kernel on the
# host, so the host variant is the one to build (JLArrays answers `false` for its own backend likewise).
KA.isgpu(::NoScalarBackend) = false
function (obj::KA.Kernel{NoScalarBackend, W, N, F})(args...;
        ndrange = nothing, workgroupsize = nothing) where {W, N, F}
    jl = KA.Kernel{JLArrays.JLBackend, W, N, F}(JLArrays.JLBackend(), obj.f)
    return jl(args...; ndrange = ndrange, workgroupsize = workgroupsize)
end

const DG_JL = CB.GPUBackend(NoScalarBackend())
const DG_SER = CB.SerialBackend()
const DG_DS = SB.DirectSumSpectralBackend()
dg_rel(a, b) = maximum(abs, a .- b) / max(maximum(abs, b), eps())
dg_uni(L, N) = range(0.0, L; length = N + 1)[1:N]
dg_str(L, N) = [L * (i - 1) / N + 0.05 * L * sinpi(2 * (i - 1) / N) for i in 1:N]
dg_cart() = FG.Geometry.CartesianGeometry{Float64}()
dg_sph() = FG.Geometry.SphericalGeometry(1.0)

# Serial and JLArrays coefficients for the same call, with the wavenumber axes each returns.
function dg_pair(g, f, ms; kwargs...)
    cs, ks = FFS.calculate_spectrum(g, f, ms; transform = DG_DS, execution = DG_SER, kwargs...)
    cd, kd = FFS.calculate_spectrum(g, f, ms; transform = DG_DS, execution = DG_JL, kwargs...)
    return cs, ks, cd, kd
end

Test.@testset "JLArray scalar indexing is disallowed" begin
    GPUArraysCore.allowscalar(false)
    a = JLArrays.JLArray([1.0, 2.0, 3.0])
    Test.@test_throws Exception a[1]
    Test.@test_throws Exception (a[2] = 0.0)
    # A whole-array copy back to the host is not scalar indexing and must still work.
    Test.@test Array(a) == [1.0, 2.0, 3.0]
end

Test.@testset "Cartesian direct sum on JLArrays" begin
    Random.seed!(51)
    L = 2π

    # Uniform tensor grid, real field.
    N = 8
    xs = dg_uni(L, N)
    gu = FG.Grids.StructuredGrid(dg_cart(), xs, xs; periodic = (true, true), period = (L, L))
    fu = [cos(2 * x + y) for x in xs, y in xs]
    cs, _, cd, _ = dg_pair(gu, fu, (N, N))
    Test.@test dg_rel(cd, cs) < 1e-12

    # Stretched tensor grid with even axes, which a real field's coefficient box widens.
    ax = dg_str(L, N)
    gs = FG.Grids.StructuredGrid(dg_cart(), ax, ax; periodic = (true, true), period = (L, L))
    fs = randn(N, N)
    cs2, ks2, cd2, kd2 = dg_pair(gs, fs, (N, N))
    Test.@test size(cd2) == size(cs2) == (N ÷ 2 + 1, N + 1)
    Test.@test dg_rel(cd2, cs2) < 1e-12
    Test.@test map(collect, kd2) == map(collect, ks2)

    # Three stretched axes, two of them widened past axis 1: the device kernel runs over the modes the
    # transform computes, and the box's modes outside them and their mirror stay zero.
    n3 = 6
    ax3 = dg_str(L, n3)
    g3 = FG.Grids.StructuredGrid(dg_cart(), ax3, ax3, ax3; periodic = (true, true, true), period = (L, L, L))
    f3 = randn(n3, n3, n3)
    cs3, _, cd3, _ = dg_pair(g3, f3, (4, 6, 4))
    Test.@test size(cd3) == size(cs3) == (3, 7, 5)
    Test.@test dg_rel(cd3, cs3) < 1e-12

    # Scattered point cloud.
    M = 60
    xv = L .* rand(M)
    yv = L .* rand(M)
    gc = FG.Grids.UnstructuredGrid(dg_cart(), (xv, yv), fill(L^2 / M, M);
        periodic = (true, true), period = (L, L))
    fc = randn(M)
    csc, _, cdc, _ = dg_pair(gc, fc, (N, N))
    Test.@test dg_rel(cdc, csc) < 1e-12

    # Curvilinear grid: its coordinates are one value per cell, so the device path takes the pointwise
    # kernel over `vec`'d coordinate arrays.
    n = 6
    X = [L * (i - 1) / n + 0.1 * L / n * sinpi(2 * (j - 1) / n) for i in 1:n, j in 1:n]
    Y = [L * (j - 1) / n + 0.1 * L / n * cospi(2 * (i - 1) / n) for i in 1:n, j in 1:n]
    meas = [1.0 + 0.3 * sinpi(2 * (i + j) / n) for i in 1:n, j in 1:n]
    gcv = FG.Grids.CurvilinearGrid(dg_cart(), X, Y, trues(n, n); measure = meas, periodic = (true, true),
        period = (L, L))
    fcv = randn(n, n)
    csv, _, cdv, _ = dg_pair(gcv, fcv, (n, n))
    Test.@test dg_rel(cdv, csv) < 1e-12

    # Complex field (the device buffer takes the field's own float type) and a batch axis.
    fx = randn(ComplexF64, N, N)
    csx, _, cdx, _ = dg_pair(gs, fx, (N, N))
    Test.@test dg_rel(cdx, csx) < 1e-12
    fb = randn(N, N, 3)
    csb, _, cdb, _ = dg_pair(gs, fb, (N, N))
    Test.@test dg_rel(cdb, csb) < 1e-12
    fcb = randn(M, 2)
    cscb, _, cdcb, _ = dg_pair(gc, fcb, (N, N))
    Test.@test dg_rel(cdcb, cscb) < 1e-12

    # D = 1 and D = 3 exercise the flat-index decode at both ends.
    for D in (1, 3)
        Nd = D == 3 ? 5 : 10
        gd = FG.Grids.StructuredGrid(dg_cart(), ntuple(_ -> dg_uni(L, Nd), D)...;
            periodic = ntuple(_ -> true, D), period = ntuple(_ -> L, D))
        fd = randn(ntuple(_ -> Nd, D)...)
        csd, _, cdd, _ = dg_pair(gd, fd, ntuple(_ -> Nd, D))
        Test.@test dg_rel(cdd, csd) < 1e-12
    end
end

Test.@testset "Spherical direct sum on JLArrays" begin
    Random.seed!(52)
    lmax = 4
    ms = (lmax + 1, 2 * lmax + 1)

    # One grid per layout: tensor, ring, and a point cloud.
    ggl = FG.Connectivity.structured_grid(Float64, FG.SphericalSampling.GaussLegendreSampling(), lmax + 1)
    ghp = FG.Grids.HEALPixGrid(dg_sph(), 2)
    Ns = 50
    ga = π * (3 - sqrt(5))
    λ = [mod(ga * i, 2π) for i in 1:Ns]
    φ = asin.([-1 + 2 * (i - 0.5) / Ns for i in 1:Ns])
    gcl = FG.Grids.UnstructuredGrid(dg_sph(), (λ, φ), fill(4π / Ns, Ns))

    for g in (ggl, ghp, gcl)
        f = randn(size(g)...)
        cs, _, cd, _ = dg_pair(g, f, ms)
        Test.@test eltype(cd) === Float64          # a real field keeps real spherical coefficients
        Test.@test dg_rel(cd, cs) < 1e-12
    end

    # Batched and complex.
    fb = randn(size(ggl)..., 2)
    csb, _, cdb, _ = dg_pair(ggl, fb, ms)
    Test.@test dg_rel(cdb, csb) < 1e-12
    fx = randn(ComplexF64, size(ggl)...)
    csx, _, cdx, _ = dg_pair(ggl, fx, ms)
    Test.@test dg_rel(cdx, csx) < 1e-12
end

dg_dev(a) = JLArrays.JLArray(a)

function dg_grids(L, N)
    ax = dg_str(L, N)
    M = 60
    n = 6
    X = [L * (i - 1) / n + 0.1 * L / n * sinpi(2 * (j - 1) / n) for i in 1:n, j in 1:n]
    Y = [L * (j - 1) / n + 0.1 * L / n * cospi(2 * (i - 1) / n) for i in 1:n, j in 1:n]
    return (
        (FG.Grids.StructuredGrid(dg_cart(), dg_uni(L, N), dg_uni(L, N); periodic = (true, true),
            period = (L, L)), (N, N)),
        (FG.Grids.StructuredGrid(dg_cart(), ax, ax; periodic = (true, true), period = (L, L)), (N, N)),
        (FG.Grids.UnstructuredGrid(dg_cart(), (L .* rand(M), L .* rand(M)), fill(L^2 / M, M);
            periodic = (true, true), period = (L, L)), (N, N)),
        (FG.Grids.CurvilinearGrid(dg_cart(), X, Y, trues(n, n);
            measure = [1.0 + 0.3 * sinpi(2 * (i + j) / n) for i in 1:n, j in 1:n], periodic = (true, true),
            period = (L, L)), (n, n)),
    )
end

Test.@testset "Cartesian direct-sum plans on JLArrays, reused" begin
    # One plan per grid and field type, executed on fresh fields from the host and from the device, into
    # host and device coefficients.
    Random.seed!(53)
    for (g, ms) in dg_grids(2π, 8), (T, batch) in ((Float64, ()), (Float64, (2,)), (ComplexF64, ()))
        p = FFS.plan_spectrum(g, T, ms; transform = DG_DS, execution = DG_JL, batch = batch)
        c = FFS.allocate_coefficients(p)
        for _ in 1:2
            f = randn(T, size(g)..., batch...)
            cs, ks = FFS.calculate_spectrum(g, f, ms; transform = DG_DS, execution = DG_SER)
            kd = FFS.calculate_spectrum!(c, p, f)
            Test.@test dg_rel(c, cs) < 1e-12
            Test.@test map(collect, kd) == map(collect, ks)
            cdev = dg_dev(zero(c))
            FFS.calculate_spectrum!(cdev, p, dg_dev(f))
            Test.@test dg_rel(Array(cdev), cs) < 1e-12
        end
    end
end

Test.@testset "Cartesian direct-sum synthesis on JLArrays" begin
    # Held to the serial direct-sum synthesis of the same coefficients, over the box the stretched grid, the
    # cloud and the curvilinear grid widen for a real field.
    Random.seed!(54)
    for (g, ms) in dg_grids(2π, 8), (T, batch) in ((Float64, ()), (Float64, (2,)), (ComplexF64, ()))
        c, _ = FFS.calculate_spectrum(g, randn(T, size(g)..., batch...), ms; transform = DG_DS,
            execution = DG_SER)
        ref = FFS.synthesize(g, c, ms; transform = DG_DS, execution = DG_SER, real_output = T <: Real)
        p = FFS.plan_synthesis(g, T, ms; transform = DG_DS, execution = DG_JL, batch = batch)
        out = FFS.allocate_field(p)
        FFS.synthesize!(out, p, c)
        Test.@test dg_rel(out, ref) < 1e-12
        outd = dg_dev(zero(out))
        FFS.synthesize!(outd, p, dg_dev(c))
        Test.@test dg_rel(Array(outd), ref) < 1e-12
    end
end

Test.@testset "Spherical direct-sum plans and synthesis on JLArrays" begin
    Random.seed!(55)
    lmax = 4
    ms = (lmax + 1, 2 * lmax + 1)
    ggl = FG.Connectivity.structured_grid(Float64, FG.SphericalSampling.GaussLegendreSampling(), lmax + 1)
    ghp = FG.Grids.HEALPixGrid(dg_sph(), 2)
    Ns = 50
    ga = π * (3 - sqrt(5))
    gcl = FG.Grids.UnstructuredGrid(dg_sph(), ([mod(ga * i, 2π) for i in 1:Ns],
        asin.([-1 + 2 * (i - 0.5) / Ns for i in 1:Ns])), fill(4π / Ns, Ns))
    for g in (ggl, ghp, gcl), (T, batch) in ((Float64, ()), (Float64, (2,)), (ComplexF64, ()))
        p = FFS.plan_spectrum(g, T, ms; transform = DG_DS, execution = DG_JL, batch = batch)
        c = FFS.allocate_coefficients(p)
        q = FFS.plan_synthesis(g, T, ms; transform = DG_DS, execution = DG_JL, batch = batch)
        out = FFS.allocate_field(q)
        for _ in 1:2
            f = randn(T, size(g)..., batch...)
            cs, _ = FFS.calculate_spectrum(g, f, ms; transform = DG_DS, execution = DG_SER)
            FFS.calculate_spectrum!(c, p, dg_dev(f))
            Test.@test dg_rel(c, cs) < 1e-12
            ref = FFS.synthesize(g, cs, ms; transform = DG_DS, execution = DG_SER, real_output = T <: Real)
            FFS.synthesize!(out, q, cs)
            Test.@test dg_rel(out, ref) < 1e-12
            outd = dg_dev(zero(out))
            FFS.synthesize!(outd, q, dg_dev(cs))
            Test.@test dg_rel(Array(outd), ref) < 1e-12
        end
    end
end

# The reductions of coefficients held in a JLArray against the same reductions of the host copy: radial
# spectra (both cutoffs, in place), cross, anisotropic, transect and degree spectra, on packed real-field
# layouts of even and odd length and on a full complex layout, two fields at once. Only the order the
# device adds a bin's terms in differs.
Test.@testset "Spectral reductions of JLArray coefficients equal the host reductions" begin
    Random.seed!(57)
    for (n, halved) in ((32, true), (33, true), (32, false))
        g = FG.Grids.StructuredGrid(dg_cart(), dg_uni(2π, n), dg_uni(2π, n); periodic = (true, true),
            period = (2π, 2π))
        ks = FFS.Grids.physical_wavenumbers(g, (n, n), Val(halved))
        c = randn(ComplexF64, length(ks[1]), length(ks[2]), 2)
        h = randn(ComplexF64, size(c))
        cd, hd = dg_dev(c), dg_dev(h)
        for cutoff in (true, false)
            kb, E = FFS.isotropic_spectrum(ks, c; cutoff)
            kbd, Ed = FFS.isotropic_spectrum(ks, cd; cutoff)
            Test.@test kbd == kb
            Test.@test dg_rel(Ed, E) < 1e-12
        end
        kb, E = FFS.isotropic_spectrum(ks, c)
        Ei, kbi = zero(E), zero(kb)
        FFS.isotropic_spectrum!(Ei, kbi, ks, cd)
        Test.@test kbi ≈ kb && dg_rel(Ei, E) < 1e-12
        _, S = FFS.cross_spectrum(ks, c, h)
        _, Sd = FFS.cross_spectrum(ks, cd, hd)
        Test.@test dg_rel(Sd, S) < 1e-12
        Test.@test dg_rel(last(FFS.anisotropic_spectrum(ks, cd)), last(FFS.anisotropic_spectrum(ks, c))) < 1e-12
        for dims in ((1,), (2,))
            Test.@test dg_rel(last(FFS.transect_spectrum(ks, cd, dims)), last(FFS.transect_spectrum(ks, c, dims))) < 1e-12
        end
    end
    lmax = 12
    C = randn(lmax + 1, 2lmax + 1, 2)
    _, El = FFS.spherical_energy_spectrum(C)
    _, Eld = FFS.spherical_energy_spectrum(dg_dev(C))
    Test.@test dg_rel(Eld, El) < 1e-12
    # A device field's one-shot spectrum stays on the device, where the reductions above read it.
    gc = FG.Grids.StructuredGrid(dg_cart(), dg_uni(2π, 12), dg_uni(2π, 10); periodic = (true, true),
        period = (2π, 2π))
    f = randn(12, 10, 2)
    cs, ks = FFS.calculate_spectrum(gc, f, (12, 10); transform = DG_DS, execution = DG_SER)
    cd, _ = FFS.calculate_spectrum(gc, dg_dev(f), (12, 10); transform = DG_DS, execution = DG_JL)
    Test.@test cd isa JLArrays.JLArray && dg_rel(Array(cd), cs) < 1e-12
    Test.@test dg_rel(last(FFS.isotropic_spectrum(ks, cd)), last(FFS.isotropic_spectrum(ks, cs))) < 1e-12
end
