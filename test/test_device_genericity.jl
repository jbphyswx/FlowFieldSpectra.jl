# Device genericity on a device array that REFUSES host scalar indexing.
#
# `GPUBackend(KA.CPU())` allocates plain `Array`s, so a path that walks a "device" array element by
# element on the host still returns the right answer there. A `JLArray` throws on scalar `getindex`, so
# the same path fails loudly here. The first testset asserts the guard is armed, so the rest cannot pass
# vacuously.
#
# SCOPE: the DirectSum device kernels. JLArrays implements no `AbstractFFTs` methods, so the GPU FFT,
# NUFFT and NUFSHT device paths — which delegate their transform to a library that needs a device FFT —
# cannot run on it at all. Those keep their `KA.CPU()` coverage in `test_gpu.jl`.

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

# Serial and JLArrays coefficients for the same call, plus the twin each carries.
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
    Test.@test size(cd) == size(cs)
    Test.@test eltype(cd) === eltype(cs)
    Test.@test dg_rel(cd, cs) < 1e-12

    # Stretched tensor grid with even axes: the twin path runs its own device kernel over the masked
    # mode set and copies each hyperplane back, so it is the densest device-indexing path here.
    ax = dg_str(L, N)
    gs = FG.Grids.StructuredGrid(dg_cart(), ax, ax; periodic = (true, true), period = (L, L))
    fs = randn(N, N)
    cs2, ks2, cd2, kd2 = dg_pair(gs, fs, (N, N))
    Test.@test dg_rel(cd2, cs2) < 1e-12
    ts = FFS.Packing.axis_twin(ks2[1])
    td = FFS.Packing.axis_twin(kd2[1])
    Test.@test (ts === nothing) == (td === nothing)
    if ts !== nothing
        for (a, b) in zip(td.slices, ts.slices)
            Test.@test size(a) == size(b)
            isempty(b) || Test.@test dg_rel(a, b) < 1e-12
        end
    end

    # Scattered point cloud.
    M = 60
    xv = L .* rand(M)
    yv = L .* rand(M)
    gc = FG.Grids.UnstructuredGrid(dg_cart(), (xv, yv), fill(L^2 / M, M);
        periodic = (true, true), period = (L, L))
    fc = randn(M)
    csc, _, cdc, _ = dg_pair(gc, fc, (N, N))
    Test.@test dg_rel(cdc, csc) < 1e-12

    # Curvilinear grid: its coordinates are one value per CELL, so the device path takes the pointwise
    # kernel over `vec`'d coordinate arrays.
    n = 6
    X = [L * (i - 1) / n + 0.1 * L / n * sinpi(2 * (j - 1) / n) for i in 1:n, j in 1:n]
    Y = [L * (j - 1) / n + 0.1 * L / n * cospi(2 * (i - 1) / n) for i in 1:n, j in 1:n]
    meas = [1.0 + 0.3 * sinpi(2 * (i + j) / n) for i in 1:n, j in 1:n]
    gcv = FG.Grids.CurvilinearGrid(dg_cart(), (X, Y), nothing, meas, trues(n, n),
        (FG.Grids.Periodic(), FG.Grids.Periodic()), (L, L))
    fcv = randn(n, n)
    csv, _, cdv, _ = dg_pair(gcv, fcv, (n, n))
    Test.@test size(cdv) == size(csv)
    Test.@test dg_rel(cdv, csv) < 1e-12

    # Complex field (the device buffer takes the field's own float type) and a batch axis.
    fx = randn(ComplexF64, N, N)
    csx, _, cdx, _ = dg_pair(gs, fx, (N, N))
    Test.@test eltype(cdx) === ComplexF64
    Test.@test dg_rel(cdx, csx) < 1e-12
    fb = randn(N, N, 3)
    csb, _, cdb, _ = dg_pair(gs, fb, (N, N))
    Test.@test size(cdb) == size(csb)
    Test.@test dg_rel(cdb, csb) < 1e-12
    fcb = randn(M, 2)
    cscb, _, cdcb, _ = dg_pair(gc, fcb, (N, N))
    Test.@test size(cdcb) == size(cscb)
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
        Test.@test size(cd) == size(cs)
        Test.@test eltype(cd) === Float64          # a real field keeps real spherical coefficients
        Test.@test dg_rel(cd, cs) < 1e-10
    end

    # Batched and complex.
    fb = randn(size(ggl)..., 2)
    csb, _, cdb, _ = dg_pair(ggl, fb, ms)
    Test.@test size(cdb) == size(csb)
    Test.@test dg_rel(cdb, csb) < 1e-10
    fx = randn(ComplexF64, size(ggl)...)
    csx, _, cdx, _ = dg_pair(ggl, fx, ms)
    Test.@test eltype(cdx) === ComplexF64
    Test.@test dg_rel(cdx, csx) < 1e-10
end
