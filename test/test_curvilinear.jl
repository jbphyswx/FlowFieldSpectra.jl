# Curvilinear grids under a Fourier transform.
#
# A curvilinear grid stores one array per direction holding a value per cell, so `x = x(i,j)` and
# `y = y(i,j)` each depend on both indices and the phase `exp(-i(kₓx + k_y y))` admits no factorization
# into a product over axes. The per-axis passes a tensor grid's transform separates into come from its
# one-axis-per-direction storage. A curvilinear map is therefore read as a point cloud.
#
# A curvilinear grid's index-space adjacency is
# exploitable, and FlowGeometries' `IndexStencilNeighbors` trait serves it; a global spectral transform
# reads no adjacency.
#
# The reference is a node cloud built over exactly the same points and measures: an independent
# construction that shares no dispatch with the curvilinear methods, so agreement pins the routing.

using Test: Test
using FlowFieldSpectra: FlowFieldSpectra as FFS
using FlowGeometries: FlowGeometries as FG
using SpectralBackends: SpectralBackends as SB
using ComputationalBackends: ComputationalBackends as CB
using NonuniformFFTs: NonuniformFFTs
using FINUFFT: FINUFFT
using OhMyThreads: OhMyThreads
using KernelAbstractions: KernelAbstractions as KA
using Random: Random

const CV_DS = SB.DirectSumSpectralBackend()
const CV_SER = CB.SerialBackend()
cv_rel(a, b) = maximum(abs, a .- b) / max(maximum(abs, b), eps())

# A curvilinear grid and the node cloud over its own points, carrying the same measures.
function cv_cartesian(n, L)
    X = [L * (i - 1) / n + 0.12 * L / n * sinpi(2 * (j - 1) / n) for i in 1:n, j in 1:n]
    Y = [L * (j - 1) / n + 0.12 * L / n * cospi(2 * (i - 1) / n) for i in 1:n, j in 1:n]
    meas = [1.0 + 0.3 * sinpi(2 * (i + j) / n) for i in 1:n, j in 1:n]
    g = FG.Grids.CurvilinearGrid(FG.Geometry.CartesianGeometry{Float64}(), (X, Y), nothing, meas,
        trues(n, n), (FG.Grids.Periodic(), FG.Grids.Periodic()), (L, L))
    cloud = FG.Grids.UnstructuredGrid(FG.Geometry.CartesianGeometry{Float64}(),
        (vec(X), vec(Y)), vec(meas); periodic = (true, true), period = (L, L))
    return g, cloud
end

Test.@testset "Curvilinear Cartesian transforms as a point cloud" begin
    Random.seed!(2027)
    n = 6
    L = 2π
    ms = (4, 4)
    g, cloud = cv_cartesian(n, L)
    f = randn(n, n)

    # DirectSum: curvilinear must equal the node cloud over the same points, on every backend.
    c, ks = FFS.calculate_spectrum(g, f, ms; transform = CV_DS, execution = CV_SER)
    cref, ksref = FFS.calculate_spectrum(cloud, vec(f), ms; transform = CV_DS, execution = CV_SER)
    Test.@test size(c) == size(cref)
    Test.@test cv_rel(c, cref) < 1e-13
    # the Nyquist twin rides on ks and must agree too
    tw = FFS.Packing.axis_twin(ks[1])
    twref = FFS.Packing.axis_twin(ksref[1])
    Test.@test (tw === nothing) == (twref === nothing)
    if tw !== nothing
        for (a, b) in zip(tw.slices, twref.slices)
            isempty(a) || Test.@test cv_rel(a, b) < 1e-13
        end
    end
    for exec in (CB.ThreadedBackend(), CB.GPUBackend(KA.CPU()))
        cx, _ = FFS.calculate_spectrum(g, f, ms; transform = CV_DS, execution = exec)
        Test.@test cv_rel(cx, c) < 1e-13
    end

    # Both NUFFT providers reach it as a point cloud, so a curvilinear grid gets a fast path.
    for tr in (FFS.NonuniformFFTsBackend(), FFS.FINUFFTBackend())
        cn, _ = FFS.calculate_spectrum(g, f, ms; transform = tr, eps = 1e-12)
        Test.@test size(cn) == size(c)
        Test.@test cv_rel(cn, c) < 1e-11
        # and a reusable plan over the same grid
        p = FFS.plan_spectrum(g, Float64, ms; transform = tr, eps = 1e-12)
        buf = zeros(ComplexF64, FFS.Packing.packed_size(ms, Val(true))...)
        FFS.calculate_spectrum!(buf, p, f)
        Test.@test cv_rel(buf, cn) < 1e-11
    end

    # Synthesis reaches the same points, both the packed-half and the full-spectrum forms.
    fs = FFS.synthesize(g, c, ms; transform = CV_DS, execution = CV_SER, ks = ks)
    fsref = FFS.synthesize(cloud, cref, ms; transform = CV_DS, execution = CV_SER, ks = ksref)
    Test.@test size(fs) == size(g)
    Test.@test cv_rel(vec(fs), vec(fsref)) < 1e-13

    cc, kc = FFS.calculate_spectrum(g, ComplexF64.(f), ms; transform = CV_DS, execution = CV_SER)
    fc = FFS.synthesize(g, cc, ms; transform = CV_DS, execution = CV_SER, real_output = false, ks = kc)
    Test.@test size(fc) == size(g)
    Test.@test eltype(fc) === ComplexF64

    # Batched.
    fb = randn(n, n, 3)
    cb, _ = FFS.calculate_spectrum(g, fb, ms; transform = CV_DS, execution = CV_SER)
    Test.@test size(cb) == (FFS.Packing.packed_size(ms, Val(true))..., 3)
    for b in 1:3
        c1, _ = FFS.calculate_spectrum(g, fb[:, :, b], ms; transform = CV_DS, execution = CV_SER)
        Test.@test cv_rel(cb[:, :, b], c1) < 1e-13
    end
end

Test.@testset "Curvilinear spherical transforms per point" begin
    Random.seed!(2028)
    n = 6
    lmax = 3
    ms = (lmax + 1, 2lmax + 1)
    Lam = [2π * (i - 1) / n for i in 1:n, j in 1:n]
    Phi = [π * (j - 0.5) / n - π / 2 for i in 1:n, j in 1:n]
    meas = [cos(Phi[i, j]) for i in 1:n, j in 1:n]
    g = FG.Grids.CurvilinearGrid(FG.Geometry.SphericalGeometry(1.0), (Lam, Phi), nothing, meas,
        trues(n, n), (FG.Grids.Periodic(), FG.Grids.Bounded()), (2π, 0.0))
    cloud = FG.Grids.UnstructuredGrid(FG.Geometry.SphericalGeometry(1.0),
        (vec(Lam), vec(Phi)), vec(meas))
    f = randn(n, n)

    # No iso-latitude structure is declared for a curvilinear grid, so it takes the per-point path.
    Test.@test FFS.Grids._sph_layout(g) isa FFS.Grids.ScatteredSphere

    c, _ = FFS.calculate_spectrum(g, f, ms; transform = CV_DS, execution = CV_SER)
    cref, _ = FFS.calculate_spectrum(cloud, vec(f), ms; transform = CV_DS, execution = CV_SER)
    Test.@test size(c) == size(cref)
    Test.@test cv_rel(c, cref) < 1e-13

    for exec in (CB.ThreadedBackend(), CB.GPUBackend(KA.CPU()))
        cx, _ = FFS.calculate_spectrum(g, f, ms; transform = CV_DS, execution = exec)
        Test.@test cv_rel(cx, c) < 1e-12
    end

    # Synthesis writes the grid's own shape.
    out = FFS.synthesize(g, c, ms; transform = CV_DS, execution = CV_SER)
    Test.@test size(out) == size(g)
end
