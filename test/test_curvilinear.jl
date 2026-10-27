# A curvilinear grid stores one coordinate value per cell, so its phase does not factorize over axes and
# it transforms as the point cloud of its cells. The reference is a node cloud over the same points and
# measures.

using Test: Test
using FlowFieldSpectra: FlowFieldSpectra as FFS
using FlowTransformBindings: FlowTransformBindings as FTB
using FlowGeometries: FlowGeometries as FG
using SpectralBackends: SpectralBackends as SB
using ComputationalBackends: ComputationalBackends as CB
using NonuniformFFTs: NonuniformFFTs
using FINUFFT: FINUFFT
using Random: Random

const CV_DS = SB.DirectSumSpectralBackend()
const CV_SER = CB.SerialBackend()
cv_rel(a, b) = maximum(abs, a .- b) / max(maximum(abs, b), eps())

# A curvilinear grid and the node cloud over its own points, carrying the same measures.
function cv_cartesian(n, L)
    X = [L * (i - 1) / n + 0.12 * L / n * sinpi(2 * (j - 1) / n) for i in 1:n, j in 1:n]
    Y = [L * (j - 1) / n + 0.12 * L / n * cospi(2 * (i - 1) / n) for i in 1:n, j in 1:n]
    meas = [1.0 + 0.3 * sinpi(2 * (i + j) / n) for i in 1:n, j in 1:n]
    g = FG.Grids.CurvilinearGrid(FG.Geometry.CartesianGeometry{Float64}(), X, Y, trues(n, n);
        measure = meas, topology = (FG.Grids.Periodic(), FG.Grids.Periodic()), period = (L, L))
    cloud = FG.Grids.UnstructuredGrid(FG.Geometry.CartesianGeometry{Float64}(),
        (vec(X), vec(Y)), vec(meas); topology = (FG.Grids.Periodic(), FG.Grids.Periodic()), period = (L, L))
    return g, cloud
end

Test.@testset "Curvilinear Cartesian transforms as a point cloud" begin
    Random.seed!(2027)
    n = 6
    L = 2π
    ms = (4, 4)
    g, cloud = cv_cartesian(n, L)
    f = randn(n, n)

    c, ks = FFS.calculate_spectrum(g, f, ms; transform = CV_DS, execution = CV_SER)
    cref, ksref = FFS.calculate_spectrum(cloud, vec(f), ms; transform = CV_DS, execution = CV_SER)
    Test.@test cv_rel(c, cref) < 1e-13

    for tr in (FTB.NonuniformFFTsBackend(), FTB.FINUFFTBackend())
        cn, _ = FFS.calculate_spectrum(g, f, ms; transform = tr, eps = 1e-12)
        Test.@test cv_rel(cn, c) < 1e-11
    end

    Test.@test map(collect, ks) == map(collect, ksref)
    fs = FFS.synthesize(g, c, ms; transform = CV_DS, execution = CV_SER)
    fsref = FFS.synthesize(cloud, cref, ms; transform = CV_DS, execution = CV_SER)
    Test.@test size(fs) == size(g)
    Test.@test cv_rel(vec(fs), vec(fsref)) < 1e-13
end

Test.@testset "Curvilinear spherical transforms per point" begin
    Random.seed!(2028)
    n = 6
    lmax = 3
    ms = (lmax + 1, 2lmax + 1)
    Lam = [2π * (i - 1) / n for i in 1:n, j in 1:n]
    Phi = [π * (j - 0.5) / n - π / 2 for i in 1:n, j in 1:n]
    meas = [cos(Phi[i, j]) for i in 1:n, j in 1:n]
    g = FG.Grids.CurvilinearGrid(FG.Geometry.SphericalGeometry(1.0), Lam, Phi, meas, trues(n, n);
        topology = (FG.Grids.Periodic(), FG.Grids.Bounded()), period = (2π, nothing))
    cloud = FG.Grids.UnstructuredGrid(FG.Geometry.SphericalGeometry(1.0),
        (vec(Lam), vec(Phi)), vec(meas))
    f = randn(n, n)

    c, _ = FFS.calculate_spectrum(g, f, ms; transform = CV_DS, execution = CV_SER)
    cref, _ = FFS.calculate_spectrum(cloud, vec(f), ms; transform = CV_DS, execution = CV_SER)
    Test.@test cv_rel(c, cref) < 1e-13
end
