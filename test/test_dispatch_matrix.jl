# A direction's Fourier length, and the error raised for a transform that does not apply to a grid.

using Test: Test
using FlowFieldSpectra: FlowFieldSpectra as FFS
using FlowTransformBindings: FlowTransformBindings as FTB
using FlowGeometries: FlowGeometries as FG
using SpectralBackends: SpectralBackends as SB
using ComputationalBackends: ComputationalBackends as CB
using FFTW: FFTW
using NonuniformFFTs: NonuniformFFTs
using FINUFFT: FINUFFT
using Random: Random

const DM_CART = FG.Geometry.CartesianGeometry{Float64}()
const DM_SPHEROID = FG.Geometry.SpheroidGeometry(1.0, inv(298.257223563))

dm_unif(L, N) = range(0.0, L; length = N + 1)[1:N]

Test.@testset "A direction's Fourier length is the length its cells cover" begin
    # A bounded direction takes the length its cells cover, `N·Δ` on a lattice, whatever `period` a node
    # cloud stores for it.
    L = 2π
    N = 8
    xs = collect(dm_unif(L, N))
    cloud = FG.Grids.UnstructuredGrid(DM_CART, (repeat(xs, N), repeat(xs; inner = N)), fill(1.0, N^2);
        topology = (FG.Grids.Periodic(), FG.Grids.Bounded()), period = (L, 5.0))
    ks = FFS.Grids.physical_wavenumbers(cloud, (N, N), Float64)
    Test.@test isapprox(ks[1][2] - ks[1][1], 2π / L; rtol = 1e-12)
    Test.@test isapprox(ks[2][2] - ks[2][1], 2π / L; rtol = 1e-12)

    # On a bounded lattice of spacing `Δ` starting off the origin, the measure-weighted sums measured from
    # the first sample are the DFT of the samples, so every transform and the shuffled node cloud equal the
    # FFT, at wavenumbers `2π/(N·Δ)`.
    Random.seed!(6060)
    Δ = 0.3
    ax = range(1.7; step = Δ, length = N)
    gb = FG.Grids.StructuredGrid(DM_CART, ax, ax)
    f = randn(N, N)
    cf, kf = FFS.calculate_spectrum(gb, f, (N, N); transform = SB.FFTSpectralBackend(),
        execution = CB.SerialBackend())
    Test.@test isapprox(kf[2][2] - kf[2][1], 2π / (N * Δ); rtol = 1e-12)
    for (t, kw) in ((SB.DirectSumSpectralBackend(), (;)), (FTB.FINUFFTBackend(), (; eps = 1e-13)),
                    (FTB.NonuniformFFTsBackend(), (; eps = 1e-13)))
        c, _ = FFS.calculate_spectrum(gb, f, (N, N); transform = t, execution = CB.SerialBackend(), kw...)
        Test.@test isapprox(c, cf; rtol = 1e-10, atol = 1e-12)
    end
    # A node cloud widens its even axis 2 to both `±N/2`, which these lattice points sample alike; at every
    # frequency of the lattice's own box it gives the FFT's coefficient.
    perm = Random.randperm(N^2)
    nc = FG.Grids.UnstructuredGrid(DM_CART, (repeat(collect(ax), N)[perm], repeat(collect(ax); inner = N)[perm]),
        fill(Δ^2, N^2))
    cn, _ = FFS.calculate_spectrum(nc, vec(f)[perm], (N, N); transform = SB.DirectSumSpectralBackend(),
        execution = CB.SerialBackend())
    Test.@test size(cn) == (N ÷ 2 + 1, N + 1)
    at = [k >= 0 ? k + 1 : N + k + 2 for k in Int.(FFTW.fftfreq(N, N))]   # frequency `k` on the `N + 1` axis
    Test.@test isapprox(cn[:, at], cf; rtol = 1e-10, atol = 1e-12)
    Test.@test isapprox(cn[:, N ÷ 2 + 1], cn[:, N ÷ 2 + 2]; rtol = 1e-10, atol = 1e-12)
end

Test.@testset "The catch-all names the grid and the transform" begin
    lmax = 3
    sms = (lmax + 1, 2 * lmax + 1)
    nlon, nlat = 2 * lmax + 1, lmax + 1
    g = FG.Grids.StructuredGrid(DM_SPHEROID,
        [2π * (i - 1) / nlon for i in 1:nlon], [π * (j - 0.5) / nlat - π / 2 for j in 1:nlat];
        topology = (FG.Grids.Periodic(), FG.Grids.Bounded()), period = (2π, 0.0))
    err = try
        FFS.calculate_spectrum(g, randn(nlon, nlat), sms; transform = SB.FSHTSpectralBackend(),
            execution = CB.SerialBackend())
        nothing
    catch e
        e
    end
    Test.@test err isa ArgumentError
    Test.@test occursin("SpheroidGeometry", err.msg)
end
