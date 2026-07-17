# Distributed execution parity (in-process workers via `addprocs`, CPU-only, CI-runnable). The
# distributed result must equal the serial result: point-partitionable transforms (DirectSum/NUFFT on a
# scattered grid) sum α_w-weighted partial coefficients; FFT batch-partitions the trailing axis.

using Test: Test
using Random: Random
using Distributed: Distributed
using FlowFieldSpectra: FlowFieldSpectra as FFS

Distributed.nprocs() == 1 && Distributed.addprocs(2; exeflags = "--project=$(Base.active_project())")
Distributed.@everywhere begin
    using FlowFieldSpectra: FlowFieldSpectra as FFS
    using FFTW: FFTW
    using FINUFFT: FINUFFT
    using OhMyThreads: OhMyThreads
end

Test.@testset "Distributed spectrum parity" begin
    Random.seed!(7)
    L = 2π
    ms = (16, 16)
    N = 200
    xv = rand(N) .* L
    yv = rand(N) .* L
    f = rand(N, 2)                                # (N, batch=2)
    sc = FFS.ScatteredCartesianGrid((xv, yv); domain_size = (L, L))

    # Point-partition: DirectSum (serial + threaded inner) and NUFFT.
    cref, kref = FFS.calculate_spectrum(sc, f, ms; transform = FFS.DirectSumBackend(), execution = FFS.SerialBackend())
    _, Eref = FFS.isotropic_spectrum(kref, cref; num_bins = 6, dims = 3)
    for inner in (FFS.SerialBackend(), FFS.ThreadedBackend())
        cd, kd = FFS.calculate_spectrum(sc, f, ms; transform = FFS.DirectSumBackend(), execution = FFS.DistributedBackend(inner))
        Test.@test isapprox(cd, cref; atol = 1e-12)
        Test.@test all(collect(kd[d]) ≈ collect(kref[d]) for d in 1:2)
        _, Ed = FFS.isotropic_spectrum(kd, cd; num_bins = 6, dims = 3)
        Test.@test isapprox(Ed, Eref; atol = 1e-12)
    end

    cn, _ = FFS.calculate_spectrum(sc, f, ms; transform = FFS.NUFFTBackend(), execution = FFS.SerialBackend(), eps = 1e-12)
    cnd, _ = FFS.calculate_spectrum(sc, f, ms; transform = FFS.NUFFTBackend(), execution = FFS.DistributedBackend(), eps = 1e-12)
    Test.@test isapprox(cnd, cn; atol = 1e-10)

    # FFT batch-partition (uniform tensor grid; batch split across workers, gathered).
    ug = FFS.UniformCartesianGrid(; domain = (L, L), n = ms)
    xs, ys = ug.axes
    u = [cos(2x) + 0.5 * sin(3y) for x in xs, y in ys]
    ub = cat(u, 2 .* u, 3 .* u, 4 .* u; dims = 3)      # (16,16,4)
    cf, _ = FFS.calculate_spectrum(ug, ub, ms; transform = FFS.FFTBackend(), execution = FFS.SerialBackend())
    cfd, _ = FFS.calculate_spectrum(ug, ub, ms; transform = FFS.FFTBackend(), execution = FFS.DistributedBackend())
    Test.@test isapprox(cfd, cf; atol = 1e-12)

    # Spherical point-partition (α_w spherical path).
    Random.seed!(8)
    θ = rand(200) .* π
    φ = rand(200) .* 2π
    fθ = rand(200)
    sph = FFS.ScatteredSphericalGrid(θ, φ)
    cs, _ = FFS.calculate_spectrum(sph, fθ, (8, 15); transform = FFS.DirectSumBackend(), execution = FFS.SerialBackend())
    csd, _ = FFS.calculate_spectrum(sph, fθ, (8, 15); transform = FFS.DirectSumBackend(), execution = FFS.DistributedBackend())
    Test.@test isapprox(csd, cs; atol = 1e-10)
end
