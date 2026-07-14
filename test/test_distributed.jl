# Distributed execution parity (in-process workers via `addprocs`, CPU-only, CI-runnable).
# Distributed result must equal the serial result: complex coefficients are additive over the
# point partition (with the α_w = N_w/N compensation), and the batch partition gathers disjoint
# field slices.

using Test: Test
using Random: Random
using Distributed: Distributed
using SharedArrays: SharedArrays
using FlowFieldSpectra: FlowFieldSpectra as FFS

# Two local workers, with the FlowFieldSpectra Distributed extension + the transform packages
# available everywhere (workers must load the extension's trigger packages so the `pmap` closure —
# whose type lives in the extension module — deserializes, and must load FINUFFT/OhMyThreads to run
# the NUFFT / threaded-inner partials).
Distributed.nprocs() == 1 && Distributed.addprocs(2; exeflags = "--project=$(Base.active_project())")
Distributed.@everywhere begin
    using Distributed: Distributed
    using SharedArrays: SharedArrays
    using FlowFieldSpectra: FlowFieldSpectra as FFS
    using FFTW: FFTW
    using FINUFFT: FINUFFT
    using OhMyThreads: OhMyThreads
end

Test.@testset "Distributed spectrum parity" begin
    Random.seed!(7)
    L = 2π
    ms = (16, 16)
    dx = L / ms[1]
    dy = L / ms[2]
    xs = range(0.0, stop = L - dx, length = ms[1])
    ys = range(0.0, stop = L - dy, length = ms[2])
    xv = vec([x for x in xs, y in ys])
    yv = vec([y for x in xs, y in ys])
    u = @. cos(2xv) + 0.5 * sin(3yv)
    v = @. sin(xv)
    g = FFS.ScatteredCartesianGrid((xv, yv); domain_size = (L, L))

    # DirectSum point-partition: distributed == serial (coefficients and binned energy), for a
    # serial inner and a threaded inner (hybrid distributed+threaded).
    cref, kref = FFS.calculate_spectrum(g, (u, v), ms; transform = FFS.DirectSumBackend(), execution = FFS.SerialBackend())
    _, Eref = FFS.isotropic_spectrum(kref, cref; num_bins = 6)
    for inner in (FFS.SerialBackend(), FFS.ThreadedBackend())
        cd, kd = FFS.calculate_spectrum(g, (u, v), ms; transform = FFS.DirectSumBackend(), execution = FFS.DistributedBackend(inner))
        Test.@test isapprox(cd, cref; atol = 1e-12)
        Test.@test all(collect(kd[d]) ≈ collect(kref[d]) for d in 1:2)
        _, Ed = FFS.isotropic_spectrum(kd, cd; num_bins = 6)
        Test.@test isapprox(Ed, Eref; atol = 1e-12)
    end

    # NUFFT point-partition parity.
    cn, _ = FFS.calculate_spectrum(g, (u, v), ms; transform = FFS.NUFFTBackend(), execution = FFS.SerialBackend(), eps = 1e-12)
    cnd, _ = FFS.calculate_spectrum(g, (u, v), ms; transform = FFS.NUFFTBackend(), execution = FFS.DistributedBackend(), eps = 1e-12)
    Test.@test isapprox(cnd, cn; atol = 1e-10)

    # FFT batch-partition parity (fields split across workers, gathered along the trailing axis).
    ug = FFS.UniformCartesianGrid((xv, yv); domain_size = (L, L))
    cf, _ = FFS.calculate_spectrum(ug, (u, v), ms; transform = FFS.FFTBackend(), execution = FFS.SerialBackend())
    cfd, _ = FFS.calculate_spectrum(ug, (u, v), ms; transform = FFS.FFTBackend(), execution = FFS.DistributedBackend())
    Test.@test isapprox(cfd, cf; atol = 1e-12)

    # Spherical uniform-weight point-partition parity (exercises the α_w = N_w/N spherical path).
    Random.seed!(8)
    θ = rand(200) .* π
    φ = rand(200) .* 2π
    fθ = rand(200)
    sph = FFS.ScatteredSphericalGrid(θ, φ)
    cs, _ = FFS.calculate_spectrum(sph, (fθ,), (8, 15); transform = FFS.DirectSumBackend(), execution = FFS.SerialBackend())
    csd, _ = FFS.calculate_spectrum(sph, (fθ,), (8, 15); transform = FFS.DirectSumBackend(), execution = FFS.DistributedBackend())
    Test.@test isapprox(csd, cs; atol = 1e-10)
end
