# Distributed execution parity (in-process workers via `addprocs`, CPU-only, CI-runnable). The
# distributed result must equal the serial result: point-partitionable transforms (DirectSum/NUFFT on a
# scattered grid) sum α_w-weighted partial coefficients; FFT batch-partitions the trailing axis.
# (Grid constructors + CB/SB aliases come from runtests.jl.)

using Test: Test
using Random: Random
using Distributed: Distributed
using FlowFieldSpectra: FlowFieldSpectra as FFS
using FlowTransformBindings: FlowTransformBindings as FTB

Distributed.nprocs() == 1 && Distributed.addprocs(2; exeflags = "--project=$(Base.active_project())")
Distributed.@everywhere begin
    using FlowFieldSpectra: FlowFieldSpectra as FFS
    using FlowGeometries: FlowGeometries       # workers deserialize/build the FlowGeometries subgrid
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
    sc = scg((xv, yv), (L, L))

    # FFT batch-partition (uniform tensor grid; batch split across workers, gathered).
    xs = ucg_axis(Float64, L, 16); ys = ucg_axis(Float64, L, 16)
    ug = ucg((L, L), ms)
    u = [cos(2x) + 0.5 * sin(3y) for x in xs, y in ys]
    ub = cat(u, 2 .* u, 3 .* u, 4 .* u; dims = 3)      # (16,16,4)
    cf, _ = FFS.calculate_spectrum(ug, ub, ms; transform = SB.FFTSpectralBackend(), execution = CB.SerialBackend())
    cfd, _ = FFS.calculate_spectrum(ug, ub, ms; transform = SB.FFTSpectralBackend(), execution = CB.DistributedBackend())
    Test.@test isapprox(cfd, cf; atol = 1e-12)

    # The round-robin shares carry different mean measures (1 on odd points, 3 on even ones), so each
    # share's coefficients weigh in by its measure total, and the measure changes the coefficients.
    meas = [isodd(j) ? 1.0 : 3.0 for j in 1:N]
    scw = FG.Grids.UnstructuredGrid(_cg(Float64), (xv, yv), meas; topology = (FG.Grids.Periodic(), FG.Grids.Periodic()), period = (L, L))
    for t in (SB.DirectSumSpectralBackend(), FTB.FINUFFTBackend())
        kw = t isa FTB.FINUFFTBackend ? (; eps = 1e-12) : (;)
        cw, _ = FFS.calculate_spectrum(scw, f, ms; transform = t, execution = CB.SerialBackend(), kw...)
        cu, _ = FFS.calculate_spectrum(sc, f, ms; transform = t, execution = CB.SerialBackend(), kw...)
        Test.@test maximum(abs, cw .- cu) > 1e-2 * maximum(abs, cu)
        cwd, _ = FFS.calculate_spectrum(scw, f, ms; transform = t, execution = CB.DistributedBackend(), kw...)
        Test.@test isapprox(cwd, cw; atol = 1e-10)
    end

    # A masked node carries no weight, NaN included; each share keeps its slice of the mask, and the
    # result is the unmasked grid's on a field zeroed there. Explicit `weights` are per node, so each
    # share takes its own.
    Random.seed!(8)
    θ = rand(200) .* π
    φ = rand(200) .* 2π
    fθ = rand(200)
    sph = sph_scat(θ, φ)
    smask = trues(200); smask[1:3:200] .= false
    fnan = copy(fθ); fnan[.!smask] .= NaN
    smeas = [isodd(j) ? 1.0 : 2.0 for j in 1:200]
    sphm = FG.Grids.UnstructuredGrid(FG.Geometry.SphericalGeometry(1.0), (φ, π / 2 .- θ), smeas, smask)
    sphu = FG.Grids.UnstructuredGrid(FG.Geometry.SphericalGeometry(1.0), (φ, π / 2 .- θ), smeas)
    cz, _ = FFS.calculate_spectrum(sphu, ifelse.(smask, fθ, 0.0), (8, 15); transform = SB.DirectSumSpectralBackend(),
        execution = CB.SerialBackend())
    cmd, _ = FFS.calculate_spectrum(sphm, fnan, (8, 15); transform = SB.DirectSumSpectralBackend(), execution = CB.DistributedBackend())
    Test.@test isapprox(cmd, cz; atol = 1e-12)
    w = rand(200)
    cx, _ = FFS.calculate_spectrum(sph, fθ, (8, 15); transform = SB.DirectSumSpectralBackend(), execution = CB.SerialBackend(), weights = w)
    cxd, _ = FFS.calculate_spectrum(sph, fθ, (8, 15); transform = SB.DirectSumSpectralBackend(), execution = CB.DistributedBackend(), weights = w)
    Test.@test isapprox(cxd, cx; atol = 1e-12)

    # On a bounded cloud every share transforms on the whole grid's origins and Fourier lengths.
    scb = FG.Grids.UnstructuredGrid(_cg(Float64), (xv, yv), ones(N))
    for t in (SB.DirectSumSpectralBackend(), FTB.FINUFFTBackend())
        kw = t isa FTB.FINUFFTBackend ? (; eps = 1e-12) : (;)
        cb, kb = FFS.calculate_spectrum(scb, f, ms; transform = t, execution = CB.SerialBackend(), kw...)
        cbd, kbd = FFS.calculate_spectrum(scb, f, ms; transform = t, execution = CB.DistributedBackend(), kw...)
        Test.@test isapprox(cbd, cb; atol = 1e-10)
        Test.@test all(collect(kbd[d]) ≈ collect(kb[d]) for d in 1:2)
    end
end

Test.@testset "Distributed plans and synthesis" begin
    # Each worker holds the local plan of its share; an execution equals the serial transform. A node
    # cloud divides its points, a uniform grid its batch columns; synthesis divides the same way.
    Random.seed!(17)
    L = 2π
    ms = (16, 16)
    N = 200
    xv = rand(N) .* L
    yv = rand(N) .* L
    f = rand(N, 2)
    dist = CB.DistributedBackend()
    meas = [isodd(j) ? 1.0 : 3.0 for j in 1:N]
    scw = FG.Grids.UnstructuredGrid(_cg(Float64), (xv, yv), meas)          # bounded, unequal measure
    for (t, kw, tol) in ((SB.DirectSumSpectralBackend(), (;), 1e-12), (FTB.FINUFFTBackend(), (; eps = 1e-12), 1e-10))
        cs, ks = FFS.calculate_spectrum(scw, f, ms; transform = t, execution = CB.SerialBackend(), kw...)
        p = FFS.plan_spectrum(scw, Float64, ms; transform = t, execution = dist, batch = (2,), kw...)
        buf = FFS.allocate_coefficients(p)
        # A point set widens both even axes of the coefficient box.
        Test.@test size(buf) == size(cs) == (ms[1] ÷ 2 + 1, ms[2] + 1, 2)
        for _ in 1:2
            kd = FFS.calculate_spectrum!(buf, p, f)
            Test.@test isapprox(buf, cs; atol = tol)
            Test.@test all(collect(kd[d]) ≈ collect(ks[d]) for d in 1:2)
        end
        FFS.close!(p)
        # Each worker evaluates the series over the whole box at its own points.
        ss = FFS.synthesize(scw, cs, ms; transform = t, execution = CB.SerialBackend(), kw...)
        sd = FFS.synthesize(scw, cs, ms; transform = t, execution = dist, kw...)
        Test.@test isapprox(sd, ss; atol = tol)
    end

    # A uniform grid divides its batch columns among the workers.
    xs = ucg_axis(Float64, L, 16)
    ug = ucg((L, L), ms)
    ub = cat((b .* [cos(2x) + 0.5 * sin(3y) for x in xs, y in xs] for b in 1:4)...; dims = 3)
    cf, _ = FFS.calculate_spectrum(ug, ub, ms; transform = SB.FFTSpectralBackend(), execution = CB.SerialBackend())
    pf = FFS.plan_spectrum(ug, Float64, ms; transform = SB.FFTSpectralBackend(), execution = dist, batch = (4,))
    bf = FFS.allocate_coefficients(pf)
    FFS.calculate_spectrum!(bf, pf, ub)
    Test.@test isapprox(bf, cf; atol = 1e-12)
    Test.@test isapprox(FFS.synthesize(ug, cf, ms; transform = SB.FFTSpectralBackend(), execution = dist), ub;
        atol = 1e-12)

    # The sphere, with a measure that varies, divides its points.
    θ = rand(N) .* π
    φ = rand(N) .* 2π
    sph = FG.Grids.UnstructuredGrid(FG.Geometry.SphericalGeometry(1.0), (φ, π / 2 .- θ), meas)
    fθ = rand(N)
    csr, _ = FFS.calculate_spectrum(sph, fθ, (8, 15); transform = SB.DirectSumSpectralBackend(),
        execution = CB.SerialBackend())
    psp = FFS.plan_spectrum(sph, Float64, (8, 15); transform = SB.DirectSumSpectralBackend(), execution = dist)
    bsp = FFS.allocate_coefficients(psp)
    FFS.calculate_spectrum!(bsp, psp, fθ)
    Test.@test isapprox(bsp, csr; atol = 1e-10)
    Test.@test isapprox(FFS.synthesize(sph, csr, (8, 15); transform = SB.DirectSumSpectralBackend(), execution = dist),
        FFS.synthesize(sph, csr, (8, 15); transform = SB.DirectSumSpectralBackend(), execution = CB.SerialBackend());
        atol = 1e-12)
end
