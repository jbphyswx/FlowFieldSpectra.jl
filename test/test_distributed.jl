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

    # Point-partition: DirectSum (serial + threaded inner) and NUFFT.
    cref, kref = FFS.calculate_spectrum(sc, f, ms; transform = SB.DirectSumSpectralBackend(), execution = CB.SerialBackend())
    _, Eref = FFS.isotropic_spectrum(kref, cref; num_bins = 6, dims = 3)
    for inner in (CB.SerialBackend(), CB.ThreadedBackend())
        cd, kd = FFS.calculate_spectrum(sc, f, ms; transform = SB.DirectSumSpectralBackend(), execution = CB.DistributedBackend(inner))
        Test.@test isapprox(cd, cref; atol = 1e-12)
        Test.@test all(collect(kd[d]) ≈ collect(kref[d]) for d in 1:2)
        _, Ed = FFS.isotropic_spectrum(kd, cd; num_bins = 6, dims = 3)
        Test.@test isapprox(Ed, Eref; atol = 1e-12)
    end

    cn, _ = FFS.calculate_spectrum(sc, f, ms; transform = FTB.FINUFFTBackend(), execution = CB.SerialBackend(), eps = 1e-12)
    cnd, _ = FFS.calculate_spectrum(sc, f, ms; transform = FTB.FINUFFTBackend(), execution = CB.DistributedBackend(), eps = 1e-12)
    Test.@test isapprox(cnd, cn; atol = 1e-10)

    # FFT batch-partition (uniform tensor grid; batch split across workers, gathered).
    xs = ucg_axis(Float64, L, 16); ys = ucg_axis(Float64, L, 16)
    ug = ucg((L, L), ms)
    u = [cos(2x) + 0.5 * sin(3y) for x in xs, y in ys]
    ub = cat(u, 2 .* u, 3 .* u, 4 .* u; dims = 3)      # (16,16,4)
    cf, _ = FFS.calculate_spectrum(ug, ub, ms; transform = SB.FFTSpectralBackend(), execution = CB.SerialBackend())
    cfd, _ = FFS.calculate_spectrum(ug, ub, ms; transform = SB.FFTSpectralBackend(), execution = CB.DistributedBackend())
    Test.@test isapprox(cfd, cf; atol = 1e-12)

    # Spherical point-partition (α_w path).
    Random.seed!(8)
    θ = rand(200) .* π
    φ = rand(200) .* 2π
    fθ = rand(200)
    sph = sph_scat(θ, φ)
    cs, _ = FFS.calculate_spectrum(sph, fθ, (8, 15); transform = SB.DirectSumSpectralBackend(), execution = CB.SerialBackend())
    csd, _ = FFS.calculate_spectrum(sph, fθ, (8, 15); transform = SB.DirectSumSpectralBackend(), execution = CB.DistributedBackend())
    Test.@test isapprox(csd, cs; atol = 1e-10)

    # The round-robin shares carry different mean measures (1 on odd points, 3 on even ones), so each
    # share's coefficients weigh in by its measure total, and the measure changes the coefficients.
    meas = [isodd(j) ? 1.0 : 3.0 for j in 1:N]
    scw = FG.Grids.UnstructuredGrid(_cg(Float64), (xv, yv), meas; periodic = (true, true), period = (L, L))
    for t in (SB.DirectSumSpectralBackend(), FTB.FINUFFTBackend())
        kw = t isa FTB.FINUFFTBackend ? (; eps = 1e-12) : (;)
        cw, _ = FFS.calculate_spectrum(scw, f, ms; transform = t, execution = CB.SerialBackend(), kw...)
        cu, _ = FFS.calculate_spectrum(sc, f, ms; transform = t, execution = CB.SerialBackend(), kw...)
        Test.@test maximum(abs, cw .- cu) > 1e-2 * maximum(abs, cu)
        cwd, _ = FFS.calculate_spectrum(scw, f, ms; transform = t, execution = CB.DistributedBackend(), kw...)
        Test.@test isapprox(cwd, cw; atol = 1e-10)
    end

    # A masked node carries no weight, NaN included; each share keeps its slice of the mask. Explicit
    # `weights` are per node, so each share takes its own.
    smask = trues(200); smask[1:3:200] .= false
    fnan = copy(fθ); fnan[.!smask] .= NaN
    smeas = [isodd(j) ? 1.0 : 2.0 for j in 1:200]
    sphm = FG.Grids.UnstructuredGrid(FG.Geometry.SphericalGeometry(1.0), (φ, π / 2 .- θ), smeas, smask)
    cm, _ = FFS.calculate_spectrum(sphm, fnan, (8, 15); transform = SB.DirectSumSpectralBackend(), execution = CB.SerialBackend())
    cmd, _ = FFS.calculate_spectrum(sphm, fnan, (8, 15); transform = SB.DirectSumSpectralBackend(), execution = CB.DistributedBackend())
    Test.@test all(isfinite, cm)
    Test.@test isapprox(cmd, cm; atol = 1e-10)
    w = rand(200)
    cx, _ = FFS.calculate_spectrum(sph, fθ, (8, 15); transform = SB.DirectSumSpectralBackend(), execution = CB.SerialBackend(), weights = w)
    cxd, _ = FFS.calculate_spectrum(sph, fθ, (8, 15); transform = SB.DirectSumSpectralBackend(), execution = CB.DistributedBackend(), weights = w)
    Test.@test isapprox(cxd, cx; atol = 1e-10)

    # On a bounded cloud a share's own origins and Fourier lengths differ from the whole grid's, and every
    # share transforms on the whole grid's.
    scb = FG.Grids.UnstructuredGrid(_cg(Float64), (xv, yv), ones(N))
    Test.@test FFS.Grids.axis_geometry(Float64, FFS._subgrid(scb, 2:2:N), 2) != FFS.Grids.axis_geometry(Float64, scb, 2)
    for t in (SB.DirectSumSpectralBackend(), FTB.FINUFFTBackend())
        kw = t isa FTB.FINUFFTBackend ? (; eps = 1e-12) : (;)
        cb, kb = FFS.calculate_spectrum(scb, f, ms; transform = t, execution = CB.SerialBackend(), kw...)
        cbd, kbd = FFS.calculate_spectrum(scb, f, ms; transform = t, execution = CB.DistributedBackend(), kw...)
        Test.@test isapprox(cbd, cb; atol = 1e-10)
        Test.@test all(collect(kbd[d]) ≈ collect(kb[d]) for d in 1:2)
    end
end
