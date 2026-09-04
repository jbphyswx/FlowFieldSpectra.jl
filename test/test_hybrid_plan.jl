# The hybrid FFT/NUFFT composite, held for reuse.
#
# A grid uniform in some directions and stretched in others takes one FFT over the uniform axes and a 1-D
# type-1 NUFFT along each stretched one. The one-shot rebuilds the FFTW plan and every axis NUFFT on each
# call; `plan_spectrum` holds them along with the working arrays, so a reused execution writes into
# memory the plan owns.
#
# `@allocated` inside a `@testset` measures the harness, so every allocation assertion here goes through
# a top-level barrier function, as `test_allocs.jl` does.

using Test: Test
using FlowFieldSpectra: FlowFieldSpectra as FFS
using FlowGeometries: FlowGeometries as FG
using SpectralBackends: SpectralBackends as SB
using ComputationalBackends: ComputationalBackends as CB
using FFTW: FFTW
using NonuniformFFTs: NonuniformFFTs
using KernelAbstractions: KernelAbstractions as KA
using Random: Random

const HP_SER = CB.SerialBackend()
const HP_FFT = SB.FFTSpectralBackend()
hp_rel(a, b) = maximum(abs, a .- b) / max(maximum(abs, b), eps())
hp_uni(L, N) = range(0.0, L; length = N + 1)[1:N]
hp_str(L, N) = [L * (i - 1) / N + 0.04 * L * sinpi(2 * (i - 1) / N) for i in 1:N]
hp_grid(L, N1, N2) = FG.Grids.StructuredGrid(FG.Geometry.CartesianGeometry{Float64}(),
    hp_uni(L, N1), hp_str(L, N2); periodic = (true, true), period = (L, L))

# Top-level allocation barriers.
hp_a_plan(out, pl, fld) = @allocated FFS.calculate_spectrum!(out, pl, fld)
hp_a_one(gr, fld, ms) = @allocated FFS.calculate_spectrum(gr, fld, ms; transform = HP_FFT, execution = HP_SER)
hp_a_region(out, r, fld) = @allocated FFS._region_fft_exec!(out, r, fld)
hp_a_publish(c, W, ph, nsx, pms, nt, neg) = @allocated FFS.Packing.publish_packed!(c, W, ph, nsx, pms, nt, neg)
hp_a_twins(tw, W, n, nt, neg) = @allocated FFS.Packing.gather_conj_twins!(tw, W, n, nt, neg)
hp_a_quad(out, fld, α, np, nt) = @allocated FFS.quadrature_weighted_into!(out, fld, α, np, nt)

Test.@testset "Hybrid plan matches the one-shot" begin
    Random.seed!(17)
    L = 2π
    for (N1, N2) in ((16, 16), (16, 15), (15, 16))     # even, odd-stretched, odd-uniform-axis-1
        g = hp_grid(L, N1, N2)
        f = randn(N1, N2)
        cref, ksref = FFS.calculate_spectrum(g, f, (N1, N2); transform = HP_FFT, execution = HP_SER)
        p = FFS.plan_spectrum(g, Float64, (N1, N2); transform = HP_FFT, execution = HP_SER)
        Test.@test p isa FFS.HybridPlan
        buf = zeros(ComplexF64, FFS.Packing.packed_size((N1, N2), Val(true))...)
        ks = FFS.calculate_spectrum!(buf, p, f)
        Test.@test hp_rel(buf, cref) < 1e-11

        # The Nyquist twin rides on `ks` and must agree with the one-shot's.
        tw = FFS.Packing.axis_twin(ks[1])
        twr = FFS.Packing.axis_twin(ksref[1])
        Test.@test (tw === nothing) == (twr === nothing)
        if tw !== nothing
            for (a, b) in zip(tw.slices, twr.slices)
                isempty(a) || Test.@test hp_rel(a, b) < 1e-11
            end
        end

        # Reuse leaves the plan intact, and a second field goes through the same one.
        for _ in 1:3
            FFS.calculate_spectrum!(buf, p, f)
            Test.@test hp_rel(buf, cref) < 1e-11
        end
        f2 = randn(N1, N2)
        c2, _ = FFS.calculate_spectrum(g, f2, (N1, N2); transform = HP_FFT, execution = HP_SER)
        FFS.calculate_spectrum!(buf, p, f2)
        Test.@test hp_rel(buf, c2) < 1e-11

        # The caller's field is never mutated.
        keep = copy(f)
        FFS.calculate_spectrum!(buf, p, f)
        Test.@test f == keep
    end
end

Test.@testset "Hybrid plan: complex, batched, and the uniform split" begin
    Random.seed!(18)
    L = 2π
    N = 16
    g = hp_grid(L, N, N)

    fb = randn(N, N, 2)
    cb, _ = FFS.calculate_spectrum(g, fb, (N, N); transform = HP_FFT, execution = HP_SER)
    pb = FFS.plan_spectrum(g, Float64, (N, N); transform = HP_FFT, execution = HP_SER, batch = (2,))
    bb = zeros(ComplexF64, FFS.Packing.packed_size((N, N), Val(true))..., 2)
    FFS.calculate_spectrum!(bb, pb, fb)
    Test.@test hp_rel(bb, cb) < 1e-11
    Test.@test_throws DimensionMismatch FFS.calculate_spectrum!(bb, pb, randn(N, N))

    fc = randn(ComplexF64, N, N)
    cc, _ = FFS.calculate_spectrum(g, fc, (N, N); transform = HP_FFT, execution = HP_SER)
    pc = FFS.plan_spectrum(g, ComplexF64, (N, N); transform = HP_FFT, execution = HP_SER)
    bc = zeros(ComplexF64, N, N)
    FFS.calculate_spectrum!(bc, pc, fc)
    Test.@test hp_rel(bc, cc) < 1e-11

    # An all-uniform grid keeps the pure FFTW plan; an all-stretched one names NUFFT.
    gu = FG.Grids.StructuredGrid(FG.Geometry.CartesianGeometry{Float64}(),
        hp_uni(L, N), hp_uni(L, N); periodic = (true, true), period = (L, L))
    Test.@test !(FFS.plan_spectrum(gu, Float64, (N, N); transform = HP_FFT, execution = HP_SER) isa
        FFS.HybridPlan)
    gs = FG.Grids.StructuredGrid(FG.Geometry.CartesianGeometry{Float64}(),
        hp_str(L, N), hp_str(L, N); periodic = (true, true), period = (L, L))
    Test.@test_throws ArgumentError FFS.plan_spectrum(gs, Float64, (N, N); transform = HP_FFT,
        execution = HP_SER)
end

Test.@testset "Hybrid plan holds every buffer it owns" begin
    # What the plan removes is FFS's own per-call allocation. Each step it owns is asserted at exactly
    # zero; the residue is the NUFFT provider's own per-execution cost inside `exec_type1!`, which the
    # one-shot pays too and which grows with the thread count.
    Random.seed!(19)
    L = 2π
    N = 16
    g = hp_grid(L, N, N)
    f = randn(N, N)
    p = FFS.plan_spectrum(g, Float64, (N, N); transform = HP_FFT, execution = HP_SER)
    buf = zeros(ComplexF64, FFS.Packing.packed_size((N, N), Val(true))...)
    for _ in 1:3
        FFS.calculate_spectrum!(buf, p, f)
    end

    # The quadrature scaling writes into the plan's buffer.
    Test.@test p.qbuf !== nothing
    hp_a_quad(p.qbuf, f, p.qw, p.npts, p.ntrans)
    Test.@test hp_a_quad(p.qbuf, f, p.qw, p.npts, p.ntrans) == 0

    # The region FFT writes into the plan's working array.
    W1 = FFS._region_fft_exec!(p.work[1], p.region, p.qbuf)
    Test.@test hp_a_region(p.work[1], p.region, p.qbuf) == 0

    # Publishing and the twin gather write into the caller's buffer and the plan's slices.
    W = FFS._axis_nufft_exec!(p.work[2], p.axes[1], W1, p.sdims[1])
    hp_a_publish(buf, W, p.phase, p.nsx, p.pms, p.ntrans, p.neg)
    Test.@test hp_a_publish(buf, W, p.phase, p.nsx, p.pms, p.ntrans, p.neg) == 0
    hp_a_twins(p.twins, W, prod(p.nsx), p.ntrans, p.neg)
    Test.@test hp_a_twins(p.twins, W, prod(p.nsx), p.ntrans, p.neg) == 0

    # So a reused execution stays well under the one-shot, which rebuilds the FFTW plan and every axis
    # NUFFT on top of that same provider cost.
    hp_a_one(g, f, (N, N))
    Test.@test hp_a_plan(buf, p, f) < 0.5 * hp_a_one(g, f, (N, N))
end

Test.@testset "Hybrid composite on a device" begin
    # The FFT over the uniform axes comes from the GPUFFT extension and each stretched axis takes the
    # device 1-D NUFFT, so the working array never leaves the backend. The derivation is shared with the
    # host composite; the twins and the publish are device-resident.
    Random.seed!(23)
    L = 2π
    gpu = CB.GPUBackend(KA.CPU())
    for (N1, N2) in ((16, 16), (16, 15), (15, 16))
        g = hp_grid(L, N1, N2)
        f = randn(N1, N2)
        ch, ksh = FFS.calculate_spectrum(g, f, (N1, N2); transform = HP_FFT, execution = HP_SER)
        cg, ksg = FFS.calculate_spectrum(g, f, (N1, N2); transform = HP_FFT, execution = gpu)
        Test.@test size(cg) == size(ch)
        Test.@test hp_rel(cg, ch) < 1e-11
        tg = FFS.Packing.axis_twin(ksg[1])
        th = FFS.Packing.axis_twin(ksh[1])
        Test.@test (tg === nothing) == (th === nothing)
        if tg !== nothing
            for (a, b) in zip(tg.slices, th.slices)
                isempty(a) || Test.@test hp_rel(a, b) < 1e-11
            end
        end
    end

    N = 16
    g = hp_grid(L, N, N)
    fb = randn(N, N, 2)
    cgb, _ = FFS.calculate_spectrum(g, fb, (N, N); transform = HP_FFT, execution = gpu)
    chb, _ = FFS.calculate_spectrum(g, fb, (N, N); transform = HP_FFT, execution = HP_SER)
    Test.@test hp_rel(cgb, chb) < 1e-11

    fc = randn(ComplexF64, N, N)
    cgc, _ = FFS.calculate_spectrum(g, fc, (N, N); transform = HP_FFT, execution = gpu)
    chc, _ = FFS.calculate_spectrum(g, fc, (N, N); transform = HP_FFT, execution = HP_SER)
    Test.@test hp_rel(cgc, chc) < 1e-11

    # An all-stretched grid names NUFFT on the device too.
    gs = FG.Grids.StructuredGrid(FG.Geometry.CartesianGeometry{Float64}(),
        hp_str(L, N), hp_str(L, N); periodic = (true, true), period = (L, L))
    Test.@test_throws ArgumentError FFS.calculate_spectrum(gs, randn(N, N), (N, N);
        transform = HP_FFT, execution = gpu)
end
