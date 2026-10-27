# The hybrid FFT/NUFFT composite, one-shot and held for reuse, and the line passes it shares with the
# separable plan.
#
# A grid uniform in some directions and stretched in others takes one FFT over the uniform axes and a 1-D
# type-1 NUFFT along each stretched one. Every result is held to the direct sum, which shares no code with
# the composite, entry for entry over the coefficient box, which holds both `±N₂/2` of an even stretched
# axis for a real field.
#
# `@allocated` inside a `@testset` measures the harness, so the allocation probe is a top-level function.

using Test: Test
using FlowFieldSpectra: FlowFieldSpectra as FFS
using FlowTransformBindings: FlowTransformBindings as FTB
using FlowGeometries: FlowGeometries as FG
using SpectralBackends: SpectralBackends as SB
using ComputationalBackends: ComputationalBackends as CB
using FFTW: FFTW
using NonuniformFFTs: NonuniformFFTs
using FINUFFT: FINUFFT
using KernelAbstractions: KernelAbstractions as KA
using Random: Random

const HP_SER = CB.SerialBackend()
const HP_FFT = SB.FFTSpectralBackend()
const HP_DS = SB.DirectSumSpectralBackend()
hp_rel(a, b) = maximum(abs, a .- b) / max(maximum(abs, b), eps())
hp_uni(L, N) = range(0.0, L; length = N + 1)[1:N]
hp_str(L, N) = [L * (i - 1) / N + 0.04 * L * sinpi(2 * (i - 1) / N) for i in 1:N]
hp_grid(L, N1, N2) = FG.Grids.StructuredGrid(FG.Geometry.CartesianGeometry{Float64}(),
    hp_uni(L, N1), hp_str(L, N2); topology = (FG.Grids.Periodic(), FG.Grids.Periodic()), period = (L, L))

# The coefficients and wavenumber axes against the direct sum's.
function hp_against_ds(c, ks, g, f, ms; tol = 1e-10)
    cd, ksd = FFS.calculate_spectrum(g, f, ms; transform = HP_DS, execution = HP_SER)
    Test.@test size(c) == size(cd)
    Test.@test all(d -> collect(ks[d]) ≈ collect(ksd[d]), eachindex(ks))
    Test.@test hp_rel(c, cd) < tol
end

# Top-level allocation barrier.
hp_a_plan(out, pl, fld) = @allocated FFS.calculate_spectrum!(out, pl, fld)

Test.@testset "Hybrid composite, one-shot and plan, on the host" begin
    Random.seed!(17)
    L = 2π
    for nu in (FTB.NonuniformFFTsBackend(), FTB.FINUFFTBackend()), (N1, N2) in ((16, 16), (16, 15), (15, 16))
        # even, odd-stretched, odd-uniform-axis-1
        kw = (; transform = HP_FFT, execution = HP_SER, nufft = nu, eps = 1e-13)
        g = hp_grid(L, N1, N2)
        f = randn(N1, N2)
        c1, ks1 = FFS.calculate_spectrum(g, f, (N1, N2); kw...)
        hp_against_ds(c1, ks1, g, f, (N1, N2))
        p = FFS.plan_spectrum(g, Float64, (N1, N2); kw...)
        buf = FFS.allocate_coefficients(p)
        keep = copy(f)
        for _ in 1:2
            ks = FFS.calculate_spectrum!(buf, p, f)
            hp_against_ds(buf, ks, g, f, (N1, N2))
        end
        Test.@test f == keep                          # the caller's field is never mutated
    end

    N = 16
    g = hp_grid(L, N, N)
    for (f, T, batch) in ((randn(N, N, 2), Float64, (2,)), (randn(ComplexF64, N, N), ComplexF64, ()))
        c1, ks1 = FFS.calculate_spectrum(g, f, (N, N); transform = HP_FFT, execution = HP_SER, eps = 1e-13)
        hp_against_ds(c1, ks1, g, f, (N, N))
        p = FFS.plan_spectrum(g, T, (N, N); transform = HP_FFT, execution = HP_SER, batch = batch, eps = 1e-13)
        buf = FFS.allocate_coefficients(p)
        ks = FFS.calculate_spectrum!(buf, p, f)
        hp_against_ds(buf, ks, g, f, (N, N))
    end
    pb = FFS.plan_spectrum(g, Float64, (N, N); transform = HP_FFT, execution = HP_SER, batch = (2,))
    Test.@test_throws DimensionMismatch FFS.calculate_spectrum!(FFS.allocate_coefficients(pb), pb, randn(N, N))

    # With FINUFFT on the stretched axes every step a reused execution takes writes into memory the plan
    # owns, so it allocates nothing.
    p = FFS.plan_spectrum(g, Float64, (N, N); transform = HP_FFT, execution = HP_SER, nufft = FTB.FINUFFTBackend())
    buf = FFS.allocate_coefficients(p)
    f = randn(N, N)
    hp_a_plan(buf, p, f)
    Test.@test hp_a_plan(buf, p, f) == 0
end

Test.@testset "Hybrid composite on a device" begin
    # The uniform axes take the device FFT through AbstractFFTs and each stretched axis the device 1-D
    # NUFFT, each library in turn.
    Random.seed!(23)
    L = 2π
    gpu = CB.GPUBackend(KA.CPU())
    for nu in (FTB.FINUFFTBackend(), FTB.NonuniformFFTsBackend()), (N1, N2) in ((16, 16), (16, 15), (15, 16))
        g = hp_grid(L, N1, N2)
        for f in (randn(N1, N2), randn(N1, N2, 2), randn(ComplexF64, N1, N2))
            c, ks = FFS.calculate_spectrum(g, f, (N1, N2); transform = HP_FFT, execution = gpu, nufft = nu,
                eps = 1e-13)
            hp_against_ds(c, ks, g, f, (N1, N2))
        end
    end
end

Test.@testset "Hybrid plan on a device, reused" begin
    # The plan holds the device FFT, the device NUFFT per stretched axis and the gather table; each
    # execution is held to the direct sum.
    Random.seed!(29)
    L = 2π
    gpu = CB.GPUBackend(KA.CPU())
    for nu in (FTB.FINUFFTBackend(), FTB.NonuniformFFTsBackend()), (N1, N2) in ((16, 16), (15, 16))
        g = hp_grid(L, N1, N2)
        for (T, batch) in ((Float64, ()), (Float64, (2,)), (ComplexF64, ()))
            p = FFS.plan_spectrum(g, T, (N1, N2); transform = HP_FFT, execution = gpu, nufft = nu, batch = batch,
                eps = 1e-13)
            buf = FFS.allocate_coefficients(p)
            for _ in 1:2
                f = randn(T, N1, N2, batch...)
                keep = copy(f)
                ks = FFS.calculate_spectrum!(buf, p, f)
                hp_against_ds(buf, ks, g, f, (N1, N2))
                Test.@test f == keep
            end
            FFS.close!(p)
        end
    end
end

Test.@testset "Hybrid synthesis against the direct-sum synthesis" begin
    # Uniform along axis 1 and stretched along axis 2, even and odd. The direct sum's coefficients, for a
    # real field the packed half of a box widened on an even stretched axis, are evaluated by the FFT over
    # axis 1 and a type-2 NUFFT along axis 2, reused and one-shot, against the direct sum's own synthesis.
    Random.seed!(31)
    L = 2π
    gpu = CB.GPUBackend(KA.CPU())
    for nu in (FTB.NonuniformFFTsBackend(), FTB.FINUFFTBackend()), (N1, N2) in ((16, 16), (16, 15), (15, 16), (15, 15)),
            exec in (HP_SER, gpu), iflag in (1, -1)
        g = hp_grid(L, N1, N2)
        ms = (N1, N2)
        for (T, batch) in ((Float64, ()), (Float64, (2,)), (ComplexF64, ()))
            R = T <: Real
            c, _ = FFS.calculate_spectrum(g, randn(T, N1, N2, batch...), ms; transform = HP_DS,
                execution = HP_SER, iflag)
            ref = FFS.synthesize(T, g, c, ms; transform = HP_DS, execution = HP_SER, iflag)
            p = FFS.plan_synthesis(g, T, ms; transform = HP_FFT, execution = exec, nufft = nu, batch, iflag,
                eps = 1e-13)
            out = FFS.allocate_field(p)
            for _ in 1:2
                FFS.synthesize!(out, p, c)
                Test.@test hp_rel(out, ref) < 1e-10
            end
            FFS.close!(p)
            once = FFS.synthesize(T, g, c, ms; transform = HP_FFT, execution = exec, nufft = nu, iflag,
                eps = 1e-13)
            Test.@test hp_rel(once, ref) < 1e-10
        end
    end
end

# `a` as a `SubArray`, which a pass takes by its device method.
hp_wrapped(a) = view(a, ntuple(_ -> Colon(), ndims(a))...)

# Each pass of a plan built for a device, run on its working arrays by the host method and by the device
# method, which stages every line as a column.
function hp_staged_passes(run!, p)
    A = randn(eltype(first(p.work)), size(first(p.work)))
    for (ap, d, W) in zip(p.axes, p.sdims, Base.tail(p.work))
        host = run!(similar(W), A, d, ap)
        staged = similar(W)
        run!(hp_wrapped(staged), hp_wrapped(A), d, ap)
        Test.@test hp_rel(staged, host) < 1e-14
        A = host
    end
end

Test.@testset "Line passes on a device's staged route equal the host route" begin
    # `GPUBackend(KA.CPU())` allocates `Array`s, which take the host route; views of them take the device
    # route, so both run here against each other: real and complex fields, both libraries, even and odd
    # stretched axes, the forward and the synthesis of the hybrid and of the separable plans.
    Random.seed!(37)
    L = 2π
    gpu = CB.GPUBackend(KA.CPU())
    P3 = (FG.Grids.Periodic(), FG.Grids.Periodic(), FG.Grids.Periodic())
    cart = FG.Geometry.CartesianGeometry{Float64}()
    for nu in (FTB.NonuniformFFTsBackend(), FTB.FINUFFTBackend()), N2 in (16, 15), T in (Float64, ComplexF64)
        gh = FG.Grids.StructuredGrid(cart, hp_uni(L, 8), hp_str(L, N2), hp_str(L, 12); topology = P3,
            period = (L, L, L))
        ms = (8, N2, 12)
        # Three lines per execution: one library transform count for every pass, and a last execution that
        # carries fewer lines than the plan's count.
        kw = (; transform = HP_FFT, execution = gpu, nufft = nu, batch = (2,), eps = 1e-13, batch_chunk = 3)
        hp_staged_passes(FFS._axis_run!, FFS.plan_spectrum(gh, T, ms; kw...))
        hp_staged_passes(FFS._axis_synth!, FFS.plan_synthesis(gh, T, ms; kw...))
        gs = FG.Grids.StructuredGrid(cart, hp_str(L, 10), hp_str(L, N2), hp_str(L, 12); topology = P3,
            period = (L, L, L))
        skw = (; transform = nu, execution = gpu, batch = (2,), eps = 1e-13, batch_chunk = 3)
        sep = FFS.plan_spectrum(gs, T, (10, N2, 12); skw...)
        hp_staged_passes(FFS._axis_run!, (; sep.work, sep.axes, sdims = (1, 2, 3)))
        hp_staged_passes(FFS._axis_synth!, FFS.plan_synthesis(gs, T, (10, N2, 12); skw...))
    end
end
