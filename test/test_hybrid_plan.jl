# The hybrid FFT/NUFFT composite, one-shot and held for reuse.
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
    hp_uni(L, N1), hp_str(L, N2); periodic = (true, true), period = (L, L))

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
            ref = FFS.synthesize(g, c, ms; transform = HP_DS, execution = HP_SER, iflag, real_output = R)
            p = FFS.plan_synthesis(g, T, ms; transform = HP_FFT, execution = exec, nufft = nu, batch, iflag,
                eps = 1e-13)
            out = FFS.allocate_field(p)
            for _ in 1:2
                FFS.synthesize!(out, p, c)
                Test.@test hp_rel(out, ref) < 1e-10
            end
            FFS.close!(p)
            once = FFS.synthesize(g, c, ms; transform = HP_FFT, execution = exec, nufft = nu, iflag,
                real_output = R, eps = 1e-13)
            Test.@test hp_rel(once, ref) < 1e-10
        end
    end
end
