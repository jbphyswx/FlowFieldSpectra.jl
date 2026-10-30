# Dispatch gates: every prebuilt plan's execution and synthesis, the in-place reductions and operators,
# and the plan builders whose result type follows their arguments' types make no runtime dispatch in this
# package's code or its extensions. A NonuniformFFTs plan's type carries its transform count, so its
# builders stay out; its executions are in. `JET` refuses pre-release Julia.
using JET: JET

const JET_MODULES = (FFS, filter(!isnothing, map(n -> Base.get_extension(FFS, n),
    (:FlowFieldSpectraAbstractFFTsExt, :FlowFieldSpectraFFTWExt, :FlowFieldSpectraKernelAbstractionsExt,
     :FlowFieldSpectraKernelAbstractionsAbstractFFTsExt, :FlowFieldSpectraNUFSHTExt,
     :FlowFieldSpectraNUFSHTKernelAbstractionsExt, :FlowFieldSpectraFastSphericalHarmonicsExt,
     :FlowFieldSpectraOhMyThreadsExt, :FlowFieldSpectraGPUArraysCoreExt)))...)

# A plan for `E` fields of batch 2 executed forward and synthesized back, both under JET.
function jet_round(g, t, ms, E, exec; kw = (;), skw = kw)
    p = FFS.plan_spectrum(t, exec, g, E, ms; batch = (2,), kw...)
    c = FFS.allocate_coefficients(p)
    f = randn(E, size(g)..., 2)
    JET.@test_opt target_modules = JET_MODULES FFS.calculate_spectrum!(c, p, f)
    q = FFS.plan_synthesis(t, exec, g, E, ms; batch = (2,), skw...)
    out = FFS.allocate_field(q)
    JET.@test_opt target_modules = JET_MODULES FFS.synthesize!(out, q, c)
    FFS.close!(p)
    FFS.close!(q)
    return nothing
end

Test.@testset "Entry points make no runtime dispatch" begin
    if isempty(VERSION.prerelease)
        Random.seed!(77)
        ser = CB.SerialBackend()
        thr = CB.ThreadedBackend()
        dev = CB.GPUBackend(KA.CPU())
        L = 2π
        N = 16
        ms = (N, N)
        str = [L * (i - 1) / N + 0.04 * L * sinpi(2 * (i - 1) / N) for i in 1:N]
        ug = ucg((L, L), ms)
        hg = FG.Grids.StructuredGrid(_cg(Float64), ucg_axis(Float64, L, N), str; topology = _periodic(ms),
            period = (L, L))
        tg = nucg((str, str), (L, L))
        sg = scg((rand(64) .* L, rand(64) .* L), (L, L))
        fin = FTB.FINUFFTBackend()
        nonu = FTB.NonuniformFFTsBackend()
        for E in (Float64, ComplexF64)
            jet_round(ug, SB.FFTSpectralBackend(), ms, E, ser)
            jet_round(ug, SB.FFTSpectralBackend(), ms, E, dev)
            jet_round(hg, SB.FFTSpectralBackend(), ms, E, ser; kw = (; nufft = fin))
            jet_round(hg, SB.FFTSpectralBackend(), ms, E, thr; kw = (; nufft = nonu))
            jet_round(tg, fin, ms, E, ser)
            jet_round(tg, nonu, ms, E, thr)
            jet_round(sg, fin, ms, E, ser)
            jet_round(sg, nonu, ms, E, ser)
            jet_round(ug, SB.DirectSumSpectralBackend(), ms, E, ser)
            jet_round(sg, SB.DirectSumSpectralBackend(), ms, E, ser)
            jet_round(tg, SB.DirectSumSpectralBackend(), ms, E, dev)
        end

        lmax = 6
        sms = (lmax + 1, 2 * lmax + 1)
        ggl = sph_gl(lmax)
        Ns = 200
        ga = π * (3 - sqrt(5))
        gss = FG.Grids.UnstructuredGrid(FG.Geometry.SphericalGeometry(1.0),
            ([mod(ga * i, 2π) for i in 1:Ns], asin.([-1 + 2 * (i - 0.5) / Ns for i in 1:Ns])), fill(4π / Ns, Ns))
        for E in (Float64, ComplexF64)
            jet_round(sph_cc(lmax), SB.FSHTSpectralBackend(), sms, E, ser)
            jet_round(ggl, SB.FSHTSpectralBackend(), sms, E, dev; kw = (; sampling = GL))
            jet_round(ggl, SB.DirectSumSpectralBackend(), sms, E, ser)
            jet_round(gss, SB.DirectSumSpectralBackend(), sms, E, dev)
            for analysis in (FFS.Quadrature(), FFS.LeastSquares()), exec in (ser, dev)
                jet_round(gss, SB.NUFSHTSpectralBackend(), sms, E, exec;
                    kw = (; nufft = fin, analysis, maxiter = 3), skw = (; nufft = fin))
            end
        end

        # Builders whose plan type follows the argument types, forward and synthesis, for a real and a
        # complex field.
        builders = ((ug, SB.FFTSpectralBackend(), ms, ser, (;)), (ug, SB.FFTSpectralBackend(), ms, dev, (;)),
            (hg, SB.FFTSpectralBackend(), ms, ser, (; nufft = fin)), (tg, fin, ms, ser, (;)), (sg, fin, ms, ser, (;)),
            (ug, SB.DirectSumSpectralBackend(), ms, ser, (;)), (tg, SB.DirectSumSpectralBackend(), ms, dev, (;)),
            (ggl, SB.DirectSumSpectralBackend(), sms, ser, (;)), (gss, SB.DirectSumSpectralBackend(), sms, dev, (;)),
            (sph_cc(lmax), SB.FSHTSpectralBackend(), sms, ser, (;)),
            (ggl, SB.FSHTSpectralBackend(), sms, dev, (; sampling = GL)),
            (gss, SB.NUFSHTSpectralBackend(), sms, ser, (; nufft = fin)))
        for (g, t, m, exec, kw) in builders, E in (Float64, ComplexF64)
            JET.@test_opt target_modules = JET_MODULES FFS.plan_spectrum(t, exec, g, E, m; batch = (2,), kw...)
            JET.@test_opt target_modules = JET_MODULES FFS.plan_synthesis(t, exec, g, E, m; batch = (2,), kw...)
        end

        # The in-place reductions, serial and threaded, and the operators on a real field's packed half, two
        # components.
        c, ks = FFS.calculate_spectrum(ug, randn(N, N, 2), ms; transform = SB.FFTSpectralBackend(), execution = ser)
        nb = 6
        for execution in (ser, thr)
            JET.@test_opt target_modules = JET_MODULES FFS.isotropic_spectrum!(zeros(nb, 2), zeros(nb), ks, c;
                num_bins = nb, execution)
            JET.@test_opt target_modules = JET_MODULES FFS.transect_spectrum!(zeros(N, 2), ks, c, (1,); execution)
            JET.@test_opt target_modules = JET_MODULES FFS.welch_power_spectrum!(zeros(nb), zeros(nb), ks, c;
                num_bins = nb, execution)
            JET.@test_opt target_modules = JET_MODULES FFS.coherence_spectrum!(zeros(nb), zeros(nb), zeros(nb), ks,
                c, c; num_bins = nb, execution)
            JET.@test_opt target_modules = JET_MODULES FFS.spherical_energy_spectrum!(zeros(lmax + 1, 2),
                randn(lmax + 1, 2 * lmax + 1, 2); execution)
        end
        o = similar(c, size(c, 1), size(c, 2), 1)
        JET.@test_opt target_modules = JET_MODULES FFS.spectral_divergence!(o, ks, c)
        JET.@test_opt target_modules = JET_MODULES FFS.spectral_vorticity!(o, ks, c)

        # A plan holding its preprocessing, built and executed on a field with an invalid point, by mask and by value:
        # a node set's detrend and window, and a uniform grid's padding.
        fsg = randn(64)
        fsg[5] = NaN
        lin = FFS.Preprocess(; detrend = FFS.LinearDetrend(), window = FFS.Tukey(0.5))
        padded = FFS.Preprocess(; detrend = FFS.LinearDetrend(), window = FFS.Hann(), pad = 2.0)
        for (g, t, f, spec) in ((sg, fin, fsg, lin), (sg, SB.DirectSumSpectralBackend(), fsg, lin),
                                (ug, SB.FFTSpectralBackend(), randn(N, N), padded))
            JET.@test_opt target_modules = JET_MODULES FFS.plan_spectrum(g, Float64, ms; transform = t, execution = ser,
                preprocess = spec)
            p = FFS.plan_spectrum(g, Float64, ms; transform = t, execution = ser, preprocess = spec)
            cp = FFS.allocate_coefficients(p)
            JET.@test_opt target_modules = JET_MODULES FFS.calculate_spectrum!(cp, p, f)
            JET.@test_opt target_modules = JET_MODULES FFS.calculate_spectrum!(cp, p, f, isfinite.(f))
            FFS.close!(p)
        end
    end
end
