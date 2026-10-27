# Radial reductions against independent references: the full native spectrum `unpacked` rebuilds, the
# inline sum over every mode of a widened coefficient box, brute-force lattice mode counts per shell, and
# the closed-form mode count of a whole shell. (Grid constructors and aliases come from runtests.jl.)

# Native integer frequencies of a full axis of `n` modes, and `½ Σ |C|²` over the native modes of `full`
# whose `|k|` (in units of the spacings `scales`) passes `keep`.
_native_freqs(n) = [j <= (n - 1) ÷ 2 ? j : j - n for j in 0:(n - 1)]
function _native_energy_ref(full::AbstractArray{<:Complex}, scales, keep)
    D = length(scales)
    freqs = ntuple(d -> _native_freqs(size(full, d)), D)
    s = 0.0
    for I in CartesianIndices(full)
        k = sqrt(sum(d -> (scales[d] * freqs[d][I[d]])^2, 1:D))
        keep(k) && (s += 0.5 * abs2(full[I]))
    end
    return s
end

Test.@testset "Radial reductions: cutoff, estimators and default bins" begin
    L = 2π
    for D in 1:3, N in (D == 3 ? (6, 7) : (12, 13))
        Random.seed!(10D + N)
        ms = ntuple(_ -> N, D)
        g = ucg(ntuple(_ -> L, D), ms)
        f = randn(ms...)
        c, ks = FFS.calculate_spectrum(g, f, ms; transform = SB.FFTSpectralBackend())
        cc, ksc = FFS.calculate_spectrum(g, ComplexF64.(f), ms; transform = SB.FFTSpectralBackend())
        full = FTB.unpacked(c, ms)
        kmax = FFS.Reductions.kmax(Float64, ks)

        # Every mode binned: the bins hold `½⟨f²⟩` (Parseval under the `1/N` normalization).
        kb, E = FFS.isotropic_spectrum(ks, c; cutoff = false)
        Test.@test sum(E) * (kb[2] - kb[1]) ≈ 0.5 * Statistics.mean(abs2, f) rtol = 1e-12
        # The cutoff keeps the modes with `|k| ≤ k_max`, counted on the full native spectrum.
        kb, E = FFS.isotropic_spectrum(ks, c)
        Test.@test sum(E) * (kb[2] - kb[1]) ≈ _native_energy_ref(full, ntuple(_ -> 1.0, D), k -> k <= kmax + 1e-12) rtol = 1e-12
        # `PowerScaling` reports each bin's power, the density times the bin width.
        _, Ep = FFS.isotropic_spectrum(ks, c; convention = FFS.SpectralConvention(scaling = FFS.PowerScaling()))
        Test.@test Ep ≈ E .* (kb[2] - kb[1]) rtol = 1e-12
        # A real field's packed half bins as its full spectrum does, with the same default bins.
        for cut in (true, false)
            kbr, Er = FFS.isotropic_spectrum(ks, c; cutoff = cut)
            kbc, Ec = FFS.isotropic_spectrum(ksc, cc; cutoff = cut)
            Test.@test kbr == kbc
            Test.@test Er ≈ Ec rtol = 1e-12
        end
    end

    # Uniform along axis 1 at its own length and stretched along axis 2, whose `±N₂/2` are two modes that
    # the coefficient box holds. With no radial cutoff the isotropic spectrum sums the whole box, and a
    # transect keeping axis 1 sums each `|k₁|` over both its signs, both held to the inline sum
    # `C(k) = Σ w f e^{-ik·(x - x₀)} / Σw` over every mode of the box.
    Random.seed!(3)
    n1, n2 = 12, 10
    xax = ucg_axis(Float64, L, n1)
    yax = sort(rand(n2)) .* L
    gn = FG.Grids.StructuredGrid(_cg(Float64), xax, yax; topology = (FG.Grids.Periodic(), FG.Grids.Periodic()), period = (L, L))
    ms = (n1, 8)
    fn = randn(n1, n2)
    for (t, kw, tol) in ((SB.DirectSumSpectralBackend(), (;), 1e-12), (FTB.FINUFFTBackend(), (; eps = 1e-13), 1e-10))
        cn, ksn = FFS.calculate_spectrum(gn, fn, ms; transform = t, kw...)
        w = collect(FG.Grids.measure(gn))
        k1s = _native_freqs(n1)                    # axis 1 at its own length: `-6 … 5`
        k2s = -(ms[2] ÷ 2):(ms[2] ÷ 2)            # axis 2 widened: `-4 … 4`
        Cref = [sum(w[I] * fn[I] * cis(-(k1 * (xax[I[1]] - xax[1]) + k2 * (yax[I[2]] - yax[1])))
                    for I in CartesianIndices(fn)) / sum(w) for k1 in k1s, k2 in k2s]
        kb, E = FFS.isotropic_spectrum(ksn, cn; cutoff = false)
        Test.@test sum(E) * (kb[2] - kb[1]) ≈ 0.5 * sum(abs2, Cref) rtol = tol
        _, Et = FFS.transect_spectrum(ksn, cn, (2,))
        Eref = [0.5 * sum(abs2(Cref[i, j]) for i in eachindex(k1s), j in eachindex(k2s) if abs(k1s[i]) == m)
                for m in 0:(n1 ÷ 2)]
        Test.@test vec(Et) ≈ Eref rtol = tol
    end

    # Unit power on every mode: a shell sum counts the lattice's modes in each bin; a mode average reports
    # a whole shell's count `π(k₊² − k₋²)/(Δk₁Δk₂)`, and a bin holding no mode is NaN.
    ks = (FTB.RFFTAxis(16, 1.0), FTB.FFTAxis(16, 1.0))
    c = ones(ComplexF64, 9, 16)
    freqs = [(i, j) for i in _native_freqs(16), j in _native_freqs(16)]
    for cut in (true, false)
        kb, Es = FFS.isotropic_spectrum(ks, c; cutoff = cut, convention = FFS.SpectralConvention(scaling = FFS.PowerScaling()))
        dk = kb[2] - kb[1]
        top = cut ? FFS.Reductions.kmax(Float64, ks) : Inf
        count = zeros(length(kb))
        for (i, j) in freqs
            k = hypot(i, j)
            k <= top + 1e-12 && (count[clamp(floor(Int, k / dk) + 1, 1, length(kb))] += 1)
        end
        Test.@test Es ≈ 0.5 .* count rtol = 1e-12
        _, Em = FFS.isotropic_spectrum(ks, c; cutoff = cut,
                                       convention = FFS.SpectralConvention(scaling = FFS.PowerScaling(), estimator = FFS.ModeAverage()))
        whole = [0.5 * π * ((b * dk)^2 - ((b - 1) * dk)^2) for b in eachindex(kb)]
        Test.@test Em[count .> 0] ≈ whole[count .> 0] rtol = 1e-12
        Test.@test all(isnan, Em[count .== 0])
    end

    # The anisotropic spectrum's sectors, the cross spectrum of a field with itself, and Welch's mean
    # over realizations reduce as the isotropic spectrum does.
    ma = FFS.SpectralConvention(estimator = FFS.ModeAverage())
    kb, θb, Ea = FFS.anisotropic_spectrum(ks, c; num_θ_bins = 8, convention = ma)
    dk = kb[2] - kb[1]; dθ = θb[2] - θb[1]
    for ik in eachindex(kb), iθ in eachindex(θb)
        isnan(Ea[ik, iθ]) && continue
        Test.@test Ea[ik, iθ] ≈ 0.5 * (dθ / 2) * ((ik * dk)^2 - ((ik - 1) * dk)^2) / (dk * dθ) rtol = 1e-12
    end
    # `k = 0` has no angle and spreads evenly over the first `k` bin's sectors, so `∫ E dθ` is the
    # isotropic spectrum in every bin, the first included. A constant field holds `k = 0` alone: each
    # sector of the first bin carries `½c²/(nθ·dk·dθ)` and every other bin nothing.
    xs16 = ucg_axis(Float64, L, 16)
    g16 = ucg((L, L), (16, 16))
    fa = [1.5 + cos(3x) + 0.7 * sin(2y) for x in xs16, y in xs16]
    ca, ksa = FFS.calculate_spectrum(g16, fa, (16, 16); transform = SB.FFTSpectralBackend())
    kba, θba, Eaa = FFS.anisotropic_spectrum(ksa, ca; num_k_bins = 6, num_θ_bins = 12)
    _, Eia = FFS.isotropic_spectrum(ksa, ca; num_bins = 6)
    Test.@test vec(sum(Eaa; dims = 2)) .* (θba[2] - θba[1]) ≈ Eia rtol = 1e-12
    c0, ks0 = FFS.calculate_spectrum(g16, fill(2.0, 16, 16), (16, 16); transform = SB.FFTSpectralBackend())
    kb0, θb0, E0 = FFS.anisotropic_spectrum(ks0, c0; num_k_bins = 6, num_θ_bins = 12)
    Test.@test E0[1, :] ≈ fill(0.5 * 4.0 / (12 * (kb0[2] - kb0[1]) * (θb0[2] - θb0[1])), 12) rtol = 1e-12
    Test.@test maximum(abs, E0[2:end, :]) < 1e-25

    Random.seed!(5)
    cr = randn(ComplexF64, 9, 16, 3)
    for conv in (FFS.SpectralConvention(), ma)
        _, Ei = FFS.isotropic_spectrum(ks, cr[:, :, 1]; convention = conv)
        _, Sx = FFS.cross_spectrum(ks, cr[:, :, 1], cr[:, :, 1]; convention = conv)
        Test.@test real.(Sx) ≈ Ei rtol = 1e-12
        for cut in (true, false)
            _, Eall = FFS.isotropic_spectrum(ks, cr; cutoff = cut, convention = conv)
            _, Ew = FFS.welch_power_spectrum(ks, cr; cutoff = cut, convention = conv)
            m = vec(Statistics.mean(Eall; dims = 2))
            Test.@test isnan.(Ew) == isnan.(m)
            Test.@test Ew[.!isnan.(m)] ≈ m[.!isnan.(m)] rtol = 1e-12
            Eip = similar(Eall); kip = zeros(size(Eall, 1))
            FFS.isotropic_spectrum!(Eip, kip, ks, cr; cutoff = cut, convention = conv)
            Test.@test isequal(Eip, Eall)
        end
    end
end

Test.@testset "Reductions on ThreadedBackend equal the serial ones" begin
    # Rows split over the tasks, each task binning into its own accumulators: the sums differ from the
    # serial ones only in the order the tasks' sums are added.
    ser = (; execution = CB.SerialBackend())
    thr = (; execution = CB.ThreadedBackend())
    Random.seed!(41)
    for (ks, c) in (((FTB.RFFTAxis(64, 1.0), FTB.FFTAxis(48, 0.7)), randn(ComplexF64, 33, 48, 3)),
                    ((FTB.FFTAxis(24, 1.0), FTB.FFTAxis(20, 1.3), FTB.FFTAxis(16, 0.9)), randn(ComplexF64, 24, 20, 16, 2)))
        D = length(ks)
        for cut in (true, false)
            kb, E = FFS.isotropic_spectrum(ks, c; cutoff = cut, ser...)
            kbt, Et = FFS.isotropic_spectrum(ks, c; cutoff = cut, thr...)
            Test.@test kbt == kb
            Test.@test Et ≈ E rtol = 1e-13
            Eip, kip = similar(E), similar(kb)
            FFS.isotropic_spectrum!(Eip, kip, ks, c; cutoff = cut, thr...)
            Test.@test Eip ≈ E rtol = 1e-13
            Test.@test last(FFS.welch_power_spectrum(ks, c; cutoff = cut, thr...)) ≈
                       last(FFS.welch_power_spectrum(ks, c; cutoff = cut, ser...)) rtol = 1e-13
        end
        Test.@test last(FFS.cross_spectrum(ks, c, reverse(c; dims = D + 1); thr...)) ≈
                   last(FFS.cross_spectrum(ks, c, reverse(c; dims = D + 1); ser...)) rtol = 1e-13
        ch = FFS.coherence_spectrum(ks, c, reverse(c; dims = D + 1); thr...)
        cs = FFS.coherence_spectrum(ks, c, reverse(c; dims = D + 1); ser...)
        Test.@test all(map((a, b) -> isapprox(a, b; rtol = 1e-12), ch, cs))
        c1 = c[ntuple(_ -> Colon(), D)..., 1]
        for dims in ((1,), (D,), ntuple(identity, D)), cc in (c, c1)
            Test.@test last(FFS.transect_spectrum(ks, cc, dims; thr...)) ≈
                       last(FFS.transect_spectrum(ks, cc, dims; ser...)) rtol = 1e-13
        end
        if D == 2
            Test.@test last(FFS.anisotropic_spectrum(ks, c; num_θ_bins = 8, thr...)) ≈
                       last(FFS.anisotropic_spectrum(ks, c; num_θ_bins = 8, ser...)) rtol = 1e-13
        end
    end
    lmax = 40
    C = randn(lmax + 1, 2lmax + 1, 3)
    Test.@test last(FFS.spherical_energy_spectrum(C; thr...)) ≈ last(FFS.spherical_energy_spectrum(C; ser...)) rtol = 1e-13
end
