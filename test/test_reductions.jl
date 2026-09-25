# Radial reductions against independent references: the full native spectrum `unpacked` rebuilds (twins
# included), brute-force lattice mode counts per shell, and the closed-form mode count of a whole shell.
# (Grid constructors and aliases come from runtests.jl.)

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
        full = FFS.unpacked(c, ms)
        kmax = FFS.Packing.kmax(Float64, ks)

        # Every mode binned: the bins hold `½⟨f²⟩` (Parseval under the `1/N` normalization).
        kb, E = FFS.isotropic_spectrum(ks, c; cutoff = false)
        Test.@test sum(E) * (kb[2] - kb[1]) ≈ 0.5 * Statistics.mean(abs2, f) rtol = 1e-12
        # The cutoff keeps the modes with `|k| ≤ k_max`, counted on the full native spectrum.
        kb, E = FFS.isotropic_spectrum(ks, c)
        Test.@test sum(E) * (kb[2] - kb[1]) ≈ _native_energy_ref(full, ntuple(_ -> 1.0, D), k -> k <= kmax + 1e-12) rtol = 1e-12
        # A real field's packed half bins as its full spectrum does, with the same default bins.
        for cut in (true, false)
            kbr, Er = FFS.isotropic_spectrum(ks, c; cutoff = cut)
            kbc, Ec = FFS.isotropic_spectrum(ksc, cc; cutoff = cut)
            Test.@test kbr == kbc
            Test.@test Er ≈ Ec rtol = 1e-12
        end
    end

    # A stretched axis: the packed half's `−N₂/2` partners are its Nyquist twins, which `unpacked` reads
    # from `ks`, so every mode binned holds the energy of the full spectrum it rebuilds.
    Random.seed!(3)
    xax = sort(rand(14)) .* L; yax = sort(rand(12)) .* L
    gn = nucg((xax, yax), (L, L))
    ms = (10, 8)
    fn = [cos(2x) + 0.5sin(3y) + 0.3cos(x + 2y) for x in xax, y in yax]
    cn, ksn = FFS.calculate_spectrum(gn, fn, ms; transform = FTB.FINUFFTBackend(), eps = 1e-13)
    Test.@test FFS.Packing.axis_twin(ksn[1]) !== nothing
    kb, E = FFS.isotropic_spectrum(ksn, cn; cutoff = false)
    Test.@test sum(E) * (kb[2] - kb[1]) ≈ 0.5 * sum(abs2, FFS.unpacked(cn, ms, ksn)) rtol = 1e-12

    # Unit power on every mode: a shell sum counts the lattice's modes in each bin; a mode average reports
    # a whole shell's count `π(k₊² − k₋²)/(Δk₁Δk₂)`, and a bin holding no mode is NaN.
    ks = (FFS.Packing.RFFTAxis(1.0, 16), FFS.Packing.FFTAxis(1.0, 16))
    c = ones(ComplexF64, 9, 16)
    freqs = [(i, j) for i in _native_freqs(16), j in _native_freqs(16)]
    for cut in (true, false)
        kb, Es = FFS.isotropic_spectrum(ks, c; cutoff = cut, convention = FFS.SpectralConvention(scaling = FFS.PowerScaling()))
        dk = kb[2] - kb[1]
        top = cut ? FFS.Packing.kmax(Float64, ks) : Inf
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
    _, Enan = FFS.isotropic_spectrum(ks, c; num_bins = 400, convention = FFS.SpectralConvention(estimator = FFS.ModeAverage()))
    Test.@test any(isnan, Enan)

    # The anisotropic spectrum's sectors, the cross spectrum of a field with itself, and Welch's mean
    # over realizations reduce as the isotropic spectrum does.
    ma = FFS.SpectralConvention(estimator = FFS.ModeAverage())
    kb, θb, Ea = FFS.anisotropic_spectrum(ks, c; num_θ_bins = 8, convention = ma)
    dk = kb[2] - kb[1]; dθ = θb[2] - θb[1]
    for ik in eachindex(kb), iθ in eachindex(θb)
        isnan(Ea[ik, iθ]) && continue
        Test.@test Ea[ik, iθ] ≈ 0.5 * (dθ / 2) * ((ik * dk)^2 - ((ik - 1) * dk)^2) / (dk * dθ) rtol = 1e-12
    end
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

    # The multitaper tapers have unit mean square, so each tapered copy keeps the field's level.
    V = FFS.dpss(128, 4.0, 7)
    Test.@test V' * V ≈ 128 * LA.I(7) atol = 1e-8
end
