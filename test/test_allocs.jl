# Every steady-state in-place / prebuilt-plan path allocates exactly zero, on a 16² and a 32² grid, so no
# count grows with the grid. Each probe goes through a function barrier and is warmed up first. (Grid
# constructors + CB/SB aliases come from runtests.jl.)

using Test: Test
using FlowFieldSpectra: FlowFieldSpectra as FFS
using FlowTransformBindings: FlowTransformBindings as FTB
using FFTW: FFTW
using FINUFFT: FINUFFT

_a_iso!(Ek, kb, ks, c, nb) = @allocated FFS.isotropic_spectrum!(Ek, kb, ks, c; num_bins = nb)
_a_sph!(El, C, l) = @allocated FFS.spherical_energy_spectrum!(El, C; lmax = l)
_a_tr!(Er, ks, c, dims) = @allocated FFS.transect_spectrum!(Er, ks, c, dims)
_a_planexec(out, plan, f) = @allocated FFS.calculate_spectrum!(out, plan, f)
_a_synth!(out, plan, c) = @allocated FFS.synthesize!(out, plan, c)
_a_div!(o, ks, c) = @allocated FFS.spectral_divergence!(o, ks, c)
_a_vort!(o, ks, c) = @allocated FFS.spectral_vorticity!(o, ks, c)
_a_welch!(Ek, kb, ks, c, nb) = @allocated FFS.welch_power_spectrum!(Ek, kb, ks, c; num_bins = nb)
_a_coh!(g2, ph, kb, ks, cf, cg, nb) = @allocated FFS.coherence_spectrum!(g2, ph, kb, ks, cf, cg; num_bins = nb)
_a_lomb!(P, t, y, freqs) = @allocated FFS.lomb_scargle!(P, t, y, freqs)

function _zero_profile(N::Int)
    L = 2π
    ms = (N, N)
    xs = ucg_axis(Float64, L, N); ys = ucg_axis(Float64, L, N)
    ug = ucg((L, L), ms)
    u = [cos(2x) + sin(3y) for x in xs, y in ys]
    xv = vec([x for x in xs, y in ys])
    yv = vec([y for x in xs, y in ys])
    sg = scg((xv, yv), (L, L))
    fscat = vec(u)
    c, ks = FFS.calculate_spectrum(ug, u, ms; transform = SB.DirectSumSpectralBackend())

    nb = 8
    Ek = zeros(Float64, nb)
    kb = zeros(Float64, nb)
    _a_iso!(Ek, kb, ks, c, nb)
    iso = _a_iso!(Ek, kb, ks, c, nb)

    lmax = 8
    Csph = zeros(ComplexF64, lmax + 1, 2lmax + 1)
    Csph[FFS.sph_mode_index(3, 1)] = 1.0
    El = zeros(Float64, lmax + 1)
    _a_sph!(El, Csph, lmax)
    sph = _a_sph!(El, Csph, lmax)

    Er = zeros(Float64, N)                                   # transect out dim 1 → kept dim 2 length N
    _a_tr!(Er, ks, c, (1,))
    tr = _a_tr!(Er, ks, c, (1,))

    # Every plan names SerialBackend: the zero-allocation contract is the serial path's.
    pms = FTB.packed_size(ms, Val(true))                      # real fields ⇒ packed half on axis 1

    fplan = FFS.plan_spectrum(ug, Float64, ms; transform = SB.FFTSpectralBackend(), execution = CB.SerialBackend())
    cf = zeros(ComplexF64, pms...)
    _a_planexec(cf, fplan, u)
    fftexec = _a_planexec(cf, fplan, u)

    # A point set and a stretched grid widen the coefficient box, so their coefficients take each plan's
    # own shape.
    nplan = FFS.plan_spectrum(sg, Float64, ms; transform = FTB.FINUFFTBackend(), execution = CB.SerialBackend())
    cnu = FFS.allocate_coefficients(nplan)
    _a_planexec(cnu, nplan, fscat)
    nuexec = _a_planexec(cnu, nplan, fscat)

    # The direct sum through its reusable plan, on a scattered cloud and on a stretched tensor grid: both
    # hold their per-axis matrices and working arrays, so a repeated execution allocates nothing.
    dsplan = FFS.plan_spectrum(sg, Float64, ms; transform = SB.DirectSumSpectralBackend(), execution = CB.SerialBackend())
    cds = FFS.allocate_coefficients(dsplan)
    _a_planexec(cds, dsplan, fscat)
    dsum = _a_planexec(cds, dsplan, fscat)

    nax = collect(xs) .+ (0.03 * L / N) .* sinpi.(2 .* (0:(N - 1)) ./ N)
    ng = nucg((nax, nax), (L, L))
    dtplan = FFS.plan_spectrum(ng, Float64, ms; transform = SB.DirectSumSpectralBackend(), execution = CB.SerialBackend())
    cdt = FFS.allocate_coefficients(dtplan)
    _a_planexec(cdt, dtplan, u)
    dsum_tensor = _a_planexec(cdt, dtplan, u)

    # Spectral operators / averaging in-place variants (`c` is a (pms…) scalar field ⇒ D = 2).
    cvec = cat(c, c; dims = 3)                               # 2-component vector field (pms…, 2)
    dout = Array{ComplexF64}(undef, pms..., 1)
    _a_div!(dout, ks, cvec)
    divg = _a_div!(dout, ks, cvec)
    wout = Array{ComplexF64}(undef, pms..., 1)
    _a_vort!(wout, ks, cvec)
    vort = _a_vort!(wout, ks, cvec)

    Ekw = zeros(Float64, nb); kbw = zeros(Float64, nb)
    _a_welch!(Ekw, kbw, ks, c, nb)
    welch = _a_welch!(Ekw, kbw, ks, c, nb)

    g2 = zeros(Float64, nb); ph = zeros(Float64, nb); kbc = zeros(Float64, nb)
    _a_coh!(g2, ph, kbc, ks, c, c, nb)
    coh = _a_coh!(g2, ph, kbc, ks, c, c, nb)                 # O(num_bins) scratch, grid-independent

    tl = collect(range(0.0, 1.0; length = 4N)); yl = sin.(3 .* tl); fl = collect(range(0.5, 4.0; length = 8))
    Pl = zeros(Float64, length(fl))
    _a_lomb!(Pl, tl, yl, fl)
    lomb = _a_lomb!(Pl, tl, yl, fl)

    # The inverse's counterpart: a synthesis plan holds its backward transform and every buffer, so a
    # reused inversion writes into the caller's array and allocates nothing.
    sfft = FFS.plan_synthesis(ug, Float64, ms; transform = SB.FFTSpectralBackend(), execution = CB.SerialBackend())
    offt = FFS.allocate_field(sfft)
    _a_synth!(offt, sfft, cf)
    synth_fft = _a_synth!(offt, sfft, cf)

    cds2, _ = FFS.calculate_spectrum(ng, u, ms; transform = SB.DirectSumSpectralBackend(), execution = CB.SerialBackend())
    sds = FFS.plan_synthesis(ng, Float64, ms; transform = SB.DirectSumSpectralBackend(), execution = CB.SerialBackend())
    ods = FFS.allocate_field(sds)
    _a_synth!(ods, sds, cds2)
    synth_dsum = _a_synth!(ods, sds, cds2)
    cdc = randn(ComplexF64, ms...)
    sdc = FFS.plan_synthesis(ng, ComplexF64, ms; transform = SB.DirectSumSpectralBackend(), execution = CB.SerialBackend())
    odc = FFS.allocate_field(sdc)
    _a_synth!(odc, sdc, cdc)
    synth_dsum_complex = _a_synth!(odc, sdc, cdc)

    return (; iso, sph, tr, fftexec, nuexec, dsum, dsum_tensor, synth_fft, synth_dsum, synth_dsum_complex,
        divg, vort, welch, lomb, coh)
end

Test.@testset "Allocations" begin
    p16 = _zero_profile(16)
    p32 = _zero_profile(32)
    Test.@testset "in-place / prebuilt-plan paths allocate zero (grid-independent)" begin
        for k in (:iso, :sph, :tr, :fftexec, :nuexec, :dsum, :dsum_tensor, :synth_fft, :synth_dsum,
                  :synth_dsum_complex, :divg, :vort, :welch, :lomb)
            Test.@test getfield(p16, k) == 0
            Test.@test getfield(p32, k) == 0
        end
    end
    Test.@testset "coherence_spectrum! uses only O(num_bins) scratch (grid-independent)" begin
        Test.@test p16.coh == p32.coh                        # constant across grid ⇒ no per-element temp
        Test.@test p16.coh <= 512                            # a few bins' worth
    end
end
