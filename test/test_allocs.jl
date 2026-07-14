# Allocation guarantees for the real FlowFieldSpectra kernels.
#
# Two things are asserted, with the numbers *measured*, not guessed:
#   1. Every steady-state in-place / prebuilt-plan path allocates EXACTLY zero — and does so
#      independent of grid size (measured at two grid sizes; a grid-dependent allocation would mean
#      a hidden per-element temporary). These are the load-bearing performance guarantees: a hot
#      loop over time steps / realizations must not feed the GC.
#   2. The functions that MUST allocate (they return freshly-sized arrays) allocate only ~their
#      output, not wasteful extra temporaries — asserted against `sizeof(output)`.
#
# Every measurement goes through a function barrier (typed args) so `@allocated` reflects the kernel
# and not closure-boxing of captured globals, and is warmed up once before it is measured.

using Test: Test
using Random: Random
using FlowFieldSpectra: FlowFieldSpectra as FFS
using FFTW: FFTW
using FINUFFT: FINUFFT
using OhMyThreads: OhMyThreads

# --- function-barrier probes (each returns the steady-state allocation of one operation) -------
_a_iso!(Ek, kb, ks, c, nb) = @allocated FFS.isotropic_spectrum!(Ek, kb, ks, c; num_bins = nb)
_a_sph!(El, C, l) = @allocated FFS.spherical_energy_spectrum!(El, C; lmax = l)
_a_tacc!(Er, c, ms, NU, dims) = @allocated FFS.Reductions._accumulate_transect!(Er, c, ms, NU, dims)
_a_tr!(Er, ks, c, dims) = @allocated FFS.transect_spectrum!(Er, ks, c, dims)
_a_planexec(out, plan, f) = @allocated FFS.calculate_spectrum!(out, plan, (f,))
_a_dsum!(out, g, f, ms, exec) =
    @allocated FFS.calculate_spectrum!(out, g, f, ms; transform = FFS.DirectSumBackend(), execution = exec)
_a_div(ks, c) = @allocated FFS.spectral_divergence(ks, c)
_a_vort(ks, c) = @allocated FFS.spectral_vorticity(ks, c)

# Build a 2D Cartesian problem (uniform + scattered grids, 2-component field) at resolution N.
function _alloc_setup(N::Int)
    L = 2π
    ms = (N, N)
    dx = L / N
    xs = range(0.0, stop = L - dx, length = N)
    ys = range(0.0, stop = L - dx, length = N)
    xv = vec([x for x in xs, y in ys])
    yv = vec([y for x in xs, y in ys])
    u = @. cos(2xv) + sin(3yv)
    v = @. sin(xv)
    ug = FFS.UniformCartesianGrid((xv, yv); domain_size = (L, L))
    sg = FFS.ScatteredCartesianGrid((xv, yv); domain_size = (L, L))
    c, ks = FFS.calculate_spectrum(ug, (u, v), ms; transform = FFS.DirectSumBackend())
    return (; L, ms, N, u, v, ug, sg, c, ks)
end

# Measure every should-be-zero path at grid size N; returns a NamedTuple of allocation counts.
function _zero_profile(N::Int)
    s = _alloc_setup(N)
    ms, c, ks = s.ms, s.c, s.ks

    nb = 16
    Ek = zeros(Float64, nb); kb = zeros(Float64, nb)
    _a_iso!(Ek, kb, ks, c, nb)                              # warmup
    iso = _a_iso!(Ek, kb, ks, c, nb)

    lmax = 8
    Csph = zeros(ComplexF64, lmax + 1, 2lmax + 1, 1)
    Csph[FFS.sph_mode_index(3, 1), 1] = 1.0
    El = zeros(Float64, lmax + 1)
    _a_sph!(El, Csph, lmax)
    sph = _a_sph!(El, Csph, lmax)

    Ered = zeros(Float64, N)                                # kept dim (2) has length N
    _a_tacc!(Ered, c, ms, 2, (1,))
    tacc = _a_tacc!(Ered, c, ms, 2, (1,))
    _a_tr!(Ered, ks, c, (1,))
    tr = _a_tr!(Ered, ks, c, (1,))

    fplan = FFS.plan_spectrum(s.ug, Float64, ms; transform = FFS.FFTBackend(), n_transf = 1)
    cf = zeros(ComplexF64, ms..., 1)
    _a_planexec(cf, fplan, s.u)
    fftexec = _a_planexec(cf, fplan, s.u)

    nplan = FFS.plan_spectrum(s.sg, Float64, ms; transform = FFS.NUFFTBackend(), n_transf = 1)
    cnu = zeros(ComplexF64, ms..., 1)
    _a_planexec(cnu, nplan, s.u)
    nuexec = _a_planexec(cnu, nplan, s.u)

    cds = zeros(ComplexF64, ms..., 2)
    _a_dsum!(cds, s.sg, (s.u, s.v), ms, FFS.SerialBackend())
    dsum_serial = _a_dsum!(cds, s.sg, (s.u, s.v), ms, FFS.SerialBackend())

    return (; iso, sph, tacc, tr, fftexec, nuexec, dsum_serial)
end

Test.@testset "Allocations" begin
    Test.@testset "in-place / prebuilt-plan paths allocate zero (grid-independent)" begin
        # Measured at two grid sizes: exactly zero at each, and equal across sizes (a grid-dependent
        # count would betray a per-element temporary that the ! contract promises not to create).
        p16 = _zero_profile(16)
        p32 = _zero_profile(32)
        for k in keys(p16)
            Test.@test getfield(p16, k) == 0
            Test.@test getfield(p32, k) == 0
        end
    end

    Test.@testset "must-allocate ops allocate only ~their output" begin
        s = _alloc_setup(32)
        # spectral_divergence / vorticity return one (ms..., 1) coefficient array; allocation should
        # be that output plus a tiny constant, never a multiple of it.
        divout = FFS.spectral_divergence(s.ks, s.c)
        vortout = FFS.spectral_vorticity(s.ks, s.c)
        _a_div(s.ks, s.c); _a_vort(s.ks, s.c)               # warmup
        Test.@test _a_div(s.ks, s.c) <= sizeof(divout) + 512
        Test.@test _a_vort(s.ks, s.c) <= sizeof(vortout) + 512
    end

    Test.@testset "threaded direct-sum: small constant scheduling overhead only" begin
        # The threaded direct sum spawns OhMyThreads tasks (unavoidable, small, grid-independent).
        # It must not scale with the grid — the coefficient accumulation itself is allocation-free.
        s16 = _alloc_setup(16); s32 = _alloc_setup(32)
        c16 = zeros(ComplexF64, s16.ms..., 2); c32 = zeros(ComplexF64, s32.ms..., 2)
        _a_dsum!(c16, s16.sg, (s16.u, s16.v), s16.ms, FFS.ThreadedBackend())
        _a_dsum!(c32, s32.sg, (s32.u, s32.v), s32.ms, FFS.ThreadedBackend())
        a16 = _a_dsum!(c16, s16.sg, (s16.u, s16.v), s16.ms, FFS.ThreadedBackend())
        a32 = _a_dsum!(c32, s32.sg, (s32.u, s32.v), s32.ms, FFS.ThreadedBackend())
        Test.@test a16 == a32          # grid-independent (task-scheduling constant, not per-element)
        Test.@test a16 <= 4096         # small, bounded
    end
end
