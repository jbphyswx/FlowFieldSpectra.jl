# Allocation guarantees for the real kernels, measured (not guessed) at two grid sizes: every
# steady-state in-place / prebuilt-plan path allocates EXACTLY zero and is grid-independent (a
# grid-dependent count would betray a per-element temporary). Each probe goes through a function
# barrier and is warmed up before measuring.

using Test: Test
using FlowFieldSpectra: FlowFieldSpectra as FFS
using FFTW: FFTW
using FINUFFT: FINUFFT
using OhMyThreads: OhMyThreads

_a_iso!(Ek, kb, ks, c, nb) = @allocated FFS.isotropic_spectrum!(Ek, kb, ks, c; num_bins = nb)
_a_sph!(El, C, l) = @allocated FFS.spherical_energy_spectrum!(El, C; lmax = l)
_a_tr!(Er, ks, c, dims) = @allocated FFS.transect_spectrum!(Er, ks, c, dims)
_a_planexec(out, plan, f) = @allocated FFS.calculate_spectrum!(out, plan, f)
_a_dsum!(out, g, f, ms, exec) =
    @allocated FFS.calculate_spectrum!(out, g, f, ms; transform = FFS.DirectSumBackend(), execution = exec)

function _zero_profile(N::Int)
    L = 2π
    ms = (N, N)
    ug = FFS.UniformCartesianGrid(; domain = (L, L), n = ms)
    xs, ys = ug.axes
    u = [cos(2x) + sin(3y) for x in xs, y in ys]
    xv = vec([x for x in xs, y in ys])
    yv = vec([y for x in xs, y in ys])
    sg = FFS.ScatteredCartesianGrid((xv, yv); domain_size = (L, L))
    fscat = vec(u)
    c, ks = FFS.calculate_spectrum(ug, u, ms; transform = FFS.DirectSumBackend())

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

    fplan = FFS.plan_spectrum(ug, Float64, ms; transform = FFS.FFTBackend())
    cf = zeros(ComplexF64, ms...)
    _a_planexec(cf, fplan, u)
    fftexec = _a_planexec(cf, fplan, u)

    nplan = FFS.plan_spectrum(sg, Float64, ms; transform = FFS.NUFFTBackend())
    cnu = zeros(ComplexF64, ms...)
    _a_planexec(cnu, nplan, fscat)
    nuexec = _a_planexec(cnu, nplan, fscat)

    cds = zeros(ComplexF64, ms...)
    _a_dsum!(cds, sg, fscat, ms, FFS.SerialBackend())
    dsum = _a_dsum!(cds, sg, fscat, ms, FFS.SerialBackend())

    return (; iso, sph, tr, fftexec, nuexec, dsum)
end

Test.@testset "Allocations" begin
    Test.@testset "in-place / prebuilt-plan paths allocate zero (grid-independent)" begin
        p16 = _zero_profile(16)
        p32 = _zero_profile(32)
        for k in keys(p16)
            Test.@test getfield(p16, k) == 0
            Test.@test getfield(p32, k) == 0
        end
    end
end
