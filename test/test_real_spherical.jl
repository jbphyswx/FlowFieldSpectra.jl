# A real field's spherical coefficients are REAL.
#
# The basis is the real spherical harmonics (`DirectSum._real_sph`, the FastSphericalHarmonics
# convention), so `Σⱼ wⱼ fⱼ Y_lm(xⱼ)` of a real field is real and a real array holds it exactly. This
# mirrors the Cartesian rule, where a real field's coefficients are the packed Hermitian half and a
# complex field's the full native cube — in both cases the field's realness picks the layout.
#
# The value contract is that the real result equals the real part of the same transform run on the
# complexified field, which is the reference these testsets use.

using Test: Test
using FlowFieldSpectra: FlowFieldSpectra as FFS
using FlowGeometries: FlowGeometries as FG
using SpectralBackends: SpectralBackends as SB
using ComputationalBackends: ComputationalBackends as CB
using FastSphericalHarmonics: FastSphericalHarmonics as FSH
using NUFSHT: NUFSHT
using FINUFFT: FINUFFT
using OhMyThreads: OhMyThreads
using KernelAbstractions: KernelAbstractions as KA
using Random: Random

const RS_DS = SB.DirectSumSpectralBackend()
const RS_SER = CB.SerialBackend()
rs_rel(a, b) = maximum(abs, a .- b) / max(maximum(abs, b), eps())

function rs_grids(lmax)
    sph = FG.Geometry.SphericalGeometry(1.0)
    N = 200
    ga = π * (3 - sqrt(5))
    λ = [mod(ga * i, 2π) for i in 1:N]
    φ = asin.([-1 + 2 * (i - 0.5) / N for i in 1:N])
    return (
        ("tensor", FG.Connectivity.structured_grid(Float64, FG.SphericalSampling.GaussLegendreSampling(), lmax + 1)),
        ("ring", FG.Grids.HEALPixGrid(sph, 2)),
        ("cloud", FG.Grids.UnstructuredGrid(sph, (λ, φ), fill(4π / N, N))),
    )
end

Test.@testset "Real field gives real spherical coefficients" begin
    Random.seed!(7)
    lmax = 4
    ms = (lmax + 1, 2 * lmax + 1)
    for (name, g) in rs_grids(lmax)                     # tensor, ring, and point-cloud layouts
        f = randn(size(g)...)
        for exec in (RS_SER, CB.ThreadedBackend(), CB.GPUBackend(KA.CPU()))
            c, _ = FFS.calculate_spectrum(g, f, ms; transform = RS_DS, execution = exec)
            Test.@test eltype(c) === Float64
            Test.@test size(c) == ms
            # The same transform on the complexified field, whose real part this must equal.
            cc, _ = FFS.calculate_spectrum(g, ComplexF64.(f), ms; transform = RS_DS, execution = exec)
            Test.@test eltype(cc) === ComplexF64
            Test.@test rs_rel(c, real.(cc)) < 1e-13
            # The complex route's imaginary part is zero, so the real layout holds the whole result.
            Test.@test maximum(abs, imag.(cc)) < 1e-12
        end
        # Half the bytes of the complex array.
        c, _ = FFS.calculate_spectrum(g, f, ms; transform = RS_DS, execution = RS_SER)
        cc, _ = FFS.calculate_spectrum(g, ComplexF64.(f), ms; transform = RS_DS, execution = RS_SER)
        Test.@test sizeof(c) * 2 == sizeof(cc)
    end
end

Test.@testset "The library spherical backends agree on the real layout" begin
    Random.seed!(8)
    lmax = 4
    ms = (lmax + 1, 2 * lmax + 1)
    sph = FG.Geometry.SphericalGeometry(1.0)

    # FastSphericalHarmonics, on its own Clenshaw–Curtis grid.
    Θ, Φ = FSH.sph_points(lmax + 1)
    gcc = FG.Grids.StructuredGrid(sph, collect(Φ), Float64(π) / 2 .- collect(Θ);
        periodic = (true, false), period = (2π, 0.0))
    fcc = randn(size(gcc)...)
    cf, _ = FFS.calculate_spectrum(gcc, fcc, ms; transform = SB.FSHTSpectralBackend(), execution = RS_SER)
    Test.@test eltype(cf) === Float64

    # NUFSHT on a point cloud.
    N = 200
    ga = π * (3 - sqrt(5))
    λ = [mod(ga * i, 2π) for i in 1:N]
    φ = asin.([-1 + 2 * (i - 0.5) / N for i in 1:N])
    gsc = FG.Grids.UnstructuredGrid(sph, (λ, φ), fill(4π / N, N))
    fsc = randn(N)
    cn, _ = FFS.calculate_spectrum(gsc, fsc, ms; transform = SB.NUFSHTSpectralBackend())
    Test.@test eltype(cn) === Float64
    cd, _ = FFS.calculate_spectrum(gsc, fsc, ms; transform = RS_DS, execution = RS_SER)
    Test.@test rs_rel(cn, cd) < 1e-8

    # The device-generic SHT stages a real field, so its coefficients are real too.
    ggl = FG.Connectivity.structured_grid(Float64, FG.SphericalSampling.GaussLegendreSampling(), lmax + 1)
    cg, _ = FFS.calculate_spectrum(ggl, randn(size(ggl)...), ms; transform = SB.FSHTSpectralBackend(),
        execution = CB.GPUBackend(KA.CPU()), sampling = FG.SphericalSampling.GaussLegendreSampling())
    Test.@test eltype(cg) === Float64
end

Test.@testset "Real coefficients flow through the reductions and the inverse" begin
    Random.seed!(9)
    lmax = 5
    ms = (lmax + 1, 2 * lmax + 1)
    g = FG.Connectivity.structured_grid(Float64, FG.SphericalSampling.GaussLegendreSampling(), lmax + 1)
    f = randn(size(g)...)
    c, _ = FFS.calculate_spectrum(g, f, ms; transform = RS_DS, execution = RS_SER)
    cc, _ = FFS.calculate_spectrum(g, ComplexF64.(f), ms; transform = RS_DS, execution = RS_SER)

    # The degree spectrum reads either layout and returns the real type.
    deg, E = FFS.spherical_energy_spectrum(c)
    degc, Ec = FFS.spherical_energy_spectrum(cc)
    Test.@test eltype(E) === Float64
    Test.@test deg == degc
    Test.@test rs_rel(E, Ec) < 1e-13
    Ei = zeros(Float64, lmax + 1)
    FFS.spherical_energy_spectrum!(Ei, c)
    Test.@test rs_rel(Ei, E) < 1e-13

    # Synthesis from real coefficients writes a real field with no complex intermediate, and matches the
    # complex route's real part.
    out = FFS.synthesize(g, c, ms; transform = RS_DS, execution = RS_SER)
    Test.@test eltype(out) === Float64
    Test.@test size(out) == size(g)
    outc = FFS.synthesize(g, cc, ms; transform = RS_DS, execution = RS_SER)
    Test.@test rs_rel(out, outc) < 1e-13
    # `real_output = false` on real coefficients still returns complex.
    outx = FFS.synthesize(g, c, ms; transform = RS_DS, execution = RS_SER, real_output = false)
    Test.@test eltype(outx) === ComplexF64
    Test.@test rs_rel(real.(outx), out) < 1e-13
end

Test.@testset "A genuinely complex spherical field" begin
    # The reference is linearity: `C(f) = C(Re f) + i·C(Im f)`, built from two REAL transforms on the same
    # path. Every testset above complexifies a real field, whose imaginary part is zero — so it cannot
    # tell apart a path that drops the imaginary part from one that carries it, and the reference has to
    # be a field with a nonzero one.
    Random.seed!(11)
    lmax = 4
    ms = (lmax + 1, 2 * lmax + 1)
    sph = FG.Geometry.SphericalGeometry(1.0)

    # `C(f)` on `path`, against the two-real-transform reference.
    function rs_linear(path, g, f, ms)
        c = path(g, f, ms)
        cr = path(g, real.(f), ms)
        ci = path(g, imag.(f), ms)
        return c, cr .+ im .* ci
    end

    # The direct sum, one grid per layout × each execution backend.
    for (name, g) in rs_grids(lmax)
        f = randn(ComplexF64, size(g)...)
        Test.@test maximum(abs, imag.(f)) > 0.1          # the reference is only meaningful for a true complex field
        for exec in (RS_SER, CB.ThreadedBackend(), CB.GPUBackend(KA.CPU()))
            c, ref = rs_linear((gg, ff, mm) ->
                FFS.calculate_spectrum(gg, ff, mm; transform = RS_DS, execution = exec)[1], g, f, ms)
            Test.@test eltype(c) === ComplexF64
            Test.@test rs_rel(c, ref) < 1e-13
        end
        # And the reusable plan, whose buffers are sized from the field's type.
        p = FFS.plan_spectrum(g, ComplexF64, ms; transform = RS_DS, execution = RS_SER)
        cp = zeros(ComplexF64, ms...)
        FFS.calculate_spectrum!(cp, p, f)
        c1, _ = FFS.calculate_spectrum(g, f, ms; transform = RS_DS, execution = RS_SER)
        Test.@test rs_rel(cp, c1) < 1e-13
    end

    # FastSphericalHarmonics on its own grid, and the device-generic SHT on a Gauss–Legendre grid: both
    # run a real analysis, so both take the field one component at a time.
    gcc = FG.Connectivity.structured_grid(Float64, FG.SphericalSampling.ClenshawCurtisSampling(), lmax + 1)
    fcc = randn(ComplexF64, size(gcc)...)
    cf, reff = rs_linear((gg, ff, mm) ->
        FFS.calculate_spectrum(gg, ff, mm; transform = SB.FSHTSpectralBackend(), execution = RS_SER)[1],
        gcc, fcc, ms)
    Test.@test eltype(cf) === ComplexF64
    Test.@test rs_rel(cf, reff) < 1e-12

    ggl = FG.Connectivity.structured_grid(Float64, FG.SphericalSampling.GaussLegendreSampling(), lmax + 1)
    fgl = randn(ComplexF64, size(ggl)...)
    cg, refg = rs_linear((gg, ff, mm) ->
        FFS.calculate_spectrum(gg, ff, mm; transform = SB.FSHTSpectralBackend(),
            execution = CB.GPUBackend(KA.CPU()),
            sampling = FG.SphericalSampling.GaussLegendreSampling())[1], ggl, fgl, ms)
    Test.@test eltype(cg) === ComplexF64
    Test.@test rs_rel(cg, refg) < 1e-12
    # The device SHT and the direct sum compute the same projection on this grid.
    cds, _ = FFS.calculate_spectrum(ggl, fgl, ms; transform = RS_DS, execution = RS_SER)
    Test.@test rs_rel(cg, cds) < 1e-10

    # NUFSHT on a point cloud, host and device, plus its reusable plans.
    N = 200
    ga = π * (3 - sqrt(5))
    λ = [mod(ga * i, 2π) for i in 1:N]
    φ = asin.([-1 + 2 * (i - 0.5) / N for i in 1:N])
    gsc = FG.Grids.UnstructuredGrid(sph, (λ, φ), fill(4π / N, N))
    fsc = randn(ComplexF64, N)
    for exec in (RS_SER, CB.GPUBackend(KA.CPU()))
        cn, refn = rs_linear((gg, ff, mm) ->
            FFS.calculate_spectrum(gg, ff, mm; transform = SB.NUFSHTSpectralBackend(), execution = exec)[1],
            gsc, fsc, ms)
        Test.@test eltype(cn) === ComplexF64
        Test.@test rs_rel(cn, refn) < 1e-9
        # NUFSHT computes the same projection the direct sum does.
        cd, _ = FFS.calculate_spectrum(gsc, fsc, ms; transform = RS_DS, execution = RS_SER)
        Test.@test rs_rel(cn, cd) < 1e-8
        pn = FFS.plan_spectrum(gsc, ComplexF64, ms; transform = SB.NUFSHTSpectralBackend(), execution = exec)
        cbuf = zeros(ComplexF64, ms...)
        FFS.calculate_spectrum!(cbuf, pn, fsc)
        Test.@test rs_rel(cbuf, cd) < 1e-8
    end
end

Test.@testset "A caller's own buffer decides its own type" begin
    # The in-place and plan entries take whatever buffer the caller brings: a real one for the natural
    # layout, or a complex one, which holds the same values.
    Random.seed!(10)
    lmax = 4
    ms = (lmax + 1, 2 * lmax + 1)
    g = FG.Connectivity.structured_grid(Float64, FG.SphericalSampling.GaussLegendreSampling(), lmax + 1)
    f = randn(size(g)...)
    cref, _ = FFS.calculate_spectrum(g, f, ms; transform = RS_DS, execution = RS_SER)

    br = zeros(Float64, ms...)
    FFS.calculate_spectrum!(br, g, f, ms; transform = RS_DS, execution = RS_SER)
    Test.@test rs_rel(br, cref) < 1e-13

    bc = zeros(ComplexF64, ms...)
    FFS.calculate_spectrum!(bc, g, f, ms; transform = RS_DS, execution = RS_SER)
    Test.@test rs_rel(real.(bc), cref) < 1e-13
    Test.@test maximum(abs, imag.(bc)) < 1e-13

    # The reusable plan likewise.
    p = FFS.plan_spectrum(g, Float64, ms; transform = RS_DS, execution = RS_SER)
    pr = zeros(Float64, ms...)
    FFS.calculate_spectrum!(pr, p, f)
    Test.@test rs_rel(pr, cref) < 1e-13
end
