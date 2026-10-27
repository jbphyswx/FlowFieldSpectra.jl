# A real field's spherical coefficients are real: the basis is the real spherical harmonics, so
# `Σⱼ wⱼ fⱼ Y_lm(xⱼ)` of a real field is real, and it equals the real part of the same transform run on the
# complexified field.

using Test: Test
using FlowFieldSpectra: FlowFieldSpectra as FFS
using FlowGeometries: FlowGeometries as FG
using SpectralBackends: SpectralBackends as SB
using ComputationalBackends: ComputationalBackends as CB
using FastSphericalHarmonics: FastSphericalHarmonics as FSH
using NUFSHT: NUFSHT
using FINUFFT: FINUFFT
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
    for (name, g) in rs_grids(lmax)
        f = randn(size(g)...)
        c, _ = FFS.calculate_spectrum(g, f, ms; transform = RS_DS, execution = RS_SER)
        cc, _ = FFS.calculate_spectrum(g, ComplexF64.(f), ms; transform = RS_DS, execution = RS_SER)
        Test.@test eltype(c) === Float64
        Test.@test rs_rel(c, real.(cc)) < 1e-13
        Test.@test maximum(abs, imag.(cc)) < 1e-12
    end
end

Test.@testset "The library spherical backends agree on the real layout" begin
    Random.seed!(8)
    lmax = 4
    ms = (lmax + 1, 2 * lmax + 1)
    sph = FG.Geometry.SphericalGeometry(1.0)

    gcc = FG.Connectivity.structured_grid(Float64, FG.SphericalSampling.Fejer1Sampling(), lmax + 1)
    cf, _ = FFS.calculate_spectrum(gcc, randn(size(gcc)...), ms; transform = SB.FSHTSpectralBackend(),
        execution = RS_SER)
    Test.@test eltype(cf) === Float64

    _, gsc = rs_grids(lmax)[3]
    cn, _ = FFS.calculate_spectrum(gsc, randn(size(gsc)...), ms; transform = SB.NUFSHTSpectralBackend())
    Test.@test eltype(cn) === Float64

    ggl = FG.Connectivity.structured_grid(Float64, FG.SphericalSampling.GaussLegendreSampling(), lmax + 1)
    cg, _ = FFS.calculate_spectrum(ggl, randn(size(ggl)...), ms; transform = SB.FSHTSpectralBackend(),
        execution = CB.GPUBackend(KA.CPU()), sampling = FG.SphericalSampling.GaussLegendreSampling())
    Test.@test eltype(cg) === Float64
end

Test.@testset "A genuinely complex spherical field" begin
    # The reference is linearity, `C(f) = C(Re f) + i·C(Im f)`, built from two real transforms.
    Random.seed!(11)
    lmax = 4
    ms = (lmax + 1, 2 * lmax + 1)
    ds(g, f) = FFS.calculate_spectrum(g, f, ms; transform = RS_DS, execution = RS_SER)[1]
    for (name, g) in rs_grids(lmax)
        f = randn(ComplexF64, size(g)...)
        Test.@test rs_rel(ds(g, f), ds(g, real.(f)) .+ im .* ds(g, imag.(f))) < 1e-13
    end

    _, gsc = rs_grids(lmax)[3]
    fsc = randn(ComplexF64, size(gsc)...)
    cd = ds(gsc, fsc)
    for exec in (RS_SER, CB.GPUBackend(KA.CPU()))
        cn, _ = FFS.calculate_spectrum(gsc, fsc, ms; transform = SB.NUFSHTSpectralBackend(), execution = exec)
        Test.@test rs_rel(cn, cd) < 1e-8
    end
end
