# Spherical-harmonic analysis of a field on a spheroid's surface. A spherical harmonic is orthonormal
# over directions, and a spheroid stores the geodetic latitude, so each node is embedded through the
# geometry's `geodetic_to_cartesian` and transformed at its geocentric direction, with the grid's own
# area measure as quadrature. The reference is a spherical point cloud over those directions carrying the
# same measures.

using Test: Test
using FlowFieldSpectra: FlowFieldSpectra as FFS
using FlowGeometries: FlowGeometries as FG
using SpectralBackends: SpectralBackends as SB
using ComputationalBackends: ComputationalBackends as CB
using NUFSHT: NUFSHT
using FINUFFT: FINUFFT

const SD_DS = SB.DirectSumSpectralBackend()
const SD_SER = CB.SerialBackend()
sd_rel(a, b) = maximum(abs, a .- b) / max(maximum(abs, b), eps())

# Earth's ellipsoid, scaled to unit semi-major axis so the coefficients stay O(1).
sd_geom() = FG.Geometry.SpheroidGeometry(1.0, inv(298.257223563))

Test.@testset "Geodetic latitude is converted to a geocentric direction" begin
    geo = sd_geom()
    sph = FG.Geometry.SphericalGeometry(1.0)
    for φ in (-1.2, -0.3, 0.0, 0.45, 1.5)
        Test.@test FFS.Grids._colatitude(sph, φ) ≈ π / 2 - φ
        p = FG.Geometry.geodetic_to_cartesian(geo, (0.0, φ))
        Test.@test FFS.Grids._colatitude(geo, φ) ≈ acos(p.z / hypot(p.x, p.y, p.z)) atol = 1e-14
    end
end

Test.@testset "Spheroid surface grid transforms against its own directions" begin
    geo = sd_geom()
    lmax = 4
    ms = (lmax + 1, 2 * lmax + 1)
    nlon, nlat = 12, 9
    λ = [2π * (i - 1) / nlon for i in 1:nlon]
    φ = [π * (j - 0.5) / nlat - π / 2 for j in 1:nlat]
    g = FG.Grids.StructuredGrid(geo, λ, φ; periodic = (true, false), period = (2π, 0.0))

    f = [1.0 + 0.5 * sin(a) + 0.3 * cos(2 * b) for a in λ, b in φ]
    c, _ = FFS.calculate_spectrum(g, f, ms; transform = SD_DS, execution = SD_SER)
    θpt, λpt = FFS.Grids._sph_points(g)
    meas = collect(FG.Grids.measure(g))
    cloud = FG.Grids.UnstructuredGrid(FG.Geometry.SphericalGeometry(1.0),
        (vec(λpt), (π / 2) .- vec(θpt)), vec(meas))
    cref, _ = FFS.calculate_spectrum(cloud, vec(f), ms; transform = SD_DS, execution = SD_SER)
    Test.@test sd_rel(c, cref) < 1e-12

    cn, _ = FFS.calculate_spectrum(g, f, ms; transform = SB.NUFSHTSpectralBackend())
    Test.@test sd_rel(cn, c) < 1e-8
    Test.@test FFS._resolve_transform(SB.AutoSpectralBackend(), g, ms, SD_SER) isa
        SB.AbstractNUFSHTSpectralBackend

    out = FFS.synthesize(g, c, ms; transform = SD_DS, execution = SD_SER)
    oref = FFS.synthesize(cloud, cref, ms; transform = SD_DS, execution = SD_SER)
    Test.@test sd_rel(vec(out), vec(oref)) < 1e-12
end

Test.@testset "Spheroid point cloud and the height direction" begin
    geo = sd_geom()
    lmax = 3
    ms = (lmax + 1, 2 * lmax + 1)
    N = 150
    ga = π * (3 - sqrt(5))
    λ = [mod(ga * i, 2π) for i in 1:N]
    φ = asin.([-1 + 2 * (i - 0.5) / N for i in 1:N])
    g = FG.Grids.UnstructuredGrid(geo, (λ, φ), fill(4π / N, N))

    f = [1.0 + 0.4 * sin(λ[i]) for i in 1:N]
    c, _ = FFS.calculate_spectrum(g, f, ms; transform = SD_DS, execution = SD_SER)
    θpt, λpt = FFS.Grids._sph_points(g)
    cloud = FG.Grids.UnstructuredGrid(FG.Geometry.SphericalGeometry(1.0),
        (λpt, (π / 2) .- θpt), fill(4π / N, N))
    cref, _ = FFS.calculate_spectrum(cloud, f, ms; transform = SD_DS, execution = SD_SER)
    Test.@test sd_rel(c, cref) < 1e-12

    # A grid carrying height as a third direction is not the domain of a surface expansion.
    g3 = FG.Grids.UnstructuredGrid(geo, (λ, φ, zeros(N)), fill(4π / N, N))
    Test.@test_throws ArgumentError FFS.Grids._sph_points(g3)
end
