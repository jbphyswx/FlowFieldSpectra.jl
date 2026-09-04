# Spherical-harmonic analysis of a field sampled on a spheroid.
#
# A spherical harmonic is orthonormal over DIRECTIONS. A spheroid stores the geodetic latitude, which is
# not the colatitude of the direction its node lies in — on Earth's ellipsoid the two differ by up to
# ~0.19° at mid-latitudes — so `Grids._colatitude` embeds the node through the geometry's own
# `geodetic_to_cartesian` and reads the geocentric direction back. Feeding those directions with the
# grid's own area measure as quadrature gives the SURFACE expansion of the field, the same one a geodetic
# field expansion uses. The radial variation of the spheroid is not modelled: that belongs to the
# solid-harmonic expansion, a different transform.
#
# The reference is a spherical point cloud built over exactly those geocentric directions carrying the
# same measures — an independent construction, so agreement pins both the conversion and the routing.

using Test: Test
using FlowFieldSpectra: FlowFieldSpectra as FFS
using FlowGeometries: FlowGeometries as FG
using SpectralBackends: SpectralBackends as SB
using ComputationalBackends: ComputationalBackends as CB
using NUFSHT: NUFSHT
using FINUFFT: FINUFFT
using OhMyThreads: OhMyThreads
using KernelAbstractions: KernelAbstractions as KA
using Random: Random

const SD_DS = SB.DirectSumSpectralBackend()
const SD_SER = CB.SerialBackend()
sd_rel(a, b) = maximum(abs, a .- b) / max(maximum(abs, b), eps())

# Earth's ellipsoid, scaled to unit semi-major axis so the coefficients stay O(1).
sd_geom() = FG.Geometry.SpheroidGeometry(1.0, inv(298.257223563))

Test.@testset "Geodetic latitude is converted to a geocentric direction" begin
    geo = sd_geom()
    sph = FG.Geometry.SphericalGeometry(1.0)

    # On a sphere the colatitude is π/2 - φ exactly.
    for φ in (-1.2, -0.3, 0.0, 0.45, 1.5)
        Test.@test FFS.Grids._colatitude(sph, 0.7, φ) ≈ π / 2 - φ
    end

    # On a spheroid it differs, vanishing at the equator and the poles and peaking near 45°.
    Test.@test FFS.Grids._colatitude(geo, 0.0, 0.0) ≈ π / 2                 # equator
    Test.@test FFS.Grids._colatitude(geo, 0.0, π / 2) ≈ 0.0 atol = 1e-12    # north pole
    dev(φ) = abs(FFS.Grids._colatitude(geo, 0.0, φ) - (π / 2 - φ))
    Test.@test dev(0.0) < 1e-15
    Test.@test dev(π / 4) > 1e-3                                            # ~0.19° at 45°
    Test.@test dev(π / 4) > dev(0.2) && dev(π / 4) > dev(1.4)               # peaks at mid-latitude
    # A spheroid is a surface of revolution, so the direction does not depend on longitude — which is
    # what keeps a latitude row one iso-latitude ring.
    Test.@test FFS.Grids._colatitude(geo, 0.0, 0.5) ≈ FFS.Grids._colatitude(geo, 2.9, 0.5)
end

Test.@testset "Spheroid surface grid transforms against its own directions" begin
    Random.seed!(3141)
    geo = sd_geom()
    lmax = 4
    ms = (lmax + 1, 2 * lmax + 1)
    nlon, nlat = 12, 9
    λ = [2π * (i - 1) / nlon for i in 1:nlon]
    φ = [π * (j - 0.5) / nlat - π / 2 for j in 1:nlat]
    g = FG.Grids.StructuredGrid(geo, λ, φ; periodic = (true, false), period = (2π, 0.0))
    Test.@test size(g) == (nlon, nlat)

    # A structured spheroid grid keeps iso-latitude rows, so it takes the ring path.
    Test.@test FFS.Grids._sph_layout(g) isa FFS.Grids.RingSphere

    f = [1.0 + 0.5 * sin(a) + 0.3 * cos(2 * b) for a in λ, b in φ]
    c, ks = FFS.calculate_spectrum(g, f, ms; transform = SD_DS, execution = SD_SER)
    Test.@test size(c) == ms
    Test.@test ks == (0:lmax, -lmax:lmax)
    Test.@test all(isfinite, c)

    # The independent reference: a spherical point cloud over the geocentric directions the spheroid's
    # nodes point in, carrying the spheroid's own per-cell measures.
    θpt, λpt = FFS.Grids._sph_points(g)
    meas = collect(FG.Grids.measure(g))
    cloud = FG.Grids.UnstructuredGrid(FG.Geometry.SphericalGeometry(1.0),
        (vec(λpt), (π / 2) .- vec(θpt)), vec(meas))
    cref, _ = FFS.calculate_spectrum(cloud, vec(f), ms; transform = SD_DS, execution = SD_SER)
    Test.@test sd_rel(c, cref) < 1e-12

    # The same cloud read at the GEODETIC latitudes is a different transform, which separates this test
    # from one the conversion could satisfy trivially.
    geod = FG.Grids.UnstructuredGrid(FG.Geometry.SphericalGeometry(1.0),
        (vec([a for a in λ, _ in φ]), vec([b for _ in λ, b in φ])), vec(meas))
    cgeod, _ = FFS.calculate_spectrum(geod, vec(f), ms; transform = SD_DS, execution = SD_SER)
    Test.@test sd_rel(cgeod, c) > 1e-6

    # Every execution backend, and the batch axis.
    for exec in (CB.ThreadedBackend(), CB.GPUBackend(KA.CPU()))
        cx, _ = FFS.calculate_spectrum(g, f, ms; transform = SD_DS, execution = exec)
        Test.@test sd_rel(cx, c) < 1e-12
    end
    fb = cat(f, 2 .* f; dims = 3)
    cb, _ = FFS.calculate_spectrum(g, fb, ms; transform = SD_DS, execution = SD_SER)
    Test.@test size(cb) == (ms..., 2)
    Test.@test sd_rel(cb[:, :, 1], c) < 1e-12

    # NUFSHT reaches it too, and agrees with the projection.
    cn, _ = FFS.calculate_spectrum(g, f, ms; transform = SB.NUFSHTSpectralBackend())
    Test.@test sd_rel(cn, c) < 1e-8
    # So `AutoSpectralBackend` selects it.
    Test.@test FFS._resolve_transform(SB.AutoSpectralBackend(), g, ms, SD_SER) isa
        SB.AbstractNUFSHTSpectralBackend

    # Synthesis evaluates at the same directions and writes the grid's own shape.
    out = FFS.synthesize(g, c, ms; transform = SD_DS, execution = SD_SER)
    Test.@test size(out) == size(g)
    oref = FFS.synthesize(cloud, cref, ms; transform = SD_DS, execution = SD_SER)
    Test.@test sd_rel(vec(out), vec(oref)) < 1e-12
end

Test.@testset "Spheroid point cloud and the height direction" begin
    Random.seed!(2718)
    geo = sd_geom()
    lmax = 3
    ms = (lmax + 1, 2 * lmax + 1)
    N = 150
    ga = π * (3 - sqrt(5))
    λ = [mod(ga * i, 2π) for i in 1:N]
    φ = asin.([-1 + 2 * (i - 0.5) / N for i in 1:N])
    g = FG.Grids.UnstructuredGrid(geo, (λ, φ), fill(4π / N, N))
    Test.@test FFS.Grids._sph_layout(g) isa FFS.Grids.ScatteredSphere

    f = [1.0 + 0.4 * sin(λ[i]) for i in 1:N]
    c, _ = FFS.calculate_spectrum(g, f, ms; transform = SD_DS, execution = SD_SER)
    θpt, λpt = FFS.Grids._sph_points(g)
    cloud = FG.Grids.UnstructuredGrid(FG.Geometry.SphericalGeometry(1.0),
        (λpt, (π / 2) .- θpt), fill(4π / N, N))
    cref, _ = FFS.calculate_spectrum(cloud, f, ms; transform = SD_DS, execution = SD_SER)
    Test.@test sd_rel(c, cref) < 1e-12

    # A grid carrying height as a third direction is not the domain of a surface expansion, and says so.
    h = zeros(N)
    g3 = FG.Grids.UnstructuredGrid(geo, (λ, φ, h), fill(4π / N, N))
    Test.@test_throws ArgumentError FFS.Grids._sph_points(g3)
end
