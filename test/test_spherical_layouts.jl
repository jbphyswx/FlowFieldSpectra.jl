# Spherical layout routing, quadrature exactness, and masks.
#
# The transform picks its spherical algorithm from the grid's SAMPLING traits
# (`SphericalSampling.is_tensor_product`, `is_iso_latitude`), so a tensor product factorizes over its
# shared longitude axis, an iso-latitude ring layout over each ring's own longitudes, and a point cloud
# has neither. These testsets gate that routing, the ring transform against the per-point projection it
# reassociates, the band-limit relation each sampling states, and mask handling.

using Test: Test
using Logging: Logging
using FlowFieldSpectra: FlowFieldSpectra as FFS
using FlowGeometries: FlowGeometries as FG
using SpectralBackends: SpectralBackends as SB
using ComputationalBackends: ComputationalBackends as CB
using KernelAbstractions: KernelAbstractions as KA
using OhMyThreads: OhMyThreads
using FastSphericalHarmonics: FastSphericalHarmonics
using FFTW: FFTW
using Random: Random

const SPH_SS = FG.SphericalSampling
const SPH_DS = SB.DirectSumSpectralBackend()
const SPH_SER = CB.SerialBackend()
sph_geom() = FG.Geometry.SphericalGeometry(1.0)
sph_rel(a, b) = maximum(abs, a .- b) / max(maximum(abs, b), eps())

# `Grids._sph_layout` and `DirectSum._sph_direct!` are internal. The claims here are which ALGORITHM a
# grid routes to, and that the ring algorithm reproduces the per-point one; a correct implementation
# gives identical coefficients either way, so the public result cannot distinguish them.

# A grid's active nodes as a standalone scattered grid carrying their own measures.
function sph_active_scattered(g)
    λ, φlat = FG.Grids.materialize(g)
    m = FG.Grids.measure(g)
    mk = FG.Grids.mask(g)
    ids = findall(i -> mk[i], eachindex(mk))
    return FG.Grids.UnstructuredGrid(sph_geom(), (collect(λ[ids]), collect(φlat[ids])),
        collect(Float64[m[i] for i in ids])), ids
end

# Warnings emitted by one call.
function sph_warncount(f)
    logger = Test.TestLogger(min_level = Logging.Warn)
    Logging.with_logger(f, logger)
    return count(r -> r.level == Logging.Warn, logger.logs)
end

Test.@testset "Spherical layout routing (from the sampling's traits)" begin
    lmax = 4
    ms = (lmax + 1, 2lmax + 1)
    gl = FG.Connectivity.structured_grid(Float64, SPH_SS.GaussLegendreSampling(), lmax + 1)
    hp = FG.Grids.HEALPixGrid(sph_geom(), 2)
    rg = FG.Grids.RingGrid(sph_geom(), SPH_SS.OctahedralGaussianSampling(2))
    Random.seed!(917)
    fib = FG.Grids.UnstructuredGrid(sph_geom(),
        (rand(40) .* 2π, asin.(2 .* rand(40) .- 1)), ones(40))

    Test.@test FFS.Grids._sph_layout(gl) isa FFS.Grids.TensorSphere
    Test.@test FFS.Grids._sph_layout(hp) isa FFS.Grids.RingSphere
    Test.@test FFS.Grids._sph_layout(rg) isa FFS.Grids.RingSphere
    Test.@test FFS.Grids._sph_layout(fib) isa FFS.Grids.ScatteredSphere

    # A pixelization must equal a hand-built grid over its own nodes and measures.
    for g in (hp, FG.Grids.HEALPixGrid(sph_geom(), 4), rg,
              FG.Grids.CubedSphereGrid(sph_geom(), 4), FG.Grids.IcosahedralGrid(sph_geom(), 2))
        f = randn(length(g))
        gs, ids = sph_active_scattered(g)
        c, _ = FFS.calculate_spectrum(g, f, ms; transform = SPH_DS, execution = SPH_SER)
        cref, _ = FFS.calculate_spectrum(gs, f[ids], ms; transform = SPH_DS, execution = SPH_SER)
        Test.@test size(c) == size(cref)
        Test.@test sph_rel(c, cref) < 1e-13
        # every execution backend agrees
        ct, _ = FFS.calculate_spectrum(g, f, ms; transform = SPH_DS, execution = CB.ThreadedBackend())
        cd, _ = FFS.calculate_spectrum(g, f, ms; transform = SPH_DS, execution = CB.GPUBackend(KA.CPU()))
        Test.@test sph_rel(ct, c) < 1e-13
        Test.@test sph_rel(cd, c) < 1e-13
    end
end

Test.@testset "Ring transform reassociates the per-point projection" begin
    lmax = 4
    ms = (lmax + 1, 2lmax + 1)
    Random.seed!(918)
    for g in (FG.Grids.HEALPixGrid(sph_geom(), 2), FG.Grids.HEALPixGrid(sph_geom(), 4),
              FG.Grids.RingGrid(sph_geom(), SPH_SS.OctahedralGaussianSampling(3)))
        f = randn(length(g))
        cring, _ = FFS.calculate_spectrum(g, f, ms; transform = SPH_DS, execution = SPH_SER)
        cpt = zeros(ComplexF64, ms...)
        FFS.DirectSum._sph_direct!(FFS.Grids.ScatteredSphere(), cpt, g, f, lmax)
        Test.@test sph_rel(cring, cpt) < 1e-13
    end
end

Test.@testset "Band limit and quadrature exactness come from the sampling" begin
    # One harmonic in, the same one out, at each sampling's own band limit.
    function roundtrip(g, lmax, l, m; sampling = nothing)
        ms = (lmax + 1, 2lmax + 1)
        C = zeros(ComplexF64, ms...)
        C[FFS.sph_mode_index(l, m)] = 1.0
        kw = sampling === nothing ? (;) : (; sampling = sampling)
        f = FFS.synthesize(g, C, ms; transform = SPH_DS, execution = SPH_SER, kw...)
        C2, _ = FFS.calculate_spectrum(g, f, ms; transform = SPH_DS, execution = SPH_SER, kw...)
        return sph_rel(C2, C)
    end

    # `admits_exact_bandlimited_quadrature` claims exactness at the sampling's own `bandlimit`.
    for (s, nlat) in ((SPH_SS.GaussLegendreSampling(), 8), (SPH_SS.GaussLegendreSampling(), 12),
                      (SPH_SS.DriscollHealySampling(), 12))
        Test.@test SPH_SS.admits_exact_bandlimited_quadrature(s)
        g = FG.Connectivity.structured_grid(Float64, s, nlat)
        lim = SPH_SS.bandlimit(s, nlat)
        Test.@test roundtrip(g, lim, max(1, lim - 1), 1; sampling = s) < 1e-10
    end

    # Clenshaw-Curtis is declared INEXACT at its band limit and supports about half of it.
    let s = SPH_SS.ClenshawCurtisSampling(), nlat = 16
        Test.@test !SPH_SS.admits_exact_bandlimited_quadrature(s)
        g = FG.Connectivity.structured_grid(Float64, s, nlat)
        lim = SPH_SS.bandlimit(s, nlat)
        half = (nlat - 1) ÷ 2
        Test.@test roundtrip(g, lim, lim - 1, 1; sampling = s) > 1e-3          # inexact at the limit
        Test.@test roundtrip(g, half, max(1, half - 1), 1; sampling = s) < 1e-10
    end

    # Driscoll-Healy's limit is nlat÷2 - 1, so a higher degree warns once and still returns coefficients.
    let s = SPH_SS.DriscollHealySampling(), nlat = 12
        g = FG.Connectivity.structured_grid(Float64, s, nlat)
        lim = SPH_SS.bandlimit(s, nlat)
        Test.@test lim == nlat ÷ 2 - 1
        f = randn(size(g)...)
        over = nlat - 1
        n_over = sph_warncount() do
            FFS.calculate_spectrum(g, f, (over + 1, 2over + 1); transform = SPH_DS,
                execution = SPH_SER, sampling = s)
        end
        Test.@test n_over >= 1
        Test.@test sph_warncount() do
            FFS.calculate_spectrum(g, f, (lim + 1, 2lim + 1); transform = SPH_DS,
                execution = SPH_SER, sampling = s)
        end == 0
        c, _ = FFS.calculate_spectrum(g, f, (over + 1, 2over + 1); transform = SPH_DS,
            execution = SPH_SER, sampling = s)
        Test.@test size(c) == (over + 1, 2over + 1)          # warns, does not raise
    end

    # McEwen-Wiaux states no latitude quadrature, so a quadrature transform on it raises.
    let s = SPH_SS.McEwenWiauxSampling(), nlat = 8
        g = FG.Connectivity.structured_grid(Float64, s, nlat)
        Test.@test_throws ArgumentError FFS.calculate_spectrum(g, randn(size(g)...),
            (nlat, 2nlat - 1); transform = SPH_DS, execution = SPH_SER, sampling = s)
    end
end

Test.@testset "Masks: the active set is what is transformed" begin
    lmax = 4
    ms = (lmax + 1, 2lmax + 1)
    Random.seed!(919)

    # Northern-hemisphere HEALPix, a partially-cut HEALPix (a ring's single weight cannot express it), a
    # masked structured grid, and masked scattered — each paired with its unmasked counterpart.
    grids = Any[]
    unmasked_of = Any[]
    for ns in (2, 4)
        npix = 12 * ns^2
        base = FG.Grids.HEALPixGrid(sph_geom(), ns)
        _, φ = FG.Grids.materialize(base)
        mk = trues(npix)
        for i in 1:npix
            φ[i] < 0 && (mk[i] = false)
        end
        push!(grids, FG.Grids.HEALPixGrid(sph_geom(), ns; mask = mk))
        push!(unmasked_of, base)
    end
    let ns = 4, npix = 12 * ns^2
        mk = trues(npix)
        for i in 1:npix
            (iseven(i) && i > npix ÷ 3) && (mk[i] = false)
        end
        push!(grids, FG.Grids.HEALPixGrid(sph_geom(), ns; mask = mk))
        push!(unmasked_of, FG.Grids.HEALPixGrid(sph_geom(), ns))
    end
    let base = FG.Connectivity.structured_grid(Float64, SPH_SS.GaussLegendreSampling(), 8)
        nlon, nl = size(base)
        mk = trues(nlon, nl)
        for j in 1:nl, i in 1:nlon
            (i > nlon ÷ 2 && j > nl ÷ 2) && (mk[i, j] = false)
        end
        push!(grids, FG.Grids.StructuredGrid(sph_geom(), FG.Grids.coordinates(base, 1),
            FG.Grids.coordinates(base, 2); sampling = SPH_SS.GaussLegendreSampling(), mask = mk))
        push!(unmasked_of, base)
    end
    let Np = 300
        θ = acos.(clamp.(2 .* rand(Np) .- 1, -1.0, 1.0))
        φ = rand(Np) .* 2π
        areas = 0.5 .+ rand(Np)
        mk = [φ[i] < π for i in 1:Np]
        coords = (collect(φ), π / 2 .- collect(θ))
        push!(grids, FG.Grids.UnstructuredGrid(sph_geom(), coords, areas, mk))
        push!(unmasked_of, FG.Grids.UnstructuredGrid(sph_geom(), coords, areas))
    end

    # The transform is linear and a node's weight is independent of the mask, so excluding a cell and
    # zeroing its datum are the same sum. The unmasked grid with a zeroed field is therefore an exact
    # reference for every layout, and it holds the sampling's own quadrature rule fixed.
    for (g, unmasked) in zip(grids, unmasked_of)
        f = randn(size(g)...)
        mk = FG.Grids.mask(g)
        c, _ = FFS.calculate_spectrum(g, f, ms; transform = SPH_DS, execution = SPH_SER)
        fz = copy(f)
        for i in eachindex(fz)
            mk[i] || (fz[i] = 0.0)
        end
        cref, _ = FFS.calculate_spectrum(unmasked, fz, ms; transform = SPH_DS, execution = SPH_SER)
        Test.@test size(c) == size(cref)
        Test.@test sph_rel(c, cref) < 1e-13

        # Masked data is commonly NaN, and `0 * NaN` is NaN, so a zero weight alone is not exclusion.
        f2 = copy(f)
        for i in eachindex(f2)
            mk[i] || (f2[i] = NaN)
        end
        c2, _ = FFS.calculate_spectrum(g, f2, ms; transform = SPH_DS, execution = SPH_SER)
        Test.@test all(isfinite, c2)
        Test.@test sph_rel(c2, c) == 0.0

        # Threaded and device paths agree with serial.
        ct, _ = FFS.calculate_spectrum(g, f, ms; transform = SPH_DS, execution = CB.ThreadedBackend())
        cd, _ = FFS.calculate_spectrum(g, f, ms; transform = SPH_DS, execution = CB.GPUBackend(KA.CPU()))
        Test.@test sph_rel(ct, c) < 1e-12
        Test.@test sph_rel(cd, c) < 1e-12
    end

    # On an equal-area pixelization the weights are the cell measures themselves, so a grid over just the
    # active nodes is a second independent reference. The f_sky factor between them states that the masked
    # grid's weights total the covered solid angle.
    let g = grids[1]
        f = randn(size(g)...)
        gs, ids = sph_active_scattered(g)
        c, _ = FFS.calculate_spectrum(g, f, ms; transform = SPH_DS, execution = SPH_SER)
        cref, _ = FFS.calculate_spectrum(gs, f[ids], ms; transform = SPH_DS, execution = SPH_SER)
        Test.@test sph_rel(c, cref .* FFS.Grids.sky_fraction(g)) < 1e-12
    end

    # A mask leaves partial longitude rows, so a masked structured grid leaves the tensor path.
    let base = FG.Connectivity.structured_grid(Float64, SPH_SS.GaussLegendreSampling(), 8)
        Test.@test FFS.Grids._sph_layout(base) isa FFS.Grids.TensorSphere
        Test.@test FFS.Grids._sph_layout(grids[4]) isa FFS.Grids.RingSphere
        Test.@test FFS.Grids._sph_layout(grids[5]) isa FFS.Grids.ScatteredSphere
    end

    # The fast SHT paths consume complete longitude rows, and both are linear in the field, so a masked
    # cell is left out by zeroing its datum. Each must match itself on the unmasked grid with a zeroed
    # field, so masking never silently includes an inactive cell.
    let lmax_f = 5, nlat = lmax_f + 1, nlon = 2 * lmax_f + 1
        base = FG.Connectivity.structured_grid(Float64, SPH_SS.ClenshawCurtisSampling(), nlat)
        Test.@test size(base) == (nlon, nlat)
        mk = trues(nlon, nlat)
        for j in 1:nlat, i in 1:nlon
            (i > nlon ÷ 2 && j == nlat) && (mk[i, j] = false)
        end
        gm = FG.Grids.StructuredGrid(sph_geom(), FG.Grids.coordinates(base, 1),
            FG.Grids.coordinates(base, 2); sampling = SPH_SS.ClenshawCurtisSampling(), mask = mk)
        msf = (nlat, 2nlat - 1)
        f = randn(nlon, nlat)
        fz = copy(f)
        for i in eachindex(fz)
            mk[i] || (fz[i] = 0.0)
        end
        # The same transform tag reaches FastSphericalHarmonics on a CPU backend and the device-generic
        # SHT on a GPU one, so only the execution varies here.
        tr = SB.FSHTSpectralBackend()
        for exec in (SPH_SER, CB.GPUBackend(KA.CPU()))
            cm, _ = FFS.calculate_spectrum(gm, f, msf; transform = tr, execution = exec,
                sampling = SPH_SS.ClenshawCurtisSampling())
            cu, _ = FFS.calculate_spectrum(base, fz, msf; transform = tr, execution = exec,
                sampling = SPH_SS.ClenshawCurtisSampling())
            Test.@test sph_rel(cm, cu) < 1e-12
            # a NaN in a masked cell must not reach the result
            fn = copy(f)
            for i in eachindex(fn)
                mk[i] || (fn[i] = NaN)
            end
            cn, _ = FFS.calculate_spectrum(gm, fn, msf; transform = tr, execution = exec,
                sampling = SPH_SS.ClenshawCurtisSampling())
            Test.@test all(isfinite, cn)
            Test.@test sph_rel(cn, cm) == 0.0
        end
    end

    # Covered extent is grid metadata, so the transform's return shape never changes.
    let ns = 2, npix = 12 * ns^2
        base = FG.Grids.HEALPixGrid(sph_geom(), ns)
        Test.@test FFS.Grids.sky_fraction(base) ≈ 1.0
        Test.@test FFS.Grids.covered_area(base) ≈ 4π
        _, φ = FG.Grids.materialize(base)
        mk = trues(npix)
        for i in 1:npix
            φ[i] < 0 && (mk[i] = false)
        end
        g = FG.Grids.HEALPixGrid(sph_geom(), ns; mask = mk)
        Test.@test FFS.Grids.sky_fraction(g) ≈ count(mk) / npix          # equal-area pixels
        Test.@test FFS.Grids.covered_area(g) ≈ 4π * count(mk) / npix

        # Synthesis evaluates at every grid point, so its output stays full length.
        C = zeros(ComplexF64, ms...)
        C[FFS.sph_mode_index(2, 1)] = 1.0
        out = FFS.synthesize(g, C, ms; transform = SPH_DS, execution = SPH_SER)
        Test.@test length(out) == npix
        Test.@test out ≈ FFS.synthesize(base, C, ms; transform = SPH_DS, execution = SPH_SER)
    end
end
