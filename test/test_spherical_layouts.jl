# Spherical layouts, quadrature exactness, and masks.
#
# The transform picks its spherical algorithm from how the grid's cells lie on iso-latitude rings, so a
# tensor product factorizes over its shared longitude axis, a ring layout over each ring's own longitudes,
# and a point cloud has neither. Each algorithm is held to the per-point projection over the same nodes and
# weights, which it reassociates.

using Test: Test
using Logging: Logging
using FlowFieldSpectra: FlowFieldSpectra as FFS
using FlowGeometries: FlowGeometries as FG
using SpectralBackends: SpectralBackends as SB
using ComputationalBackends: ComputationalBackends as CB
using KernelAbstractions: KernelAbstractions as KA
using OhMyThreads: OhMyThreads
using FastSphericalHarmonics: FastSphericalHarmonics
using NUFSHT: NUFSHT
using FINUFFT: FINUFFT
using NonuniformFFTs: NonuniformFFTs
using FFTW: FFTW
using Random: Random

const SPH_SS = FG.SphericalSampling
const SPH_DS = SB.DirectSumSpectralBackend()
const SPH_SER = CB.SerialBackend()
sph_geom() = FG.Geometry.SphericalGeometry(1.0)
sph_rel(a, b) = maximum(abs, a .- b) / max(maximum(abs, b), eps())

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

Test.@testset "A pixelization transforms as the point set of its own nodes" begin
    lmax = 4
    ms = (lmax + 1, 2lmax + 1)
    Random.seed!(917)
    # In NESTED order a ring's pixels are not contiguous ids, so that grid has no ring path.
    for g in (FG.Grids.HEALPixGrid(sph_geom(), 2), FG.Grids.HEALPixGrid(sph_geom(), 4; scheme = SPH_SS.Nested()),
              FG.Grids.RingGrid(sph_geom(), SPH_SS.OctahedralGaussianSampling(2)),
              FG.Grids.CubedSphereGrid(sph_geom(), 4))
        f = randn(length(g))
        gs, ids = sph_active_scattered(g)
        cref, _ = FFS.calculate_spectrum(gs, f[ids], ms; transform = SPH_DS, execution = SPH_SER)
        for exec in (SPH_SER, CB.ThreadedBackend())
            c, _ = FFS.calculate_spectrum(g, f, ms; transform = SPH_DS, execution = exec)
            Test.@test sph_rel(c, cref) < 1e-13
        end
    end
end

Test.@testset "Ring transform reassociates the per-point projection" begin
    # Against the per-point projection over the same nodes, for the grid's own quadrature and for one the
    # caller states: `sampling` or per-latitude `weights` on a structured grid (a mask sends it to the
    # ring path), per-node `weights` on a ring pixelization.
    lmax = 4
    ms = (lmax + 1, 2lmax + 1)
    Random.seed!(918)
    GL = SPH_SS.GaussLegendreSampling()
    base = FG.Connectivity.structured_grid(Float64, GL, 8)
    nlon, nl = size(base)
    mk = trues(nlon, nl)
    mk[2:4, 3] .= false
    graw = FG.Grids.StructuredGrid(sph_geom(), FG.Grids.coordinates(base, 1), FG.Grids.coordinates(base, 2);
        mask = mk)                                   # records no sampling
    cases = Any[(graw, randn(size(graw)...), (; sampling = GL)),
                (graw, randn(size(graw)...), (; weights = 2 .* rand(nl) ./ nl))]
    for g in (FG.Grids.HEALPixGrid(sph_geom(), 2), FG.Grids.HEALPixGrid(sph_geom(), 4),
              FG.Grids.RingGrid(sph_geom(), SPH_SS.OctahedralGaussianSampling(3)))
        push!(cases, (g, randn(length(g)), (;)))
        push!(cases, (g, randn(length(g)), (; weights = 4π .* (0.5 .+ rand(length(g))) ./ length(g))))
    end
    for (g, f, kw) in cases
        cpt = zeros(ms...)
        FFS.DirectSum._sph_direct!(FG.Grids.NoRings(), cpt, g, f, lmax; kw...)
        for exec in (SPH_SER, CB.ThreadedBackend())
            c, _ = FFS.calculate_spectrum(g, f, ms; transform = SPH_DS, execution = exec, kw...)
            Test.@test sph_rel(c, cpt) < 1e-13
        end
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

    for (s, nlat) in ((SPH_SS.GaussLegendreSampling(), 8), (SPH_SS.GaussLegendreSampling(), 12),
                      (SPH_SS.DriscollHealySampling(), 12))
        g = FG.Connectivity.structured_grid(Float64, s, nlat)
        lim = SPH_SS.bandlimit(s, nlat)
        Test.@test roundtrip(g, lim, max(1, lim - 1), 1; sampling = s) < 1e-10
    end

    # Clenshaw–Curtis is exact to about half its band limit.
    let s = SPH_SS.ClenshawCurtisSampling(), nlat = 16
        g = FG.Connectivity.structured_grid(Float64, s, nlat)
        half = (nlat - 1) ÷ 2
        Test.@test roundtrip(g, half, max(1, half - 1), 1; sampling = s) < 1e-10
    end

    # A degree above the sampling's band limit warns; the limit itself does not.
    let s = SPH_SS.DriscollHealySampling(), nlat = 12
        g = FG.Connectivity.structured_grid(Float64, s, nlat)
        lim = SPH_SS.bandlimit(s, nlat)
        f = randn(size(g)...)
        over = nlat - 1
        Test.@test sph_warncount() do
            FFS.calculate_spectrum(g, f, (over + 1, 2over + 1); transform = SPH_DS,
                execution = SPH_SER, sampling = s)
        end >= 1
        Test.@test sph_warncount() do
            FFS.calculate_spectrum(g, f, (lim + 1, 2lim + 1); transform = SPH_DS,
                execution = SPH_SER, sampling = s)
        end == 0
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
        push!(grids, FG.Grids.HEALPixGrid(sph_geom(), ns; mask = [φ[i] >= 0 for i in 1:npix]))
        push!(unmasked_of, base)
    end
    let ns = 4, npix = 12 * ns^2
        push!(grids, FG.Grids.HEALPixGrid(sph_geom(), ns; mask = [!(iseven(i) && i > npix ÷ 3) for i in 1:npix]))
        push!(unmasked_of, FG.Grids.HEALPixGrid(sph_geom(), ns))
    end
    let base = FG.Connectivity.structured_grid(Float64, SPH_SS.GaussLegendreSampling(), 8)
        nlon, nl = size(base)
        mk = [!(i > nlon ÷ 2 && j > nl ÷ 2) for i in 1:nlon, j in 1:nl]
        push!(grids, FG.Grids.StructuredGrid(sph_geom(), FG.Grids.coordinates(base, 1),
            FG.Grids.coordinates(base, 2); sampling = SPH_SS.GaussLegendreSampling(), mask = mk))
        push!(unmasked_of, base)
    end
    let Np = 300
        θ = acos.(clamp.(2 .* rand(Np) .- 1, -1.0, 1.0))
        φ = rand(Np) .* 2π
        areas = 0.5 .+ rand(Np)
        coords = (collect(φ), π / 2 .- collect(θ))
        push!(grids, FG.Grids.UnstructuredGrid(sph_geom(), coords, areas, [φ[i] < π for i in 1:Np]))
        push!(unmasked_of, FG.Grids.UnstructuredGrid(sph_geom(), coords, areas))
    end

    # The transform is linear and a node's weight is independent of the mask, so excluding a cell and
    # zeroing its datum are the same sum. The unmasked grid with a zeroed field is therefore an exact
    # reference for every layout, and it holds the sampling's own quadrature rule fixed. Masked data is
    # commonly NaN, and `0 * NaN` is NaN, so a zero weight alone is not exclusion: a NaN in a masked cell
    # must leave the coefficients unchanged, in the direct sum and in NUFSHT's quadrature analysis alike.
    # This is `inactive = ZeroFill()`, the default, named here.
    gpu = CB.GPUBackend(KA.CPU())
    nu = SB.NUFSHTSpectralBackend()
    zf = (; inactive = FFS.ZeroFill())
    for (g, unmasked) in zip(grids, unmasked_of)
        f = randn(size(g)...)
        mk = FG.Grids.mask(g)
        fz = [mk[i] ? f[i] : 0.0 for i in eachindex(f)]
        fn = [mk[i] ? f[i] : NaN for i in eachindex(f)]
        cref, _ = FFS.calculate_spectrum(unmasked, reshape(fz, size(f)), ms; transform = SPH_DS, execution = SPH_SER)
        for (tr, exec, kw, tol) in ((SPH_DS, SPH_SER, zf, 1e-12), (SPH_DS, gpu, zf, 1e-12),
                                    (nu, SPH_SER, (; tol = 1e-12, zf...), 1e-10), (nu, gpu, (; tol = 1e-12, zf...), 1e-10))
            c, _ = FFS.calculate_spectrum(g, reshape(fn, size(f)), ms; transform = tr, execution = exec, kw...)
            Test.@test sph_rel(c, cref) < tol
        end
    end

    # On an equal-area pixelization the weights are the cell measures themselves, so a grid over just the
    # active nodes is a second independent reference. The f_sky factor between them states that the masked
    # grid's weights total the covered solid angle.
    let g = grids[1]
        f = randn(size(g)...)
        gs, ids = sph_active_scattered(g)
        c, _ = FFS.calculate_spectrum(g, f, ms; transform = SPH_DS, execution = SPH_SER)
        cref, _ = FFS.calculate_spectrum(gs, f[ids], ms; transform = SPH_DS, execution = SPH_SER)
        Test.@test sph_rel(c, cref .* FG.Grids.sky_fraction(g)) < 1e-12
    end

    # `Exclude` transforms the active nodes as their own point set: on an equal-area pixelization and on a
    # scattered grid it is the transform of a grid of the active nodes, carrying their own measures.
    df = (; inactive = FFS.Exclude())
    for g in (grids[1], grids[end])
        f = randn(size(g)...)
        mk = FG.Grids.mask(g)
        fn = [mk[i] ? f[i] : NaN for i in eachindex(f)]
        gs, ids = sph_active_scattered(g)
        cref, _ = FFS.calculate_spectrum(gs, f[ids], ms; transform = SPH_DS, execution = SPH_SER)
        for (tr, exec, kw, tol) in ((SPH_DS, SPH_SER, df, 1e-12), (SPH_DS, CB.ThreadedBackend(), df, 1e-12),
                                    (SPH_DS, gpu, df, 1e-12), (nu, SPH_SER, (; tol = 1e-12, df...), 1e-10),
                                    (nu, gpu, (; tol = 1e-12, df...), 1e-10))
            c, _ = FFS.calculate_spectrum(g, reshape(fn, size(f)), ms; transform = tr, execution = exec, kw...)
            Test.@test sph_rel(c, cref) < tol
        end
    end

    # The fast SHT paths consume complete longitude rows, and both are linear in the field, so a masked
    # cell is left out by zeroing its datum. Each must match itself on the unmasked grid with a zeroed
    # field, a NaN in a masked cell included.
    let lmax_f = 5, nlat = lmax_f + 1, nlon = 2 * lmax_f + 1
        base = FG.Connectivity.structured_grid(Float64, SPH_SS.ClenshawCurtisSampling(), nlat)
        mk = [!(i > nlon ÷ 2 && j == nlat) for i in 1:nlon, j in 1:nlat]
        gm = FG.Grids.StructuredGrid(sph_geom(), FG.Grids.coordinates(base, 1),
            FG.Grids.coordinates(base, 2); sampling = SPH_SS.ClenshawCurtisSampling(), mask = mk)
        msf = (nlat, 2nlat - 1)
        f = randn(nlon, nlat)
        fz = ifelse.(mk, f, 0.0)
        fn = ifelse.(mk, f, NaN)
        tr = SB.FSHTSpectralBackend()
        # `Exclude` rescales the quadrature from the active cells' share of the latitude rule to the
        # whole sphere's `4π`.
        wq = SPH_SS.latitude_weights(Float64, SPH_SS.ClenshawCurtisSampling(), nlat)
        share = sum(mk[i, j] * wq[j] * 2π / nlon for i in 1:nlon, j in 1:nlat)
        for exec in (SPH_SER, CB.GPUBackend(KA.CPU()))
            cu, _ = FFS.calculate_spectrum(base, fz, msf; transform = tr, execution = exec,
                sampling = SPH_SS.ClenshawCurtisSampling())
            cn, _ = FFS.calculate_spectrum(gm, fn, msf; transform = tr, execution = exec,
                sampling = SPH_SS.ClenshawCurtisSampling())
            Test.@test sph_rel(cn, cu) < 1e-12
            cd, _ = FFS.calculate_spectrum(gm, fn, msf; transform = tr, execution = exec,
                sampling = SPH_SS.ClenshawCurtisSampling(), inactive = FFS.Exclude())
            Test.@test sph_rel(cd, cu .* (4π / share)) < 1e-12
        end
    end

    # Synthesis evaluates at every grid point, masked or not.
    let ns = 2
        base = FG.Grids.HEALPixGrid(sph_geom(), ns)
        g = grids[1]
        C = zeros(ms...)
        C[FFS.sph_mode_index(2, 1)] = 1.0
        Test.@test FFS.synthesize(g, C, ms; transform = SPH_DS, execution = SPH_SER) ≈
                   FFS.synthesize(base, C, ms; transform = SPH_DS, execution = SPH_SER)
    end
end
