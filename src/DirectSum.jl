module DirectSum

using FlowGeometries: FlowGeometries
using ..Grids: Grids
using ..SphericalKernels: SphericalKernels

export sph_mode_index

# The direct-sum (DFT / SHT-projection) reference transform. Dependency-free, `O(∏ms · Npts · ∏batch)`
# — correct for any grid, never FFT-competitive by design. It reads a field tensor `(spatial…, batch…)`
# and writes coefficients `(spectral…, batch…)`; the batch is the innermost, contiguous loop. Structured
# grids read coordinates through their 1-D axes (`coordinates(g)[d][I[d]]`) — NO per-point coordinate
# blob is materialized; unstructured grids read their per-node vectors (`coordinates(g, d)[j]`).

"""
    sph_mode_index(l, m) -> CartesianIndex

`CartesianIndex` of degree `l`, order `m` in the `(lmax+1, 2lmax+1)` coefficient array.
"""
@inline function sph_mode_index(l::Int, m::Int)
    row = l - abs(m) + 1
    col = m == 0 ? 1 : (m < 0 ? 2 * abs(m) : 2 * m + 1)
    return CartesianIndex(row, col)
end

# Phase argument Σ_d ks[d][K[d]] · x_d, Val-unrolled over the type-parameter dim count so each tuple
# index is a compile-time constant. This stays type-stable and allocation-free even when `@inbounds`
# is disabled (as under `--check-bounds=yes`, how `Pkg.test` runs) — a plain `for d in 1:D` loop
# indexing the heterogeneous `ks`/`axes`/`coords` tuples at a runtime `d` can otherwise box.
@inline _phase_tensor(ks::Tuple, axes::Tuple, K::CartesianIndex{D}, P::CartesianIndex{D}, ::Type{FT}) where {D, FT} =
    sum(ntuple(d -> FT(ks[d][K[d]]) * FT(axes[d][P[d]]), Val(D)))
@inline _phase_scattered(ks::Tuple, coords::Tuple, K::CartesianIndex{D}, j::Int, ::Type{FT}) where {D, FT} =
    sum(ntuple(d -> FT(ks[d][K[d]]) * FT(coords[d][j]), Val(D)))

# =============================================================================
# Cartesian forward (analysis):  C[k, b] = (1/Npts) Σ_p f[p, b] · exp(-iflag · i · k·x_p)
# =============================================================================

# Structured tensor-product grid (uniform or nonuniform): spatial point `P` is a CartesianIndex over the
# axis lengths; its coordinate along `d` is `coordinates(g)[d][P[d]]` (no materialization).
function _calculate_spectrum_cartesian_direct!(
    coeffs::AbstractArray{Complex{FT}},
    g::FlowGeometries.Grids.AbstractStructuredGrid{<:FlowGeometries.Geometry.AbstractCartesianGeometry},
    field::AbstractArray,
    ms::NTuple{D, Int},
    iflag::Int,
) where {FT, D}
    axes = FlowGeometries.Grids.coordinates(g)
    ss = size(g)
    Npts = prod(ss)
    M = prod(ms)
    B = length(coeffs) ÷ M
    ks = Grids.physical_wavenumbers(g, ms)
    fill!(coeffs, zero(Complex{FT}))
    spat = CartesianIndices(ss)
    # Linear indexing (no `reshape`): the (spatial, batch) split is column-major, so mode `mi` of batch
    # `b` is `coeffs[mi + (b-1)·M]` and point `pj` of batch `b` is `field[pj + (b-1)·Npts]`. Avoids the
    # non-escaping reshape-header alloc that appears when `@inbounds` is off (`--check-bounds=yes`).
    @inbounds for (mi, I) in enumerate(CartesianIndices(ms))
        for (pj, P) in enumerate(spat)
            phi = _phase_tensor(ks, axes, I, P, FT)
            W = cis(-iflag * phi)
            for b in 1:B
                coeffs[mi + (b - 1) * M] += field[pj + (b - 1) * Npts] * W
            end
        end
    end
    coeffs ./= Npts
    return ks
end

# Unstructured point cloud: spatial point `j` reads `coordinates(g, d)[j]`.
function _calculate_spectrum_cartesian_direct!(
    coeffs::AbstractArray{Complex{FT}},
    g::FlowGeometries.Grids.AbstractUnstructuredGrid{<:FlowGeometries.Geometry.AbstractCartesianGeometry},
    field::AbstractArray,
    ms::NTuple{D, Int},
    iflag::Int,
) where {FT, D}
    coords = FlowGeometries.Grids.coordinates(g)
    N = length(coords[1])
    M = prod(ms)
    B = length(coeffs) ÷ M
    ks = Grids.physical_wavenumbers(g, ms)
    fill!(coeffs, zero(Complex{FT}))
    @inbounds for (mi, I) in enumerate(CartesianIndices(ms))
        for j in 1:N
            phi = _phase_scattered(ks, coords, I, j, FT)
            W = cis(-iflag * phi)
            for b in 1:B
                coeffs[mi + (b - 1) * M] += field[j + (b - 1) * N] * W
            end
        end
    end
    coeffs ./= N
    return ks
end

# =============================================================================
# Cartesian inverse (synthesis):  f[p, b] = Σ_k C[k, b] · exp(+iflag · i · k·x_p)
# =============================================================================

function _synthesize_cartesian_direct!(
    out::AbstractArray{Complex{FT}},
    g::FlowGeometries.Grids.AbstractStructuredGrid{<:FlowGeometries.Geometry.AbstractCartesianGeometry},
    coeffs::AbstractArray,
    ms::NTuple{D, Int},
    iflag::Int,
) where {FT, D}
    axes = FlowGeometries.Grids.coordinates(g)
    ss = size(g)
    Npts = prod(ss)
    M = prod(ms)
    B = length(out) ÷ Npts
    ks = Grids.physical_wavenumbers(g, ms)
    fill!(out, zero(Complex{FT}))
    spat = CartesianIndices(ss)
    @inbounds for (pj, P) in enumerate(spat)
        for (mi, I) in enumerate(CartesianIndices(ms))
            phi = _phase_tensor(ks, axes, I, P, FT)
            W = cis(iflag * phi)
            for b in 1:B
                out[pj + (b - 1) * Npts] += coeffs[mi + (b - 1) * M] * W
            end
        end
    end
    return out
end

function _synthesize_cartesian_direct!(
    out::AbstractArray{Complex{FT}},
    g::FlowGeometries.Grids.AbstractUnstructuredGrid{<:FlowGeometries.Geometry.AbstractCartesianGeometry},
    coeffs::AbstractArray,
    ms::NTuple{D, Int},
    iflag::Int,
) where {FT, D}
    coords = FlowGeometries.Grids.coordinates(g)
    N = length(coords[1])
    M = prod(ms)
    B = length(out) ÷ N
    ks = Grids.physical_wavenumbers(g, ms)
    fill!(out, zero(Complex{FT}))
    @inbounds for j in 1:N
        for (mi, I) in enumerate(CartesianIndices(ms))
            phi = _phase_scattered(ks, coords, I, j, FT)
            W = cis(iflag * phi)
            for b in 1:B
                out[j + (b - 1) * N] += coeffs[mi + (b - 1) * M] * W
            end
        end
    end
    return out
end

# =============================================================================
# Spherical forward / inverse (SHT projection / synthesis).
# Spherical grids are small (N = nlon·nlat or the scattered count), so per-point (θ, φ, weight) lists
# are materialized once — this is NOT the ∏N_d Cartesian blob, just a modest point list. θ/φ come from
# the FlowGeometries adapter's convention bridge (θ = colatitude, φ = longitude).
# =============================================================================

# Structured `(nlon, nlat)` grid: point p = iλ + (jφ-1)·nlon (longitude fastest, matching the field
# layout), weight w_lat[jφ]·(2π/nlon) when a quadrature `sampling`/`weights` is supplied, else uniform.
function _sph_point_data(
    g::FlowGeometries.Grids.AbstractStructuredGrid{<:FlowGeometries.Geometry.AbstractSphericalGeometry},
    ::Type{FT}; sampling = nothing, weights = nothing) where {FT}
    θpt, φpt = Grids._sph_points(g)
    nlon = length(FlowGeometries.Grids.coordinates(g, 1))
    nlat = length(FlowGeometries.Grids.coordinates(g, 2))
    N = nlon * nlat
    wlat = Grids._sht_weights(g, nlat; sampling = sampling, weights = weights)
    dλ = FT(2π) / nlon
    wpt = Vector{FT}(undef, N)
    @inbounds for p in 1:N
        jφ = ((p - 1) ÷ nlon) + 1
        wpt[p] = wlat === nothing ? FT(4π) / N : FT(wlat[jφ]) * dλ
    end
    return FT.(θpt), FT.(φpt), wpt
end

# Scattered points: per-node (θ, φ); uniform weight unless an explicit per-node `weights` is supplied.
function _sph_point_data(
    g::FlowGeometries.Grids.AbstractUnstructuredGrid{<:FlowGeometries.Geometry.AbstractSphericalGeometry},
    ::Type{FT}; sampling = nothing, weights = nothing) where {FT}
    θpt, φpt = Grids._sph_points(g)
    N = length(θpt)
    wpt = weights === nothing ? fill(FT(4π) / N, N) : FT.(weights)
    return FT.(θpt), FT.(φpt), wpt
end

# Real spherical harmonic value in the FastSphericalHarmonics convention (verified to match
# `FSH.sph_evaluate` to round-off): Y_lm = s(m)·P̄_l^|m|·trig, with s(0)=1, s(m≠0)=(-1)^|m|√2, and
# trig = cos(mφ) for m≥0, sin(|m|φ) for m<0. `Plm` holds the normalized associated Legendre
# P̄_l^|m|(cosθ). Coefficients are real (stored in the complex array with zero imaginary part).
@inline function _real_sph(Plm::AbstractMatrix{FT}, l::Int, m::Int, φ::FT) where {FT}
    abs_m = abs(m)
    P = Plm[l+1, abs_m+1]
    m == 0 && return P
    s = isodd(abs_m) ? -sqrt(FT(2)) : sqrt(FT(2))
    return s * P * (m > 0 ? cos(m * φ) : sin(abs_m * φ))
end

function _calculate_spectrum_spherical_direct!(
    coeffs::AbstractArray{Complex{FT}},
    g::FlowGeometries.Grids.AbstractGrid{<:FlowGeometries.Geometry.AbstractSphericalGeometry},
    field::AbstractArray,
    lmax::Int; sampling = nothing, weights = nothing,
) where {FT}
    θpt, φpt, wpt = _sph_point_data(g, FT; sampling = sampling, weights = weights)
    N = length(θpt)
    Nθc = lmax + 1
    Nφc = 2 * lmax + 1
    B = length(coeffs) ÷ (Nθc * Nφc)
    F = reshape(field, N, B)
    C = reshape(coeffs, Nθc, Nφc, B)
    fill!(coeffs, zero(Complex{FT}))
    tables = SphericalKernels.legendre_tables(FT, lmax)
    Plm = Matrix{FT}(undef, lmax + 1, lmax + 1)
    @inbounds for p in 1:N
        xj = cos(θpt[p])
        sj = sin(θpt[p])
        φp = φpt[p]
        wp = wpt[p]
        SphericalKernels.fill_legendre!(Plm, tables, xj, sj, lmax)
        for l in 0:lmax
            for m in -l:l
                Ylm = _real_sph(Plm, l, m, φp)          # real SH (FSH convention)
                idx = sph_mode_index(l, m)
                gw = Ylm * wp
                for b in 1:B
                    C[idx, b] += F[p, b] * gw
                end
            end
        end
    end
    return (0:lmax, -lmax:lmax)
end

function _synthesize_spherical_direct!(
    out::AbstractArray{Complex{FT}},
    g::FlowGeometries.Grids.AbstractGrid{<:FlowGeometries.Geometry.AbstractSphericalGeometry},
    coeffs::AbstractArray,
    lmax::Int,
) where {FT}
    θpt, φpt, _ = _sph_point_data(g, FT)
    N = length(θpt)
    Nθc = lmax + 1
    Nφc = 2 * lmax + 1
    B = length(out) ÷ N
    O = reshape(out, N, B)
    C = reshape(coeffs, Nθc, Nφc, B)
    fill!(O, zero(Complex{FT}))
    tables = SphericalKernels.legendre_tables(FT, lmax)
    Plm = Matrix{FT}(undef, lmax + 1, lmax + 1)
    @inbounds for p in 1:N
        xj = cos(θpt[p])
        sj = sin(θpt[p])
        φp = φpt[p]
        SphericalKernels.fill_legendre!(Plm, tables, xj, sj, lmax)
        for l in 0:lmax
            for m in -l:l
                Ylm = _real_sph(Plm, l, m, φp)          # real SH (FSH convention)
                idx = sph_mode_index(l, m)
                for b in 1:B
                    O[p, b] += C[idx, b] * Ylm
                end
            end
        end
    end
    return out
end

end # module DirectSum
