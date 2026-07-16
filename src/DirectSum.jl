module DirectSum

using ..Grids: Grids, AbstractCartesianGrid, UniformCartesianGrid, NonuniformCartesianGrid,
    ScatteredCartesianGrid, AbstractSphericalGrid, StructuredSphericalGrid, ScatteredSphericalGrid,
    physical_wavenumbers
using ..SphericalKernels: SphericalKernels

export sph_mode_index

# The direct-sum (DFT / SHT-projection) reference transform. Dependency-free, `O(∏ms · Npts · ∏batch)`
# — correct for any grid, never FFT-competitive by design. It reads a field tensor `(spatial…, batch…)`
# and writes coefficients `(spectral…, batch…)`; the batch is the innermost, contiguous loop. Tensor
# grids read coordinates through their 1-D axes (`axes[d][I[d]]`) — NO per-point coordinate blob is
# materialized; scattered grids read their per-point vectors (`coords[d][j]`).

"""
    sph_mode_index(l, m) -> CartesianIndex

`CartesianIndex` of degree `l`, order `m` in the `(lmax+1, 2lmax+1)` coefficient array.
"""
@inline function sph_mode_index(l::Int, m::Int)
    row = l - abs(m) + 1
    col = m == 0 ? 1 : (m < 0 ? 2 * abs(m) : 2 * m + 1)
    return CartesianIndex(row, col)
end

# =============================================================================
# Cartesian forward (analysis):  C[k, b] = (1/Npts) Σ_p f[p, b] · exp(-iflag · i · k·x_p)
# =============================================================================

# Tensor-product grid (uniform or nonuniform): spatial point `P` is a CartesianIndex over the axis
# lengths; its coordinate along `d` is `axes[d][P[d]]` (no materialization).
function _calculate_spectrum_cartesian_direct!(
    coeffs::AbstractArray{Complex{FT}},
    g::Union{UniformCartesianGrid{FT, D}, NonuniformCartesianGrid{FT, D}},
    field::AbstractArray,
    ms::NTuple{D, Int},
    iflag::Int,
) where {FT, D}
    axes = g.axes
    ss = map(length, axes)
    Npts = prod(ss)
    M = prod(ms)
    B = length(coeffs) ÷ M
    ks = physical_wavenumbers(g.domain_size, ms, FT)
    F = reshape(field, Npts, B)
    C = reshape(coeffs, M, B)
    fill!(C, zero(Complex{FT}))
    spat = CartesianIndices(ss)
    @inbounds for (mi, I) in enumerate(CartesianIndices(ms))
        for (pj, P) in enumerate(spat)
            phi = zero(FT)
            for d in 1:D
                phi += ks[d][I[d]] * FT(axes[d][P[d]])
            end
            W = cis(-iflag * phi)
            for b in 1:B
                C[mi, b] += F[pj, b] * W
            end
        end
    end
    C ./= Npts
    return ks
end

# Scattered point cloud: spatial point `j` reads `coords[d][j]`.
function _calculate_spectrum_cartesian_direct!(
    coeffs::AbstractArray{Complex{FT}},
    g::ScatteredCartesianGrid{FT, D},
    field::AbstractArray,
    ms::NTuple{D, Int},
    iflag::Int,
) where {FT, D}
    coords = g.coords
    N = length(coords[1])
    M = prod(ms)
    B = length(coeffs) ÷ M
    ks = physical_wavenumbers(g.domain_size, ms, FT)
    F = reshape(field, N, B)
    C = reshape(coeffs, M, B)
    fill!(C, zero(Complex{FT}))
    @inbounds for (mi, I) in enumerate(CartesianIndices(ms))
        for j in 1:N
            phi = zero(FT)
            for d in 1:D
                phi += ks[d][I[d]] * FT(coords[d][j])
            end
            W = cis(-iflag * phi)
            for b in 1:B
                C[mi, b] += F[j, b] * W
            end
        end
    end
    C ./= N
    return ks
end

# =============================================================================
# Cartesian inverse (synthesis):  f[p, b] = Σ_k C[k, b] · exp(+iflag · i · k·x_p)
# =============================================================================

function _synthesize_cartesian_direct!(
    out::AbstractArray{Complex{FT}},
    g::Union{UniformCartesianGrid{FT, D}, NonuniformCartesianGrid{FT, D}},
    coeffs::AbstractArray,
    ms::NTuple{D, Int},
    iflag::Int,
) where {FT, D}
    axes = g.axes
    ss = map(length, axes)
    Npts = prod(ss)
    M = prod(ms)
    B = length(out) ÷ Npts
    ks = physical_wavenumbers(g.domain_size, ms, FT)
    O = reshape(out, Npts, B)
    C = reshape(coeffs, M, B)
    fill!(O, zero(Complex{FT}))
    spat = CartesianIndices(ss)
    @inbounds for (pj, P) in enumerate(spat)
        for (mi, I) in enumerate(CartesianIndices(ms))
            phi = zero(FT)
            for d in 1:D
                phi += ks[d][I[d]] * FT(axes[d][P[d]])
            end
            W = cis(iflag * phi)
            for b in 1:B
                O[pj, b] += C[mi, b] * W
            end
        end
    end
    return out
end

function _synthesize_cartesian_direct!(
    out::AbstractArray{Complex{FT}},
    g::ScatteredCartesianGrid{FT, D},
    coeffs::AbstractArray,
    ms::NTuple{D, Int},
    iflag::Int,
) where {FT, D}
    coords = g.coords
    N = length(coords[1])
    M = prod(ms)
    B = length(out) ÷ N
    ks = physical_wavenumbers(g.domain_size, ms, FT)
    O = reshape(out, N, B)
    C = reshape(coeffs, M, B)
    fill!(O, zero(Complex{FT}))
    @inbounds for j in 1:N
        for (mi, I) in enumerate(CartesianIndices(ms))
            phi = zero(FT)
            for d in 1:D
                phi += ks[d][I[d]] * FT(coords[d][j])
            end
            W = cis(iflag * phi)
            for b in 1:B
                O[j, b] += C[mi, b] * W
            end
        end
    end
    return out
end

# =============================================================================
# Spherical forward / inverse (SHT projection / synthesis).
# Spherical grids are small (N = Nθ·Nφ or the scattered count), so per-point (θ, φ, weight) lists are
# materialized once — this is NOT the ∏N_d Cartesian blob, just a modest point list.
# =============================================================================

# Per-point (θ, φ, weight) lists, in the SAME column-major order as a reshaped field:
#   structured (Nθ, Nφ): point p = iθ + (iφ-1)·Nθ, weight w_θ[iθ]·(2π/Nφ) or uniform 4π/N;
#   scattered:           point k, weight weights[k] or uniform 4π/N.
function _sph_point_data(g::StructuredSphericalGrid{FT}) where {FT}
    Nθ = length(g.θ)
    Nφ = length(g.φ)
    N = Nθ * Nφ
    θpt = Vector{FT}(undef, N)
    φpt = Vector{FT}(undef, N)
    wpt = Vector{FT}(undef, N)
    dφ = FT(2π) / Nφ
    @inbounds for iφ in 1:Nφ, iθ in 1:Nθ
        p = iθ + (iφ - 1) * Nθ
        θpt[p] = FT(g.θ[iθ])
        φpt[p] = FT(g.φ[iφ])
        wpt[p] = g.weights === nothing ? FT(4π) / N : FT(g.weights[iθ]) * dφ
    end
    return θpt, φpt, wpt
end

function _sph_point_data(g::ScatteredSphericalGrid{FT}) where {FT}
    θ = g.coords[1]
    φ = g.coords[2]
    N = length(θ)
    θpt = FT.(θ)
    φpt = FT.(φ)
    wpt = g.weights === nothing ? fill(FT(4π) / N, N) : FT.(g.weights)
    return θpt, φpt, wpt
end

function _calculate_spectrum_spherical_direct!(
    coeffs::AbstractArray{Complex{FT}},
    g::AbstractSphericalGrid{FT},
    field::AbstractArray,
    lmax::Int,
) where {FT}
    θpt, φpt, wpt = _sph_point_data(g)
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
                abs_m = abs(m)
                factor = (m < 0 && isodd(abs_m)) ? -one(FT) : one(FT)
                Ylm = factor * Plm[l+1, abs_m+1] * cis(m * φp)
                idx = sph_mode_index(l, m)
                gw = conj(Ylm) * wp
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
    g::AbstractSphericalGrid{FT},
    coeffs::AbstractArray,
    lmax::Int,
) where {FT}
    θpt, φpt, _ = _sph_point_data(g)
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
                abs_m = abs(m)
                factor = (m < 0 && isodd(abs_m)) ? -one(FT) : one(FT)
                Ylm = factor * Plm[l+1, abs_m+1] * cis(m * φp)
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
