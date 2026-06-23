module SphericalKernels

export LegendreTables, legendre_tables, fill_legendre!, normalized_legendre

"""
    LegendreTables{FT}

Precomputed, point-independent recurrence coefficients for the normalized associated Legendre
functions ``\\bar P_\\ell^m`` up to degree `lmax`. Built once per transform via
[`legendre_tables`](@ref); the per-point table is then filled in `O(lmax²)` with no `sqrt`
calls by [`fill_legendre!`](@ref).
"""
struct LegendreTables{FT}
    lmax::Int
    P00::FT            # \bar P_0^0 = 1/sqrt(4π)
    c::Vector{FT}      # sectoral: \bar P_m^m = \bar P_{m-1}^{m-1} * c[m] * sinθ   (m = 1:lmax)
    d::Vector{FT}      # first step: \bar P_{m+1}^m = x * d[m+1] * \bar P_m^m       (m = 0:lmax-1)
    a::Matrix{FT}      # upward recurrence coefficient a[l+1, m+1]
    b::Matrix{FT}      # upward recurrence coefficient b[l+1, m+1]
end

"""
    legendre_tables(::Type{FT}, lmax::Int) -> LegendreTables{FT}

Precompute the recurrence coefficients up to degree `lmax`.
"""
function legendre_tables(::Type{FT}, lmax::Int) where {FT}
    c = Vector{FT}(undef, max(lmax, 0))
    @inbounds for m in 1:lmax
        c[m] = -sqrt(FT(2m + 1) / FT(2m))
    end
    d = Vector{FT}(undef, max(lmax, 0))
    @inbounds for m in 0:(lmax-1)
        d[m+1] = sqrt(FT(2m + 3))
    end
    a = zeros(FT, lmax + 1, lmax + 1)
    b = zeros(FT, lmax + 1, lmax + 1)
    @inbounds for m in 0:lmax
        for l in (m+2):lmax
            a[l+1, m+1] = sqrt(FT(4l^2 - 1) / FT(l^2 - m^2))
            b[l+1, m+1] = sqrt(FT(2l + 1) * FT((l - 1)^2 - m^2) / (FT(2l - 3) * FT(l^2 - m^2)))
        end
    end
    return LegendreTables{FT}(lmax, one(FT) / sqrt(FT(4π)), c, d, a, b)
end

"""
    fill_legendre!(Plm::AbstractMatrix, t::LegendreTables, x, s, lmax)

Fill `Plm[l+1, m+1] = \\bar P_\\ell^m(x)` for `m = 0:lmax`, `l = m:lmax` at a single point with
`x = cosθ`, `s = sinθ`, reusing the precomputed coefficients in `t`. Entries with `l < m` are
left untouched (the projection never reads them).
"""
function fill_legendre!(Plm::AbstractMatrix{FT}, t::LegendreTables{FT}, x::FT, s::FT, lmax::Int) where {FT}
    @inbounds begin
        Plm[1, 1] = t.P00
        # Sectoral diagonal \bar P_m^m
        for m in 1:lmax
            Plm[m+1, m+1] = Plm[m, m] * t.c[m] * s
        end
        # Each fixed-m column swept upward in l
        for m in 0:lmax
            if m + 1 <= lmax
                Plm[m+2, m+1] = x * t.d[m+1] * Plm[m+1, m+1]
            end
            for l in (m+2):lmax
                Plm[l+1, m+1] = x * t.a[l+1, m+1] * Plm[l, m+1] - t.b[l+1, m+1] * Plm[l-1, m+1]
            end
        end
    end
    return Plm
end

"""
    normalized_legendre(l, m, x, s) -> FT

Single normalized associated Legendre value ``\\bar P_\\ell^m(x)`` (`m ≥ 0`) computed by
on-the-fly recurrence. Reference implementation used for validation; the hot path uses
[`fill_legendre!`](@ref).
"""
@inline function normalized_legendre(l::Int, m::Int, x::FT, s::FT)::FT where {FT}
    m > l && return zero(FT)
    P_mm = one(FT) / sqrt(FT(4π))
    @inbounds for mm in 1:m
        P_mm *= -sqrt(FT(2mm + 1) / FT(2mm)) * s
    end
    l == m && return P_mm
    P_lm = x * sqrt(FT(2m + 3)) * P_mm
    P_lminus1_m = P_mm
    l == m + 1 && return P_lm
    @inbounds for ll in (m+2):l
        coeff1 = sqrt(FT(4ll^2 - 1) / FT(ll^2 - m^2))
        coeff2 = sqrt(FT(2ll + 1) * FT((ll - 1)^2 - m^2) / (FT(2ll - 3) * FT(ll^2 - m^2)))
        P_lminus1_m, P_lm = P_lm, x * coeff1 * P_lm - coeff2 * P_lminus1_m
    end
    return P_lm
end

end # module SphericalKernels
