module SphericalKernels

using FlowTransformBindings: FlowTransformBindings as FTB

export LegendreTables, legendre_tables, fill_legendre!, normalized_legendre, legendre_index, legendre_count,
    device_legendre_table, legendre_start_step, legendre_order!

# The normalized associated Legendre functions are `P̄_ℓ^m(cos θ) = √((2ℓ+1)/4π) d^ℓ_{m0}(θ)` (with the
# Condon–Shortley phase). `d^ℓ_{m0}` runs up the degrees by FlowTransformBindings' recurrence at `n = 0`
# (`FTB.wigner_d_coeffs`), in Reinsch's form (`FTB.wigner_d_step`), from `d^m_{m0}`, which runs up the
# orders by `d^m_{m0} = −sin θ √((2m−1)/(2m)) d^{m−1}_{m−1,0}` from `d^0_{00} = 1`. Each order step and each
# degree step rounds a bounded number of times, so a value carries an error of order `ℓ eps`. `d^m_{m0}`
# falls below the smallest float for high orders near the poles, so a value is held as `y·2^{Kk}` with an
# integer scale `k ≤ 0`, `K = FTB.wigner_d_span`, rescaled as it grows (`FTB.wigner_d_rescale`), and stored
# once `k ≥ −1`; a value left at zero is below `2^{−3K/2}`.

"""
    LegendreTables{FT}

The step coefficients of `d^ℓ_{m0}` up to degree `lmax` from `FTB.wigner_d_coeffs`, `α`, `c = c₊` and `γ`
at `[ℓ+1, m+1]` (`c₋ = −c₊` at `n = 0`), the normalizations `nrm[ℓ+1] = √((2ℓ+1)/4π)`, and the order
factors `f[m] = √((2m−1)/(2m))`. Built once per transform by [`legendre_tables`](@ref) and read by
[`fill_legendre!`](@ref) and the device table.
"""
struct LegendreTables{FT, V<:AbstractVector{FT}, M<:AbstractMatrix{FT}}
    lmax::Int
    nrm::V
    f::V
    α::M
    c::M
    γ::M
end

"""
    legendre_tables(::Type{FT}, lmax::Int) -> LegendreTables{FT}

The step coefficients, normalizations and order factors up to degree `lmax`, in `FT`.
"""
function legendre_tables(::Type{FT}, lmax::Int) where {FT}
    nrm = [sqrt(FT(2ℓ + 1) / (4 * FT(π))) for ℓ in 0:lmax]
    f = [sqrt(FT(2m - 1) / FT(2m)) for m in 1:lmax]
    α = zeros(FT, lmax + 1, lmax + 1); c = zeros(FT, lmax + 1, lmax + 1); γ = zeros(FT, lmax + 1, lmax + 1)
    for m in 0:lmax, ℓ in m:lmax
        a, _, g, cp, _ = FTB.wigner_d_coeffs(FT, ℓ, m, 0)
        α[ℓ + 1, m + 1] = a; c[ℓ + 1, m + 1] = cp; γ[ℓ + 1, m + 1] = g
    end
    return LegendreTables(lmax, nrm, f, α, c, γ)
end

"""
    legendre_start_step(y, k, g) -> (y, k)

The next order's start from `d^{m−1}_{m−1,0} = y·2^{Kk}`: its product by `g = −sin θ f[m]`, held again as
`y·2^{Kk}` with `2^{−K/2} < |y| ≤ 1`, or `(0, k)` where it vanishes. Starting from `(1, 0)` at `m = 0`.
"""
@inline function legendre_start_step(y::FT, k::Int32, g::FT) where {FT}
    span = FTB.wigner_d_span(FT)
    lo = ldexp(one(FT), -(span ÷ 2))
    up = ldexp(one(FT), span)
    y *= g
    while !iszero(y) && abs(y) <= lo
        y *= up
        k -= Int32(1)
    end
    return y, k
end

"""
    legendre_order!(P, ix, α, c, γ, nrm, m, lmax, y, k, pos, u) -> P

`P[ix(ℓ)] = P̄_ℓ^m` for `ℓ = m … lmax` at one node, from the start `d^m_{m0} = y·2^{Kk}`
([`legendre_start_step`](@ref)), on the side `pos = cos θ ≥ 0` with `u = 1 − |cos θ|`, the coefficient
tables of [`LegendreTables`](@ref). The host fill and the device table both run it.
"""
@inline function legendre_order!(P, ix::I, α, c, γ, nrm, m::Int, lmax::Int, y::FT, k::Int32, pos::Bool,
                                 u::FT) where {I, FT}
    up, dn = FTB.wigner_d_bounds(FT)
    e = y
    @inbounds for ℓ in m:lmax
        P[ix(ℓ)] = nrm[ℓ + 1] * (y * ifelse(k == Int32(0), one(FT), ifelse(k == Int32(-1), dn, zero(FT))))
        ℓ == lmax && break
        cp = c[ℓ + 1, m + 1]
        σα, cσ, σγ, σ = FTB.wigner_d_signed(pos, α[ℓ + 1, m + 1], cp, -cp, γ[ℓ + 1, m + 1])
        y, e = FTB.wigner_d_step(σα, cσ, σγ, σ, u, y, e)
        y, e, k = FTB.wigner_d_rescale(y, e, k, up, dn)
    end
    return P
end

# `1 − |x|` from `x = cos θ` and `s = sin θ`, accurate where it is small.
@inline _one_minus_abs(x, s) = s^2 / (1 + abs(x))

"""
    fill_legendre!(Plm::AbstractMatrix, t::LegendreTables, x, s, lmax)

Fill `Plm[l+1, m+1] = \\bar P_\\ell^m(x)` for `m = 0:lmax`, `l = m:lmax` at a single point with
`x = cosθ`, `s = sinθ`, reusing the precomputed coefficients in `t`. Entries with `l < m` are
left untouched (the projection never reads them).
"""
function fill_legendre!(Plm::AbstractMatrix{FT}, t::LegendreTables{FT}, x::FT, s::FT, lmax::Int) where {FT}
    pos = x >= 0
    u = _one_minus_abs(x, s)
    y = one(FT)
    k = Int32(0)
    @inbounds for m in 0:lmax
        m > 0 && ((y, k) = legendre_start_step(y, k, -s * t.f[m]))
        legendre_order!(Plm, ℓ -> CartesianIndex(ℓ + 1, m + 1), t.α, t.c, t.γ, t.nrm, m, lmax, y, k, pos, u)
    end
    return Plm
end

"""
    legendre_index(l, m, lmax) -> Int

Row of `P̄_ℓ^m` (`0 ≤ m ≤ ℓ ≤ lmax`) in a packed Legendre table, which stores each order's degrees
`ℓ = m:lmax` in turn: `m(lmax + 1) − m(m − 1)/2 + (ℓ − m) + 1`.
"""
@inline legendre_index(l::Int, m::Int, lmax::Int) = m * (lmax + 1) - (m * (m - 1)) ÷ 2 + (l - m) + 1

"""
    legendre_count(lmax) -> Int

Rows of a packed Legendre table, `(lmax + 1)(lmax + 2)/2`.
"""
legendre_count(lmax::Int) = (lmax + 1) * (lmax + 2) ÷ 2

"""
    device_legendre_table(backend, ::Type{FT}, lmax, θ) -> AbstractMatrix{FT}

The packed table `P[legendre_index(ℓ, m, lmax), j] = P̄_ℓ^m(cos θ_j)` on the KernelAbstractions device
`backend`, computed there in `FT`: one thread per node forms the starts by [`legendre_start_step`](@ref),
one thread per order and node runs [`legendre_order!`](@ref). The KernelAbstractions extension defines it.
"""
function device_legendre_table end

"""
    normalized_legendre(l, m, x, s) -> FT

Single normalized associated Legendre value ``\\bar P_\\ell^m(x)`` (`m ≥ 0`), `x = cos θ`, `s = sin θ`, by
the recurrences of [`fill_legendre!`](@ref) for one order.
"""
function normalized_legendre(l::Int, m::Int, x::FT, s::FT)::FT where {FT}
    m > l && return zero(FT)
    t = legendre_tables(FT, l)
    y = one(FT)
    k = Int32(0)
    for j in 1:m
        y, k = legendre_start_step(y, k, -s * t.f[j])
    end
    P = zeros(FT, 1)
    legendre_order!(P, _ -> 1, t.α, t.c, t.γ, t.nrm, m, l, y, k, x >= 0, _one_minus_abs(x, s))
    return P[1]
end

end # module SphericalKernels
