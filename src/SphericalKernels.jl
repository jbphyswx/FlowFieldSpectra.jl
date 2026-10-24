module SphericalKernels

export LegendreTables, legendre_tables, fill_legendre!, normalized_legendre, legendre_index, legendre_count,
    device_legendre_table, legendre_start, legendre_order!

# The normalized associated Legendre functions are `P̄_ℓ^m(cos θ) = √((2ℓ+1)/4π) d^ℓ_{m0}(θ)` (with the
# Condon–Shortley phase), and `d^ℓ_{m0}` runs up the degrees by
#
#     d^{ℓ+1} = α x d^ℓ − γ d^{ℓ−1},    α = (2ℓ+1)/R,  γ = r/R,  r = √(ℓ²−m²),  R = √((ℓ+1)²−m²),
#
# from `d^m_{m0} = (−1)^m √(C(2m,m)) (sin θ / 2)^m`. In this form a rounding error `ε` in `αx` moves the angle
# the recurrence oscillates at by `ε / sin θ`. It instead carries `e_ℓ = y_ℓ − σ y_{ℓ−1}` with
# `u = 1 − σx = 1 − |x|`, `σ = sign x` (Reinsch's modification):
#
#     e_{ℓ+1} = σ(c − αu) y_ℓ + σγ e_ℓ,    y_{ℓ+1} = σ y_ℓ + e_{ℓ+1},    c = α − 1 − γ,
#
# `c = (m²/R)(1/(ℓ+r) + 1/(ℓ+1+R))` a sum of positive terms, so the angle moves by `ε tan(θ/2)` per step
# and the values carry an error of order `ℓ eps`. `d^m_{m0}` falls below the smallest float for high
# orders near the poles, so a value is held as `y·2^{Kk}` with an integer scale `k ≤ 0`, rescaled as it
# grows, and stored once `k ≥ −1`; a value left at zero is below `2^{−3K/2}`.

# The scale step `K`: half the exponent range of `T`, at most 256.
@inline _span(::Type{T}) where {T<:AbstractFloat} = min(256, exponent(floatmax(T)) ÷ 2)

"""
    LegendreTables{FT}

The step coefficients of `d^ℓ_{m0}` up to degree `lmax`, `α`, `c` and `γ` at `[ℓ+1, m+1]`, the
normalizations `nrm[ℓ+1] = √((2ℓ+1)/4π)`, and `lD[m+1] = log2 √(C(2m,m)) − m`, so that
`|d^m_{m0}(θ)| = 2^{lD + m log2 sin θ}`. Built once per transform by [`legendre_tables`](@ref) and read by
[`fill_legendre!`](@ref) and the device table.
"""
struct LegendreTables{FT, V<:AbstractVector{FT}, VS<:AbstractVector, M<:AbstractMatrix{FT}}
    lmax::Int
    nrm::V
    lD::VS
    α::M
    c::M
    γ::M
end

# `(α, c, γ)` of the step from degree `ℓ` at order `m` in `F`.
@inline function _step(::Type{F}, ℓ::Int, m::Int) where {F}
    ℓ == 0 && return (one(F), zero(F), zero(F))
    m2 = F(m)^2
    r = sqrt(max(F(ℓ)^2 - m2, zero(F)))
    R = sqrt(F(ℓ + 1)^2 - m2)
    c = m == 0 ? zero(F) : (m2 / R) * (inv(F(ℓ) + r) + inv(F(ℓ + 1) + R))
    return (F(2ℓ + 1) / R, c, r / R)
end

"""
    legendre_tables(::Type{FT}, lmax::Int) -> LegendreTables{FT}

The step coefficients and normalizations up to degree `lmax`, formed in `Float64` at least.
"""
function legendre_tables(::Type{FT}, lmax::Int) where {FT}
    F = promote_type(FT, Float64)
    nrm = [FT(sqrt(F(2ℓ + 1) / (4 * F(π)))) for ℓ in 0:lmax]
    lD = zeros(F, lmax + 1)
    for m in 1:lmax
        lD[m + 1] = lD[m] + log2(F(2m - 1) / F(2m)) / 2
    end
    α = zeros(FT, lmax + 1, lmax + 1); c = zeros(FT, lmax + 1, lmax + 1); γ = zeros(FT, lmax + 1, lmax + 1)
    for m in 0:lmax, ℓ in m:lmax
        a, cc, g = _step(F, ℓ, m)
        α[ℓ + 1, m + 1] = a; c[ℓ + 1, m + 1] = cc; γ[ℓ + 1, m + 1] = g
    end
    return LegendreTables(lmax, nrm, lD, α, c, γ)
end

"""
    legendre_start(FT, lD, m, ls) -> (y, k)

`d^m_{m0}` as `y·2^{Kk}`, `2^{−K/2} < |y| ≤ 2^{K/2}`, from `lD = t.lD[m+1]` and `ls = log2 sin θ`, in the
precision of `lD`; `(0, 0)` where it vanishes.
"""
@inline function legendre_start(::Type{FT}, lD::S, m::Int, ls::S) where {FT,S}
    K = _span(FT)
    ex = m == 0 ? lD : lD + m * ls
    isfinite(ex) || return (zero(FT), Int32(0))
    k = min(0, ceil(Int, (ex - K ÷ 2) / K))
    y = exp2(ex - K * k)
    return (FT(isodd(m) ? -y : y), Int32(k))
end

"""
    legendre_order!(P, ix, α, c, γ, nrm, m, lmax, y, k, pos, u) -> P

`P[ix(ℓ)] = P̄_ℓ^m` for `ℓ = m … lmax` at one node, from the start `(y, k)` of
[`legendre_start`](@ref), on the side `pos = cos θ ≥ 0` with `u = 1 − |cos θ|`, the coefficient tables of
[`LegendreTables`](@ref). The host fill and the device table both run it.
"""
@inline function legendre_order!(P, ix::I, α, c, γ, nrm, m::Int, lmax::Int, y::FT, k::Int32, pos::Bool,
                                 u::FT) where {I, FT}
    K = _span(FT)
    up = ldexp(one(FT), K ÷ 2); dn = ldexp(one(FT), -K)
    σ = ifelse(pos, one(FT), -one(FT))
    e = y
    @inbounds for ℓ in m:lmax
        P[ix(ℓ)] = nrm[ℓ + 1] * (y * ifelse(k == Int32(0), one(FT), ifelse(k == Int32(-1), dn, zero(FT))))
        ℓ == lmax && break
        a = α[ℓ + 1, m + 1]; g = γ[ℓ + 1, m + 1]
        e = muladd(σ * muladd(-a, u, c[ℓ + 1, m + 1]), y, σ * g * e)
        y = muladd(σ, y, e)
        big = (k < Int32(0)) & (abs(y) > up)
        sc = ifelse(big, dn, one(FT))
        y *= sc; e *= sc; k += Int32(big)
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
    ls = log2(abs(eltype(t.lD)(s)))
    @inbounds for m in 0:lmax
        y, k = legendre_start(FT, t.lD[m + 1], m, ls)
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
`backend`, one thread per order and node running [`legendre_order!`](@ref) from starts formed on the
host. The KernelAbstractions extension defines it.
"""
function device_legendre_table end

"""
    normalized_legendre(l, m, x, s) -> FT

Single normalized associated Legendre value ``\\bar P_\\ell^m(x)`` (`m ≥ 0`), `x = cos θ`, `s = sin θ`, by
the recurrence of [`fill_legendre!`](@ref) for one order.
"""
function normalized_legendre(l::Int, m::Int, x::FT, s::FT)::FT where {FT}
    m > l && return zero(FT)
    t = legendre_tables(FT, l)
    P = zeros(FT, 1)
    y, k = legendre_start(FT, t.lD[m + 1], m, log2(abs(eltype(t.lD)(s))))
    legendre_order!(P, _ -> 1, t.α, t.c, t.γ, t.nrm, m, l, y, k, x >= 0, _one_minus_abs(x, s))
    return P[1]
end

end # module SphericalKernels
