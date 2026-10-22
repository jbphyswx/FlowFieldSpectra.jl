module Preprocessing

using LinearAlgebra: SymTridiagonal, eigen, pinv

export AbstractWindow, NoWindow, Hann, Hamming, Blackman, Tukey,
    AbstractDetrend, NoDetrend, Demean, LinearDetrend,
    Preprocess, window_value, window_function, window_function!, window_correction, detrend!, dpss,
    is_identity, axis_taper, point_window, apply_window!, detrend_spatial!

# =============================================================================
# Window tapers
# =============================================================================

"""
    AbstractWindow

Supertype for apodization windows applied before transforming. Concrete windows dispatch
[`window_value`](@ref), the window at a fractional position `t ∈ [0, 1]` across the domain's extent in one
direction; a field is multiplied by the product over directions at each point's position
([`point_window`](@ref)). Reduces spectral leakage for non-periodic data.
"""
abstract type AbstractWindow end

"""`NoWindow()` — rectangular window (all ones); no apodization."""
struct NoWindow <: AbstractWindow end

"""`Hann()` — Hann (raised-cosine) taper."""
struct Hann <: AbstractWindow end

"""`Hamming()` — Hamming taper."""
struct Hamming <: AbstractWindow end

"""`Blackman()` — Blackman taper."""
struct Blackman <: AbstractWindow end

"""
    Tukey(alpha=0.5)

Tukey (tapered-cosine) window with taper fraction `alpha ∈ [0,1]`; `alpha=0` is rectangular
and `alpha=1` is Hann.
"""
struct Tukey{T<:Real} <: AbstractWindow
    alpha::T
end
Tukey() = Tukey(0.5)

"""
    window_value(win::AbstractWindow, t::T) -> T

The taper `win` at the fractional position `t ∈ [0, 1]` along an axis, `0` and `1` its two ends: Hann
`½(1 − cos 2πt)`, Hamming `0.54 − 0.46 cos 2πt`, Blackman `0.42 − ½cos 2πt + 0.08 cos 4πt`, and Tukey(α) a
Hann edge over `t < α/2` and `t > 1 − α/2`, one between.
"""
function window_value end

window_value(::NoWindow, t::T) where {T} = one(T)
window_value(::Hann, t::T) where {T} = T(0.5) * (1 - cospi(2t))
window_value(::Hamming, t::T) where {T} = T(0.54) - T(0.46) * cospi(2t)
window_value(::Blackman, t::T) where {T} = T(0.42) - T(0.5) * cospi(2t) + T(0.08) * cospi(4t)
function window_value(win::Tukey, t::T) where {T}
    α = T(win.alpha)
    α <= 0 && return one(T)
    α >= 1 && return window_value(Hann(), t)
    s = min(t, 1 - t)
    return s < α / 2 ? T(0.5) * (1 + cospi(2s / α - 1)) : one(T)
end

"""
    window_function!(w::AbstractVector, win::AbstractWindow) -> w

Fill `w` (length `n`) in place with the taper `win` at the `n` evenly spaced positions
`t = (i − 1)/(n − 1)`; a single sample takes one.
"""
function window_function!(w::AbstractVector{T}, win::AbstractWindow) where {T}
    n = length(w)
    n == 1 && return fill!(w, one(T))
    @inbounds for i in 1:n
        w[i] = window_value(win, T(i - 1) / (n - 1))
    end
    return w
end

"""
    window_function(win::AbstractWindow, n::Integer, ::Type{T}=Float64) -> Vector{T}

Allocate and return the length-`n` taper `win`.
"""
function window_function(win::AbstractWindow, n::Integer, ::Type{T} = Float64) where {T}
    n <= 0 && return T[]
    return window_function!(Vector{T}(undef, n), win)
end

"""
    window_correction(w::AbstractVector) -> (S1, S2)

Coherent-gain factor `S1 = (Σ w)/n` and power factor `S2 = (Σ w²)/n`. Amplitude spectra
divide by `S1`; power/energy spectra divide by `S2` to preserve variance.
"""
function window_correction(w::AbstractVector{T}) where {T}
    n = length(w)
    n == 0 && return (one(T), one(T))
    return (sum(w) / n, sum(abs2, w) / n)
end

# =============================================================================
# Detrending
# =============================================================================

"""
    AbstractDetrend

Supertype for detrending operations applied before transforming. Concrete subtypes dispatch
`detrend!`.
"""
abstract type AbstractDetrend end

"""`NoDetrend()` — leave the data unchanged."""
struct NoDetrend <: AbstractDetrend end

"""`Demean()` — subtract the mean (remove the DC component). The default."""
struct Demean <: AbstractDetrend end

"""`LinearDetrend()` — subtract the least-squares linear trend."""
struct LinearDetrend <: AbstractDetrend end

"""
    detrend!(x::AbstractVector, d::AbstractDetrend) -> x

Detrend `x` in place according to `d`.
"""
function detrend! end

detrend!(x::AbstractVector, ::NoDetrend) = x

function detrend!(x::AbstractVector{T}, ::Demean) where {T}
    m = sum(x) / length(x)
    @inbounds @. x -= m
    return x
end

function detrend!(x::AbstractVector{T}, ::LinearDetrend) where {T}
    n = length(x)
    n < 2 && return x
    t̄ = T(n - 1) / 2
    x̄ = sum(x) / n
    sxt = zero(T)
    stt = zero(T)
    @inbounds for i in 1:n
        dt = T(i - 1) - t̄
        sxt += dt * x[i]
        stt += dt * dt
    end
    slope = stt == 0 ? zero(T) : sxt / stt
    intercept = x̄ - slope * t̄
    @inbounds for i in 1:n
        x[i] -= intercept + slope * T(i - 1)
    end
    return x
end

# =============================================================================
# Preprocess spec
# =============================================================================

"""
    Preprocess(; detrend=Demean(), window=NoWindow(), pad=1.0)

Preprocessing applied to a field before transforming.

- `detrend::AbstractDetrend`: `Demean()` (default), `NoDetrend()`, or `LinearDetrend()`.
- `window::AbstractWindow`: `NoWindow()` (default), `Hann()`, `Hamming()`, `Blackman()`, `Tukey(α)`.
- `pad::Float64`: zero-padding factor (`≥ 1`); `1.0` means none.

A window is scaled to unit mean square under the grid's measure ([`point_window`](@ref)), so the
windowed field keeps the variance of the original.
"""
struct Preprocess{D<:AbstractDetrend, W<:AbstractWindow}
    detrend::D
    window::W
    pad::Float64
end

function Preprocess(; detrend::AbstractDetrend = Demean(), window::AbstractWindow = NoWindow(),
        pad::Real = 1.0)
    pad >= 1 || throw(ArgumentError("pad must be ≥ 1 (1.0 means no padding); got $pad"))
    return Preprocess(detrend, window, Float64(pad))
end

"""`is_identity(spec)` — whether `spec` leaves a field untouched, so the transform reads it directly."""
is_identity(p::Preprocess) = p.detrend isa NoDetrend && p.window isa NoWindow && p.pad == 1

# =============================================================================
# Applying a spec to a field
# =============================================================================

"""
    axis_taper(::Type{FT}, window, n) -> Vector{FT}

Length-`n` taper for one axis, scaled to unit mean square (`Σw²/n == 1`).

Under that scaling Parseval holds on the tapered coefficients: with `C_w(k) = (1/N) Σ w f e^{-ikx}`, the
folded sum `Σ_k |C_w|²` is `mean|w·f|²`, which returns `mean|f|²` when `Σw²/n = 1` on every axis (the
tensor product's mean square is the product of the axes'). `NoWindow` is an exact no-op.
"""
function axis_taper(::Type{FT}, window::AbstractWindow, n::Integer) where {FT}
    w = window_function(window, n, FT)
    ms = sum(abs2, w) / n
    ms > 0 || throw(ArgumentError("$(nameof(typeof(window))) has zero energy over $n samples"))
    return w .* inv(sqrt(ms))
end

"""
    point_window(::Type{FT}, window, xs, w) -> Vector{FT}

The window at points with coordinates `xs` (one vector per direction, one entry per point): the product
over directions of [`window_value`](@ref) at `t_d = (x_d − min x_d)/(max x_d − min x_d)`, scaled to unit
mean square under the point weights `w`, `Σ w τ² / Σ w = 1`. On a uniform lattice of equal weights this
is the tensor product of the [`axis_taper`](@ref)s.
"""
function point_window(::Type{FT}, window::AbstractWindow, xs::Tuple, w) where {FT}
    N = length(first(xs))
    τ = ones(FT, N)
    for x in xs
        lo, hi = extrema(x)
        hi > lo || continue
        @inbounds for j in 1:N
            τ[j] *= window_value(window, FT((x[j] - lo) / (hi - lo)))
        end
    end
    ms = sum(j -> FT(w[j]) * τ[j]^2, 1:N) / sum(j -> FT(w[j]), 1:N)
    ms > 0 || throw(ArgumentError("$(nameof(typeof(window))) has zero energy over these $N points"))
    return τ .* inv(sqrt(ms))
end

"""
    apply_window!(field, τ, nsp)

Multiply every batch slice of `field` point by point by the window `τ` of [`point_window`](@ref), one
entry per spatial point of the leading `nsp` dims in column-major order.
"""
function apply_window!(field::AbstractArray, τ::AbstractVector, nsp::Int)
    N = prod(ntuple(d -> size(field, d), nsp))
    length(τ) == N || throw(DimensionMismatch("a window of $(length(τ)) entries for $N points"))
    B = length(field) ÷ N
    @inbounds for b in 1:B, j in 1:N
        field[(b - 1) * N + j] *= τ[j]
    end
    return field
end

"""
    detrend_spatial!(field, ::Demean, nsp, w)
    detrend_spatial!(field, ::LinearDetrend, nsp, w, xs)

Detrend the leading `nsp` spatial dims of `field` in place, per batch slice, under the point weights `w`
(one per spatial point in column-major order). A point of zero weight is left out of every fit, the
`NaN` masked data commonly holds included.

`Demean` subtracts the weighted mean `Σ w f / Σ w`. `LinearDetrend` subtracts the weighted
least-squares fit `a + Σ_d b_d x_d` in the point coordinates `xs` (one vector per direction), which
leaves the residual orthogonal under `w` to the constant and to every coordinate.
"""
detrend_spatial!(field, ::NoDetrend, nsp::Int, args...) = field

function detrend_spatial!(field::AbstractArray{T}, ::Demean, nsp::Int, w) where {T}
    N = prod(ntuple(d -> size(field, d), nsp))
    length(w) == N || throw(DimensionMismatch("$(length(w)) weights for $N points"))
    R = real(float(T))
    B = length(field) ÷ N
    @inbounds for b in 1:B
        off = (b - 1) * N
        s = zero(T)
        W = zero(R)
        for j in 1:N
            wj = R(w[j])
            iszero(wj) && continue
            s += wj * field[off + j]
            W += wj
        end
        m = s / W
        for j in 1:N
            field[off + j] -= m
        end
    end
    return field
end

function detrend_spatial!(field::AbstractArray{T}, ::LinearDetrend, nsp::Int, w, xs::Tuple) where {T}
    N = prod(ntuple(d -> size(field, d), nsp))
    length(w) == N || throw(DimensionMismatch("$(length(w)) weights for $N points"))
    all(x -> length(x) == N, xs) || throw(DimensionMismatch("every coordinate needs one value per point"))
    R = real(float(T))
    K = length(xs)
    # The coordinates centred on their weighted means, so the constant separates from the slopes, and
    # their weighted Gram matrix, shared by every slice.
    W = sum(j -> iszero(w[j]) ? zero(R) : R(w[j]), 1:N)
    x̄ = [sum(j -> iszero(w[j]) ? zero(R) : R(w[j]) * R(x[j]), 1:N) / W for x in xs]
    G = zeros(R, K, K)
    @inbounds for j in 1:N
        iszero(w[j]) && continue
        for a in 1:K, c in 1:K
            G[a, c] += R(w[j]) * (R(xs[a][j]) - x̄[a]) * (R(xs[c][j]) - x̄[c])
        end
    end
    Gp = pinv(G)
    B = length(field) ÷ N
    r = zeros(T, K)
    @inbounds for b in 1:B
        off = (b - 1) * N
        fill!(r, zero(T))
        s = zero(T)
        for j in 1:N
            iszero(w[j]) && continue
            v = R(w[j]) * field[off + j]
            s += v
            for a in 1:K
                r[a] += v * (R(xs[a][j]) - x̄[a])
            end
        end
        m = s / W
        slope = Gp * r
        for j in 1:N
            t = m
            for a in 1:K
                t += slope[a] * (R(xs[a][j]) - x̄[a])
            end
            field[off + j] -= t
        end
    end
    return field
end

# =============================================================================
# Multitaper: discrete prolate spheroidal sequences (Slepian tapers)
# =============================================================================

"""
    dpss(N::Integer, NW::Real, K::Integer = max(1, floor(Int, 2NW) - 1); T = Float64) -> Matrix{T}

Discrete prolate spheroidal sequences (Slepian tapers): the `K` length-`N` sequences with maximal
spectral concentration in the half-bandwidth `W = NW/N`, returned as an `N×K` matrix whose column `k` is
the order-`(k-1)` taper. The columns are orthogonal with unit mean square (`V'V = N·I`), the scaling
[`axis_taper`](@ref) gives a window, so a tapered copy's periodogram has the untapered level. `NW` is the
time–bandwidth product (typical `2.5`–`4`); `K ≈ 2·NW − 1` tapers are usefully concentrated.

The tapers are the eigenvectors of a symmetric tridiagonal matrix (Percival & Walden). A multitaper
estimate transforms the `K` tapered copies as a batch and averages their periodograms with
[`welch_power_spectrum`](@ref).
"""
function dpss(N::Integer, NW::Real, K::Integer = max(1, floor(Int, 2 * NW) - 1);
        T::Type = Float64)
    N >= 2 || throw(ArgumentError("N must be ≥ 2"))
    1 <= K <= N || throw(ArgumentError("need 1 ≤ K ≤ N"))
    W = T(NW) / N
    diag = T[((N - 1 - 2 * n) / 2)^2 * cospi(2W) for n in 0:(N-1)]
    offd = T[(n * (N - n)) / 2 for n in 1:(N-1)]
    F = eigen(SymTridiagonal(diag, offd))
    # eigenvalues ascending → DPSS orders 0..K-1 are the K largest (take last K, reverse).
    V = Matrix{T}(F.vectors[:, (N-K+1):N][:, end:-1:1])
    # Sign convention: an even order has a positive sum, an odd order a positive first moment about the
    # centre.
    @inbounds for k in 1:K
        v = @view V[:, k]
        s = isodd(k) ? sum(v) : sum(j -> (j - (N + 1) / 2) * v[j], 1:N)
        s < 0 && (v .*= -one(T))
    end
    V .*= sqrt(T(N))
    return V
end

end # module Preprocessing
