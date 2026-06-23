module Preprocessing

export AbstractWindow, NoWindow, Hann, Hamming, Blackman, Tukey,
    AbstractDetrend, NoDetrend, Demean, LinearDetrend,
    Preprocess, window_function, window_function!, window_correction, detrend!

# =============================================================================
# Window tapers (dispatch on type, not Symbol)
# =============================================================================

"""
    AbstractWindow

Supertype for apodization tapers applied per spectral axis before transforming. Concrete
windows dispatch `window_function!`. Reduces spectral leakage for non-periodic data;
only meaningful on uniform axes.
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
    window_function!(w::AbstractVector, win::AbstractWindow) -> w

Fill `w` (length `n`) in place with the taper `win`.
"""
function window_function! end

function window_function!(w::AbstractVector{T}, ::NoWindow) where {T}
    fill!(w, one(T))
    return w
end

function window_function!(w::AbstractVector{T}, ::Hann) where {T}
    n = length(w)
    n == 1 && return fill!(w, one(T))
    @inbounds for i in 1:n
        w[i] = T(0.5) * (1 - cos(2 * T(π) * (i - 1) / (n - 1)))
    end
    return w
end

function window_function!(w::AbstractVector{T}, ::Hamming) where {T}
    n = length(w)
    n == 1 && return fill!(w, one(T))
    @inbounds for i in 1:n
        w[i] = T(0.54) - T(0.46) * cos(2 * T(π) * (i - 1) / (n - 1))
    end
    return w
end

function window_function!(w::AbstractVector{T}, ::Blackman) where {T}
    n = length(w)
    n == 1 && return fill!(w, one(T))
    @inbounds for i in 1:n
        x = T(π) * (i - 1) / (n - 1)
        w[i] = T(0.42) - T(0.5) * cos(2x) + T(0.08) * cos(4x)
    end
    return w
end

function window_function!(w::AbstractVector{T}, win::Tukey) where {T}
    n = length(w)
    α = T(win.alpha)
    (α <= 0 || n == 1) && return fill!(w, one(T))
    α >= 1 && return window_function!(w, Hann())
    fill!(w, one(T))
    edge = floor(Int, α * (n - 1) / 2)
    @inbounds for i in 0:edge
        x = T(2i) / (α * (n - 1))
        taper = T(0.5) * (1 + cos(T(π) * (x - 1)))
        w[i+1] = taper
        w[n-i] = taper
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
# Detrending (dispatch on type, not Symbol)
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
# Preprocess spec (typed fields → compile-time dispatch downstream)
# =============================================================================

"""
    Preprocess(; detrend=Demean(), window=NoWindow(), pad=1.0)

Preprocessing applied to a field (per spectral axis) before transforming. Fields are typed
(not symbols) so downstream code dispatches at compile time.

- `detrend::AbstractDetrend`: `Demean()` (default), `NoDetrend()`, or `LinearDetrend()`.
- `window::AbstractWindow`: `NoWindow()` (default), `Hann()`, `Hamming()`, `Blackman()`, `Tukey(α)`.
- `pad::Float64`: zero-padding factor (`≥ 1`); `1.0` means none.

Window power/amplitude corrections for variance preservation are applied by the normalization
layer via `window_correction`.
"""
struct Preprocess{D<:AbstractDetrend, W<:AbstractWindow}
    detrend::D
    window::W
    pad::Float64
end

function Preprocess(; detrend::AbstractDetrend = Demean(), window::AbstractWindow = NoWindow(),
        pad::Real = 1.0)
    return Preprocess(detrend, window, Float64(pad))
end

end # module Preprocessing
