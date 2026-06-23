module Normalization

export AbstractSidedness, OneSided, TwoSided,
    AbstractScaling, DensityScaling, PowerScaling,
    SpectralConvention, sided_factor

# =============================================================================
# Sidedness (typed, not Symbol)
# =============================================================================

"""
    AbstractSidedness

Whether a spectrum keeps both signs of wavenumber (`TwoSided`) or folds negatives onto
positives (`OneSided`). Dispatches `sided_factor`.
"""
abstract type AbstractSidedness end

"""`TwoSided()` — keep ± wavenumbers (no folding)."""
struct TwoSided <: AbstractSidedness end

"""`OneSided()` — fold negative wavenumbers onto positives (doubles interior bins). The usual
convention for real fields."""
struct OneSided <: AbstractSidedness end

"""
    sided_factor(s::AbstractSidedness, k, kmax) -> Real

Folding multiplier. `TwoSided` → `1` everywhere. `OneSided` → `2` for interior wavenumbers,
`1` at DC (`k≈0`) and Nyquist (`k≈kmax`) which have no negative-frequency partner.
"""
@inline sided_factor(::TwoSided, k::T, kmax::T) where {T} = one(T)
@inline function sided_factor(::OneSided, k::T, kmax::T) where {T}
    (k <= eps(T) || k >= kmax - eps(T)) && return one(T)
    return T(2)
end

# =============================================================================
# Scaling (density vs power; typed)
# =============================================================================

"""
    AbstractScaling

Whether a reduced spectrum is reported as a spectral *density* (`DensityScaling`, divided by the
bin width so `∫E dk` recovers variance) or as per-bin/per-mode `PowerScaling`.
"""
abstract type AbstractScaling end

"""`DensityScaling()` — spectral density (per unit wavenumber); `∫E dk = Var(f)`."""
struct DensityScaling <: AbstractScaling end

"""`PowerScaling()` — per-bin/per-mode power (no `dk` division)."""
struct PowerScaling <: AbstractScaling end

# =============================================================================
# Convention object
# =============================================================================

"""
    SpectralConvention(; sided=OneSided(), scaling=DensityScaling(), parseval_check=false)

Convention governing how spectral coefficients become reported spectra, so the package never
silently guesses normalization. Fields are typed for compile-time dispatch.

- `sided::AbstractSidedness`: `OneSided()` (default) or `TwoSided()`.
- `scaling::AbstractScaling`: `DensityScaling()` (default) or `PowerScaling()`.
- `parseval_check::Bool`: when `true`, callers assert `Σ E·Δk ≈ Var(field)` (after mean
  removal) as a correctness self-test.

The variance-preservation property — `∫ E(k) dk = Var(f)` after demeaning — is the invariant
the test suite enforces across every backend and grid.
"""
struct SpectralConvention{S<:AbstractSidedness, C<:AbstractScaling}
    sided::S
    scaling::C
    parseval_check::Bool
end

function SpectralConvention(; sided::AbstractSidedness = OneSided(),
        scaling::AbstractScaling = DensityScaling(), parseval_check::Bool = false)
    return SpectralConvention(sided, scaling, parseval_check)
end

end # module Normalization
