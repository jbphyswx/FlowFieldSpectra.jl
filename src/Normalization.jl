module Normalization

export AbstractSidedness, OneSided, TwoSided,
    AbstractScaling, DensityScaling, PowerScaling,
    AbstractShellEstimator, ShellSum, ModeAverage,
    SpectralConvention

# =============================================================================
# Sidedness
# =============================================================================

"""
    AbstractSidedness

Whether a spectrum keeps both signs of wavenumber (`TwoSided`) or folds negatives onto positives
(`OneSided`).
"""
abstract type AbstractSidedness end

"""`TwoSided()` — keep ± wavenumbers (no folding)."""
struct TwoSided <: AbstractSidedness end

"""`OneSided()` — fold negative wavenumbers onto positives. The usual convention for real fields."""
struct OneSided <: AbstractSidedness end

# =============================================================================
# Scaling
# =============================================================================

"""
    AbstractScaling

Whether a binned spectrum is a spectral *density* (`DensityScaling`, divided by the bin width) or the
power in each bin (`PowerScaling`).
"""
abstract type AbstractScaling end

"""`DensityScaling()` — spectral density per unit wavenumber: `Σ E·dk` is the energy the bins hold."""
struct DensityScaling <: AbstractScaling end

"""`PowerScaling()` — the energy in each bin (no `dk` division)."""
struct PowerScaling <: AbstractScaling end

# =============================================================================
# Radial-bin estimator
# =============================================================================

"""
    AbstractShellEstimator

How a radial bin turns the modes it holds into a spectrum value.
"""
abstract type AbstractShellEstimator end

"""
`ShellSum()` — the sum of `½|C|²` over the bin's modes, so the bins add up to the energy they hold. A
shell the mode box holds only in part (radius beyond `k_max = min_d max|k_d|`) sums fewer modes than a
whole one.
"""
struct ShellSum <: AbstractShellEstimator end

"""
`ModeAverage()` — the mean of `½|C|²` over the bin's modes times the number of modes a whole shell of
that width holds, `V_D (k₊ᴰ − k₋ᴰ) / ∏_d Δk_d` with `V_D` the unit-ball volume. For an isotropic field a
shell the mode box holds in part then estimates the whole shell, and the scatter of the lattice's mode
count per shell at low `k` averages out; the bins add up to the energy only approximately. A bin holding
no mode is `NaN`.
"""
struct ModeAverage <: AbstractShellEstimator end

# =============================================================================
# Convention object
# =============================================================================

"""
    SpectralConvention(; sided=OneSided(), scaling=DensityScaling(), estimator=ShellSum())

How spectral coefficients become reported spectra: the sidedness, the scaling, and the radial-bin
estimator. Fields are typed for compile-time dispatch.
"""
struct SpectralConvention{S<:AbstractSidedness, C<:AbstractScaling, E<:AbstractShellEstimator}
    sided::S
    scaling::C
    estimator::E
end

SpectralConvention(; sided::AbstractSidedness = OneSided(), scaling::AbstractScaling = DensityScaling(),
        estimator::AbstractShellEstimator = ShellSum()) = SpectralConvention(sided, scaling, estimator)

end # module Normalization
