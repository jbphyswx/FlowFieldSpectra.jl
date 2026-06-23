module Plans

export AbstractSpectralPlan, plan_spectrum

"""
    AbstractSpectralPlan

Supertype for reusable transform plans. A plan is tied to the *fixed* geometry of a problem —
the grid coordinates, the spectral resolution `ms`, the number of batched transforms `n_transf`,
and the element type — but not to the field *values*. Build a plan once with
[`plan_spectrum`](@ref) and reuse it across many fields / batch slices / time steps via
`calculate_spectrum!`, avoiding repeated FFTW/FINUFFT plan construction and point sorting.

Concrete plan types are defined in the backend extensions (e.g. the FFTW and FINUFFT
extensions); this module only declares the shared interface.
"""
abstract type AbstractSpectralPlan end

"""
    plan_spectrum(backend, grid, ::Type{T}, ms; n_transf=1, kwargs...) -> AbstractSpectralPlan

Construct a reusable [`AbstractSpectralPlan`](@ref) for `backend` on `grid` at spectral
resolution `ms`, transforming `n_transf` co-located fields/slices of element type `T` in one
batched execution. Requires the backend's extension to be loaded.

Execute a plan with `calculate_spectrum!(coeffs, plan, fields)`.
"""
function plan_spectrum end

end # module Plans
