module Plans

export AbstractSpectralPlan, plan_spectrum

"""
    AbstractSpectralPlan

Supertype for reusable transform plans. A plan is tied to the *fixed* geometry of a problem —
the grid coordinates, the spectral resolution `ms`, the trailing `batch` shape, and the element type —
but not to the field *values*. Build a plan once with
[`plan_spectrum`](@ref) and reuse it across many fields / batch slices / time steps via
`calculate_spectrum!`, avoiding repeated FFTW/FINUFFT plan construction and point sorting.

Concrete plan types are defined in the backend extensions (e.g. the FFTW and FINUFFT
extensions); this module only declares the shared interface.
"""
abstract type AbstractSpectralPlan end

"""
    plan_spectrum(grid, ::Type{T}, ms; transform=AutoSpectralBackend(), execution=AutoBackend(), batch=(), kwargs...)
    plan_spectrum(transform, execution, grid, ::Type{T}, ms; batch=(), kwargs...)

Construct a reusable [`AbstractSpectralPlan`](@ref) for the `transform`×`execution` backend pair on
`grid` at spectral resolution `ms`, transforming a field with trailing batch shape `batch` of element
type `T` in one batched execution. The keyword form resolves `execution` and forwards to the canonical
positional form implemented by the backend extensions. Requires the transform's extension to be
loaded.

Execute a plan with `calculate_spectrum!(coeffs, plan, fields)`.
"""
function plan_spectrum end

end # module Plans
