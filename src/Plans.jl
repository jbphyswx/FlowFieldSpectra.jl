module Plans

export AbstractSpectralPlan, plan_spectrum,
    coefficient_size, coefficient_type, wavenumbers, allocate_coefficients, analysis_status,
    AbstractSynthesisPlan, plan_synthesis, synthesize!, field_size, field_type, allocate_field, close!

"""
    AbstractSpectralPlan

Supertype for reusable transform plans, tied to the grid, the spectral resolution `ms`, the trailing
`batch` shape and the element type. Build one with [`plan_spectrum`](@ref) and execute it on many fields
with `calculate_spectrum!`.

A plan holds working buffers that each execution overwrites, so one task at a time may execute it;
tasks running concurrently each build their own plan.
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

"""
    coefficient_size(plan) -> Dims

Shape of the coefficient array `calculate_spectrum!(coeffs, plan, field)` fills, batch axes included.

A real Cartesian field's is the packed half `(box[1]÷2+1, box[2:D]…, batch…)` of its coefficient box
(`Grids.mode_box`) and a complex one's the full native `(ms…, batch…)`; a spherical plan's is
`(lmax+1, 2lmax+1, batch…)`.
"""
function coefficient_size end

"""
    coefficient_type(plan) -> Type

Element type of the [`coefficient_size`](@ref) array. Cartesian coefficients are complex; a spherical
plan's are real for a real field, the basis being the real spherical harmonics.
[`allocate_coefficients`](@ref) reads both.
"""
function coefficient_type end

"""
    wavenumbers(plan) -> Tuple

The physical wavenumber axes the matching `calculate_spectrum` returns as its second value, in the same
packed native order as the coefficients: `rfftfreq`-like on a halved axis, `fftfreq` on a full one, and
`(0:lmax, -lmax:lmax)` for a spherical plan.
"""
function wavenumbers end

"""
    allocate_coefficients(plan) -> Array

A zeroed coefficient array of this plan's own size and element type, ready for
`calculate_spectrum!(coeffs, plan, field)`.
"""
allocate_coefficients(plan::AbstractSpectralPlan) =
    zeros(coefficient_type(plan), coefficient_size(plan)...)

"""
    analysis_status(plan) -> (; converged, iterations, residual)

The outcome of the plan's last `calculate_spectrum!`. A plan computing a least-squares fit
(`analysis = LeastSquares()`) passes on the fitting library's report over the whole batch (NUFSHT's
`nusht_analyze!` returns it per call): `converged` for every field, the most LSMR iterations, and the
largest relative residual `‖A†r‖/‖A†f‖`. A fit that stops short still writes its coefficients and is
reported here. A quadrature is `(true, 0, 0)`.
"""
analysis_status(plan::AbstractSpectralPlan) = _quadrature_status(real(coefficient_type(plan)))

# The report `analysis_status` returns; a plan running its fit in several calls combines theirs.
const AnalysisStatus{T} = @NamedTuple{converged::Bool, iterations::Int, residual::T}
_quadrature_status(::Type{T}) where {T} = AnalysisStatus{T}((true, 0, zero(T)))
_combine_status(a::AnalysisStatus{T}, b::AnalysisStatus{T}) where {T} =
    AnalysisStatus{T}((a.converged & b.converged, max(a.iterations, b.iterations), max(a.residual, b.residual)))

# A transform writes its modes by linear index, so `coeffs` must have the shape it writes.
function _check_coefficients(coeffs::AbstractArray, plan::AbstractSpectralPlan)
    size(coeffs) == coefficient_size(plan) || throw(DimensionMismatch(
        "coeffs is $(size(coeffs)); this plan writes $(coefficient_size(plan)) — allocate it with " *
        "`allocate_coefficients(plan)`"))
    return nothing
end

function _check_coefficients(coeffs::AbstractArray, want::Tuple)
    size(coeffs) == want || throw(DimensionMismatch(
        "coeffs is $(size(coeffs)); this transform writes $want"))
    return nothing
end

# =============================================================================
# Synthesis: the series evaluated at the grid's points.
# =============================================================================

"""
    AbstractSynthesisPlan

Supertype for reusable synthesis plans, the counterpart to [`AbstractSpectralPlan`](@ref), tied to the
grid, the spectral resolution `ms`, the trailing `batch` shape and the element type. Build one with
[`plan_synthesis`](@ref) and execute it on many coefficient sets with [`synthesize!`](@ref).

A plan holds working buffers that each execution overwrites, so one task at a time may execute it;
tasks running concurrently each build their own plan.
"""
abstract type AbstractSynthesisPlan end

"""
    plan_synthesis(grid, ::Type{T}, ms; transform=AutoSpectralBackend(), execution=AutoBackend(),
                   batch=(), iflag=1, kwargs...)
    plan_synthesis(transform, execution, grid, ::Type{T}, ms; batch=(), kwargs...)

Construct a reusable [`AbstractSynthesisPlan`](@ref) evaluating an `ms`-resolution series at the points
of `grid`, a field with trailing batch shape `batch`: what `synthesize` computes, the adjoint of the
forward transform under the grid's quadrature and its inverse only where that quadrature is exact. `T`
is the field's element type, which fixes both the coefficient layout the plan consumes (the packed half
for a real field, the full native cube for a complex one) and the output it writes.

Execute with `synthesize!(out, plan, coeffs)`.
"""
function plan_synthesis end

"""
    synthesize!(out, plan::AbstractSynthesisPlan, coeffs) -> out

Fill preallocated `out` with the series `coeffs` evaluated at the plan's grid points, reusing everything
`plan` holds. `out` is [`field_size`](@ref)-shaped of [`field_type`](@ref); [`allocate_field`](@ref)
provides one.
"""
function synthesize! end

"""
    field_size(plan::AbstractSynthesisPlan) -> Dims

Shape of the field `synthesize!` writes: the grid's own spatial shape followed by the batch axes, so a
round trip returns the shape the forward transform consumed.
"""
function field_size end

"""
    field_type(plan::AbstractSynthesisPlan) -> Type

Element type of that field — real when the plan was built for a real field, complex otherwise.
"""
function field_type end

"""
    allocate_field(plan) -> Array

A zeroed field array of this plan's own size and element type, ready for
`synthesize!(out, plan, coeffs)`.
"""
allocate_field(plan::AbstractSynthesisPlan) = zeros(field_type(plan), field_size(plan)...)

"""
    close!(plan) -> nothing

Release the library resources `plan` holds; idempotent. A plan over FINUFFT holds C memory that only this
frees. A plan over a Julia-only transform holds none, and this does nothing.
"""
close!(::Union{AbstractSpectralPlan, AbstractSynthesisPlan}) = nothing

end # module Plans
