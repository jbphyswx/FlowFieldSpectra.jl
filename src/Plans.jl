module Plans

export AbstractSpectralPlan, plan_spectrum,
    coefficient_size, coefficient_type, wavenumbers, allocate_coefficients,
    AbstractSynthesisPlan, plan_synthesis, synthesize!, field_size, field_type, allocate_field

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

"""
    coefficient_size(plan) -> Dims

Shape of the coefficient array `calculate_spectrum!(coeffs, plan, field)` fills, batch axes included.

A real Cartesian field's is the packed half `(ms[1]÷2+1, ms[2:D]…, batch…)` and a complex one's the full
native `(ms…, batch…)`; a spherical plan's is `(lmax+1, 2lmax+1, batch…)`.
"""
function coefficient_size end

"""
    coefficient_type(plan) -> Type

Element type of that array. Cartesian coefficients are complex; a spherical plan's follow the field, real
for a real one, because the basis is the real spherical harmonics.

An allocation needs this as well as [`coefficient_size`](@ref), and
[`allocate_coefficients`](@ref) takes both.
"""
function coefficient_type end

"""
    wavenumbers(plan) -> Tuple

The physical wavenumber axes the matching `calculate_spectrum` returns as its second value, in the same
packed native order as the coefficients: `rfftfreq`-like on a halved axis, `fftfreq` on a full one, and
`(0:lmax, -lmax:lmax)` for a spherical plan.

A halved axis carries the Nyquist twin the transform attached, so the axes a reduction needs come from
here as well.
"""
function wavenumbers end

"""
    allocate_coefficients(plan) -> Array

A zeroed coefficient array of this plan's own size and element type, ready for
`calculate_spectrum!(coeffs, plan, field)`.
"""
allocate_coefficients(plan::AbstractSpectralPlan) =
    zeros(coefficient_type(plan), coefficient_size(plan)...)

# =============================================================================
# Synthesis: the inverse's counterpart to the analysis pair above.
# =============================================================================

"""
    AbstractSynthesisPlan

Supertype for reusable inverse-transform plans, the counterpart to [`AbstractSpectralPlan`](@ref). A
synthesis plan is tied to the grid, the spectral resolution `ms`, the trailing `batch` shape, the element
type, and whether the output is real — everything except the coefficient *values*.

Build one with [`plan_synthesis`](@ref) and reuse it across many coefficient sets via
[`synthesize!`](@ref), so a snapshot series on one grid builds its backward transform once. The forward
and inverse are separate objects, so a caller that only inverts never builds a forward plan.
"""
abstract type AbstractSynthesisPlan end

"""
    plan_synthesis(grid, ::Type{T}, ms; transform=AutoSpectralBackend(), execution=AutoBackend(),
                   batch=(), real_output=true, iflag=1, kwargs...)
    plan_synthesis(transform, execution, grid, ::Type{T}, ms; batch=(), real_output=true, kwargs...)

Construct a reusable [`AbstractSynthesisPlan`](@ref) inverting an `ms`-resolution spectrum on `grid` back
to a field with trailing batch shape `batch`. `T` is the FIELD's element type, which fixes both the
coefficient layout the plan consumes (the packed half for a real field, the full native cube for a
complex one) and the output it writes.

Execute with `synthesize!(out, plan, coeffs)`.
"""
function plan_synthesis end

"""
    synthesize!(out, plan::AbstractSynthesisPlan, coeffs; ks=nothing) -> out

Fill preallocated `out` with the inverse transform of `coeffs`, reusing everything `plan` holds. `out` is
[`field_size`](@ref)-shaped of [`field_type`](@ref); [`allocate_field`](@ref) provides one.

`ks` is the wavenumber tuple the forward returned. A plan holds what the GRID fixes, while the Nyquist
twin a packed inverse needs on a nonuniformly-sampled grid is a functional of the coefficients, so it
arrives with them on `ks[1]`. A uniform grid needs none.
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

end # module Plans
