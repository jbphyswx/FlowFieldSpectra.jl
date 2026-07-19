module Problem

using ..Grids: Grids, AbstractGrid, ndims_spatial, spatial_size

export TransformProblem, spatial_shape, batch_shape, n_batch, batch_length,
    coeff_output_size, coeff_eltype, stack_fields

# =============================================================================
# The shape contract, taken from the grid — never guessed.
#
# A field on `grid` is an AbstractArray shaped `(spatial…, batch…)`: the first `ndims_spatial(grid)`
# dims are spatial (and must equal `spatial_size(grid)`), and EVERY trailing dim is a batch dim
# (components / vertical levels / time / ensemble — any number), carried through the transform and
# (by default) through the reductions. The coefficient array is `(spectral…, batch…)`.
# =============================================================================

"""
    TransformProblem{NS, B}

Shape of one transform: the leading `NS` spatial array dims (matching the grid) and the trailing `B`
batch dims. Built from `(grid, field)` via [`TransformProblem`](@ref); drives buffer ranks and the
coefficient output size.
"""
struct TransformProblem{NS, B}
    spatial::NTuple{NS, Int}      # == spatial_size(grid)
    batch::NTuple{B, Int}         # trailing batch sizes (possibly empty)
end

"""
    TransformProblem(grid, field::AbstractArray) -> TransformProblem

Validate `field` against `grid` and split its axes: the first `ndims_spatial(grid)` dims must equal
`spatial_size(grid)`; the remainder are the batch shape.
"""
@inline function TransformProblem(grid::AbstractGrid, field::AbstractArray{T, N}) where {T, N}
    ss = spatial_size(grid)
    ns = ndims_spatial(grid)
    N >= ns || throw(DimensionMismatch(
        "field has $N dims but grid $(nameof(typeof(grid))) needs at least $ns leading spatial dim(s) of size $ss"))
    @inbounds for d in 1:ns
        size(field, d) == ss[d] || throw(DimensionMismatch(
            "field leading dim $d = $(size(field, d)) ≠ grid spatial size $(ss[d]) (spatial_size = $ss)"))
    end
    batch = ntuple(d -> size(field, ns + d), N - ns)
    return TransformProblem(ss, batch)
end

"""`spatial_shape(prob)` — the leading spatial array sizes `(N_1, …)`."""
spatial_shape(p::TransformProblem) = p.spatial

"""`batch_shape(prob)` — the trailing batch sizes (`()` if none)."""
batch_shape(p::TransformProblem) = p.batch

"""`n_batch(prob)` — number of trailing batch dims."""
n_batch(::TransformProblem{NS, B}) where {NS, B} = B

"""`batch_length(prob)` — total batch slices, `prod(batch)` (1 if no batch axes)."""
batch_length(p::TransformProblem) = prod(p.batch; init = 1)

"""
    coeff_output_size(spectral::Tuple, prob) -> NTuple

Shape of the coefficient array: `(spectral…, batch…)`.
"""
coeff_output_size(spectral::Tuple, p::TransformProblem) = (spectral..., p.batch...)

"""
    coeff_eltype(grid) -> Type

Complex coefficient element type for a grid (`Complex{FT}`).
"""
coeff_eltype(::AbstractGrid{FT}) where {FT} = Complex{FT}

"""
    stack_fields(fields::Tuple) -> AbstractArray

Convenience for the multi-field call form: stack equal-shaped field arrays along a **new trailing
batch axis** (`(spatial…, batch…, NU)`). This materializes a combined array — pass a single
`(spatial…, batch…)` array to avoid the copy.
"""
function stack_fields(fields::Tuple)
    length(fields) >= 1 || throw(ArgumentError("need at least one field"))
    sz = size(fields[1])
    @inbounds for u in 2:length(fields)
        size(fields[u]) == sz ||
            throw(DimensionMismatch("all fields must share shape (field $u is $(size(fields[u])), field 1 is $sz)"))
    end
    return stack(fields)
end

end # module Problem
