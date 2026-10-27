module Problem

using FlowGeometries: FlowGeometries

export TransformProblem, spatial_shape, batch_shape, n_batch, batch_length,
    coeff_output_size, stack_fields

# =============================================================================
# The shape contract, read from the grid. A field on `grid` is shaped `(spatial…, batch…)`: the first
# `ndims(grid)` dims equal `size(grid)`, and every trailing dim is a batch dim (components, vertical
# levels, time, ensemble), carried through the transform and the reductions. The coefficient array is
# `(spectral…, batch…)`. `ndims(grid)` is `N` for a structured grid and `1` for a point cloud.
# =============================================================================

"""
    TransformProblem{NS, B}

Shape of one transform: the leading `NS` spatial array dims (matching the grid) and the trailing `B`
batch dims. Built from `(grid, field)` via [`TransformProblem`](@ref); drives buffer ranks and the
coefficient output size.
"""
struct TransformProblem{NS, B}
    spatial::NTuple{NS, Int}      # == size(grid)
    batch::NTuple{B, Int}         # trailing batch sizes (possibly empty)
end

"""
    TransformProblem(grid, field::AbstractArray) -> TransformProblem

Validate `field` against `grid` and split its axes: the first `ndims(grid)` dims must equal
`size(grid)`; the remainder are the batch shape.
"""
@inline function TransformProblem(grid::FlowGeometries.Grids.AbstractGrid, field::AbstractArray{T, N}) where {T, N}
    ss = size(grid)
    ns = ndims(grid)
    N >= ns || throw(DimensionMismatch(
        "field has $N dims but grid $(nameof(typeof(grid))) needs at least $ns leading spatial dim(s) of size $ss"))
    @inbounds for d in 1:ns
        size(field, d) == ss[d] || throw(DimensionMismatch(
            "field leading dim $d = $(size(field, d)) ≠ grid spatial size $(ss[d]) (size = $ss)"))
    end
    batch = ntuple(d -> size(field, ns + d), Val(N - ns))
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
    stack_fields(fields::Tuple) -> AbstractArray

Convenience for the multi-field call form: stack equal-shaped field arrays along a **new trailing
batch axis** (`(spatial…, batch…, NU)`). This materializes a combined array — pass a single
`(spatial…, batch…)` array to avoid the copy.
"""
stack_fields(fields::Tuple) = (check_same_shape(fields); stack(fields))

"""
    check_same_shape(fields::Tuple) -> fields

Raises unless there is at least one field and every field has the first one's shape.
"""
function check_same_shape(fields::Tuple)
    length(fields) >= 1 || throw(ArgumentError("need at least one field"))
    sz = size(fields[1])
    @inbounds for u in 2:length(fields)
        size(fields[u]) == sz ||
            throw(DimensionMismatch("all fields must share shape (field $u is $(size(fields[u])), field 1 is $sz)"))
    end
    return fields
end

end # module Problem
