module Problem

using ..Grids: AbstractGrid

export TransformProblem, n_spectral, n_batch, batch_size, output_size,
    pack_fields, coeff_eltype

"""
    TransformProblem{D, B}

Describes the shape of a spectral transform: which input axes are *spectral* (transformed,
matching the grid's `D` dimensions) and which trailing axes are *batch* axes (carried through
untouched — components, vertical levels, time, ensemble members). `B` is the number of batch
axes.

# Fields
- `ms::NTuple{D,Int}`: number of spectral modes along each spectral axis.
- `batch::NTuple{B,Int}`: size of each trailing batch axis.

The data-layout contract is **column-major, spectral dims leading, batch dims trailing**: a
field array has shape `(N_spatial-or-spectral…, batch…)` and the coefficient array has shape
`(ms…, batch…)`. The legacy "tuple of `NU` field vectors" form maps to a single batch axis of
length `NU`.
"""
struct TransformProblem{D, B}
    ms::NTuple{D, Int}
    batch::NTuple{B, Int}
end

"""
    TransformProblem(ms::NTuple{D,Int}; batch=())

Construct a problem with spectral resolution `ms` and trailing batch-axis sizes `batch`.
"""
function TransformProblem(ms::NTuple{D, Int}; batch::Tuple = ()) where {D}
    B = length(batch)
    return TransformProblem{D, B}(ms, NTuple{B, Int}(batch))
end

"""`n_spectral(prob)` — number of spectral (transformed) axes."""
n_spectral(::TransformProblem{D, B}) where {D, B} = D

"""`n_batch(prob)` — number of trailing batch axes."""
n_batch(::TransformProblem{D, B}) where {D, B} = B

"""`batch_size(prob)` — total number of batch slices, `prod(batch)` (1 if no batch axes)."""
batch_size(p::TransformProblem) = prod(p.batch; init = 1)

"""
    output_size(prob) -> NTuple

Shape of the coefficient array: `(ms…, batch…)`.
"""
output_size(p::TransformProblem) = (p.ms..., p.batch...)

# -----------------------------------------------------------------------------
# Legacy field-tuple adapter
# -----------------------------------------------------------------------------

"""
    coeff_eltype(grid) -> Type

Complex coefficient element type for a grid (`Complex{FT}`).
"""
coeff_eltype(::AbstractGrid{FT}) where {FT} = Complex{FT}

"""
    pack_fields(fields_vecs::Tuple) -> (data, batch)

Normalize the legacy "tuple of `NU` equal-length field vectors" into a packed
`(N, NU)` matrix plus the batch shape `(NU,)`. A single vector packs to `(N, 1)` with batch
`(1,)`. When `fields_vecs` is already an `AbstractArray`, it is returned as-is with an empty
batch (the caller supplies the layout).
"""
function pack_fields(fields_vecs::Tuple)
    NU = length(fields_vecs)
    N = length(fields_vecs[1])
    @inbounds for u in 2:NU
        length(fields_vecs[u]) == N ||
            throw(DimensionMismatch("all field vectors must have equal length (got $(length(fields_vecs[u])) ≠ $N)"))
    end
    FT = float(eltype(fields_vecs[1]))
    data = Matrix{FT}(undef, N, NU)
    @inbounds for u in 1:NU
        col = fields_vecs[u]
        for j in 1:N
            data[j, u] = col[j]
        end
    end
    return data, (NU,)
end

end # module Problem
