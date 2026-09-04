module FlowFieldSpectraDistributedExt

using Distributed: Distributed
using FlowFieldSpectra: FlowFieldSpectra as FFS
using ComputationalBackends: ComputationalBackends
using SpectralBackends: SpectralBackends
using FlowGeometries: FlowGeometries

# =============================================================================
# DistributedBackend execution, tensor-native. Point-partitionable transforms (DirectSum / NUFFT /
# NUFSHT-projection) on a SCATTERED (unstructured) grid split the point axis (dim 1) and sum α_w-weighted
# partial COEFFICIENTS (complex coeffs are additive over a disjoint point partition; |coeff|² is not).
# Every other case (FFT / SHT, or NUFSHT solve which couples all points) splits the trailing BATCH axis
# into disjoint slices and gathers. Each worker runs the inner LOCAL backend, so distribution composes
# with Serial/Threaded/GPU. `pmap`-into-a-typed-buffer (not `@distributed (+)`, which infers `Any`).
# =============================================================================

# A grid whose cells carry their own coordinates admits a disjoint point partition. A curvilinear grid
# does too: its coordinate arrays hold one value per cell and index linearly, so a subset of its cells is
# a node cloud.
_is_scattered(::FlowGeometries.Grids.AbstractUnstructuredGrid) = true
_is_scattered(::FlowGeometries.Grids.AbstractCurvilinearGrid) = true
_is_scattered(::FlowGeometries.Grids.AbstractGrid) = false

# Round-robin point chunks (balanced across the domain); contiguous batch chunks.
_index_chunks(N::Integer, nw::Integer) = [collect(w:max(1, nw):N) for w in 1:max(1, nw)]
_batch_chunks(B::Integer, nw::Integer) = [(((w - 1) * B) ÷ nw + 1):((w * B) ÷ nw) for w in 1:nw]

# ---- point-partition: Σ_w α_w · coeff_w ----
function _distributed_pointsum(inner, transform, g::FlowGeometries.Grids.AbstractGrid, field::AbstractArray, ms::Tuple; kwargs...)
    # The partition is over POINTS, so the field's spatial dims collapse to one point axis; the batch dims
    # ride through untouched, keeping the coefficient shape. `reshape` is a no-op on a node cloud, whose
    # spatial part is already one axis.
    Nglob = prod(ntuple(d -> size(field, d), ndims(g)))
    fieldP = reshape(field, Nglob, FFS.Grids.field_batch_shape(g, field)...)
    nw = max(1, Distributed.nworkers())
    chunks = _index_chunks(Nglob, nw)
    partials = Distributed.pmap(chunks) do idx
        sg = FFS._subgrid(g, idx)
        sf = collect(selectdim(fieldP, 1, idx))
        cw, kw = FFS.calculate_spectrum(transform, inner, sg, sf, ms; kwargs...)
        (Array(cw), FFS._ks_twin(kw), length(idx))
    end
    FT = real(eltype(partials[1][1]))
    # The workers' own element type: complex for a Cartesian spectrum, real for a real field's spherical
    # coefficients.
    coeffs = zeros(eltype(partials[1][1]), size(partials[1][1]))
    @inbounds for (cw, _, Nw) in partials
        coeffs .+= FT(FFS._partition_alpha(g, Nw, Nglob)) .* cw
    end
    ks = FFS._partition_ks(g, ms, eltype(field) <: Real)
    weights = [FT(FFS._partition_alpha(g, p[3], Nglob)) for p in partials]
    return coeffs, FFS._combine_twins(ks, Tuple(p[2] for p in partials), weights)
end

# ---- batch-partition: disjoint batch slices, gathered along the (flattened) batch axis ----
function _distributed_batch(inner, transform, g::FlowGeometries.Grids.AbstractGrid, field::AbstractArray, ms::Tuple; kwargs...)
    ns = ndims(g)
    sp = ntuple(d -> size(field, d), ns)
    batch = ntuple(i -> size(field, ns + i), ndims(field) - ns)
    B = prod(batch; init = 1)
    fieldB = reshape(field, sp..., B)
    nw = max(1, Distributed.nworkers())
    chunks = filter(!isempty, _batch_chunks(B, nw))
    R = eltype(field) <: Real
    spatial_out = FFS._coeff_spatial(g, ms, R)
    parts = Distributed.pmap(chunks) do bc
        fslice = collect(selectdim(fieldB, ns + 1, bc))
        cw, kw = FFS.calculate_spectrum(transform, inner, g, fslice, ms; kwargs...)
        (Array(cw), FFS._ks_twin(kw), collect(bc))
    end
    coeffsB = Array{eltype(parts[1][1])}(undef, spatial_out..., B)
    twinsB = FFS._alloc_batch_twins(parts[1][2], B)
    for (cw, tw, bc) in parts
        @inbounds selectdim(coeffsB, ndims(coeffsB), bc) .= reshape(cw, spatial_out..., length(bc))
        FFS._scatter_batch_twins!(twinsB, tw, bc, B)
    end
    coeffs = reshape(coeffsB, spatial_out..., batch...)
    ks = FFS._partition_ks(g, ms, R)
    return coeffs, FFS._reshape_batch_twins(ks, twinsB, batch)
end

function FFS._calculate_spectrum_distributed(transform, exec::ComputationalBackends.AbstractDistributedBackend,
        g::FlowGeometries.Grids.AbstractGrid, field::AbstractArray, ms::Tuple; kwargs...)
    inner = ComputationalBackends.local_backend(exec)
    if transform isa SpectralBackends.AbstractNUFSHTSpectralBackend && get(kwargs, :solve, false)
        return _distributed_batch(inner, transform, g, field, ms; kwargs...)   # CG couples all points
    end
    return (FFS._partitionable(transform) && _is_scattered(g)) ?
        _distributed_pointsum(inner, transform, g, field, ms; kwargs...) :
        _distributed_batch(inner, transform, g, field, ms; kwargs...)
end

end # module FlowFieldSpectraDistributedExt
