module FlowFieldSpectraDistributedExt

using Distributed: Distributed
using FlowFieldSpectra: FlowFieldSpectra as FFS

# =============================================================================
# DistributedBackend execution, tensor-native. Point-partitionable transforms (DirectSum / NUFFT /
# NUFSHT-projection) on a SCATTERED grid split the point axis (dim 1) and sum α_w-weighted partial
# COEFFICIENTS (complex coeffs are additive over a disjoint point partition; |coeff|² is not). Every
# other case (FFT / SHT, or NUFSHT solve which couples all points) splits the trailing BATCH axis into
# disjoint slices and gathers. Each worker runs the inner LOCAL backend, so distribution composes with
# Serial/Threaded/GPU. `pmap`-into-a-typed-buffer (not `@distributed (+)`, which infers `Any`).
# =============================================================================

_is_scattered(::FFS.ScatteredCartesianGrid) = true
_is_scattered(::FFS.ScatteredSphericalGrid) = true
_is_scattered(::FFS.AbstractGrid) = false

# Round-robin point chunks (balanced across the domain); contiguous batch chunks.
_index_chunks(N::Integer, nw::Integer) = [collect(w:max(1, nw):N) for w in 1:max(1, nw)]
_batch_chunks(B::Integer, nw::Integer) = [(((w - 1) * B) ÷ nw + 1):((w * B) ÷ nw) for w in 1:nw]

# ---- point-partition: Σ_w α_w · coeff_w ----
function _distributed_pointsum(inner, transform, g::FFS.AbstractGrid, field::AbstractArray, ms::Tuple; kwargs...)
    Nglob = size(field, 1)
    nw = max(1, Distributed.nworkers())
    chunks = _index_chunks(Nglob, nw)
    partials = Distributed.pmap(chunks) do idx
        sg = FFS._subgrid(g, idx)
        sf = collect(selectdim(field, 1, idx))
        cw, _ = FFS.calculate_spectrum(transform, inner, sg, sf, ms; kwargs...)
        (Array(cw), length(idx))
    end
    FT = real(eltype(partials[1][1]))
    coeffs = zeros(complex(FT), size(partials[1][1]))
    @inbounds for (cw, Nw) in partials
        coeffs .+= FT(FFS._partition_alpha(g, Nw, Nglob)) .* cw
    end
    return coeffs, FFS._partition_ks(g, ms)
end

# ---- batch-partition: disjoint batch slices, gathered along the (flattened) batch axis ----
function _distributed_batch(inner, transform, g::FFS.AbstractGrid, field::AbstractArray, ms::Tuple; kwargs...)
    ns = FFS.ndims_spatial(g)
    sp = ntuple(d -> size(field, d), ns)
    batch = ntuple(i -> size(field, ns + i), ndims(field) - ns)
    B = prod(batch; init = 1)
    fieldB = reshape(field, sp..., B)
    nw = max(1, Distributed.nworkers())
    chunks = filter(!isempty, _batch_chunks(B, nw))
    spatial_out = FFS._coeff_spatial(g, ms)
    parts = Distributed.pmap(chunks) do bc
        fslice = collect(selectdim(fieldB, ns + 1, bc))
        cw, _ = FFS.calculate_spectrum(transform, inner, g, fslice, ms; kwargs...)
        (Array(cw), collect(bc))
    end
    FT = real(eltype(parts[1][1]))
    coeffsB = Array{complex(FT)}(undef, spatial_out..., B)
    for (cw, bc) in parts
        @inbounds selectdim(coeffsB, ndims(coeffsB), bc) .= reshape(cw, spatial_out..., length(bc))
    end
    coeffs = reshape(coeffsB, spatial_out..., batch...)
    return coeffs, FFS._partition_ks(g, ms)
end

function FFS._calculate_spectrum_distributed(transform, exec::FFS.DistributedBackend,
        g::FFS.AbstractGrid, field::AbstractArray, ms::Tuple; kwargs...)
    inner = FFS.Types.local_backend(exec)
    if transform isa FFS.NUFSHTBackend && get(kwargs, :solve, false)
        return _distributed_batch(inner, transform, g, field, ms; kwargs...)   # CG couples all points
    end
    return (FFS._partitionable(transform) && _is_scattered(g)) ?
        _distributed_pointsum(inner, transform, g, field, ms; kwargs...) :
        _distributed_batch(inner, transform, g, field, ms; kwargs...)
end

end # module FlowFieldSpectraDistributedExt
