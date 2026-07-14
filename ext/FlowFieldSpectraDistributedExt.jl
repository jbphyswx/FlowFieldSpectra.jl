module FlowFieldSpectraDistributedExt

using Distributed: Distributed
using SharedArrays: SharedArrays
using FlowFieldSpectra: FlowFieldSpectra as FFS

# =============================================================================
# DistributedBackend execution: split the spectrum over Julia worker processes, each computing a
# partial with its inner LOCAL backend (Serial/Threaded/GPU), then reduce. Point-partitionable
# transforms (DirectSum/NUFFT/NUFSHT-projection) split the point axis and sum α_w-weighted partial
# COEFFICIENTS (never partial energies — |coeff|² is not additive; the complex coeffs are). FFT/SHT
# split the batch (field) axis and gather disjoint slices. Requires `addprocs()` +
# `@everywhere using FlowFieldSpectra` (plus the transform's package for non-DirectSum transforms).
#
# `pmap`-into-a-typed-buffer is used rather than `@distributed (+)`: the latter's reduction infers
# as `Any` and would make AutoBackend+Distributed type-unstable.
# =============================================================================

# Round-robin point-index chunks: worker w owns indices w, w+nw, w+2nw, … (balanced for any per-
# point cost, and keeps each worker's slice spread across the domain).
_index_chunks(N::Integer, nw::Integer) = [collect(w:max(1, nw):N) for w in 1:max(1, nw)]

# Batch (field) chunks: contiguous, balanced disjoint field ranges.
_batch_chunks(NU::Integer, nw::Integer) =
    [(((w - 1) * NU) ÷ nw + 1):((w * NU) ÷ nw) for w in 1:nw]

# ---- point-partition: Σ_w α_w · coeff_w ----
function _distributed_pointsum(inner, transform, g::FFS.AbstractGrid, fields::Tuple, ms::Tuple; kwargs...)
    Nglob = FFS.Grids.npoints(g)
    nw = max(1, Distributed.nworkers())
    chunks = _index_chunks(Nglob, nw)
    partials = Distributed.pmap(chunks) do idx
        sg = FFS._subgrid(g, idx)
        sf = ntuple(u -> view(fields[u], idx), length(fields))
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

# ---- batch-partition: disjoint field slices, gathered along the trailing axis (no reduction) ----
function _distributed_batch(inner, transform, g::FFS.AbstractGrid, fields::Tuple, ms::Tuple; kwargs...)
    NU = length(fields)
    nw = max(1, Distributed.nworkers())
    chunks = filter(!isempty, _batch_chunks(NU, nw))
    parts = Distributed.pmap(chunks) do bc
        cw, _ = FFS.calculate_spectrum(transform, inner, g, ntuple(k -> fields[bc[k]], length(bc)), ms; kwargs...)
        (Array(cw), collect(bc))
    end
    FT = real(eltype(parts[1][1]))
    spatial = FFS._coeff_spatial(g, ms)
    coeffs = Array{complex(FT)}(undef, spatial..., NU)
    for (cw, bc) in parts
        @inbounds selectdim(coeffs, ndims(coeffs), bc) .= cw
    end
    return coeffs, FFS._partition_ks(g, ms)
end

function FFS._calculate_spectrum_distributed(transform, exec::FFS.DistributedBackend,
        g::FFS.AbstractGrid, fields::Tuple, ms::Tuple; kwargs...)
    inner = FFS.Types.local_backend(exec)
    # The NUFSHT CG solve couples all points, so it is not point-partitionable → batch axis.
    if transform isa FFS.NUFSHTBackend && get(kwargs, :solve, false)
        return _distributed_batch(inner, transform, g, fields, ms; kwargs...)
    end
    return FFS._partitionable(transform) ?
        _distributed_pointsum(inner, transform, g, fields, ms; kwargs...) :
        _distributed_batch(inner, transform, g, fields, ms; kwargs...)
end

end # module FlowFieldSpectraDistributedExt
