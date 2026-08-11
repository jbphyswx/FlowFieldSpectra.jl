module FlowFieldSpectraMPIExt

using MPI: MPI
using FlowFieldSpectra: FlowFieldSpectra as FFS
using ComputationalBackends: ComputationalBackends
using SpectralBackends: SpectralBackends
using FlowGeometries: FlowGeometries

# =============================================================================
# MPIBackend execution, tensor-native. Same partition strategy as the Distributed backend, combined
# in place with `MPI.Allreduce!` so every rank ends with the full result. Point-partitionable
# transforms on a SCATTERED (unstructured) grid split the point axis and sum α_w-weighted partial
# coefficients; everything else (FFT/SHT, NUFSHT-solve) splits disjoint BATCH slices into a zero buffer.
# Each rank runs the inner LOCAL backend. Launch under `mpiexec` with `MPI.Init()`; grid/field
# replicated (SPMD).
# =============================================================================

_is_scattered(::FlowGeometries.Grids.AbstractUnstructuredGrid) = true
_is_scattered(::FlowGeometries.Grids.AbstractGrid) = false

@inline _comm(b::ComputationalBackends.MPIBackend) = b.comm === nothing ? MPI.COMM_WORLD : b.comm

# ---- point-partition: α_w-weighted partial coeffs summed via in-place Allreduce ----
function _mpi_pointsum(b::ComputationalBackends.MPIBackend, transform, g::FlowGeometries.Grids.AbstractGrid, field::AbstractArray, ms::Tuple; kwargs...)
    comm = _comm(b)
    rank = MPI.Comm_rank(comm)
    nrank = MPI.Comm_size(comm)
    Nglob = size(field, 1)
    idx = (rank + 1):nrank:Nglob                      # round-robin point share
    sg = FFS._subgrid(g, idx)
    sf = collect(selectdim(field, 1, idx))
    cw, _ = FFS.calculate_spectrum(transform, ComputationalBackends.local_backend(b), sg, sf, ms; kwargs...)
    FT = real(eltype(cw))
    coeffs = Array{complex(FT)}(undef, size(cw))       # contiguous buffer for in-place Allreduce
    coeffs .= FT(FFS._partition_alpha(g, length(idx), Nglob)) .* Array(cw)
    MPI.Allreduce!(coeffs, +, comm)
    return coeffs, FFS._partition_ks(g, ms)
end

# ---- batch-partition: disjoint batch slices Allreduced into a zero buffer ----
function _mpi_batch(b::ComputationalBackends.MPIBackend, transform, g::FlowGeometries.Grids.AbstractGrid, field::AbstractArray, ms::Tuple; kwargs...)
    comm = _comm(b)
    rank = MPI.Comm_rank(comm)
    nrank = MPI.Comm_size(comm)
    ns = ndims(g)
    sp = ntuple(d -> size(field, d), ns)
    batch = ntuple(i -> size(field, ns + i), ndims(field) - ns)
    B = prod(batch; init = 1)
    fieldB = reshape(field, sp..., B)
    bc = (rank * B ÷ nrank + 1):((rank + 1) * B ÷ nrank)   # this rank's disjoint batch range
    FT = real(float(eltype(field)))
    spatial_out = FFS._coeff_spatial(g, ms)
    coeffsB = zeros(complex(FT), spatial_out..., B)
    if !isempty(bc)
        fslice = collect(selectdim(fieldB, ns + 1, bc))
        cw, _ = FFS.calculate_spectrum(transform, ComputationalBackends.local_backend(b), g, fslice, ms; kwargs...)
        @inbounds selectdim(coeffsB, ndims(coeffsB), bc) .= reshape(Array(cw), spatial_out..., length(bc))
    end
    MPI.Allreduce!(coeffsB, +, comm)                   # disjoint slices ⇒ sum assembles the full array
    coeffs = reshape(coeffsB, spatial_out..., batch...)
    return coeffs, FFS._partition_ks(g, ms)
end

function FFS._calculate_spectrum_mpi(transform, exec::ComputationalBackends.MPIBackend,
        g::FlowGeometries.Grids.AbstractGrid, field::AbstractArray, ms::Tuple; kwargs...)
    MPI.Initialized() || throw(ArgumentError("MPI is not initialized — call `MPI.Init()` before using MPIBackend."))
    if transform isa SpectralBackends.AbstractNUFSHTSpectralBackend && get(kwargs, :solve, false)
        return _mpi_batch(exec, transform, g, field, ms; kwargs...)
    end
    return (FFS._partitionable(transform) && _is_scattered(g)) ?
        _mpi_pointsum(exec, transform, g, field, ms; kwargs...) :
        _mpi_batch(exec, transform, g, field, ms; kwargs...)
end

end # module FlowFieldSpectraMPIExt
