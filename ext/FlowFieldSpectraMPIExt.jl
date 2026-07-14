module FlowFieldSpectraMPIExt

using MPI: MPI
using FlowFieldSpectra: FlowFieldSpectra as FFS

# =============================================================================
# MPIBackend execution: each rank computes a balanced share with its inner LOCAL backend, then the
# partial coefficient buffers are combined in place with `MPI.Allreduce!` so every rank ends with
# the full result. Point-partitionable transforms (DirectSum/NUFFT/NUFSHT-projection) split the
# point axis and sum α_w-weighted partial coefficients; FFT/SHT split the disjoint batch (field)
# axis. `MPIBackend(GPUBackend(dev))` runs one GPU per rank (host-staged Allreduce; a CUDA-aware
# Allreduce on the device buffer is a drop-in optimization). Launch under `mpiexec` with
# `MPI.Init()` called and the grid/fields replicated on every rank (SPMD).
# =============================================================================

@inline _comm(b::FFS.MPIBackend) = b.comm === nothing ? MPI.COMM_WORLD : b.comm

# ---- point-partition: partial coeffs summed via in-place Allreduce ----
function _mpi_pointsum(b::FFS.MPIBackend, transform, g::FFS.AbstractGrid, fields::Tuple, ms::Tuple; kwargs...)
    comm = _comm(b)
    rank = MPI.Comm_rank(comm)
    nrank = MPI.Comm_size(comm)
    Nglob = FFS.Grids.npoints(g)

    idx = (rank + 1):nrank:Nglob                     # round-robin balanced share of the points
    sg = FFS._subgrid(g, idx)
    sf = ntuple(u -> view(fields[u], idx), length(fields))
    cw, _ = FFS.calculate_spectrum(transform, FFS.Types.local_backend(b), sg, sf, ms; kwargs...)

    FT = real(eltype(cw))
    coeffs = Array{complex(FT)}(undef, size(cw))     # contiguous buffer for in-place Allreduce
    coeffs .= FT(FFS._partition_alpha(g, length(idx), Nglob)) .* Array(cw)
    MPI.Allreduce!(coeffs, +, comm)                  # Σ_rank ⇒ full global coeffs on every rank
    return coeffs, FFS._partition_ks(g, ms)
end

# ---- batch-partition: disjoint field slices Allreduced into a zero buffer ----
function _mpi_batch(b::FFS.MPIBackend, transform, g::FFS.AbstractGrid, fields::Tuple, ms::Tuple; kwargs...)
    comm = _comm(b)
    rank = MPI.Comm_rank(comm)
    nrank = MPI.Comm_size(comm)
    NU = length(fields)
    bc = (rank * NU ÷ nrank + 1):((rank + 1) * NU ÷ nrank)   # this rank's disjoint field range

    FT = real(float(eltype(fields[1])))
    spatial = FFS._coeff_spatial(g, ms)
    coeffs = zeros(complex(FT), spatial..., NU)
    if !isempty(bc)
        cw, _ = FFS.calculate_spectrum(transform, FFS.Types.local_backend(b), g,
            ntuple(k -> fields[bc[k]], length(bc)), ms; kwargs...)
        @inbounds selectdim(coeffs, ndims(coeffs), bc) .= Array(cw)
    end
    MPI.Allreduce!(coeffs, +, comm)                  # disjoint slices ⇒ sum assembles the full array
    return coeffs, FFS._partition_ks(g, ms)
end

function FFS._calculate_spectrum_mpi(transform, exec::FFS.MPIBackend,
        g::FFS.AbstractGrid, fields::Tuple, ms::Tuple; kwargs...)
    MPI.Initialized() || throw(ArgumentError("MPI is not initialized — call `MPI.Init()` before using MPIBackend."))
    if transform isa FFS.NUFSHTBackend && get(kwargs, :solve, false)
        return _mpi_batch(exec, transform, g, fields, ms; kwargs...)
    end
    return FFS._partitionable(transform) ?
        _mpi_pointsum(exec, transform, g, fields, ms; kwargs...) :
        _mpi_batch(exec, transform, g, fields, ms; kwargs...)
end

end # module FlowFieldSpectraMPIExt
