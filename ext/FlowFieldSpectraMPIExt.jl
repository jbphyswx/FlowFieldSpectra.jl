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

# A grid whose cells carry their own coordinates admits a disjoint point partition. A curvilinear grid
# does too: its coordinate arrays hold one value per cell and index linearly, so a subset of its cells is
# a node cloud.
_is_scattered(::FlowGeometries.Grids.AbstractUnstructuredGrid) = true
_is_scattered(::FlowGeometries.Grids.AbstractCurvilinearGrid) = true
_is_scattered(::FlowGeometries.Grids.AbstractGrid) = false

@inline _comm(b::ComputationalBackends.MPIBackend) = b.comm === nothing ? MPI.COMM_WORLD : b.comm

# ---- point-partition: α_w-weighted partial coeffs summed via in-place Allreduce ----
function _mpi_pointsum(b::ComputationalBackends.MPIBackend, transform, g::FlowGeometries.Grids.AbstractGrid, field::AbstractArray, ms::Tuple; kwargs...)
    comm = _comm(b)
    rank = MPI.Comm_rank(comm)
    nrank = MPI.Comm_size(comm)
    # The partition is over POINTS, so the field's spatial dims collapse to one point axis; the batch dims
    # ride through untouched, keeping the coefficient shape. `reshape` is a no-op on a node cloud, whose
    # spatial part is already one axis.
    Nglob = prod(ntuple(d -> size(field, d), ndims(g)))
    fieldP = reshape(field, Nglob, FFS.Grids.field_batch_shape(g, field)...)
    idx = (rank + 1):nrank:Nglob                      # round-robin point share
    sg = FFS._subgrid(g, idx)
    sf = collect(selectdim(fieldP, 1, idx))
    cw, kw = FFS.calculate_spectrum(transform, ComputationalBackends.local_backend(b), sg, sf, ms; kwargs...)
    FT = real(eltype(cw))
    α = FT(FFS._partition_alpha(g, length(idx), Nglob))
    coeffs = Array{eltype(cw)}(undef, size(cw))        # contiguous buffer for in-place Allreduce
    coeffs .= α .* Array(cw)
    MPI.Allreduce!(coeffs, +, comm)
    ks = FFS._partition_ks(g, ms, eltype(field) <: Real)
    # Every rank owns a point share, so all of them hold twin slices of the same shape; the twins are
    # linear in the field, so the same α-weighted sum that assembles the coefficients assembles them.
    return coeffs, _allreduce_twin(FFS._ks_twin(kw), ks, α, comm)
end

_allreduce_twin(::Nothing, ks::Tuple, α, comm) = ks
function _allreduce_twin(t::FFS.Packing.NyquistTwin, ks::Tuple, α, comm)
    slices = map(t.slices) do s
        buf = Array{eltype(s)}(undef, size(s))
        buf .= α .* s
        isempty(buf) || MPI.Allreduce!(buf, +, comm)
        buf
    end
    return (FFS.Packing.with_twin(ks[1], FFS.Packing.NyquistTwin(slices)), Base.tail(ks)...)
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
    R = eltype(field) <: Real
    spatial_out = FFS._coeff_spatial(g, ms, R)
    coeffsB = zeros(FFS._coeff_eltype(g, eltype(field), FT), spatial_out..., B)
    tw = nothing
    if !isempty(bc)
        fslice = collect(selectdim(fieldB, ns + 1, bc))
        cw, kw = FFS.calculate_spectrum(transform, ComputationalBackends.local_backend(b), g, fslice, ms; kwargs...)
        @inbounds selectdim(coeffsB, ndims(coeffsB), bc) .= reshape(Array(cw), spatial_out..., length(bc))
        tw = FFS._ks_twin(kw)
    end
    MPI.Allreduce!(coeffsB, +, comm)                   # disjoint slices ⇒ sum assembles the full array
    coeffs = reshape(coeffsB, spatial_out..., batch...)
    ks = FFS._partition_ks(g, ms, R)
    # A rank with no batch columns never built a twin, so whether one exists is agreed by reduction and
    # the buffer shapes come from `ms`; disjoint batch slices then sum into the full-batch twin.
    if MPI.Allreduce(tw === nothing ? 0 : 1, max, comm) == 1
        msn = NTuple{length(ms), Int}(Tuple(ms))
        bufs = FFS._alloc_twins(Complex{FT}, msn, B)
        tw === nothing || FFS._scatter_batch_twins!(bufs, tw, bc, B)
        for buf in bufs
            isempty(buf) || MPI.Allreduce!(buf, +, comm)
        end
        ks = FFS._reshape_batch_twins(ks, bufs, batch)
    end
    return coeffs, ks
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
