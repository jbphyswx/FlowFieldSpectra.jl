module FlowFieldSpectraMPIExt

using MPI: MPI
using FlowFieldSpectra: FlowFieldSpectra as FFS
using ComputationalBackends: ComputationalBackends
using SpectralBackends: SpectralBackends
using FlowGeometries: FlowGeometries

# =============================================================================
# MPIBackend execution: the partition of the Distributed extension, combined in place with
# `MPI.Allreduce!` so every rank ends with the full result. A batch partition sums disjoint slices into a
# zero buffer. Each rank runs the inner local backend. Launch under `mpiexec` with `MPI.Init()`; every
# rank holds the whole grid and field.
# =============================================================================

@inline _comm(b::ComputationalBackends.MPIBackend) = b.comm === nothing ? MPI.COMM_WORLD : b.comm

_require_init() = MPI.Initialized() ||
    throw(ArgumentError("MPI is not initialized — call `MPI.Init()` before using MPIBackend."))

# This rank's share: round-robin point indices for a point partition, a contiguous range of batch columns
# otherwise.
_rank_share(points::Bool, rank::Int, nrank::Int, Nglob::Int, B::Int) =
    points ? ((rank + 1):nrank:Nglob) : ((rank * B ÷ nrank + 1):((rank + 1) * B ÷ nrank))

# ---- point-partition: α_w-weighted partial coeffs summed via in-place Allreduce ----
function _mpi_pointsum(b::ComputationalBackends.MPIBackend, transform, g::FlowGeometries.Grids.AbstractGrid, field::AbstractArray, ms::Tuple; kwargs...)
    comm = _comm(b)
    rank = MPI.Comm_rank(comm)
    nrank = MPI.Comm_size(comm)
    Nglob = prod(ntuple(d -> size(field, d), ndims(g)))
    fieldP = reshape(field, Nglob, FFS.Grids.field_batch_shape(g, field)...)
    idx = (rank + 1):nrank:Nglob                      # round-robin point share
    sg = FFS._subgrid(g, idx)
    sf = collect(selectdim(fieldP, 1, idx))
    cw, _ = FFS.calculate_spectrum(transform, ComputationalBackends.local_backend(b), sg, sf, ms;
                                   FFS._partition_kwargs(kwargs, g, idx)...)
    FT = real(eltype(cw))
    α = FT(FFS._partition_weight(g, idx, get(kwargs, :weights, nothing)))
    coeffs = Array{eltype(cw)}(undef, size(cw))        # contiguous buffer for in-place Allreduce
    coeffs .= α .* Array(cw)
    MPI.Allreduce!(coeffs, +, comm)
    return coeffs, FFS._partition_ks(g, ms, eltype(field) <: Real)
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
    if !isempty(bc)
        fslice = collect(selectdim(fieldB, ns + 1, bc))
        cw, _ = FFS.calculate_spectrum(transform, ComputationalBackends.local_backend(b), g, fslice, ms; kwargs...)
        @inbounds selectdim(coeffsB, ndims(coeffsB), bc) .= reshape(Array(cw), spatial_out..., length(bc))
    end
    MPI.Allreduce!(coeffsB, +, comm)                   # disjoint slices ⇒ sum assembles the full array
    return reshape(coeffsB, spatial_out..., batch...), FFS._partition_ks(g, ms, R)
end

function FFS._calculate_spectrum_mpi(transform, exec::ComputationalBackends.MPIBackend,
        g::FlowGeometries.Grids.AbstractGrid, field::AbstractArray, ms::Tuple; kwargs...)
    _require_init()
    return FFS._point_partition(transform, g, kwargs) ?
        _mpi_pointsum(exec, transform, g, field, ms; kwargs...) :
        _mpi_batch(exec, transform, g, field, ms; kwargs...)
end

# =============================================================================
# Reusable plans, called on every rank. Each rank builds the local plan of its share once; an execution
# transforms the share and combines the ranks' results in place with `MPI.Allreduce!`, as the one-shot
# does. A rank whose share is empty holds no plan and contributes zeros.
# =============================================================================

"""
    MPISpectralPlan

A spectrum plan divided among the ranks of an MPI communicator. This rank holds the local plan of its
share, a disjoint set of points whose coefficients carry the weight `α`, or a set of batch columns, with
buffers for the share's field and coefficients. Every rank calls `calculate_spectrum!(coeffs, plan,
field)` with the whole field and ends with the whole spectrum.
"""
struct MPISpectralPlan{CT, NC, NB, KS, LP, FB, CB, SH, A, C, SP <: Tuple} <: FFS.AbstractSpectralPlan
    plan::LP                      # this rank's local plan; `nothing` when its share is empty
    fbuf::FB                      # this rank's share of the field
    cbuf::CB                      # its coefficients
    share::SH                     # point indices (a point partition) or batch columns
    α::A                          # the point share's weight; `nothing` for a batch partition
    comm::C
    spatial::SP
    spatial_out::NTuple{NC, Int}
    batch::NTuple{NB, Int}
    Nglob::Int
    B::Int
    ks::KS
end

Base.show(io::IO, p::MPISpectralPlan) =
    print(io, "MPISpectralPlan(", p.α === nothing ? "batch" : "points", ", share ", length(p.share), ")")

FFS.Plans.coefficient_size(p::MPISpectralPlan) = (p.spatial_out..., p.batch...)
FFS.Plans.coefficient_type(::MPISpectralPlan{CT}) where {CT} = CT
FFS.Plans.wavenumbers(p::MPISpectralPlan) = p.ks
FFS.Plans.close!(p::MPISpectralPlan) = (p.plan === nothing || FFS.close!(p.plan); nothing)

function FFS._plan_spectrum_mpi(t::SpectralBackends.AbstractSpectralBackend,
        exec::ComputationalBackends.MPIBackend, g::FlowGeometries.Grids.AbstractGrid, ::Type{T}, ms::Tuple;
        batch::Tuple = (), kwargs...) where {T}
    _require_init()
    comm = _comm(exec)
    inner = ComputationalBackends.local_backend(exec)
    FT = real(float(T))
    R = T <: Real
    bt = NTuple{length(batch), Int}(batch)
    B = prod(bt; init = 1)
    sp = size(g)
    Nglob = prod(sp)
    points = FFS._point_partition(t, g, kwargs)
    share = _rank_share(points, MPI.Comm_rank(comm), MPI.Comm_size(comm), Nglob, B)
    plan, fbuf = if isempty(share)
        nothing, nothing
    elseif points
        FFS.plan_spectrum(t, inner, FFS._subgrid(g, share), T, ms; batch = bt,
            FFS._partition_kwargs(kwargs, g, share)...), Array{T}(undef, length(share), B)
    else
        FFS.plan_spectrum(t, inner, g, T, ms; batch = (length(share),), kwargs...),
            Array{T}(undef, sp..., length(share))
    end
    cbuf = plan === nothing ? nothing : FFS.allocate_coefficients(plan)
    α = points ? FT(FFS._partition_weight(g, share, get(kwargs, :weights, nothing))) : nothing
    so = FFS._coeff_spatial(g, ms, R)
    ks = FFS._partition_ks(g, ms, R)
    return MPISpectralPlan{FFS._coeff_eltype(g, T, FT), length(so), length(bt), typeof(ks), typeof(plan),
            typeof(fbuf), typeof(cbuf), typeof(share), typeof(α), typeof(comm), typeof(sp)}(
        plan, fbuf, cbuf, share, α, comm, sp, so, bt, Nglob, B, ks)
end

# The rows `idx` of the `(Nglob, B)` field, into `buf`.
function _gather_points!(buf, field, idx, Nglob::Int, B::Int)
    @inbounds for b in 1:B, (j, i) in enumerate(idx)
        buf[j, b] = field[i + (b - 1) * Nglob]
    end
    return buf
end

"""
    calculate_spectrum!(coeffs, plan::MPISpectralPlan, field) -> ks

Fill `coeffs` with the spectrum of the whole `field` on every rank: this rank transforms its share, and
`MPI.Allreduce!` sums the shares.
"""
function FFS.calculate_spectrum!(coeffs::AbstractArray{<:Number}, p::MPISpectralPlan, field)
    FFS.Plans._check_coefficients(coeffs, p)
    length(field) == p.Nglob * p.B || throw(DimensionMismatch(
        "field holds $(length(field)) values; this plan transforms $(p.B) field(s) of $(p.Nglob) points — " *
        "pass the matching `batch=` to plan_spectrum"))
    fill!(coeffs, zero(eltype(coeffs)))
    if p.plan !== nothing
        if p.α !== nothing
            _gather_points!(p.fbuf, field, p.share, p.Nglob, p.B)
            FFS.calculate_spectrum!(p.cbuf, p.plan, p.fbuf)
            coeffs .= p.α .* reshape(p.cbuf, size(coeffs))
        else
            copyto!(p.fbuf, selectdim(reshape(field, p.spatial..., p.B), length(p.spatial) + 1, p.share))
            FFS.calculate_spectrum!(p.cbuf, p.plan, p.fbuf)
            selectdim(reshape(coeffs, p.spatial_out..., p.B), length(p.spatial_out) + 1, p.share) .=
                reshape(p.cbuf, p.spatial_out..., length(p.share))
        end
    end
    MPI.Allreduce!(coeffs, +, p.comm)
    return p.ks
end

"""
    MPISynthesisPlan

A synthesis divided among the ranks of an MPI communicator: this rank holds the local synthesis plan of
its share, the points of a point set it evaluates the series at, or a set of batch columns. Every rank
calls `synthesize!(out, plan, coeffs)` with the whole spectrum and ends with the whole field.
"""
struct MPISynthesisPlan{OT, NB, LP, OB, SH, C, SP <: Tuple} <: FFS.AbstractSynthesisPlan
    plan::LP                      # this rank's local plan; `nothing` when its share is empty
    obuf::OB                      # the field it writes
    share::SH
    points::Bool
    comm::C
    spatial::SP
    nspec::Int                    # spectral dims of the coefficients
    batch::NTuple{NB, Int}
    Nglob::Int
    B::Int
end

Base.show(io::IO, p::MPISynthesisPlan) =
    print(io, "MPISynthesisPlan(", p.points ? "points" : "batch", ", share ", length(p.share), ")")

FFS.Plans.field_size(p::MPISynthesisPlan) = (p.spatial..., p.batch...)
FFS.Plans.field_type(::MPISynthesisPlan{OT}) where {OT} = OT
FFS.Plans.close!(p::MPISynthesisPlan) = (p.plan === nothing || FFS.close!(p.plan); nothing)

function FFS._plan_synthesis_mpi(t::SpectralBackends.AbstractSpectralBackend,
        exec::ComputationalBackends.MPIBackend, g::FlowGeometries.Grids.AbstractGrid, ::Type{T}, ms::Tuple;
        batch::Tuple = (), kwargs...) where {T}
    _require_init()
    comm = _comm(exec)
    inner = ComputationalBackends.local_backend(exec)
    bt = NTuple{length(batch), Int}(batch)
    B = prod(bt; init = 1)
    sp = size(g)
    Nglob = prod(sp)
    msn = NTuple{length(ms), Int}(Tuple(ms))
    points = FFS._point_partition_synthesis(t, g)
    share = _rank_share(points, MPI.Comm_rank(comm), MPI.Comm_size(comm), Nglob, B)
    plan = if isempty(share)
        nothing
    elseif points
        FFS.plan_synthesis(t, inner, FFS._subgrid(g, share), T, msn; batch = bt,
            FFS._partition_kwargs(kwargs, g, share)...)
    else
        FFS.plan_synthesis(t, inner, g, T, msn; batch = (length(share),), kwargs...)
    end
    obuf = plan === nothing ? nothing : FFS.Plans.allocate_field(plan)
    FT = real(float(eltype(g)))
    OT = T <: Real ? FT : Complex{FT}
    return MPISynthesisPlan{OT, length(bt), typeof(plan), typeof(obuf), typeof(share), typeof(comm), typeof(sp)}(
        plan, obuf, share, points, comm, sp, length(FFS._coeff_spatial(g, ms, T <: Real)), bt, Nglob, B)
end

"""
    synthesize!(out, plan::MPISynthesisPlan, coeffs) -> out

Evaluate the series `coeffs` at the grid's points into `out` on every rank: this rank evaluates its
share, and `MPI.Allreduce!` assembles the disjoint shares.
"""
function FFS.Plans.synthesize!(out::AbstractArray, p::MPISynthesisPlan, coeffs::AbstractArray)
    size(out) == FFS.Plans.field_size(p) || throw(DimensionMismatch(
        "out is $(size(out)); this plan writes $(FFS.Plans.field_size(p))"))
    fill!(out, zero(eltype(out)))
    if p.plan !== nothing
        if p.points
            FFS.Plans.synthesize!(p.obuf, p.plan, coeffs)
            reshape(out, p.Nglob, p.B)[p.share, :] .= reshape(p.obuf, length(p.share), p.B)
        else
            spec = ntuple(i -> size(coeffs, i), p.nspec)
            c = collect(selectdim(reshape(coeffs, spec..., p.B), p.nspec + 1, p.share))
            FFS.Plans.synthesize!(p.obuf, p.plan, c)
            selectdim(reshape(out, p.spatial..., p.B), length(p.spatial) + 1, p.share) .=
                reshape(p.obuf, p.spatial..., length(p.share))
        end
    end
    MPI.Allreduce!(out, +, p.comm)
    return out
end

end # module FlowFieldSpectraMPIExt
