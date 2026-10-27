module FlowFieldSpectraDistributedExt

using Distributed: Distributed
using FlowFieldSpectra: FlowFieldSpectra as FFS
using ComputationalBackends: ComputationalBackends
using SpectralBackends: SpectralBackends
using FlowGeometries: FlowGeometries
using FlowTransformBindings: FlowTransformBindings as FTB

# =============================================================================
# DistributedBackend execution. A point-partitionable transform (DirectSum, NUFFT, NUFSHT projection) on
# a scattered grid splits the point axis and sums the α_w-weighted partial coefficients, which are
# additive over a disjoint point partition. Every other case (FFT, SHT, a NUFSHT solve, which couples all
# points) splits the trailing batch axis into disjoint slices and gathers them. Each worker runs the inner
# local backend, so distribution composes with Serial, Threaded and GPU. Results gather through `pmap`
# into a typed buffer; `@distributed (+)` infers `Any`.
# =============================================================================

# A chunk on a worker runs FastTransforms on one OpenMP thread there, the processes carrying the
# parallelism. The scope is set inside the `pmap` body because a scoped value does not cross processes.
_serial_ft(f) = Base.ScopedValues.with(f, FTB.FASTTRANSFORMS_THREADS => 1)

# Round-robin point chunks (balanced across the domain); contiguous batch chunks.
_index_chunks(N::Integer, nw::Integer) = [collect(w:max(1, nw):N) for w in 1:max(1, nw)]
_batch_chunks(B::Integer, nw::Integer) = [(((w - 1) * B) ÷ nw + 1):((w * B) ÷ nw) for w in 1:nw]

# ---- point-partition: Σ_w α_w · coeff_w ----
function _distributed_pointsum(inner, transform, g::FlowGeometries.Grids.AbstractGrid, field::AbstractArray, ms::Tuple; kwargs...)
    # The spatial dims collapse to one point axis; the batch dims ride through.
    Nglob = prod(size(g))
    fieldP = reshape(field, Nglob, FFS.Grids.field_batch_shape(g, field)...)
    nw = max(1, Distributed.nworkers())
    chunks = _index_chunks(Nglob, nw)
    partials = Distributed.pmap(chunks) do idx
        sg = FFS._subgrid(g, idx)
        sf = collect(selectdim(fieldP, 1, idx))
        kwi = FFS._partition_kwargs(kwargs, g, idx)
        cw, _ = _serial_ft(() -> FFS.calculate_spectrum(transform, inner, sg, sf, ms; kwi...))
        Array(cw)
    end
    FT = real(eltype(partials[1]))
    w = get(kwargs, :weights, nothing)
    α = [FT(FFS._partition_weight(g, idx, w)) for idx in chunks]
    # The workers' own element type: complex for a Cartesian spectrum, real for a real field's spherical
    # coefficients.
    coeffs = zeros(eltype(partials[1]), size(partials[1]))
    @inbounds for (k, cw) in enumerate(partials)
        coeffs .+= α[k] .* cw
    end
    return coeffs, FFS._partition_ks(g, ms, float(eltype(field)))
end

# ---- batch-partition: disjoint batch slices, gathered along the (flattened) batch axis ----
function _distributed_batch(inner, transform, g::FlowGeometries.Grids.AbstractGrid, field::AbstractArray, ms::Tuple; kwargs...)
    ns = ndims(g)
    sp = size(g)
    batch = FFS.Grids.field_batch_shape(g, field)
    B = prod(batch; init = 1)
    fieldB = reshape(field, sp..., B)
    nw = max(1, Distributed.nworkers())
    chunks = filter(!isempty, _batch_chunks(B, nw))
    E = float(eltype(field))
    spatial_out = FFS._coeff_spatial(g, ms, E)
    parts = Distributed.pmap(chunks) do bc
        fslice = collect(selectdim(fieldB, ns + 1, bc))
        cw, _ = _serial_ft(() -> FFS.calculate_spectrum(transform, inner, g, fslice, ms; kwargs...))
        (Array(cw), collect(bc))
    end
    coeffsB = Array{eltype(parts[1][1])}(undef, spatial_out..., B)
    for (cw, bc) in parts
        @inbounds selectdim(coeffsB, ndims(coeffsB), bc) .= reshape(cw, spatial_out..., length(bc))
    end
    return reshape(coeffsB, spatial_out..., batch...), FFS._partition_ks(g, ms, E)
end

function FFS._calculate_spectrum_distributed(transform, exec::ComputationalBackends.AbstractDistributedBackend,
        g::FlowGeometries.Grids.AbstractGrid, field::AbstractArray, ms::Tuple; kwargs...)
    inner = ComputationalBackends.local_backend(exec)
    return FFS._point_partition(transform, g, kwargs) ?
        _distributed_pointsum(inner, transform, g, field, ms; kwargs...) :
        _distributed_batch(inner, transform, g, field, ms; kwargs...)
end

# =============================================================================
# Reusable plans. Each worker builds the local plan for its share once and keeps it; an execution sends
# each worker its share of the field (or the coefficients its points read) and combines what comes back
# as the one-shot does. A `Future` from `remotecall` holds its value on the worker that computed it, so
# `fetch` there returns the plan itself.
# =============================================================================

# The workers and their nonempty shares: point indices for a point partition, batch columns otherwise.
function _shares(points::Bool, Nglob::Int, B::Int)
    ws = Distributed.workers()
    chunks = points ? _index_chunks(Nglob, length(ws)) : _batch_chunks(B, length(ws))
    keep = findall(!isempty, chunks)
    return ws[keep], chunks[keep]
end

# One local plan per share, each built on its worker; `build(share)` runs there.
function _remote_plans(build, pids, shares)
    futs = [Distributed.remotecall(() -> _serial_ft(() -> build(s)), w) for (w, s) in zip(pids, shares)]
    # Raise a worker's build error here, unwrapped, before any execution reads the plan.
    for (w, f) in zip(pids, futs)
        try
            Distributed.remotecall_fetch(_built, w, f)
        catch e
            throw(e isa Distributed.RemoteException ? e.captured.ex : e)
        end
    end
    return futs
end
_built(fut) = (fetch(fut); nothing)
_close_share(fut) = (FFS.close!(fetch(fut)); nothing)

"""
    DistributedSpectralPlan

A spectrum plan divided among the Distributed.jl workers. Each worker holds the local plan of its share,
a disjoint set of points whose coefficients are summed with the weights `α`, or a set of batch columns,
built once and kept on that worker. Execute with `calculate_spectrum!(coeffs, plan, field)`;
[`close!`](@ref) releases every worker's plan.
"""
struct DistributedSpectralPlan{CT, NC, NB, KS, PW <: AbstractVector{Int}, FS <: AbstractVector,
        SH <: AbstractVector, A, SP <: Tuple} <: FFS.AbstractSpectralPlan
    pids::PW
    plans::FS                     # one `Future` per worker, holding its local plan there
    shares::SH                    # each worker's point indices (a point partition) or batch columns
    α::A                          # each point share's weight; `nothing` for a batch partition
    spatial::SP                   # the grid's spatial shape
    spatial_out::NTuple{NC, Int}
    batch::NTuple{NB, Int}
    Nglob::Int
    B::Int
    ks::KS
end

Base.show(io::IO, p::DistributedSpectralPlan) =
    print(io, "DistributedSpectralPlan(", length(p.pids), " workers, ",
        p.α === nothing ? "batch" : "points", ")")

FFS.Plans.coefficient_size(p::DistributedSpectralPlan) = (p.spatial_out..., p.batch...)
FFS.Plans.coefficient_type(::DistributedSpectralPlan{CT}) where {CT} = CT
FFS.Plans.wavenumbers(p::DistributedSpectralPlan) = p.ks
function FFS.Plans.close!(p::DistributedSpectralPlan)
    foreach((w, f) -> Distributed.remotecall_wait(_close_share, w, f), p.pids, p.plans)
    return nothing
end

function FFS._plan_spectrum_distributed(t::SpectralBackends.AbstractSpectralBackend,
        exec::ComputationalBackends.AbstractDistributedBackend, g::FlowGeometries.Grids.AbstractGrid,
        ::Type{T}, ms::Tuple; batch::Tuple = (), kwargs...) where {T}
    inner = ComputationalBackends.local_backend(exec)
    FT = real(float(T))
    bt = NTuple{length(batch), Int}(batch)
    B = prod(bt; init = 1)
    Nglob = prod(size(g))
    points = FFS._point_partition(t, g, kwargs)
    pids, shares = _shares(points, Nglob, B)
    build = points ?
        (s -> FFS.plan_spectrum(t, inner, FFS._subgrid(g, s), T, ms; batch = bt,
            FFS._partition_kwargs(kwargs, g, s)...)) :
        (s -> FFS.plan_spectrum(t, inner, g, T, ms; batch = (length(s),), kwargs...))
    plans = _remote_plans(build, pids, shares)
    w = get(kwargs, :weights, nothing)
    α = points ? [FT(FFS._partition_weight(g, s, w)) for s in shares] : nothing
    so = FFS._coeff_spatial(g, ms, T)
    ks = FFS._partition_ks(g, ms, T)
    sp = size(g)
    return DistributedSpectralPlan{FFS._coeff_eltype(g, T, FT), length(so), length(bt), typeof(ks),
            typeof(pids), typeof(plans), typeof(shares), typeof(α), typeof(sp)}(
        pids, plans, shares, α, sp, so, bt, Nglob, B, ks)
end

# On a worker: its plan executed on its share, returning the coefficients.
function _spectrum_share(fut, data)
    plan = fetch(fut)
    c = FFS.allocate_coefficients(plan)
    _serial_ft(() -> FFS.calculate_spectrum!(c, plan, data))
    return c
end

"""
    calculate_spectrum!(coeffs, plan::DistributedSpectralPlan, field) -> ks

Fill `coeffs` with the spectrum of `field`, each worker transforming its share with the plan it holds.
"""
function FFS.calculate_spectrum!(coeffs::AbstractArray{<:Number}, p::DistributedSpectralPlan, field)
    FFS.Plans._check_coefficients(coeffs, p)
    length(field) == p.Nglob * p.B || throw(DimensionMismatch(
        "field holds $(length(field)) values; this plan transforms $(p.B) field(s) of $(p.Nglob) points — " *
        "pass the matching `batch=` to plan_spectrum"))
    points = p.α !== nothing
    parts = Vector{Any}(undef, length(p.pids))
    @sync for k in eachindex(p.pids)
        data = points ? reshape(field, p.Nglob, p.B)[p.shares[k], :] :
            collect(selectdim(reshape(field, p.spatial..., p.B), length(p.spatial) + 1, p.shares[k]))
        @async parts[k] = Distributed.remotecall_fetch(_spectrum_share, p.pids[k], p.plans[k], data)
    end
    points ? _sum_shares!(coeffs, p, parts) : _gather_shares!(coeffs, p, parts)
    return p.ks
end

# A point partition: `Σ_w α_w C_w`.
function _sum_shares!(coeffs, p::DistributedSpectralPlan, parts)
    fill!(coeffs, zero(eltype(coeffs)))
    for (k, c) in enumerate(parts)
        coeffs .+= p.α[k] .* reshape(c, size(coeffs))
    end
    return coeffs
end

# A batch partition: each worker's columns in place.
function _gather_shares!(coeffs, p::DistributedSpectralPlan, parts)
    coeffsB = reshape(coeffs, p.spatial_out..., p.B)
    for (k, c) in enumerate(parts)
        bc = p.shares[k]
        selectdim(coeffsB, ndims(coeffsB), bc) .= reshape(c, p.spatial_out..., length(bc))
    end
    return coeffs
end

"""
    DistributedSynthesisPlan

A synthesis divided among the Distributed.jl workers: each worker holds the local synthesis plan of its
share, the points of a point set it evaluates the series at, or a set of batch columns. Execute with
`synthesize!(out, plan, coeffs)`; [`close!`](@ref) releases every worker's plan.
"""
struct DistributedSynthesisPlan{OT, NB, PW <: AbstractVector{Int}, FS <: AbstractVector,
        SH <: AbstractVector, SP <: Tuple} <: FFS.AbstractSynthesisPlan
    pids::PW
    plans::FS
    shares::SH
    points::Bool
    spatial::SP
    batch::NTuple{NB, Int}
    Nglob::Int
    B::Int
end

Base.show(io::IO, p::DistributedSynthesisPlan) =
    print(io, "DistributedSynthesisPlan(", length(p.pids), " workers, ", p.points ? "points" : "batch", ")")

FFS.Plans.field_size(p::DistributedSynthesisPlan) = (p.spatial..., p.batch...)
FFS.Plans.field_type(::DistributedSynthesisPlan{OT}) where {OT} = OT
function FFS.Plans.close!(p::DistributedSynthesisPlan)
    foreach((w, f) -> Distributed.remotecall_wait(_close_share, w, f), p.pids, p.plans)
    return nothing
end

function FFS._plan_synthesis_distributed(t::SpectralBackends.AbstractSpectralBackend,
        exec::ComputationalBackends.AbstractDistributedBackend, g::FlowGeometries.Grids.AbstractGrid,
        ::Type{T}, ms::Tuple; batch::Tuple = (), kwargs...) where {T}
    inner = ComputationalBackends.local_backend(exec)
    bt = NTuple{length(batch), Int}(batch)
    B = prod(bt; init = 1)
    Nglob = prod(size(g))
    msn = NTuple{length(ms), Int}(Tuple(ms))
    points = FFS._point_partition_synthesis(t, g)
    pids, shares = _shares(points, Nglob, B)
    build = points ?
        (s -> FFS.plan_synthesis(t, inner, FFS._subgrid(g, s), T, msn; batch = bt,
            FFS._partition_kwargs(kwargs, g, s)...)) :
        (s -> FFS.plan_synthesis(t, inner, g, T, msn; batch = (length(s),), kwargs...))
    plans = _remote_plans(build, pids, shares)
    FT = real(float(eltype(g)))
    OT = T <: Real ? FT : Complex{FT}
    sp = size(g)
    return DistributedSynthesisPlan{OT, length(bt), typeof(pids), typeof(plans), typeof(shares), typeof(sp)}(
        pids, plans, shares, points, sp, bt, Nglob, B)
end

# On a worker: its synthesis plan evaluated on its coefficients.
function _synthesis_share(fut, c)
    plan = fetch(fut)
    out = FFS.Plans.allocate_field(plan)
    _serial_ft(() -> FFS.Plans.synthesize!(out, plan, c))
    return out
end

"""
    synthesize!(out, plan::DistributedSynthesisPlan, coeffs) -> out

Evaluate the series `coeffs` at the grid's points into `out`, each worker evaluating its share.
"""
function FFS.Plans.synthesize!(out::AbstractArray, p::DistributedSynthesisPlan{OT, NB},
        coeffs::AbstractArray) where {OT, NB}
    size(out) == FFS.Plans.field_size(p) || throw(DimensionMismatch(
        "out is $(size(out)); this plan writes $(FFS.Plans.field_size(p))"))
    # The coefficients' spectral dims precede the plan's `NB` batch dims.
    spec = ntuple(i -> size(coeffs, i), Val(ndims(coeffs) - NB))
    nsp = length(spec)
    parts = Vector{Any}(undef, length(p.pids))
    @sync for k in eachindex(p.pids)
        c = p.points ? coeffs : collect(selectdim(reshape(coeffs, spec..., p.B), nsp + 1, p.shares[k]))
        @async parts[k] = Distributed.remotecall_fetch(_synthesis_share, p.pids[k], p.plans[k], c)
    end
    if p.points
        O = reshape(out, p.Nglob, p.B)
        for (k, o) in enumerate(parts)
            O[p.shares[k], :] .= reshape(o, length(p.shares[k]), p.B)
        end
    else
        O = reshape(out, p.spatial..., p.B)
        for (k, o) in enumerate(parts)
            bc = p.shares[k]
            selectdim(O, ndims(O), bc) .= reshape(o, p.spatial..., length(bc))
        end
    end
    return out
end

end # module FlowFieldSpectraDistributedExt
