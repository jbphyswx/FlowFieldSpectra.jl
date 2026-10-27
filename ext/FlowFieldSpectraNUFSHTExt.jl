module FlowFieldSpectraNUFSHTExt

using NUFSHT: NUFSHT
using FlowTransformBindings: FlowTransformBindings as FTB
using FlowFieldSpectra: FlowFieldSpectra as FFS
using ComputationalBackends: ComputationalBackends
using SpectralBackends: SpectralBackends as SB
using FlowGeometries: FlowGeometries

# Nonuniform spherical harmonic transform via NUFSHT. The field `(spatial…, batch…)` transforms `C`
# co-located slices per plan call (`batch_chunk`, the whole batch by default). NUFSHT's coefficient layout
# (FastSphericalHarmonics' `sph_mode`) is FFS's `sph_mode_index`, so its coefficients are FFS's. The plan
# over the fixed nodes is built once. Points `(θ, φ)` (colatitude, longitude) come from
# `Grids._sph_points`, a structured grid expanded in its field's column-major order.
#
# Both analyses are `NUFSHT.nusht_analyze!`. `Quadrature()` passes the grid's weights: the projection
# `Σⱼ wⱼ fⱼ Y_lm(xⱼ)` `calculate_spectrum` returns, against the direct sum's per-point weights
# (`DirectSum._sph_point_data`), a structured grid's latitude quadrature or a scattered grid's rescaled node
# measure. `LeastSquares()` passes none: the fit `A c ≈ f`, which recovers the coefficients a field was
# synthesized from. A masked grid's inactive cells enter as `inactive` says (`Grids.ZeroFill`,
# `Grids.Exclude`): the plan holds the nodes `Grids._transform_nodes` names, so a masked cell's `NaN` is
# never read.

FFS._available(::FFS._HostNUFSHT) = true

"""
    NUFSHTSphericalPlan{T}

Reusable NUFSHT plan: the fixed-node NUFSHT plan (point preset + NUFFT setup) over the nodes its
`inactive` holds, the coefficient and staged-field buffers, those nodes' indices and which of them read
the field, their quadrature weights or the LSMR workspace of a fit, and the last fit's outcome, all built
once for a fixed point set and batch shape. Reuse across many fields via `calculate_spectrum!`.
"""
struct NUFSHTSphericalPlan{T, CT, NB, A, P, CR, WS, QW, FW, AC, LV, KS} <: FFS.AbstractSpectralPlan
    plan::P                       # NUFSHT.NUSHTplan over the held nodes, ntrans = C
    C_real::CR                    # (Nθ, Nφ, C) real coefficient buffer
    ws::WS                        # FTB.LSMRWorkspace of a fit; `nothing` for a quadrature
    qw::QW                        # (n,) the held nodes' quadrature weights; `nothing` for a fit
    fw::FW                        # (n, C) one component of one chunk of the field at the held nodes
    act::AC                       # the held nodes' linear indices, `nothing` for every node
    live::LV                      # per held node, whether it reads the field; `nothing` where every one does
    status::Base.RefValue{FFS.Plans.AnalysisStatus{T}}  # NUFSHT's report of the last call
    N::Int                        # the grid's node count, the field's stride between batch slices
    lmax::Int
    Nθ::Int
    Nφ::Int
    batch::NTuple{NB, Int}
    B::Int                        # the declared total batch
    C::Int                        # transforms per execution
    analysis::A
    maxiter::Int
    rtol::T
    ks::KS
end

# Transforms per NUFSHT execution, `<= 0` meaning the whole batch in one. The best chunk depends on the
# NUFFT engine NUFSHT resolves and on the core count: NonuniformFFTs threads its CPU spreading over point
# blocks and treats `ntrans` as a per-buffer component axis, so a small chunk bounds its
# `nthreads × ntrans` working set; FINUFFT threads over `ntrans`, where a small chunk idles cores. The
# default is the whole batch; `benchmark/` carries the sweep.
const DEFAULT_BATCH_CHUNK = 0

# Custom show: the wrapped NUFSHT plan holds a NUFFT library plan, whose default printing can segfault.
Base.show(io::IO, p::NUFSHTSphericalPlan{T}) where {T} =
    print(io, "NUFSHTSphericalPlan{", T, "}(lmax=", p.lmax, ", B=", p.B, ", ", nameof(typeof(p.analysis)), ")")

FFS.Plans.coefficient_size(p::NUFSHTSphericalPlan) = (p.Nθ, p.Nφ, p.batch...)
FFS.Plans.coefficient_type(::NUFSHTSphericalPlan{T, CT}) where {T, CT} = CT
FFS.Plans.wavenumbers(p::NUFSHTSphericalPlan) = p.ks
FFS.Plans.analysis_status(p::NUFSHTSphericalPlan) = p.status[]
FFS.Plans.close!(p::NUFSHTSphericalPlan) = (NUFSHT.close!(p.plan); nothing)

# What an analysis holds: a fit's LSMR workspace, a quadrature's weights at the held nodes.
_fit_workspace(::FFS.Quadrature, plan) = nothing
_fit_workspace(::FFS.LeastSquares, plan) = FTB.LSMRWorkspace(plan)
_held_weights(::FFS.Quadrature, w, act) = FFS.Grids._at_points(w, act)
_held_weights(::FFS.LeastSquares, w, act) = nothing

# The report `nusht_analyze!` returns, in the plan's real type.
_status(r, ::Type{T}) where {T} =
    FFS.Plans.AnalysisStatus{T}((Bool(r.converged), Int(r.iterations), T(r.residual)))

# `fw[j, b] = part(field[node j, lo+b-1])` for the `nvalid` slices of one chunk, node `j` the plan's
# `j`-th held node and `part` the field component NUFSHT's real transform reads, zero at a node that reads
# no field; the chunk's unused slices are zero, so a partial final chunk contributes nothing through them.
function _stage_component!(fw::AbstractMatrix{FT}, field, act, live, N::Int, part, lo::Int,
        nvalid::Int) where {FT}
    n = size(fw, 1)
    @inbounds for b in axes(fw, 2)
        if b > nvalid
            for j in 1:n
                fw[j, b] = zero(FT)
            end
            continue
        end
        off = (lo + b - 2) * N
        for j in 1:n
            fw[j, b] = _reads(live, j) ? FT(part(field[off + _node(act, j)])) : zero(FT)
        end
    end
    return fw
end

@inline _node(::Nothing, j::Int) = j
@inline _node(act::AbstractVector{Int}, j::Int) = @inbounds act[j]
@inline _reads(::Nothing, ::Int) = true
@inline _reads(live::AbstractVector{Bool}, j::Int) = @inbounds live[j]

function _nusht_plan(::Type{CT}, ::Type{FT}, g, ms::Tuple, batch::NTuple{NB, Int}, exec;
        tol::Real, analysis::FFS.AbstractAnalysis, maxiter::Int, rtol::Real, nufft, batch_chunk::Int,
        sampling = nothing, weights = nothing, inactive::FFS.AbstractInactiveCells) where {CT, FT, NB}
    lmax = ms[1] - 1
    Nθ = lmax + 1
    Nφ = 2 * lmax + 1
    θ, φ, w = FFS.DirectSum._sph_point_data(g, FT; sampling,
        weights = FFS.Grids._inactive_weights(inactive, g, FT; sampling, weights))
    N = length(θ)
    act, live = FFS.Grids._transform_nodes(inactive, g, analysis)
    θa = FFS.Grids._at_points(θ, act)
    B = prod(batch; init = 1)
    C = FFS._chunk(B, B, batch_chunk)
    plan = NUFSHT.make_plan(FT, θa, FFS.Grids._at_points(φ, act), lmax; execution = exec, tol = tol,
        ntrans = C, nufft = nufft)
    try
        C_real = zeros(FT, Nθ, Nφ, C)
        ws = _fit_workspace(analysis, plan)
        qw = _held_weights(analysis, collect(FT, w), act)
        fw = zeros(FT, length(θa), C)
        ks = (0:lmax, -lmax:lmax)
        return NUFSHTSphericalPlan{FT, CT, NB, typeof(analysis), typeof(plan), typeof(C_real), typeof(ws),
                typeof(qw), typeof(fw), typeof(act), typeof(live), typeof(ks)}(plan, C_real, ws, qw, fw, act,
            live, Ref(FFS.Plans._quadrature_status(FT)), N, lmax, Nθ, Nφ, batch, B, C, analysis, maxiter,
            FT(rtol), ks)
    catch
        NUFSHT.close!(plan)
        rethrow()
    end
end

"""
    plan_spectrum(NUFSHTSpectralBackend(), execution, grid, T, ms; batch=(), tol, analysis=Quadrature(),
                  maxiter, rtol, nufft, batch_chunk=0, sampling=nothing, weights=nothing,
                  inactive=ZeroFill())

Reusable [`FFS.AbstractSpectralPlan`](@ref) for NUFSHT, running on one thread under `SerialBackend()` and
on `Threads.nthreads()` under `ThreadedBackend()`; execute it with `calculate_spectrum!(coeffs, plan,
field)` and release it with `close!`. `analysis` is the quadrature or the least-squares fit, which
`maxiter` and `rtol` bound and [`analysis_status`](@ref) reports. `nufft` selects NUFSHT's NUFFT library
(a FlowTransformBindings tag; `AutoSpectralBackend()` lets NUFSHT pick), `sampling` and `weights` set the
quadrature as `Grids._sph_point_weights` reads them, `inactive` how a masked grid's inactive cells enter,
and `batch_chunk` how many `batch` slices one NUFSHT execution carries, the whole batch by default.
"""
function FFS.plan_spectrum(::SB.AbstractNUFSHTSpectralBackend,
        exec::Union{ComputationalBackends.AbstractSerialBackend, ComputationalBackends.AbstractThreadedBackend},
        g::FlowGeometries.Grids.AbstractGrid{<:FFS.Grids.SphericalHarmonicGeometry},
        ::Type{T}, ms::Tuple; batch::Tuple = (), tol::Real = 1.0e-8,
        analysis::FFS.AbstractAnalysis = FFS.Quadrature(), maxiter::Int = 500, rtol::Real = 1.0e-6,
        nufft::SB.AbstractSpectralBackend = SB.AutoSpectralBackend(), batch_chunk::Int = DEFAULT_BATCH_CHUNK,
        sampling = nothing, weights = nothing, inactive::FFS.AbstractInactiveCells = FFS.ZeroFill()) where {T}
    FT = real(float(T))
    return _nusht_plan(FFS.sph_coeff_type(T, FT), FT, g, ms, NTuple{length(batch), Int}(batch), exec;
        tol, analysis, maxiter, rtol, nufft, batch_chunk, sampling, weights, inactive)
end

"""
    calculate_spectrum!(coeffs, plan::NUFSHTSphericalPlan, field) -> ks

Fill preallocated `coeffs` `(Nθ, Nφ, batch…)` with the scattered-spherical spectrum of `field`
`(N, batch…)`, reusing `plan`'s nodes / NUFSHT plan / LSMR solve workspace (no re-planning).
"""
function FFS.calculate_spectrum!(coeffs::AbstractArray{<:Number}, plan::NUFSHTSphericalPlan{T}, field) where {T}
    FFS.Plans._check_coefficients(coeffs, plan)
    length(field) == plan.N * plan.B || throw(DimensionMismatch(
        "field holds $(length(field)) values; this plan transforms $(plan.B) field(s) of $(plan.N) points — " *
        "pass the matching `batch=` to plan_spectrum"))
    fill!(coeffs, zero(eltype(coeffs)))
    s = _nusht_pass!(coeffs, plan, field, real, one(T))
    eltype(field) <: Real || (s = FFS.Plans._combine_status(s, _nusht_pass!(coeffs, plan, field, imag, im)))
    plan.status[] = s
    return plan.ks
end

# One real field component through NUFSHT, added into `coeffs` at weight `scale`, `plan.C` batch slices
# per execution; returns the outcome over the chunks. NUFSHT's transform is real and linear, so a real
# field runs this once with `real` (the identity on it) at weight 1, and a complex field runs it again with
# `imag` at weight `im`. NUFSHT's slots above degree `lmax` are zero, as FFS's are.
function _nusht_pass!(coeffs, plan::NUFSHTSphericalPlan{T}, field, part, scale) where {T}
    C = plan.C
    Cc = reshape(coeffs, plan.Nθ, plan.Nφ, plan.B)
    s = FFS.Plans._quadrature_status(T)
    for lo in 1:C:plan.B
        nvalid = min(C, plan.B - lo + 1)
        r = NUFSHT.nusht_analyze!(plan.C_real,
            _stage_component!(plan.fw, field, plan.act, plan.live, plan.N, part, lo, nvalid), plan.plan;
            weights = plan.qw, ws = plan.ws, maxiter = plan.maxiter, rtol = plan.rtol)
        s = FFS.Plans._combine_status(s, _status(r, T))
        view(Cc, :, :, lo:(lo + nvalid - 1)) .+= scale .* view(plan.C_real, :, :, 1:nvalid)
    end
    return s
end

# One-shot: build a plan for this field's batch shape, then execute it once.
function FFS._calculate_spectrum_nufsht(exec::Union{ComputationalBackends.AbstractSerialBackend, ComputationalBackends.AbstractThreadedBackend},
        g::FlowGeometries.Grids.AbstractGrid{<:FFS.Grids.SphericalHarmonicGeometry}, field::AbstractArray, ms::Tuple;
        tol::Real = 1.0e-8, analysis::FFS.AbstractAnalysis = FFS.Quadrature(), maxiter::Int = 500,
        rtol::Real = 1.0e-6, nufft::SB.AbstractSpectralBackend = SB.AutoSpectralBackend(),
        batch_chunk::Int = DEFAULT_BATCH_CHUNK,
        sampling = nothing, weights = nothing, inactive::FFS.AbstractInactiveCells = FFS.ZeroFill())
    FT = real(float(eltype(field)))
    batch = FFS.Grids.field_batch_shape(g, field)
    plan = _nusht_plan(FFS.sph_coeff_type(eltype(field), FT), FT, g, ms, batch, exec; tol, analysis, maxiter,
        rtol, nufft, batch_chunk, sampling, weights, inactive)
    try
        coeffs = FFS.allocate_coefficients(plan)
        return coeffs, FFS.calculate_spectrum!(coeffs, plan, field)
    finally
        FFS.close!(plan)
    end
end

# =============================================================================
# Synthesis via `nusht_type2!`, the series evaluated at the nodes, whose adjoint the quadrature analysis
# applies to the weighted field. NUFSHT's coefficients are real, so a complex coefficient array is
# evaluated one real component at a time, which the transform's linearity permits. NUFSHT's host plan
# reads every slot of its coefficient array, so the slots above degree `lmax` are staged as zero.
# =============================================================================

# Reusable synthesis: the fixed-node NUFSHT plan (points preset once) plus the real coefficient buffer
# its type-2 reads, the slots of degree `l ≤ lmax`, and the field buffer it writes.
struct NUFSHTSynthesisPlan{FT, R, NB, P, CR, VM, FB, SP} <: FFS.AbstractSynthesisPlan
    plan::P
    Cr::CR
    valid::VM
    f::FB
    lmax::Int
    Nθ::Int
    Nφ::Int
    N::Int
    spatial::SP
    batch::NTuple{NB, Int}
    B::Int
end

# A default show of a struct holding a NUFFT library plan can segfault.
Base.show(io::IO, p::NUFSHTSynthesisPlan{FT, R}) where {FT, R} =
    print(io, "NUFSHTSynthesisPlan{", FT, "}(lmax=", p.lmax, ", ", R ? "real" : "complex", ")")

FFS.Plans.field_size(p::NUFSHTSynthesisPlan) = (p.spatial..., p.batch...)
FFS.Plans.field_type(::NUFSHTSynthesisPlan{FT, R}) where {FT, R} = R ? FT : Complex{FT}
FFS.Plans.close!(p::NUFSHTSynthesisPlan) = (NUFSHT.close!(p.plan); nothing)

function FFS.Plans.plan_synthesis(::SB.AbstractNUFSHTSpectralBackend,
        exec::Union{ComputationalBackends.AbstractSerialBackend, ComputationalBackends.AbstractThreadedBackend},
        g::FlowGeometries.Grids.AbstractGrid{<:FFS.Grids.SphericalHarmonicGeometry},
        ::Type{T}, ms::NTuple{2, Int}; batch::Tuple = (), tol::Real = 1.0e-8,
        nufft::SB.AbstractSpectralBackend = SB.AutoSpectralBackend(), kwargs...) where {T}
    FT = real(float(T))
    lmax = ms[1] - 1
    Nθ = lmax + 1
    Nφ = 2 * lmax + 1
    θ, φ = FFS.Grids._sph_points(g)
    N = length(θ)
    bt = NTuple{length(batch), Int}(batch)
    B = prod(bt; init = 1)
    plan = NUFSHT.make_plan(FT, θ, φ, lmax; execution = exec, tol = tol, ntrans = B, nufft = nufft)
    sp = size(g)
    try
        Cr = zeros(FT, Nθ, Nφ, B)
        f = zeros(FT, N, B)
        valid = _valid_slots(lmax)
        return NUFSHTSynthesisPlan{FT, T <: Real, length(bt), typeof(plan), typeof(Cr), typeof(valid), typeof(f),
                typeof(sp)}(plan, Cr, valid, f, lmax, Nθ, Nφ, N, sp, bt, B)
    catch
        NUFSHT.close!(plan)
        rethrow()
    end
end

# The slots of the `(lmax+1, 2lmax+1)` coefficient array holding a degree `l ≤ lmax`: column `c` holds order
# `|m| = c ÷ 2` from degree `|m|` at row 1 (`FFS.sph_mode_index`).
_valid_slots(lmax::Int) = [row - 1 + col ÷ 2 <= lmax for row in 1:(lmax + 1), col in 1:(2 * lmax + 1)]

# NUFSHT's coefficients are real, so a complex array is evaluated one component at a time and
# recombined, which the transform's linearity permits.
function FFS.Plans.synthesize!(out::AbstractArray, plan::NUFSHTSynthesisPlan{FT, R},
        coeffs::AbstractArray) where {FT, R}
    size(out) == FFS.Plans.field_size(plan) || throw(DimensionMismatch(
        "out is $(size(out)); this plan writes $(FFS.Plans.field_size(plan))"))
    size(coeffs)[1:2] == (plan.Nθ, plan.Nφ) || throw(DimensionMismatch(
        "spherical coefficients must be (Nθ, Nφ) = ($(plan.Nθ), $(plan.Nφ)) on the spectral dims; " *
        "got $(size(coeffs)[1:2])"))
    Cc = reshape(coeffs, plan.Nθ, plan.Nφ, plan.B)
    O = reshape(out, plan.N, plan.B)
    plan.Cr .= ifelse.(plan.valid, real.(Cc), zero(FT))
    NUFSHT.nusht_type2!(plan.f, plan.Cr, plan.plan)
    O .= plan.f
    if !R
        plan.Cr .= ifelse.(plan.valid, imag.(Cc), zero(FT))
        NUFSHT.nusht_type2!(plan.f, plan.Cr, plan.plan)
        O .+= im .* plan.f
    end
    return out
end

# GPU NUFSHT comes from the NUFSHT × KernelAbstractions extension; this less specific method raises when
# KernelAbstractions is not loaded.
function FFS._calculate_spectrum_nufsht(::ComputationalBackends.AbstractGPUBackend,
        g::FlowGeometries.Grids.AbstractGrid{<:FFS.Grids.SphericalHarmonicGeometry},
        field::AbstractArray, ms::Tuple; kwargs...)
    throw(ArgumentError("GPU NUFSHT requires `using KernelAbstractions` (to place the transform on the execution backend)."))
end

end # module FlowFieldSpectraNUFSHTExt
