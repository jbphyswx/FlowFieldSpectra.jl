module FlowFieldSpectraNUFSHTKernelAbstractionsExt

using NUFSHT: NUFSHT
using FlowTransformBindings: FlowTransformBindings as FTB
using KernelAbstractions: KernelAbstractions as KA
using FlowFieldSpectra: FlowFieldSpectra as FFS
using ComputationalBackends: ComputationalBackends
using SpectralBackends: SpectralBackends as SB
using FlowGeometries: FlowGeometries

# =============================================================================
# GPU NUFSHT on a scattered sphere: `NUFSHT.make_plan(…; execution = GPUBackend(b))` puts the nodes, and so
# the plan, in `b`'s memory, its every step running on that device; `KA.CPU()` runs the same code.
# NUFSHT's coefficient layout is FFS's (`sph_mode_index`). The reusable plan holds the device plan, the
# device field and coefficient buffers, and host staging for a host field or coefficient array. Points
# `(θ, φ)` come from `Grids._sph_points`.
# =============================================================================

FFS._available(::FFS._DeviceNUFSHT) = true

"""
    NUFSHTSphericalGPUPlan{T}

Device-resident reusable NUFSHT plan (GPU execution): the device NUFSHT plan over the nodes its
`inactive` holds, carrying `C` transforms per execution, the device field and coefficient buffers, the
host staging for a host field or coefficient array, those nodes' indices and which of them read the
field, their device quadrature weights or the device LSMR workspace of a fit, and NUFSHT's report of the
last fit. Built once for a fixed point set + batch shape; reuse via `calculate_spectrum!`.
"""
struct NUFSHTSphericalGPUPlan{T, CT, NB, A, P, FD, FH, CD, HB, WS, QW, AC, AD, LV, LD, KS} <: FFS.AbstractSpectralPlan
    plan::P            # device-resident NUFSHT plan over the held nodes, ntrans = C
    fd::FD             # device (n, C) field buffer at the held nodes
    fh::FH             # host (n, C) staging for a host field's held nodes or a complex field's component
    Cd::CD             # device (Nθ, Nφ, C) real coeff buffer
    Cr_host::HB        # host (Nθ, Nφ, C) copy of `Cd` for a host coefficient array
    ws::WS             # device FTB.LSMRWorkspace of a fit; `nothing` for a quadrature
    wd::QW             # device (n,) the held nodes' quadrature weights; `nothing` for a fit
    act::AC            # host indices of the held nodes, `nothing` for every node
    actd::AD           # the same indices on the device
    live::LV           # host flag per held node, whether it reads the field; `nothing` where every one does
    lived::LD          # the same flags on the device
    status::Base.RefValue{FFS.Plans.AnalysisStatus{T}}  # NUFSHT's report of the last call
    N::Int             # the grid's node count, the field's stride between batch slices
    lmax::Int
    Nθ::Int
    Nφ::Int
    batch::NTuple{NB, Int}
    B::Int             # the declared total batch
    C::Int             # transforms per execution
    analysis::A
    maxiter::Int
    rtol::T
    ks::KS
end

# Custom show: the wrapped NUFSHT plan holds a NUFFT library plan, whose default printing can segfault.
Base.show(io::IO, p::NUFSHTSphericalGPUPlan{T}) where {T} =
    print(io, "NUFSHTSphericalGPUPlan{", T, "}(lmax=", p.lmax, ", B=", p.B, ", ", nameof(typeof(p.analysis)), ")")

# What an analysis holds: a fit's LSMR workspace, a quadrature's weights at the held nodes in the plan's
# memory.
_fit_workspace(::FFS.Quadrature, plan) = nothing
_fit_workspace(::FFS.LeastSquares, plan) = FTB.LSMRWorkspace(plan)
_held_weights(::FFS.Quadrature, backend, w) = copyto!(KA.allocate(backend, eltype(w), length(w)), w)
_held_weights(::FFS.LeastSquares, backend, w) = nothing

# The report `nusht_analyze!` returns, in the plan's real type.
_status(r, ::Type{T}) where {T} =
    FFS.Plans.AnalysisStatus{T}((Bool(r.converged), Int(r.iterations), T(r.residual)))

function _nusht_gpu_plan(::Type{CT}, ::Type{FT}, exec, g, ms::Tuple, batch::NTuple{NB, Int};
        tol::Real, analysis::FFS.AbstractAnalysis, maxiter::Int, rtol::Real, nufft, batch_chunk::Int,
        sampling = nothing, weights = nothing, inactive::FFS.AbstractInactiveCells) where {CT, FT, NB}
    backend = exec.backend
    lmax = ms[1] - 1
    Nθ = lmax + 1
    Nφ = 2 * lmax + 1
    θ, φ, w = FFS.DirectSum._sph_point_data(g, FT; sampling,
        weights = FFS.Grids._inactive_weights(inactive, g, FT; sampling, weights))
    N = length(θ)
    act, live = FFS.Grids._transform_nodes(inactive, g, analysis)
    n = act === nothing ? N : length(act)
    B = prod(batch; init = 1)
    C = FFS._chunk(B, B, batch_chunk)
    plan = NUFSHT.make_plan(FT, FFS.Grids._at_points(θ, act), FFS.Grids._at_points(φ, act), lmax;
        execution = exec, tol = tol, ntrans = C, nufft = nufft, nthreads = FFS._backend_nthreads(exec))
    try
        fd = KA.allocate(backend, FT, n, C)
        fh = zeros(FT, n, C)
        Cd = KA.zeros(backend, FT, Nθ, Nφ, C)
        Cr_host = zeros(FT, Nθ, Nφ, C)
        ws = _fit_workspace(analysis, plan)
        wd = _held_weights(analysis, backend, FFS.Grids._at_points(collect(FT, w), act))
        actd = act === nothing ? nothing : copyto!(KA.allocate(backend, Int, n), act)
        lived = live === nothing ? nothing : copyto!(KA.allocate(backend, Bool, n), collect(Bool, live))
        ks = (0:lmax, -lmax:lmax)
        return NUFSHTSphericalGPUPlan{FT, CT, NB, typeof(analysis), typeof(plan), typeof(fd), typeof(fh),
                typeof(Cd), typeof(Cr_host), typeof(ws), typeof(wd), typeof(act), typeof(actd), typeof(live),
                typeof(lived), typeof(ks)}(
            plan, fd, fh, Cd, Cr_host, ws, wd, act, actd, live, lived, Ref(FFS.Plans._quadrature_status(FT)),
            N, lmax, Nθ, Nφ, batch, B, C, analysis, maxiter, FT(rtol), ks)
    catch
        NUFSHT.close!(plan)
        rethrow()
    end
end

FFS.Plans.coefficient_size(p::NUFSHTSphericalGPUPlan) = (p.Nθ, p.Nφ, p.batch...)
FFS.Plans.coefficient_type(::NUFSHTSphericalGPUPlan{T, CT}) where {T, CT} = CT
FFS.Plans.wavenumbers(p::NUFSHTSphericalGPUPlan) = p.ks
FFS.Plans.analysis_status(p::NUFSHTSphericalGPUPlan) = p.status[]
FFS.Plans.close!(p::NUFSHTSphericalGPUPlan) = (NUFSHT.close!(p.plan); nothing)

"""
    plan_spectrum(NUFSHTSpectralBackend(), GPUBackend(dev), grid, T, ms; batch=(), tol, analysis=Quadrature(),
                  maxiter, rtol, nufft, batch_chunk=0, sampling=nothing, weights=nothing)

The device form of the NUFSHT plan, with the nodes and every transform buffer on `dev`. `batch_chunk` is
how many `batch` slices one NUFSHT execution carries, as on the host; the default runs the whole batch in
one.
"""
function FFS.plan_spectrum(::SB.AbstractNUFSHTSpectralBackend, exec::ComputationalBackends.GPUBackend{<:KA.Backend},
        g::FlowGeometries.Grids.AbstractGrid{<:FFS.Grids.SphericalHarmonicGeometry},
        ::Type{T}, ms::Tuple; batch::Tuple = (), tol::Real = 1.0e-8,
        analysis::FFS.AbstractAnalysis = FFS.Quadrature(), maxiter::Int = 500, rtol::Real = 1.0e-6,
        nufft::SB.AbstractSpectralBackend = SB.AutoSpectralBackend(), batch_chunk::Int = 0,
        sampling = nothing, weights = nothing, inactive::FFS.AbstractInactiveCells = FFS.ZeroFill()) where {T}
    FT = real(float(T))
    return _nusht_gpu_plan(FFS.sph_coeff_type(T, FT), FT, exec, g, ms, NTuple{length(batch), Int}(batch);
        tol, analysis, maxiter, rtol, nufft, batch_chunk, sampling, weights, inactive)
end

function FFS.calculate_spectrum!(coeffs::AbstractArray{<:Number}, plan::NUFSHTSphericalGPUPlan{T}, field) where {T}
    FFS.Plans._check_coefficients(coeffs, plan)
    length(field) == plan.N * plan.B || throw(DimensionMismatch(
        "field holds $(length(field)) values; this plan transforms $(plan.B) field(s) of " *
        "$(plan.N) points — pass the matching `batch=` to plan_spectrum"))
    fill!(coeffs, zero(eltype(coeffs)))
    s = _nusht_gpu_pass!(coeffs, plan, field, real, one(T))
    eltype(field) <: Real ||
        (s = FFS.Plans._combine_status(s, _nusht_gpu_pass!(coeffs, plan, field, imag, im)))
    plan.status[] = s
    return plan.ks
end

# Slices `lo … lo + nvalid - 1` of one field component at the plan's held nodes in the device buffer, zero
# at a node that reads no field and in the columns past `nvalid`, so a partial final chunk contributes
# nothing through them. A real field on every node crosses in one copy, and on the active nodes of a
# device field in one device gather; a host field's active nodes, or one component of a complex field,
# are written into the host buffer first.
_stage_chunk!(p::NUFSHTSphericalGPUPlan, field::AbstractArray{<:Real}, part, lo::Int, nvalid::Int) =
    _stage_real!(p, p.act, field, lo, nvalid)
_stage_chunk!(p::NUFSHTSphericalGPUPlan, field::AbstractArray{<:Complex}, part, lo::Int, nvalid::Int) =
    _stage_host!(p, field, part, lo, nvalid)

function _stage_real!(p::NUFSHTSphericalGPUPlan, ::Nothing, field, lo::Int, nvalid::Int)
    n, C = size(p.fd)
    copyto!(p.fd, 1, field, (lo - 1) * n + 1, nvalid * n)
    nvalid < C && fill!(view(p.fd, :, (nvalid + 1):C), zero(eltype(p.fd)))
    # `ifelse` selects a value, so a non-reading node's `NaN` stays out.
    p.lived === nothing || (p.fd .= ifelse.(p.lived, p.fd, zero(eltype(p.fd))))
    return p.fd
end
_stage_real!(p::NUFSHTSphericalGPUPlan, ::AbstractVector{Int}, field::Array, lo::Int, nvalid::Int) =
    _stage_host!(p, field, real, lo, nvalid)
function _stage_real!(p::NUFSHTSphericalGPUPlan, ::AbstractVector{Int}, field, lo::Int, nvalid::Int)
    n, C = size(p.fd)
    FFS._gather_scaled!(p.fd, 0, field, (lo - 1) * p.N, p.N, p.actd, one(eltype(p.fd)), n, nvalid, false,
        false)
    nvalid < C && fill!(view(p.fd, :, (nvalid + 1):C), zero(eltype(p.fd)))
    return p.fd
end

function _stage_host!(p::NUFSHTSphericalGPUPlan, field, part, lo::Int, nvalid::Int)
    fh = p.fh
    FT = eltype(fh)
    n, C = size(fh)
    @inbounds for b in 1:C
        off = (lo + b - 2) * p.N
        for j in 1:n
            fh[j, b] = (b <= nvalid && _reads(p.live, j)) ? FT(part(field[off + _node(p.act, j)])) : zero(FT)
        end
    end
    copyto!(p.fd, fh)
    return p.fd
end

@inline _node(::Nothing, j::Int) = j
@inline _node(act::AbstractVector{Int}, j::Int) = @inbounds act[j]
@inline _reads(::Nothing, ::Int) = true
@inline _reads(live::AbstractVector{Bool}, j::Int) = @inbounds live[j]

# One real field component through the device NUFSHT, `plan.C` batch slices per execution, added into
# `coeffs` at weight `scale`; returns NUFSHT's report over the chunks. NUFSHT's transform is real and
# linear, so a real field runs this once with `real` (the identity on it) at weight 1, and a complex field
# runs it again with `imag` at weight `im`. NUFSHT's slots above degree `lmax` are zero, as FFS's are.
function _nusht_gpu_pass!(coeffs, plan::NUFSHTSphericalGPUPlan{T}, field, part, scale) where {T}
    C = plan.C
    Cc = reshape(coeffs, plan.Nθ, plan.Nφ, plan.B)
    s = FFS.Plans._quadrature_status(T)
    for lo in 1:C:plan.B
        nvalid = min(C, plan.B - lo + 1)
        _stage_chunk!(plan, field, part, lo, nvalid)
        r = NUFSHT.nusht_analyze!(plan.Cd, plan.fd, plan.plan; weights = plan.wd, ws = plan.ws,
            maxiter = plan.maxiter, rtol = plan.rtol)
        s = FFS.Plans._combine_status(s, _status(r, T))
        view(Cc, :, :, lo:(lo + nvalid - 1)) .+= scale .* view(_coeffs_like(Cc, plan), :, :, 1:nvalid)
    end
    return s
end

# The plan's coefficients in the memory of `A`: `Cd` itself beside a device array, its host copy beside a
# host one.
_coeffs_like(::Array, p::NUFSHTSphericalGPUPlan) = copyto!(p.Cr_host, p.Cd)
_coeffs_like(::AbstractArray, p::NUFSHTSphericalGPUPlan) = p.Cd

# One-shot GPU: build a device plan for this field's batch shape, then execute it once.
# More specific than the NUFSHT-only `AbstractGPUBackend` stub, so it wins when KernelAbstractions is
# loaded.
function FFS._calculate_spectrum_nufsht(exec::ComputationalBackends.GPUBackend{<:KA.Backend},
        g::FlowGeometries.Grids.AbstractGrid{<:FFS.Grids.SphericalHarmonicGeometry},
        field::AbstractArray, ms::Tuple;
        tol::Real = 1.0e-8, analysis::FFS.AbstractAnalysis = FFS.Quadrature(), maxiter::Int = 500,
        rtol::Real = 1.0e-6, nufft::SB.AbstractSpectralBackend = SB.AutoSpectralBackend(), batch_chunk::Int = 0,
        sampling = nothing, weights = nothing, inactive::FFS.AbstractInactiveCells = FFS.ZeroFill())
    FT = real(float(eltype(field)))
    batch = FFS.Grids.field_batch_shape(g, field)
    plan = _nusht_gpu_plan(FFS.sph_coeff_type(eltype(field), FT), FT, exec, g, ms, batch;
        tol, analysis, maxiter, rtol, nufft, batch_chunk, sampling, weights, inactive)
    try
        coeffs = FFS._allocate_coefficients_like(field, plan)
        return coeffs, FFS.calculate_spectrum!(coeffs, plan, field)
    finally
        FFS.close!(plan)
    end
end

# =============================================================================
# Synthesis on a device backend via `nusht_type2!`, the series evaluated at the nodes. The plan is built
# under the execution backend, so NUFSHT places it in that device's memory. NUFSHT's coefficients are
# real, so a complex coefficient array is evaluated one real component at a time, which the transform's
# linearity permits. The slots above degree `lmax` are staged as zero: NUFSHT's host plan (`KA.CPU()`)
# reads every slot.
# =============================================================================

# Reusable device synthesis: the device-resident NUFSHT plan (nodes preset once), its coefficient and
# field buffers, the slots of degree `l ≤ lmax` in device and host memory, and host staging for host
# coefficient and field arrays.
struct NUFSHTSynthesisGPUPlan{FT, R, NB, P, CD, FD, VD, HB, VH, FH, SP} <: FFS.AbstractSynthesisPlan
    plan::P
    Cd::CD
    fd::FD
    validd::VD
    Cr_host::HB
    valid::VH
    f_host::FH
    lmax::Int
    Nθ::Int
    Nφ::Int
    N::Int
    spatial::SP
    batch::NTuple{NB, Int}
    B::Int
end

# A default show of a struct holding a NUFFT library plan can segfault.
Base.show(io::IO, p::NUFSHTSynthesisGPUPlan{FT, R}) where {FT, R} =
    print(io, "NUFSHTSynthesisGPUPlan{", FT, "}(lmax=", p.lmax, ", ", R ? "real" : "complex", ")")

FFS.Plans.field_size(p::NUFSHTSynthesisGPUPlan) = (p.spatial..., p.batch...)
FFS.Plans.field_type(::NUFSHTSynthesisGPUPlan{FT, R}) where {FT, R} = R ? FT : Complex{FT}
FFS.Plans.close!(p::NUFSHTSynthesisGPUPlan) = (NUFSHT.close!(p.plan); nothing)

function FFS.Plans.plan_synthesis(::SB.AbstractNUFSHTSpectralBackend,
        exec::ComputationalBackends.GPUBackend{<:KA.Backend},
        g::FlowGeometries.Grids.AbstractGrid{<:FFS.Grids.SphericalHarmonicGeometry},
        ::Type{T}, ms::NTuple{2, Int}; batch::Tuple = (), tol::Real = 1.0e-8,
        nufft::SB.AbstractSpectralBackend = SB.AutoSpectralBackend(), kwargs...) where {T}
    FT = real(float(T))
    backend = exec.backend
    lmax = ms[1] - 1
    Nθ = lmax + 1
    Nφ = 2 * lmax + 1
    θ, φ = FFS.Grids._sph_points(g)
    N = length(θ)
    bt = NTuple{length(batch), Int}(batch)
    B = prod(bt; init = 1)
    plan = NUFSHT.make_plan(FT, FT.(θ), FT.(φ), lmax; execution = exec, tol = tol, ntrans = B, nufft = nufft,
        nthreads = FFS._backend_nthreads(exec))
    sp = size(g)
    try
        Cd = KA.zeros(backend, FT, Nθ, Nφ, B)
        fd = KA.allocate(backend, FT, N, B)
        valid = [row - 1 + col ÷ 2 <= lmax for row in 1:Nθ, col in 1:Nφ]   # `FFS.sph_mode_index`'s slots
        validd = copyto!(KA.allocate(backend, Bool, Nθ, Nφ), valid)
        Cr_host = zeros(FT, Nθ, Nφ, B)
        f_host = zeros(FT, N, B)
        return NUFSHTSynthesisGPUPlan{FT, T <: Real, length(bt), typeof(plan), typeof(Cd), typeof(fd), typeof(validd),
                typeof(Cr_host), typeof(valid), typeof(f_host), typeof(sp)}(plan, Cd, fd, validd, Cr_host, valid,
            f_host, lmax, Nθ, Nφ, N, sp, bt, B)
    catch
        NUFSHT.close!(plan)
        rethrow()
    end
end

function FFS.Plans.synthesize!(out::AbstractArray, plan::NUFSHTSynthesisGPUPlan{FT, R},
        coeffs::AbstractArray) where {FT, R}
    size(out) == FFS.Plans.field_size(plan) || throw(DimensionMismatch(
        "out is $(size(out)); this plan writes $(FFS.Plans.field_size(plan))"))
    size(coeffs)[1:2] == (plan.Nθ, plan.Nφ) || throw(DimensionMismatch(
        "spherical coefficients must be (Nθ, Nφ) = ($(plan.Nθ), $(plan.Nφ)) on the spectral dims; " *
        "got $(size(coeffs)[1:2])"))
    Cc = reshape(coeffs, plan.Nθ, plan.Nφ, plan.B)
    O = reshape(out, plan.N, plan.B)
    _stage_coeffs!(plan, Cc, real)
    NUFSHT.nusht_type2!(plan.fd, plan.Cd, plan.plan)
    O .= _field_like(O, plan)
    if !R
        _stage_coeffs!(plan, Cc, imag)
        NUFSHT.nusht_type2!(plan.fd, plan.Cd, plan.plan)
        O .+= im .* _field_like(O, plan)
    end
    return out
end

# `part` of the coefficients `Cc` into the plan's device buffer, zero above degree `lmax`: through the host
# staging from a host array, in place from a device one.
_stage_coeffs!(p::NUFSHTSynthesisGPUPlan{FT}, Cc::Array, part) where {FT} =
    copyto!(p.Cd, (p.Cr_host .= ifelse.(p.valid, part.(Cc), zero(FT))))
_stage_coeffs!(p::NUFSHTSynthesisGPUPlan{FT}, Cc::AbstractArray, part) where {FT} =
    (p.Cd .= ifelse.(p.validd, part.(Cc), zero(FT)))

# The synthesized component in the memory of `O`: the device buffer beside a device array, its host copy
# beside a host one.
_field_like(::Array, p::NUFSHTSynthesisGPUPlan) = copyto!(p.f_host, p.fd)
_field_like(::AbstractArray, p::NUFSHTSynthesisGPUPlan) = p.fd

end # module FlowFieldSpectraNUFSHTKernelAbstractionsExt
