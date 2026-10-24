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
# the plan, in `b`'s memory, its NUFFT running through FlowTransformBindings on that device; `KA.CPU()` runs
# the same code. NUFSHT's real coefficients map into FFS's `(Nθ, Nφ, batch…)` (FastSphericalHarmonics
# `sph_mode` layout). NUFSHT's FastTransforms sph2fourier step, whose size is independent of the node
# count, runs on the host. The reusable plan holds the device plan, the device field and coefficient
# buffers and the host staging. Points `(θ, φ)` come from `Grids._sph_points`.
# =============================================================================

"""
    NUSHTSphericalGPUPlan{T}

Device-resident reusable NUFSHT plan (GPU execution): the device NUFSHT plan over the nodes its
`inactive` holds, carrying `C` transforms per execution, the device field/coeff buffers, the host staging
buffers for a host or complex field and the layout remap, those nodes' indices and which of them read the
field, and their device quadrature weights or, for `solve=true`, the device LSMR workspace. Built once for
a fixed point set + batch shape; reuse via `calculate_spectrum!`.
"""
struct NUSHTSphericalGPUPlan{T, CT, NB, P, FD, FH, CD, HB, WS, QW, AC, AD, LV, LD, KS} <: FFS.AbstractSpectralPlan
    plan::P            # device-resident NUFSHT plan over the held nodes, ntrans = C
    fd::FD             # device (n, C) field buffer at the held nodes
    fh::FH             # host (n, C) staging for a host field's held nodes or a complex field's component
    Cd::CD             # device (Nθ, Nφ, C) real coeff buffer
    Cr_host::HB        # host (Nθ, Nφ, C) staging for the small layout remap
    ws::WS             # device FTB.LSMRWorkspace for solve=true; nothing otherwise
    wd::QW             # device (n,) the held nodes' quadrature weights; nothing for solve=true
    act::AC            # host indices of the held nodes, `nothing` for every node
    actd::AD           # the same indices on the device
    live::LV           # host flag per held node, whether it reads the field; `nothing` where every one does
    lived::LD          # the same flags on the device
    N::Int             # the grid's node count, the field's stride between batch slices
    lmax::Int
    Nθ::Int
    Nφ::Int
    batch::NTuple{NB, Int}
    B::Int             # the declared total batch
    C::Int             # transforms per execution
    solve::Bool
    maxiter::Int
    rtol::T
    ks::KS
end

# Custom show: the wrapped NUFSHT plan holds a NUFFT library plan, whose default printing can segfault.
Base.show(io::IO, p::NUSHTSphericalGPUPlan{T}) where {T} =
    print(io, "NUSHTSphericalGPUPlan{", T, "}(lmax=", p.lmax, ", B=", p.B, p.solve ? ", solve" : "", ")")

function _nusht_gpu_plan(::Type{CT}, ::Type{FT}, exec, g, ms::Tuple, batch::NTuple{NB, Int};
        tol::Real, solve::Bool, maxiter::Int, rtol::Real, nufft, batch_chunk::Int,
        sampling = nothing, weights = nothing, inactive::FFS.AbstractInactiveCells) where {CT, FT, NB}
    backend = exec.backend
    lmax = ms[1] - 1
    Nθ = lmax + 1
    Nφ = 2 * lmax + 1
    θ, φ, w = FFS.DirectSum._sph_point_data(g, FT; sampling,
        weights = FFS.Grids._inactive_weights(inactive, g, FT; sampling, weights))
    N = length(θ)
    act, live = FFS.Grids._transform_nodes(inactive, g, solve)
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
        ws = solve ? FTB.LSMRWorkspace(plan) : nothing
        # The quadrature's weights in the plan's memory; the solve reads the raw field.
        wd = solve ? nothing : copyto!(KA.allocate(backend, FT, n), FFS.Grids._at_points(collect(FT, w), act))
        actd = act === nothing ? nothing : copyto!(KA.allocate(backend, Int, n), act)
        lived = live === nothing ? nothing : copyto!(KA.allocate(backend, Bool, n), collect(Bool, live))
        ks = (0:lmax, -lmax:lmax)
        return NUSHTSphericalGPUPlan{FT, CT, NB, typeof(plan), typeof(fd), typeof(fh), typeof(Cd),
                typeof(Cr_host), typeof(ws), typeof(wd), typeof(act), typeof(actd), typeof(live), typeof(lived),
                typeof(ks)}(
            plan, fd, fh, Cd, Cr_host, ws, wd, act, actd, live, lived, N, lmax, Nθ, Nφ, batch, B, C, solve,
            maxiter, FT(rtol), ks)
    catch
        NUFSHT.close!(plan)
        rethrow()
    end
end

FFS.Plans.coefficient_size(p::NUSHTSphericalGPUPlan) = (p.Nθ, p.Nφ, p.batch...)
FFS.Plans.coefficient_type(::NUSHTSphericalGPUPlan{T, CT}) where {T, CT} = CT
FFS.Plans.wavenumbers(p::NUSHTSphericalGPUPlan) = p.ks
FFS.Plans.close!(p::NUSHTSphericalGPUPlan) = (NUFSHT.close!(p.plan); nothing)

"""
    plan_spectrum(NUFSHTSpectralBackend(), GPUBackend(dev), grid, T, ms; batch=(), tol, solve, maxiter, rtol,
                  nufft, batch_chunk=0, sampling=nothing, weights=nothing)

The device form of the NUFSHT plan, with the nodes and every transform buffer on `dev`. `batch_chunk` is
how many `batch` slices one NUFSHT execution carries, as on the host; the default runs the whole batch in
one.
"""
function FFS.plan_spectrum(::SB.AbstractNUFSHTSpectralBackend, exec::ComputationalBackends.GPUBackend{<:KA.Backend},
        g::FlowGeometries.Grids.AbstractGrid{<:FFS.Grids.SphericalHarmonicGeometry},
        ::Type{T}, ms::Tuple; batch::Tuple = (), tol::Real = 1.0e-8, solve::Bool = false,
        maxiter::Int = 500, rtol::Real = 1.0e-6, nufft::SB.AbstractSpectralBackend = SB.AutoSpectralBackend(),
        batch_chunk::Int = 0, sampling = nothing, weights = nothing,
        inactive::FFS.AbstractInactiveCells = FFS.ZeroFill()) where {T}
    FT = real(float(T))
    return _nusht_gpu_plan(FFS.sph_coeff_type(T, FT), FT, exec, g, ms, NTuple{length(batch), Int}(batch);
        tol, solve, maxiter, rtol, nufft, batch_chunk, sampling, weights, inactive)
end

function FFS.calculate_spectrum!(coeffs::AbstractArray{<:Number}, plan::NUSHTSphericalGPUPlan{T}, field) where {T}
    FFS.Plans._check_coefficients(coeffs, plan)
    length(field) == plan.N * plan.B || throw(DimensionMismatch(
        "field holds $(length(field)) values; this plan transforms $(plan.B) field(s) of " *
        "$(plan.N) points — pass the matching `batch=` to plan_spectrum"))
    fill!(coeffs, zero(eltype(coeffs)))
    _nusht_gpu_pass!(coeffs, plan, field, real, one(T))
    eltype(field) <: Real || _nusht_gpu_pass!(coeffs, plan, field, imag, im)
    return plan.ks
end

# Slices `lo … lo + nvalid - 1` of one field component at the plan's held nodes in the device buffer, zero
# at a node that reads no field and in the columns past `nvalid`, so a partial final chunk contributes
# nothing through them. A real field on every node crosses in one copy, and on the active nodes of a
# device field in one device gather; a host field's active nodes, or one component of a complex field,
# are written into the host buffer first.
_stage_chunk!(p::NUSHTSphericalGPUPlan, field::AbstractArray{<:Real}, part, lo::Int, nvalid::Int) =
    _stage_real!(p, p.act, field, lo, nvalid)
_stage_chunk!(p::NUSHTSphericalGPUPlan, field::AbstractArray{<:Complex}, part, lo::Int, nvalid::Int) =
    _stage_host!(p, field, part, lo, nvalid)

function _stage_real!(p::NUSHTSphericalGPUPlan, ::Nothing, field, lo::Int, nvalid::Int)
    n, C = size(p.fd)
    copyto!(p.fd, 1, field, (lo - 1) * n + 1, nvalid * n)
    nvalid < C && fill!(view(p.fd, :, (nvalid + 1):C), zero(eltype(p.fd)))
    # `ifelse` selects a value, so a non-reading node's `NaN` stays out.
    p.lived === nothing || (p.fd .= ifelse.(p.lived, p.fd, zero(eltype(p.fd))))
    return p.fd
end
_stage_real!(p::NUSHTSphericalGPUPlan, ::AbstractVector{Int}, field::Array, lo::Int, nvalid::Int) =
    _stage_host!(p, field, real, lo, nvalid)
function _stage_real!(p::NUSHTSphericalGPUPlan, ::AbstractVector{Int}, field, lo::Int, nvalid::Int)
    n, C = size(p.fd)
    FFS._gather_scaled!(p.fd, 0, field, (lo - 1) * p.N, p.N, p.actd, one(eltype(p.fd)), n, nvalid, false,
        false)
    nvalid < C && fill!(view(p.fd, :, (nvalid + 1):C), zero(eltype(p.fd)))
    return p.fd
end

function _stage_host!(p::NUSHTSphericalGPUPlan, field, part, lo::Int, nvalid::Int)
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
# `coeffs` at weight `scale`. NUFSHT's transform is real and linear, so a real field runs this once with
# `real` (the identity on it) at weight 1, and a complex field runs it again with `imag` at weight `im`.
function _nusht_gpu_pass!(coeffs, plan::NUSHTSphericalGPUPlan, field, part, scale)
    lmax = plan.lmax
    C = plan.C
    Cr = reshape(plan.Cr_host, plan.Nθ, plan.Nφ, C)
    Cc = reshape(coeffs, plan.Nθ, plan.Nφ, plan.B)
    for lo in 1:C:plan.B
        nvalid = min(C, plan.B - lo + 1)
        _stage_chunk!(plan, field, part, lo, nvalid)
        NUFSHT.nusht_analyze!(plan.Cd, plan.fd, plan.plan; weights = plan.wd, ws = plan.ws,
            maxiter = plan.maxiter, rtol = plan.rtol)
        copyto!(plan.Cr_host, plan.Cd)                         # device → host for the small layout remap
        @inbounds for b in 1:nvalid
            for l in 0:lmax
                for m in -l:l
                    Cc[FFS.sph_mode_index(l, m), lo + b - 1] +=
                        scale * Cr[NUFSHT.FastSphericalHarmonics.sph_mode(l, m), b]
                end
            end
        end
    end
    return coeffs
end

# One-shot GPU: build a device plan for this field's batch shape, then execute it once.
# More specific than the NUFSHT-only `AbstractGPUBackend` stub, so it wins when KernelAbstractions is
# loaded.
function FFS._calculate_spectrum_nufsht(exec::ComputationalBackends.GPUBackend{<:KA.Backend},
        g::FlowGeometries.Grids.AbstractGrid{<:FFS.Grids.SphericalHarmonicGeometry},
        field::AbstractArray, ms::Tuple;
        tol::Real = 1.0e-8, solve::Bool = false, maxiter::Int = 500, rtol::Real = 1.0e-6,
        nufft::SB.AbstractSpectralBackend = SB.AutoSpectralBackend(), batch_chunk::Int = 0,
        sampling = nothing, weights = nothing, inactive::FFS.AbstractInactiveCells = FFS.ZeroFill())
    FT = real(float(eltype(field)))
    batch = FFS.Grids.field_batch_shape(g, field)
    plan = _nusht_gpu_plan(FFS.sph_coeff_type(eltype(field), FT), FT, exec, g, ms, batch;
        tol, solve, maxiter, rtol, nufft, batch_chunk, sampling, weights, inactive)
    try
        coeffs = FFS._allocate_coefficients_like(field, plan)
        return coeffs, FFS.calculate_spectrum!(coeffs, plan, field)
    finally
        FFS.close!(plan)
    end
end

# =============================================================================
# Synthesis on a device backend via `nusht_type2!`, the series evaluated at the nodes. The plan is built
# under the execution backend, so NUFSHT places it in that device's memory. Coefficients arrive in
# `sph_mode_index` layout and are remapped on the host into NUFSHT's FastSphericalHarmonics `sph_mode` real
# buffer, which is then staged to the device; the remap is over the mode set, whose size is independent of
# the node count. NUFSHT's coefficients are real, so a complex coefficient array is evaluated one real
# component at a time, which the transform's linearity permits.
# =============================================================================

# Reusable device synthesis: the device-resident NUFSHT plan (nodes preset once) plus the host staging
# for the mode remap, whose size is independent of the node count.
struct NUSHTSynthesisGPUPlan{FT, R, NB, P, CD, FD, HB, FH, SP} <: FFS.AbstractSynthesisPlan
    plan::P
    Cd::CD
    fd::FD
    Cr_host::HB
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
Base.show(io::IO, p::NUSHTSynthesisGPUPlan{FT, R}) where {FT, R} =
    print(io, "NUSHTSynthesisGPUPlan{", FT, "}(lmax=", p.lmax, ", ", R ? "real" : "complex", ")")

FFS.Plans.field_size(p::NUSHTSynthesisGPUPlan) = (p.spatial..., p.batch...)
FFS.Plans.field_type(::NUSHTSynthesisGPUPlan{FT, R}) where {FT, R} = R ? FT : Complex{FT}
FFS.Plans.close!(p::NUSHTSynthesisGPUPlan) = (NUFSHT.close!(p.plan); nothing)

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
        return NUSHTSynthesisGPUPlan{FT, T <: Real, length(bt), typeof(plan), typeof(Cd), typeof(fd), Array{FT, 3},
                Matrix{FT}, typeof(sp)}(plan, Cd, fd, zeros(FT, Nθ, Nφ, B), zeros(FT, N, B), lmax, Nθ, Nφ, N, sp,
            bt, B)
    catch
        NUFSHT.close!(plan)
        rethrow()
    end
end

function FFS.Plans.synthesize!(out::AbstractArray, plan::NUSHTSynthesisGPUPlan{FT, R},
        coeffs::AbstractArray) where {FT, R}
    size(out) == FFS.Plans.field_size(plan) || throw(DimensionMismatch(
        "out is $(size(out)); this plan writes $(FFS.Plans.field_size(plan))"))
    size(coeffs)[1:2] == (plan.Nθ, plan.Nφ) || throw(DimensionMismatch(
        "spherical coefficients must be (Nθ, Nφ) = ($(plan.Nθ), $(plan.Nφ)) on the spectral dims; " *
        "got $(size(coeffs)[1:2])"))
    lmax = plan.lmax
    B = plan.B
    Cc = reshape(coeffs, plan.Nθ, plan.Nφ, B)
    O = reshape(out, plan.N, B)
    ET = R ? FT : Complex{FT}
    fill!(O, zero(ET))
    ncomp = R ? 1 : 2
    @inbounds for comp in 1:ncomp
        fill!(plan.Cr_host, zero(FT))
        for b in 1:B, l in 0:lmax, m in -l:l
            z = Cc[FFS.sph_mode_index(l, m), b]
            plan.Cr_host[NUFSHT.FastSphericalHarmonics.sph_mode(l, m), b] = comp == 1 ? real(z) : imag(z)
        end
        copyto!(plan.Cd, plan.Cr_host)
        NUFSHT.nusht_type2!(plan.fd, plan.Cd, plan.plan)
        copyto!(plan.f_host, plan.fd)
        for b in 1:B, j in 1:plan.N
            O[j, b] += comp == 1 ? ET(plan.f_host[j, b]) : ET(im * plan.f_host[j, b])
        end
    end
    return out
end

end # module FlowFieldSpectraNUFSHTKernelAbstractionsExt
