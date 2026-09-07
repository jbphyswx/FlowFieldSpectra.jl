module FlowFieldSpectraNonuniformFFTsKernelAbstractionsExt

# `@index`/`@Const` are imported by bare name: the `@kernel` macro only recognizes them that way inside
# a kernel body. Everything else stays module-qualified.
using KernelAbstractions: KernelAbstractions as KA, @index, @Const
using NonuniformFFTs: NonuniformFFTs
using FlowFieldSpectra: FlowFieldSpectra as FFS
using ComputationalBackends: ComputationalBackends
using SpectralBackends: SpectralBackends
using FlowGeometries: FlowGeometries

# =============================================================================
# Device-generic NonuniformFFTs (type-1) for a device execution backend, on the packed-native model:
# thread the execution backend into `PlanNUFFT(…; backend=…)`, stage the points/phase/twin tables to that
# backend, and publish the transform output directly. Plans are built with `fftshift = false`, so a real
# field's `fk_half` is already the packed half `(N₁÷2+1, N₂…)` and a complex field's `fk` is the full
# native spectrum — the same layout, convention and `ks` the host ext returns (`Σⱼ vⱼ e^{-i k·xⱼ}`, ×1/M),
# so the execution axis does not change the shape of `coeffs`. A real field's halved axis also carries its
# Nyquist twins, gathered on device out of the oversampled spectrum (see below). One code path serves any
# KA backend; it never scalar-indexes a device array on the host. The host Serial/Threaded fast paths live
# in the base ext; dispatch here is strictly more specific.
# =============================================================================

_default_eps(::Type{Float32}) = 1.0f-6
_default_eps(::Type{T}) where {T <: Real} = 1.0e-8
_default_eps(::Type{Complex{T}}) where {T} = _default_eps(T)

"""
    DEFAULT_BATCH_CHUNK

Default transforms per `exec_type1!` on a device, overridable per call as `batch_chunk=` on
`plan_spectrum` and `calculate_spectrum`. Zero runs the caller's whole batch in one execution, which
keeps the device saturated: spreading here goes through the shared-memory block path, whose working set
is set by the plan's own block and batch sizes, so concurrent transforms add parallelism. The count does
set how many oversampled grids the plan holds, so a positive `batch_chunk` splits the batch into
successive executions when device memory is the binding constraint.
"""
const DEFAULT_BATCH_CHUNK = 0

function _plan_accuracy(eps::Real, ms::NTuple)
    m = clamp(ceil(Int, -log10(eps)) + 1, 1, 16)
    σ = float(max(2, cld(2 * m, minimum(ms))))
    return NonuniformFFTs.HalfSupport(m), σ
end

# Native (unshifted) fftfreq integer frequencies of a full axis of length N.
_fftfreq_ints(N::Int) = Int[j <= (N - 1) ÷ 2 ? j : j - N for j in 0:(N - 1)]

# Transforms a field carries, and the native chunk a plan executes.
@inline _nbatch(batch::Tuple) = prod(batch; init = 1)
@inline _chunk(B::Int, batch_chunk::Int) = batch_chunk <= 0 ? max(B, 1) : clamp(batch_chunk, 1, max(B, 1))

function _check_ntrans(len::Int, M::Int, B::Int)
    len == M * B || throw(DimensionMismatch(
        "field carries $(len ÷ M) transforms; this plan was built for $B — pass the matching `batch=` " *
        "to plan_spectrum (`batch_chunk=` sets how many run per execution, independently of this)"))
    return nothing
end

# Scaled points (→ [0, 2π)), the packed offset-correction phase (× 1/M; built for `iflag = +1`, the real
# path applies the sign by a final conjugate), and packed physical wavenumbers. `R` marks a real field
# (axis 1 halved to rfftfreq `0:N₁÷2`); the other axes are native fftfreq — matching the `fftshift=false`
# transform output. Everything here is host-side; the device buffers are staged from it once.
function _scaled_phase_ks(::Type{Tr}, coords::Tuple, ms::NTuple{D, Int}, Ls::NTuple{D}, iflag::Int, R::Bool) where {Tr <: Real, D}
    M = length(coords[1])
    for d in 1:D
        length(coords[d]) == M || throw(DimensionMismatch("coordinate $d length mismatch"))
    end
    offsets = ntuple(d -> Tr(minimum(coords[d])), D)
    ranges = ntuple(d -> Tr(Ls[d]), D)   # already resolved by `Grids.axis_range`
    scaled = ntuple(d -> Tr(2π) .* (Tr.(coords[d]) .- offsets[d]) ./ ranges[d], D)
    kint = ntuple(d -> (d == 1 && R) ? collect(0:(ms[1] ÷ 2)) : _fftfreq_ints(ms[d]), D)
    pms = FFS.Packing.packed_size(ms, Val(R))
    inv_M = one(Tr) / M
    phase = Array{Complex{Tr}, D}(undef, pms...)
    @inbounds for I in CartesianIndices(pms)
        p = one(Complex{Tr})
        for d in 1:D
            p *= cis(-iflag * kint[d][I[d]] * (offsets[d] * Tr(2π) / ranges[d]))
        end
        phase[I] = p * inv_M
    end
    ks_phys = FFS.Grids.physical_wavenumbers(ranges, ms, Val(R))
    return scaled, phase, ks_phys, M, offsets, ranges
end

# Stage a host array/tuple onto the KA backend.
_to_dev(backend, ::Type{S}, v) where {S} = (d = KA.allocate(backend, S, size(v)...); copyto!(d, S.(v)); d)

# =============================================================================
# Nyquist twins. Index negation on a packed half sends an even axis at `−N_d/2` to `+N_d/2`, which is off
# the native axis. `exec_type1!` FFTs onto the oversampled grid `data.ûs` and its deconvolution step only
# reads that array while truncating to the native box, so those coefficients are already computed. Each
# twin value is one oversampled entry times the deconvolution the stored entries get; `Packing.twin_table`
# builds the (index, factor) pair per value on the host and both are staged to the device once.
# =============================================================================

struct TwinPeekGPU{DV, SL, IX, FC}
    dev::DV          # device (S,) gathered values
    slice::SL        # host (shape…, batch…) published values
    src::IX          # device (S,) indices into the oversampled `ûs`
    fac::FC          # device (S,) deconvolution × offset phase × 1/M
    n::Int           # S, zero for a mask that no mode reaches
end

KA.@kernel function _twin_gather_kernel!(dev, @Const(us), @Const(src), @Const(fac), neg::Bool)
    s = @index(Global)
    @inbounds begin
        v = us[src[s]] * fac[s]
        dev[s] = neg ? conj(v) : v
    end
end

function _twin_peek_gpu(::Type{Tr}, backend, p, ms::NTuple{D, Int}, offsets, ranges, M::Int, mask::Int,
        batch::Tuple) where {Tr <: Real, D}
    ovs = size(first(p.data.ûs))
    normfactor = prod(Ñ -> 2π / Ñ, size(first(p.data.us)))
    phis = map(k -> collect(NonuniformFFTs.fourier_coefficients(k)), p.kernels)
    shape, src, fac = FFS.Packing.twin_table(Tr, ms, mask, ovs, offsets, ranges, M;
        phis = phis, normfactor = normfactor)
    S = length(src)
    return TwinPeekGPU(
        KA.allocate(backend, Complex{Tr}, max(S, 1)),
        zeros(Complex{Tr}, shape..., batch...),
        _to_dev(backend, Int, S == 0 ? Int[1] : src),
        _to_dev(backend, Complex{Tr}, S == 0 ? Complex{Tr}[0] : fac),
        S,
    )
end

function _with_twins_gpu(::Type{Tr}, backend, ks_phys::Tuple, p, ms::NTuple{D, Int}, offsets, ranges,
        M::Int, batch::Tuple) where {Tr <: Real, D}
    D >= 2 || return ks_phys, ()
    peeks = ntuple(FFS.Packing.n_twin_slices(Val(D))) do mask
        _twin_peek_gpu(Tr, backend, p, ms, offsets, ranges, M, mask, batch)
    end
    twin = FFS.Packing.NyquistTwin(map(t -> t.slice, peeks))
    return (FFS.Packing.with_twin(ks_phys[1], twin), Base.tail(ks_phys)...), peeks
end

# `ûs` is overwritten by the next `exec_type1!`, so a chunk's twins are gathered before the next one.
# Chunk-local transform `c` is the caller's transform `base + c`.
@inline _gather_twins!(::Tuple{}, ûs, base::Int, nvalid::Int, neg::Bool, backend) = nothing
@inline function _gather_twins!(peeks::Tuple, ûs, base::Int, nvalid::Int, neg::Bool, backend)
    _gather_twin!(first(peeks), ûs, base, nvalid, neg, backend)
    return _gather_twins!(Base.tail(peeks), ûs, base, nvalid, neg, backend)
end

function _gather_twin!(tp::TwinPeekGPU, ûs, base::Int, nvalid::Int, neg::Bool, backend)
    S = tp.n
    S == 0 && return nothing
    kernel! = _twin_gather_kernel!(backend)
    @inbounds for c in 1:nvalid
        kernel!(tp.dev, ûs[c], tp.src, tp.fac, neg; ndrange = S)
        KA.synchronize(backend)
        soff = (base + c - 1) * S
        copyto!(view(vec(tp.slice), (soff + 1):(soff + S)), view(tp.dev, 1:S))
    end
    return nothing
end

# ---- device plans (buffers + tables live on the execution backend) ----

struct NUFFTNonuniformComplexGPUPlan{T, D, NB, P, CJ, FK, OB, HB, PH, KS, QW} <: FFS.AbstractSpectralPlan
    batch::NTuple{NB, Int}
    plan::P
    cj::CJ                           # C × device (M,) complex strengths
    fk::FK                           # C × device (ms…) full native spectra
    out::OB                          # device (ms…) complex output (fk × phase)
    out_host::HB                     # host (ms…) staging for the copy back to `coeffs`
    ms::NTuple{D, Int}
    M::Int
    nbatch::Int                      # transforms the caller's field must carry
    iflag::Int
    phase::PH                        # device (ms…) offset phase × 1/M
    ks_phys::KS
    qw::QW                           # device (M,) grid quadrature factor, or `nothing`
end

FFS.Plans.coefficient_size(p::NUFFTNonuniformComplexGPUPlan) = (p.ms..., p.batch...)
FFS.Plans.coefficient_type(::NUFFTNonuniformComplexGPUPlan{T}) where {T} = Complex{T}
FFS.Plans.wavenumbers(p::NUFFTNonuniformComplexGPUPlan) = p.ks_phys

struct NUFFTNonuniformRealGPUPlan{T, D, NB, P, CJ, FK, OB, HB, PH, KS, TW, QW} <: FFS.AbstractSpectralPlan
    batch::NTuple{NB, Int}
    plan::P                          # device PlanNUFFT{T<:Real}, packed half
    cj::CJ                           # C × device (M,) REAL strengths
    fk_half::FK                      # C × device (N₁÷2+1, N₂…) packed half-spectra
    out::OB                          # device packed output (fk_half × phase)
    out_host::HB                     # host packed staging
    ms::NTuple{D, Int}
    M::Int
    nbatch::Int
    iflag::Int
    phase::PH                        # device packed phase × 1/M (built for iflag=+1)
    ks_phys::KS                      # halved axis carries the twin whose slices `twins` refill
    twins::TW
    qw::QW                           # device (M,) grid quadrature factor, or `nothing`
end

Base.show(io::IO, ::NUFFTNonuniformComplexGPUPlan{T, D}) where {T, D} = print(io, "NUFFTNonuniformComplexGPUPlan{$T, $D}(NonuniformFFTs, device)")
Base.show(io::IO, ::NUFFTNonuniformRealGPUPlan{T, D}) where {T, D} = print(io, "NUFFTNonuniformRealGPUPlan{$T, $D}(NonuniformFFTs, device)")

function _nu_gpu_plan(::Type{Complex{Tr}}, backend, coords::Tuple, ms::NTuple{D, Int}, Ls::NTuple{D},
        iflag::Int, eps::Real, batch::Tuple, batch_chunk::Int, qw_h = nothing) where {Tr <: Real, D}
    scaled_h, phase_h, ks_phys, M, _, _ = _scaled_phase_ks(Tr, coords, ms, Ls, iflag, false)
    hs, σ = _plan_accuracy(eps, ms)
    B = _nbatch(batch)
    C = _chunk(B, batch_chunk)
    plan = NonuniformFFTs.PlanNUFFT(Complex{Tr}, ms; ntransforms = Val(C), fftshift = false, m = hs, σ = σ, backend = backend)
    NonuniformFFTs.set_points!(plan, ntuple(d -> _to_dev(backend, Tr, scaled_h[d]), D))
    cj = ntuple(_ -> KA.allocate(backend, Complex{Tr}, M), C)
    fk = ntuple(_ -> KA.allocate(backend, Complex{Tr}, ms...), C)
    qw = qw_h === nothing ? nothing : _to_dev(backend, Complex{Tr}, qw_h)
    out = KA.allocate(backend, Complex{Tr}, ms...)
    out_host = Array{Complex{Tr}, D}(undef, ms...)
    phase = _to_dev(backend, Complex{Tr}, phase_h)
    bt = NTuple{length(batch), Int}(batch)
    return NUFFTNonuniformComplexGPUPlan{Tr, D, length(bt), typeof(plan), typeof(cj), typeof(fk), typeof(out), typeof(out_host), typeof(phase), typeof(ks_phys), typeof(qw)}(
        bt, plan, cj, fk, out, out_host, ms, M, B, iflag, phase, ks_phys, qw)
end

function _nu_gpu_plan(::Type{Tr}, backend, coords::Tuple, ms::NTuple{D, Int}, Ls::NTuple{D},
        iflag::Int, eps::Real, batch::Tuple, batch_chunk::Int, qw_h = nothing) where {Tr <: Real, D}
    scaled_h, phase_h, ks_phys, M, offsets, ranges = _scaled_phase_ks(Tr, coords, ms, Ls, 1, true)   # sign via final conj
    hs, σ = _plan_accuracy(eps, ms)
    B = _nbatch(batch)
    C = _chunk(B, batch_chunk)
    plan = NonuniformFFTs.PlanNUFFT(Tr, ms; ntransforms = Val(C), fftshift = false, m = hs, σ = σ, backend = backend)
    NonuniformFFTs.set_points!(plan, ntuple(d -> _to_dev(backend, Tr, scaled_h[d]), D))
    pms = FFS.Packing.packed_size(ms, Val(true))
    cj = ntuple(_ -> KA.allocate(backend, Tr, M), C)
    fk_half = ntuple(_ -> KA.allocate(backend, Complex{Tr}, size(plan)...), C)
    qw = qw_h === nothing ? nothing : _to_dev(backend, Tr, qw_h)
    out = KA.allocate(backend, Complex{Tr}, pms...)
    out_host = Array{Complex{Tr}, D}(undef, pms...)
    phase = _to_dev(backend, Complex{Tr}, phase_h)
    ks_twin, twins = _with_twins_gpu(Tr, backend, ks_phys, plan, ms, offsets, ranges, M, batch)
    bt = NTuple{length(batch), Int}(batch)
    return NUFFTNonuniformRealGPUPlan{Tr, D, length(bt), typeof(plan), typeof(cj), typeof(fk_half), typeof(out), typeof(out_host), typeof(phase), typeof(ks_twin), typeof(twins), typeof(qw)}(
        bt, plan, cj, fk_half, out, out_host, ms, M, B, iflag, phase, ks_twin, twins, qw)
end

FFS.Plans.coefficient_size(p::NUFFTNonuniformRealGPUPlan) =
    (FFS.Packing.packed_size(p.ms, Val(true))..., p.batch...)
FFS.Plans.coefficient_type(::NUFFTNonuniformRealGPUPlan{T}) where {T} = Complex{T}
FFS.Plans.wavenumbers(p::NUFFTNonuniformRealGPUPlan) = p.ks_phys

# ---- complex path: full native spectrum ----

function FFS.calculate_spectrum!(coeffs::AbstractArray{Complex{T}}, plan::NUFFTNonuniformComplexGPUPlan{T, D},
        field) where {T, D}
    M = plan.M
    Pm = prod(plan.ms)
    B = plan.nbatch
    C = length(plan.cj)
    _check_ntrans(length(field), M, B)
    neg = plan.iflag < 0
    @inbounds for base in 0:C:(B - 1)
        nvalid = min(C, B - base)
        for c in 1:C
            cjc = plan.cj[c]
            if c > nvalid
                fill!(cjc, zero(Complex{T}))          # pad the final chunk to the plan's transform count
                continue
            end
            foff = (base + c - 1) * M
            copyto!(cjc, view(field, (foff + 1):(foff + M)))     # host/device field → device
            plan.qw === nothing || (cjc .*= plan.qw)             # grid quadrature factor
            neg && (cjc .= conj.(cjc))
        end
        NonuniformFFTs.exec_type1!(plan.fk, plan.plan, plan.cj)
        for c in 1:nvalid
            if neg
                plan.out .= conj.(plan.fk[c]) .* plan.phase
            else
                plan.out .= plan.fk[c] .* plan.phase
            end
            copyto!(plan.out_host, plan.out)                      # device → host
            coff = (base + c - 1) * Pm
            copyto!(view(coeffs, (coff + 1):(coff + Pm)), vec(plan.out_host))
        end
    end
    return plan.ks_phys
end

# ---- real path: publish the packed half directly ----

function FFS.calculate_spectrum!(coeffs::AbstractArray{Complex{T}}, plan::NUFFTNonuniformRealGPUPlan{T, D},
        field) where {T, D}
    M = plan.M
    Ph = length(plan.out)                                         # packed mode count
    B = plan.nbatch
    C = length(plan.cj)
    _check_ntrans(length(field), M, B)
    backend = KA.get_backend(plan.out)
    neg = plan.iflag < 0
    # real field: the iflag=-1 result is the conjugate of the iflag=+1 result (for which `phase` is built)
    @inbounds for base in 0:C:(B - 1)
        nvalid = min(C, B - base)
        for c in 1:C
            cjc = plan.cj[c]
            if c > nvalid
                fill!(cjc, zero(T))                   # pad the final chunk to the plan's transform count
                continue
            end
            foff = (base + c - 1) * M
            copyto!(cjc, view(field, (foff + 1):(foff + M)))     # real strengths → device
            plan.qw === nothing || (cjc .*= plan.qw)             # grid quadrature factor
        end
        NonuniformFFTs.exec_type1!(plan.fk_half, plan.plan, plan.cj)
        _gather_twins!(plan.twins, plan.plan.data.ûs, base, nvalid, neg, backend)
        for c in 1:nvalid
            if neg
                plan.out .= conj.(plan.fk_half[c] .* plan.phase)
            else
                plan.out .= plan.fk_half[c] .* plan.phase
            end
            KA.synchronize(backend)
            copyto!(plan.out_host, plan.out)                      # device → host
            coff = (base + c - 1) * Ph
            copyto!(view(coeffs, (coff + 1):(coff + Ph)), vec(plan.out_host))
        end
    end
    return plan.ks_phys
end

# ---- plan_spectrum + one-shot entries on a device backend; real/complex dispatched on the plan eltype ----

function FFS.plan_spectrum(::FFS.NonuniformFFTsBackend, exec::ComputationalBackends.GPUBackend{<:KA.Backend},
        g::FFS.Grids.PointwiseCartesian, ::Type{T}, ms::NTuple{D, Int};
        batch::Tuple = (), iflag::Int = 1, eps::Real = _default_eps(T),
        batch_chunk::Int = DEFAULT_BATCH_CHUNK) where {T, D}
    coords, _ = FFS.Grids.point_coordinates(real(float(T)), g, D)
    Ls = ntuple(d -> FFS.Grids.axis_range(real(float(T)), g, d), D)
    qw = FFS.Grids.quadrature_scale(g, real(float(T)), length(coords[1]))
    return _nu_gpu_plan(T, exec.backend, coords, ms, Ls, iflag, eps, batch, batch_chunk, qw)
end

function FFS._calculate_spectrum_gpu_nufft(::FFS.NonuniformFFTsBackend, exec::ComputationalBackends.GPUBackend{<:KA.Backend},
        g::FFS.Grids.PointwiseCartesian,
        field::AbstractArray, ms::Tuple; iflag::Int = 1, eps::Union{Nothing, Real} = nothing,
        batch_chunk::Int = DEFAULT_BATCH_CHUNK, kwargs...)
    D = length(ms)
    E = float(eltype(field))
    Tr = real(E)
    coords, _ = FFS.Grids.point_coordinates(Tr, g, D)
    Ls = ntuple(d -> FFS.Grids.axis_range(Tr, g, d), D)
    batch = FFS.Grids.field_batch_shape(g, field)
    epsv = eps === nothing ? _default_eps(E) : eps
    qw = FFS.Grids.quadrature_scale(g, Tr, length(coords[1]))
    plan = _nu_gpu_plan(E, exec.backend, coords, NTuple{D, Int}(ms), Ls, iflag, epsv, batch, batch_chunk, qw)
    pms = FFS.Packing.packed_size(NTuple{D, Int}(ms), Val(eltype(field) <: Real))
    coeffs = zeros(Complex{Tr}, pms..., batch...)
    ks = FFS.calculate_spectrum!(coeffs, plan, field)
    return coeffs, ks
end

# =============================================================================
# Separable transform for a nonuniform tensor-product (structured) Cartesian grid on a device. The
# D-dimensional Fourier kernel factorizes over a tensor grid, so the transform is `D` successive 1-D
# type-1 NUFFTs — one per axis, every OTHER spatial and batch dim carried as the transform batch. Each
# pass needs only its own length-`N_d` axis, so no `∏N_d` coordinate cloud is built and spreading costs
# `D·2m` per point in place of `(2m)^D`.
#
# A pass stages the transform axis to the front with `permutedims!` into preallocated device buffers, so
# each line is a contiguous column that the plan reads as a strength vector. Every pass is complex and
# unshifted, so the composed result is the full native spectrum, from which a real field's packed half is
# the leading axis-1 entries (`hermitian_request_size` puts `±N₁/2` on the axis) and its Nyquist twins
# are CONJUGATE reads: a real field's output is Hermitian, so each twin's negated frequency is native.
# The oversampled-grid peek the scattered path uses does not apply here, since these passes are complex.
# =============================================================================

# =============================================================================
# The hybrid composite on a device. The FFT over the uniform axes comes from the GPUFFT extension
# (`AbstractFFTs` on a device array) and each stretched axis takes the device 1-D NUFFT below, so the
# working array never leaves the backend. The derivation — which axes go to which pass, whether a twin is
# needed, the publish extents and phase — is `FFS._hybrid_derive`, shared with the host composite; only
# the twins and the publish differ, since both are device-resident here.
# =============================================================================

# One stretched axis of a working array whose other axes are already transformed.
function FFS._axis_nufft(::FFS.NonuniformFFTsBackend, exec::ComputationalBackends.GPUBackend{<:KA.Backend},
        A::AbstractArray{Complex{Tr}}, d::Int, axis::AbstractVector{Tr}, m::Int, rng::Tr, off::Tr,
        eps::Real; batch_chunk::Int = DEFAULT_BATCH_CHUNK, kwargs...) where {Tr}
    return _nu_axis_gpu(A, d, axis, m, rng, off, eps, batch_chunk, exec.backend)
end

function FFS._calculate_spectrum_hybrid(t::FFS.NonuniformFFTsBackend,
        exec::ComputationalBackends.GPUBackend{<:KA.Backend},
        g::FlowGeometries.Grids.AbstractStructuredGrid{<:FlowGeometries.Geometry.AbstractCartesianGeometry},
        field::AbstractArray, ms::NTuple{D, Int}, umask::NTuple{D, Bool};
        iflag::Int = 1, eps::Union{Nothing, Real} = nothing, batch_chunk::Int = DEFAULT_BATCH_CHUNK,
        kwargs...) where {D}
    backend = exec.backend
    Tr = real(float(eltype(g)))
    R = eltype(field) <: Real
    batch = ntuple(i -> size(field, D + i), ndims(field) - D)
    ntrans = prod(batch; init = 1)
    h = FFS._hybrid_derive(g, Tr, ms, umask, R, iflag, eps, batch)

    W = FFS._region_fft(exec, FFS.quadrature_weighted(g, field), h.udims, h.halve, !R && h.neg)
    for d in h.sdims
        W = FFS._axis_nufft(t, exec, W, d, h.axs[d], ms[d], h.ranges[d], h.offs_all[d], h.epsv;
            batch_chunk = batch_chunk)
    end
    if R
        phase = _to_dev(backend, Complex{Tr}, h.phase)
        ks, twins = h.need_twin ?
            _conj_twins_gpu(Tr, backend, h.ks_phys, ms, h.nsx, h.offs, h.ranges, h.npts, batch) :
            (h.ks_phys, ())
        half = FFS.Packing.packed_half_view(W, h.nsx, h.pms, ntrans)
        out = KA.allocate(backend, Complex{Tr}, h.pms..., ntrans)
        if h.neg
            out .= conj.(half .* phase)
        else
            out .= half .* phase
        end
        _gather_conj_gpu!(twins, W, prod(h.nsx), ntrans, h.neg, backend)
        KA.synchronize(backend)
        coeffs = Array{Complex{Tr}}(undef, h.pms..., batch...)
        copyto!(reshape(coeffs, :), reshape(out, :))
        return coeffs, ks
    end
    h.neg && (W .= conj.(W))                                   # closes the input conjugation
    W .*= _to_dev(backend, Complex{Tr}, h.phase)
    KA.synchronize(backend)
    coeffs = Array{Complex{Tr}}(undef, ms..., batch...)
    copyto!(reshape(coeffs, :), reshape(W, :))
    return coeffs, h.ks
end

@inline _axis_perm(nd::Int, d::Int) = (d, ntuple(i -> i < d ? i : i + 1, nd - 1)...)

# One 1-D device type-1 NUFFT along dim `d`; returns a new device array whose dim `d` holds `m` modes.
function _nu_axis_gpu(A::AbstractArray{Complex{Tr}}, d::Int, axis::AbstractVector{Tr}, m::Int,
        rng::Tr, off::Tr, eps::Real, batch_chunk::Int, backend) where {Tr}
    nd = ndims(A)
    Nd = length(axis)
    sz = size(A)
    sz[d] == Nd ||
        throw(DimensionMismatch("axis $d: field length $(sz[d]) ≠ grid axis length $Nd"))
    outsz = FFS.Packing.axis_out_size(sz, d, m)
    perm = _axis_perm(nd, d)
    iperm = invperm(collect(perm))
    rest = prod(sz) ÷ Nd
    C = _chunk(rest, batch_chunk)
    padded = cld(rest, C) * C            # whole chunks, so every execution sees C real columns
    hs, σ = _plan_accuracy(eps, (m,))
    p1 = NonuniformFFTs.PlanNUFFT(Complex{Tr}, (m,); ntransforms = Val(C), fftshift = false,
        m = hs, σ = σ, backend = backend)
    NonuniformFFTs.set_points!(p1, (_to_dev(backend, Tr, Tr(2π) .* (axis .- off) ./ rng),))
    pin = KA.zeros(backend, Complex{Tr}, Nd, padded)
    pfk = KA.zeros(backend, Complex{Tr}, m, padded)
    permutedims!(reshape(view(pin, :, 1:rest), ntuple(i -> sz[perm[i]], nd)), A, perm)
    for base in 0:C:(rest - 1)
        NonuniformFFTs.exec_type1!(ntuple(c -> view(pfk, :, base + c), C), p1,
            ntuple(c -> view(pin, :, base + c), C))
    end
    KA.synchronize(backend)
    out = KA.allocate(backend, Complex{Tr}, outsz...)
    permutedims!(out, reshape(view(pfk, :, 1:rest), ntuple(i -> outsz[perm[i]], nd)), iperm)
    return out
end

# One mask's twin slice, filled by conjugate reads of the Hermitian native spectrum on the device.
struct ConjTwinGPU{DV, SL, IX, FC}
    dev::DV          # device (S,) gathered values
    slice::SL        # host (shape…, batch…) published values
    src::IX          # device (S,) indices into one native spectrum
    fac::FC          # device (S,) offset phase × 1/M
    n::Int           # S, zero for a mask no mode reaches
end

KA.@kernel function _conj_twin_kernel!(dev, @Const(fk), @Const(src), @Const(fac), foff::Int, neg::Bool)
    s = @index(Global)
    @inbounds begin
        v = conj(fk[foff + src[s]]) * fac[s]
        dev[s] = neg ? conj(v) : v
    end
end

function _conj_twins_gpu(::Type{Tr}, backend, ks_phys::Tuple, ms::NTuple{D, Int}, ns::NTuple{D, Int},
        offsets, ranges, M::Int, batch::Tuple) where {Tr, D}
    D >= 2 || return ks_phys, ()
    gs = ntuple(FFS.Packing.n_twin_slices(Val(D))) do mask
        shape, src, fac = FFS.Packing.twin_table(Tr, ms, mask, ns, offsets, ranges, M; conjugate = true)
        S = length(src)
        ConjTwinGPU(KA.allocate(backend, Complex{Tr}, max(S, 1)),
            zeros(Complex{Tr}, shape..., batch...),
            _to_dev(backend, Int, S == 0 ? Int[1] : src),
            _to_dev(backend, Complex{Tr}, S == 0 ? Complex{Tr}[0] : fac), S)
    end
    return (FFS.Packing.with_twin(ks_phys[1], FFS.Packing.NyquistTwin(map(g -> g.slice, gs))), Base.tail(ks_phys)...), gs
end

@inline _gather_conj_gpu!(::Tuple{}, fk, Pm::Int, ntrans::Int, neg::Bool, backend) = nothing
@inline function _gather_conj_gpu!(gs::Tuple, fk, Pm::Int, ntrans::Int, neg::Bool, backend)
    _gather_one_conj_gpu!(first(gs), fk, Pm, ntrans, neg, backend)
    return _gather_conj_gpu!(Base.tail(gs), fk, Pm, ntrans, neg, backend)
end

function _gather_one_conj_gpu!(g::ConjTwinGPU, fk, Pm::Int, ntrans::Int, neg::Bool, backend)
    S = g.n
    S == 0 && return nothing
    kernel! = _conj_twin_kernel!(backend)
    flat = reshape(fk, :)
    @inbounds for t in 1:ntrans
        kernel!(g.dev, flat, g.src, g.fac, (t - 1) * Pm, neg; ndrange = S)
        KA.synchronize(backend)
        soff = (t - 1) * S
        copyto!(view(vec(g.slice), (soff + 1):(soff + S)), view(g.dev, 1:S))
    end
    return nothing
end

function FFS._calculate_spectrum_gpu_nufft(::FFS.NonuniformFFTsBackend, exec::ComputationalBackends.GPUBackend{<:KA.Backend},
        g::FlowGeometries.Grids.AbstractStructuredGrid{<:FlowGeometries.Geometry.AbstractCartesianGeometry},
        field::AbstractArray, ms::Tuple; iflag::Int = 1, eps::Union{Nothing, Real} = nothing,
        batch_chunk::Int = DEFAULT_BATCH_CHUNK, kwargs...)
    D = ndims(g)
    E = float(eltype(field))
    Tr = real(E)
    R = eltype(field) <: Real
    backend = exec.backend
    msn = NTuple{D, Int}(ms)
    axs = ntuple(d -> Tr.(FlowGeometries.Grids.coordinates(g, d)), D)
    offsets, ranges = FFS.Grids.axis_geometry(Tr, g, D)
    Ns = size(g)
    npts = prod(Ns)
    batch = ntuple(i -> size(field, D + i), ndims(field) - D)
    ntrans = prod(batch; init = 1)
    epsv = eps === nothing ? _default_eps(E) : eps
    ns = R ? FFS.Packing.hermitian_request_size(msn) : msn
    neg = !R && iflag < 0
    A = KA.allocate(backend, Complex{Tr}, size(field)...)
    copyto!(A, field)                                          # host/device → device, widen
    qwh = FFS.Grids.quadrature_scale(g, Tr, npts)              # grid quadrature factor
    if qwh !== nothing
        # `A` carries the grid shape here, so the factor reshapes to `Ns` and broadcasts over the batch.
        A .*= reshape(_to_dev(backend, Complex{Tr}, qwh), Ns..., ntuple(_ -> 1, ndims(A) - D)...)
    end
    neg && (A .= conj.(A))                                     # one conjugation covers all D passes
    for d in 1:D
        A = _nu_axis_gpu(A, d, axs[d], ns[d], ranges[d], offsets[d], epsv, batch_chunk, backend)
    end
    if R
        pms = FFS.Packing.packed_size(msn, Val(true))
        phase = _to_dev(backend, Complex{Tr},
            FFS.Packing.offset_phase(Tr, msn, offsets, ranges, npts, Val(true)))
        ks_phys = FFS.Grids.physical_wavenumbers(ranges, msn, Val(true))
        ks, twins = _conj_twins_gpu(Tr, backend, ks_phys, msn, ns, offsets, ranges, npts, batch)
        half = view(reshape(A, ns..., ntrans), 1:pms[1], ntuple(_ -> Colon(), D - 1)..., Colon())
        out = KA.allocate(backend, Complex{Tr}, pms..., ntrans)
        if iflag < 0
            out .= conj.(half .* phase)
        else
            out .= half .* phase
        end
        _gather_conj_gpu!(twins, A, prod(ns), ntrans, iflag < 0, backend)
        KA.synchronize(backend)
        coeffs = Array{Complex{Tr}}(undef, pms..., batch...)
        copyto!(reshape(coeffs, :), reshape(out, :))
        return coeffs, ks
    end
    neg && (A .= conj.(A))                                     # closes the input conjugation
    A .*= _to_dev(backend, Complex{Tr},
        FFS.Packing.offset_phase(Tr, msn, offsets, ranges, npts, Val(false), iflag))
    KA.synchronize(backend)
    coeffs = Array{Complex{Tr}}(undef, msn..., batch...)
    copyto!(reshape(coeffs, :), reshape(A, :))
    return coeffs, FFS.Grids.physical_wavenumbers(ranges, msn, Val(false))
end

# =============================================================================
# Inverse (synthesis) on a device backend via `exec_type2!`. The forward publishes `C = fk · p / M` with
# `|p| = 1` the grid-offset phase, and a type-2 returns `Σ_k û_k e^{+i k x̃}` against its own scaled
# points, so `û = C · conj(p)`.
#
# A real field goes through the COMPLEX type-2 on the full native spectrum (completed from the packed
# half by `Packing.unpacked` with the halved axis's Nyquist twin), taking the real part. The real-input
# plan's type-2 deconvolves the half into the OVERSAMPLED half, where the native `k₁ = N₁/2` mode sits in
# the interior, so its c2r step emits that mode as a conjugate pair at both `±N₁/2` while the native set
# holds only `−N₁/2`; on a nonuniformly sampled grid those are different functions of `x`.
#
# Coefficients arrive on the host, matching what the forward returns; each chunk's spectra are staged to
# the device and its strengths read back.
# =============================================================================

# Points folded to [0, 2π) plus the per-axis offset and period, host-side.
function _scaled_points(::Type{Tr}, coords::Tuple, Ls::NTuple{D}) where {Tr, D}
    M = length(coords[1])
    for d in 1:D
        length(coords[d]) == M || throw(DimensionMismatch("coordinate $d length mismatch"))
    end
    offsets = ntuple(d -> Tr(minimum(coords[d])), D)
    ranges = ntuple(d -> Tr(Ls[d]), D)   # already resolved by `Grids.axis_range`
    scaled = ntuple(d -> Tr(2π) .* (Tr.(coords[d]) .- offsets[d]) ./ ranges[d], D)
    return scaled, offsets, ranges, M
end

function FFS._synthesize(::FFS.NonuniformFFTsBackend, exec::ComputationalBackends.GPUBackend{<:KA.Backend},
        g::Union{FlowGeometries.Grids.AbstractStructuredGrid{<:FlowGeometries.Geometry.AbstractCartesianGeometry},
                 FFS.Grids.PointwiseCartesian},
        coeffs::AbstractArray{Complex{Tr}}, ms::NTuple{D, Int}; real_output::Bool = true, iflag::Int = 1,
        ks = nothing, eps::Union{Nothing, Real} = nothing,
        batch_chunk::Int = DEFAULT_BATCH_CHUNK, kwargs...) where {Tr <: Real, D}
    backend = exec.backend
    coords, spatial = FFS.Grids.point_coordinates(Tr, g, D)
    Ls = ntuple(d -> FFS.Grids.axis_range(Tr, g, d), D)
    epsv = eps === nothing ? _default_eps(Tr) : eps
    sz = FFS.Packing.packed_size(ms, Val(real_output))
    size(coeffs)[1:D] == sz || throw(DimensionMismatch(
        "real_output=$(real_output) expects $(sz) on the spectral dims; got $(size(coeffs)[1:D])."))
    batch = ntuple(i -> size(coeffs, D + i), ndims(coeffs) - D)
    ntrans = prod(batch; init = 1)
    scaled, offsets, ranges, M = _scaled_points(Tr, coords, Ls)
    full = real_output ? FFS.Packing.unpacked(reshape(coeffs, sz..., ntrans), ms, ks) :
                         reshape(coeffs, ms..., ntrans)
    p = FFS.Packing.offset_phase(Tr, ms, offsets, ranges, M, Val(false), iflag) .* M
    hs, σ = _plan_accuracy(epsv, ms)
    C = _chunk(ntrans, batch_chunk)
    plan = NonuniformFFTs.PlanNUFFT(Complex{Tr}, ms; ntransforms = Val(C), fftshift = false,
        m = hs, σ = σ, backend = backend)
    NonuniformFFTs.set_points!(plan, ntuple(d -> _to_dev(backend, Tr, scaled[d]), D))
    fks = ntuple(_ -> KA.allocate(backend, Complex{Tr}, ms...), C)
    cjs = ntuple(_ -> KA.allocate(backend, Complex{Tr}, M), C)
    stage = Array{Complex{Tr}, D}(undef, ms...)
    cj_host = Vector{Complex{Tr}}(undef, M)
    # Branch on the output element type, so `_synth_type2_gpu!` specializes on a concrete array.
    return real_output ?
        _synth_type2_gpu!(Array{Tr}(undef, spatial..., batch...), full, p, plan, fks, cjs, stage,
            cj_host, ms, M, ntrans, C, iflag < 0, backend) :
        _synth_type2_gpu!(Array{Complex{Tr}}(undef, spatial..., batch...), full, p, plan, fks, cjs,
            stage, cj_host, ms, M, ntrans, C, iflag < 0, backend)
end

# Type-2 execution loop. `Z` fixes whether the destination takes the real part or the whole value, so the
# per-element branch folds away. Staging keeps every device array indexed only by `copyto!`.
function _synth_type2_gpu!(out::AbstractArray{Z}, full, p, plan, fks::Tuple, cjs::Tuple, stage, cj_host,
        ms::NTuple{D, Int}, M::Int, ntrans::Int, C::Int, neg::Bool, backend) where {Z, D}
    Pm = prod(ms)
    for base in 0:C:(ntrans - 1)
        nvalid = min(C, ntrans - base)
        for c in 1:C
            fk = fks[c]
            if c > nvalid
                fill!(fk, zero(eltype(fk)))          # pad the final chunk to the plan's transform count
                continue
            end
            coff = (base + c - 1) * Pm
            @inbounds for i in 1:Pm
                z = full[coff + i]
                stage[i] = (neg ? conj(z) : z) * conj(p[i])
            end
            copyto!(fk, stage)                        # host → device
        end
        NonuniformFFTs.exec_type2!(cjs, plan, fks)
        KA.synchronize(backend)
        for c in 1:nvalid
            copyto!(cj_host, cjs[c])                  # device → host
            foff = (base + c - 1) * M
            @inbounds for j in 1:M
                v = neg ? conj(cj_host[j]) : cj_host[j]
                out[foff + j] = Z <: Real ? real(v) : v
            end
        end
    end
    return out
end

end # module FlowFieldSpectraNonuniformFFTsKernelAbstractionsExt
