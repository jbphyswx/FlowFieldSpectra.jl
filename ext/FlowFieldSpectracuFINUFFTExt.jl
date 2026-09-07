module FlowFieldSpectracuFINUFFTExt

using FINUFFT: FINUFFT
using CUDA: CUDA
using FlowFieldSpectra: FlowFieldSpectra as FFS
using ComputationalBackends: ComputationalBackends
using SpectralBackends: SpectralBackends
using FlowGeometries: FlowGeometries

# =============================================================================
# Fast GPU NUFFT for Cartesian grids on the packed-native model: NUFFTSpectralBackend ×
# GPUBackend{<:CUDABackend}, batched. cuFINUFFT (FINUFFT.jl's CUDA extension) with `ntrans = ∏batch`; a
# scattered (unstructured) grid uses the guru plan directly, a nonuniform tensor-product (structured)
# grid the separable per-axis 1-D path (bottom). Nonuniform points fixed by the grid ⇒
# `cufinufft_makeplan` + `cufinufft_setpts!` once, `cufinufft_exec!` per call reusing device buffers.
#
# Plans pass `modeord = 1`, so the output is already native (unshifted) order, and the published layout
# matches the host FINUFFT path exactly: a real field gives the packed half `(N₁÷2+1, N₂…)`, a complex
# field the full native spectrum. The index and phase tables come from `FFS.Packing`
# (`hermitian_request_size`, `offset_phase`, `twin_table`), which the host FINUFFT ext also uses, so the
# two providers cannot drift apart and the host test gate covers this file's mode arithmetic.
#
# CUDA-only; a non-CUDA device hits the core error stub. Validated only on real NVIDIA hardware via
# `gpu/`, never on CI — this file's device plumbing has NOT been executed.
# =============================================================================

_default_eps(::Type{T}) where {T} = T === Float32 ? 1.0e-6 : 1.0e-8

# Point scaling (host): points folded to [0, 2π) plus the per-axis offset and period.
function _scaled_points(::Type{T}, coords::Tuple, Ls::NTuple{D}) where {T, D}
    M = length(coords[1])
    for d in 1:D
        length(coords[d]) == M || throw(DimensionMismatch("coordinate $d length mismatch"))
    end
    offsets = ntuple(d -> T(minimum(coords[d])), D)
    ranges = ntuple(d -> T(Ls[d]), D)    # already resolved by `Grids.axis_range`
    scaled = ntuple(d -> T(2π) .* (T.(collect(coords[d])) .- offsets[d]) ./ ranges[d], D)
    return scaled, offsets, ranges, M
end

# Native-order guru type-1 plan over fixed points, with the points staged to the device.
function _guru(::Type{T}, scaled::Tuple, ns::NTuple{D, Int}, ntrans::Int, iflag::Int,
        eps::Real) where {T, D}
    CUDA.functional() || throw(ArgumentError("cuFINUFFT requires a functional CUDA device."))
    dev = ntuple(d -> CUDA.CuArray(scaled[d]), D)
    guru = FINUFFT.cufinufft_makeplan(1, collect(ns), -iflag, ntrans, T(eps); dtype = T, modeord = 1)
    if D == 1
        FINUFFT.cufinufft_setpts!(guru, dev[1])
    elseif D == 2
        FINUFFT.cufinufft_setpts!(guru, dev[1], dev[2])
    elseif D == 3
        FINUFFT.cufinufft_setpts!(guru, dev[1], dev[2], dev[3])
    else
        FINUFFT.cufinufft_destroy!(guru)
        throw(ArgumentError("cuFINUFFT supports up to 3 dimensions; got $D"))
    end
    finalizer(FINUFFT.cufinufft_destroy!, guru)   # free the C plan when the guru is GC'd
    return guru
end

# ---- Nyquist twins ----
# Same conjugate read as the host FINUFFT path: real strengths make the output Hermitian, so a twin's
# negated frequency is native. The gather runs on device into a small buffer; only that hyperplane
# crosses to the host slice the `NyquistTwin` publishes.
struct TwinGatherGPU{DV, SL, IX, FC}
    dev::DV          # device (S,) gathered values
    slice::SL        # host (shape…, batch…) published values
    src::IX          # device (S,) indices into one native spectrum
    fac::FC          # device (S,) offset phase × 1/M
    n::Int           # S, zero for a mask no mode reaches
end

function _twin_gpu(::Type{T}, ms::NTuple{D, Int}, ns::NTuple{D, Int}, mask::Int, offsets, ranges,
        M::Int, batch::Tuple) where {T, D}
    shape, src, fac = FFS.Packing.twin_table(T, ms, mask, ns, offsets, ranges, M; conjugate = true)
    S = length(src)
    return TwinGatherGPU(
        CUDA.zeros(Complex{T}, max(S, 1)),
        zeros(Complex{T}, shape..., batch...),
        CUDA.CuArray(S == 0 ? Int[1] : src),
        CUDA.CuArray(S == 0 ? Complex{T}[0] : fac),
        S,
    )
end

function _with_twins(::Type{T}, ks_phys::Tuple, ms::NTuple{D, Int}, ns::NTuple{D, Int}, offsets, ranges,
        M::Int, batch::Tuple) where {T, D}
    D >= 2 || return ks_phys, ()
    gs = ntuple(FFS.Packing.n_twin_slices(Val(D))) do mask
        _twin_gpu(T, ms, ns, mask, offsets, ranges, M, batch)
    end
    twin = FFS.Packing.NyquistTwin(map(g -> g.slice, gs))
    return (FFS.Packing.with_twin(ks_phys[1], twin), Base.tail(ks_phys)...), gs
end

@inline _gather_twins!(::Tuple{}, fk, Pm::Int, ntrans::Int, neg::Bool) = nothing
@inline function _gather_twins!(gs::Tuple, fk, Pm::Int, ntrans::Int, neg::Bool)
    _gather_twin!(first(gs), fk, Pm, ntrans, neg)
    return _gather_twins!(Base.tail(gs), fk, Pm, ntrans, neg)
end

function _gather_twin!(g::TwinGatherGPU, fk, Pm::Int, ntrans::Int, neg::Bool)
    S = g.n
    S == 0 && return nothing
    flat = vec(fk)
    dv = view(g.dev, 1:S)
    for t in 1:ntrans
        spec = view(flat, ((t - 1) * Pm + 1):(t * Pm))
        if neg
            map!((s, f) -> conj(conj(spec[s]) * f), dv, g.src, g.fac)
        else
            map!((s, f) -> conj(spec[s]) * f, dv, g.src, g.fac)
        end
        soff = (t - 1) * S
        copyto!(view(vec(g.slice), (soff + 1):(soff + S)), dv)
    end
    return nothing
end

# ---- device plans (buffers + tables live on the CUDA device) ----

struct cuFINUFFTRealPlan{T, D, NB, G, CJ, FK, OB, PH, KS, TW, QW} <: FFS.AbstractSpectralPlan
    batch::NTuple{NB, Int}
    guru::G                              # cuFINUFFT guru plan (C resource; self-finalizing, see _guru).
    cj::CJ                               # device (M, ntrans) strengths buffer
    fk::FK                               # device (ns…, ntrans) requested-size native spectrum
    out::OB                              # device (pms…, ntrans) packed output
    ms::NTuple{D, Int}
    ns::NTuple{D, Int}                   # requested mode counts (axis 1 extended at even N₁)
    pms::NTuple{D, Int}                  # packed output size
    ntrans::Int
    M::Int
    neg::Bool                            # caller asked for iflag < 0
    phase::PH                            # device (pms…) offset phase × 1/M
    ks_phys::KS                          # halved axis carries the twin whose slices `twins` refill
    twins::TW
    qw::QW                               # device (M,) grid quadrature factor, or `nothing`
end

FFS.Plans.coefficient_size(p::cuFINUFFTRealPlan) = (p.pms..., p.batch...)
FFS.Plans.coefficient_type(::cuFINUFFTRealPlan{T}) where {T} = Complex{T}
FFS.Plans.wavenumbers(p::cuFINUFFTRealPlan) = p.ks_phys

struct cuFINUFFTComplexPlan{T, D, NB, G, CJ, FK, PH, KS, QW} <: FFS.AbstractSpectralPlan
    batch::NTuple{NB, Int}
    guru::G
    cj::CJ                               # device (M, ntrans) strengths buffer
    fk::FK                               # device (ms…, ntrans) full native spectrum
    ms::NTuple{D, Int}
    ntrans::Int
    M::Int
    phase::PH                            # device (ms…) translation-correction phase × (1/M)
    ks_phys::KS
    qw::QW                               # device (M,) grid quadrature factor, or `nothing`
end

Base.show(io::IO, ::cuFINUFFTRealPlan{T, D}) where {T, D} = print(io, "cuFINUFFTRealPlan{$T, $D}(cuFINUFFT)")
Base.show(io::IO, ::cuFINUFFTComplexPlan{T, D}) where {T, D} = print(io, "cuFINUFFTComplexPlan{$T, $D}(cuFINUFFT)")

function _gpu_real_plan(::Type{T}, coords::Tuple, ms::NTuple{D, Int}, Ls::NTuple{D},
        batch::Tuple, iflag::Int, eps::Real, qw_h = nothing) where {T, D}
    scaled, offsets, ranges, M = _scaled_points(T, coords, Ls)
    ntrans = prod(batch; init = 1)
    ns = FFS.Packing.hermitian_request_size(ms)
    guru = _guru(T, scaled, ns, ntrans, 1, eps)          # tables are built for +1; publish conjugates
    pms = FFS.Packing.packed_size(ms, Val(true))
    phase = CUDA.CuArray(FFS.Packing.offset_phase(T, ms, offsets, ranges, M, Val(true)))
    ks_phys = FFS.Grids.physical_wavenumbers(ranges, ms, Val(true))
    ks_twin, twins = _with_twins(T, ks_phys, ms, ns, offsets, ranges, M, batch)
    cj = CUDA.zeros(Complex{T}, M, ntrans)
    fk = CUDA.zeros(Complex{T}, ns..., ntrans)
    out = CUDA.zeros(Complex{T}, pms..., ntrans)
    qw = qw_h === nothing ? nothing : CUDA.CuArray(Complex{T}.(qw_h))
    bt = NTuple{length(batch), Int}(batch)
    return cuFINUFFTRealPlan{T, D, length(bt), typeof(guru), typeof(cj), typeof(fk), typeof(out), typeof(phase), typeof(ks_twin), typeof(twins), typeof(qw)}(
        bt, guru, cj, fk, out, ms, ns, pms, ntrans, M, iflag < 0, phase, ks_twin, twins, qw,
    )
end

function _gpu_complex_plan(::Type{T}, coords::Tuple, ms::NTuple{D, Int}, Ls::NTuple{D},
        batch::Tuple, iflag::Int, eps::Real, qw_h = nothing) where {T, D}
    scaled, offsets, ranges, M = _scaled_points(T, coords, Ls)
    ntrans = prod(batch; init = 1)
    guru = _guru(T, scaled, ms, ntrans, iflag, eps)
    phase = CUDA.CuArray(FFS.Packing.offset_phase(T, ms, offsets, ranges, M, Val(false), iflag))
    ks_phys = FFS.Grids.physical_wavenumbers(ranges, ms, Val(false))
    cj = CUDA.zeros(Complex{T}, M, ntrans)
    fk = CUDA.zeros(Complex{T}, ms..., ntrans)
    qw = qw_h === nothing ? nothing : CUDA.CuArray(Complex{T}.(qw_h))
    bt = NTuple{length(batch), Int}(batch)
    return cuFINUFFTComplexPlan{T, D, length(bt), typeof(guru), typeof(cj), typeof(fk), typeof(phase), typeof(ks_phys), typeof(qw)}(
        bt, guru, cj, fk, ms, ntrans, M, phase, ks_phys, qw,
    )
end

"""
    calculate_spectrum!(coeffs, plan::cuFINUFFTRealPlan, field) -> ks_phys

Execute a prebuilt cuFINUFFT guru plan in place for a real field. `field` `(N, batch…)` fills the device
strengths buffer; `coeffs` receives the packed half `(N₁÷2+1, N₂…, batch…)`. The half is the leading
`pms[1]` entries of axis 1 of the requested-size spectrum, so it is a strided device view.
"""
function FFS.calculate_spectrum!(coeffs::AbstractArray{Complex{T}}, plan::cuFINUFFTRealPlan{T, D},
        field) where {T, D}
    copyto!(plan.cj, field)                                      # host/device → device, linear (widen)
    plan.qw === nothing || (plan.cj .*= plan.qw)                 # grid quadrature factor
    FINUFFT.cufinufft_exec!(plan.guru, plan.cj, plan.fk)
    half = view(plan.fk, 1:plan.pms[1], ntuple(_ -> Colon(), D - 1)..., Colon())
    if plan.neg
        plan.out .= conj.(half .* plan.phase)
    else
        plan.out .= half .* plan.phase
    end
    _gather_twins!(plan.twins, plan.fk, prod(plan.ns), plan.ntrans, plan.neg)
    Ph = prod(plan.pms) * plan.ntrans
    copyto!(view(reshape(coeffs, :), 1:Ph), view(reshape(plan.out, :), 1:Ph))
    return plan.ks_phys
end

"""
    calculate_spectrum!(coeffs, plan::cuFINUFFTComplexPlan, field) -> ks_phys

Execute a prebuilt cuFINUFFT guru plan in place for a complex field; `coeffs` receives the full native
spectrum `(ms…, batch…)`.
"""
function FFS.calculate_spectrum!(coeffs::AbstractArray{Complex{T}}, plan::cuFINUFFTComplexPlan{T, D},
        field) where {T, D}
    copyto!(plan.cj, field)
    plan.qw === nothing || (plan.cj .*= plan.qw)                 # grid quadrature factor
    FINUFFT.cufinufft_exec!(plan.guru, plan.cj, plan.fk)
    plan.fk .*= plan.phase                                       # (ms…) broadcasts over ntrans, folds 1/M
    Pm = prod(plan.ms) * plan.ntrans
    copyto!(view(reshape(coeffs, :), 1:Pm), view(reshape(plan.fk, :), 1:Pm))
    return plan.ks_phys
end

function FFS.plan_spectrum(::FFS.FINUFFTBackend, ::ComputationalBackends.GPUBackend{<:CUDA.CUDABackend},
        g::FFS.Grids.PointwiseCartesian, ::Type{T}, ms::NTuple{D, Int};
        batch::Tuple = (), iflag::Int = 1, eps::Real = _default_eps(real(float(T)))) where {T, D}
    Tr = real(float(T))
    coords, _ = FFS.Grids.point_coordinates(Tr, g, D)
    Ls = ntuple(d -> FFS.Grids.axis_range(Tr, g, d), D)
    qw = FFS.Grids.quadrature_scale(g, Tr, length(coords[1]))
    return T <: Real ? _gpu_real_plan(Tr, coords, ms, Ls, batch, iflag, eps, qw) :
           _gpu_complex_plan(Tr, coords, ms, Ls, batch, iflag, eps, qw)
end

# One-shot — scattered (unstructured) Cartesian grid → guru cuFINUFFT.
function FFS._calculate_spectrum_gpu_nufft(::FFS.FINUFFTBackend, ::ComputationalBackends.GPUBackend{<:CUDA.CUDABackend},
        g::FFS.Grids.PointwiseCartesian,
        field::AbstractArray, ms::Tuple; iflag::Int = 1, eps::Union{Nothing, Real} = nothing, kwargs...)
    D = length(ms)
    R = eltype(field) <: Real
    T = float(real(eltype(g)))
    coords, _ = FFS.Grids.point_coordinates(T, g, D)
    Ls = ntuple(d -> FFS.Grids.axis_range(T, g, d), D)
    batch = FFS.Grids.field_batch_shape(g, field)
    epsv = eps === nothing ? _default_eps(T) : eps
    msn = NTuple{D, Int}(ms)
    qw = FFS.Grids.quadrature_scale(g, T, length(coords[1]))
    plan = R ? _gpu_real_plan(T, coords, msn, Ls, batch, iflag, epsv, qw) :
               _gpu_complex_plan(T, coords, msn, Ls, batch, iflag, epsv, qw)
    coeffs = zeros(Complex{T}, FFS.Packing.packed_size(msn, Val(R))..., batch...)
    ks = FFS.calculate_spectrum!(coeffs, plan, field)
    return coeffs, ks
end

# =============================================================================
# Separable GPU NUFFT for a nonuniform tensor-product (structured) Cartesian grid: the D-dim kernel
# factorizes, so it is a sequence of 1-D cuFINUFFT type-1 transforms — one per axis — with every other
# spatial + batch dim carried as the `ntrans` batch. Device-side analog of the host separable path
# (FINUFFTExt._nufft_axis); each 1-D transform needs only its own length-N_d axis (no ∏N_d coords).
# Because every transform is 1-D, this supports any D. Each axis pass leaves the raw transform in native
# order; the grid-offset phase and the `1/∏N_d` normalization are applied once at the end, by the same
# `Packing.offset_phase` / `Packing.twin_table` the scattered path uses. CUDA-only; never run on CI.
# =============================================================================
function FFS._calculate_spectrum_gpu_nufft(::FFS.FINUFFTBackend, ::ComputationalBackends.GPUBackend{<:CUDA.CUDABackend},
        g::FlowGeometries.Grids.AbstractStructuredGrid{<:FlowGeometries.Geometry.AbstractCartesianGeometry},
        field::AbstractArray, ms::Tuple; iflag::Int = 1, eps::Union{Nothing, Real} = nothing, kwargs...)
    CUDA.functional() || throw(ArgumentError("cuFINUFFT requires a functional CUDA device."))
    D = ndims(g)
    T = float(real(eltype(g)))
    R = eltype(field) <: Real
    epsv = T(eps === nothing ? _default_eps(T) : eps)
    msn = NTuple{D, Int}(ms)
    axs = ntuple(d -> T.(collect(FlowGeometries.Grids.coordinates(g, d))), D)
    offsets, ranges = FFS.Grids.axis_geometry(T, g, D)
    npts = length(g)
    batch = ntuple(i -> size(field, D + i), ndims(field) - D)
    ntrans = prod(batch; init = 1)
    sgn = R ? 1 : iflag                  # a real field's tables are built for +1, conjugated on publish
    ns = R ? FFS.Packing.hermitian_request_size(msn) : msn
    A = CUDA.CuArray{Complex{T}}(undef, size(field)...)
    copyto!(A, field)                                            # host/device → device, widen real→complex
    qwh = FFS.Grids.quadrature_scale(g, T, npts)                 # grid quadrature factor
    if qwh !== nothing
        # `A` carries the grid shape here, so the factor reshapes to it and broadcasts over the batch.
        A .*= reshape(CUDA.CuArray(Complex{T}.(qwh)), size(g)..., ntuple(_ -> 1, ndims(A) - D)...)
    end
    for d in 1:D
        A = _gpu_nufft_axis(A, d, axs[d], ns[d], ranges[d], offsets[d], sgn, epsv)
    end
    if R
        pms = FFS.Packing.packed_size(msn, Val(true))
        phase = CUDA.CuArray(FFS.Packing.offset_phase(T, msn, offsets, ranges, npts, Val(true)))
        ks_phys = FFS.Grids.physical_wavenumbers(ranges, msn, Val(true))
        ks, twins = _with_twins(T, ks_phys, msn, ns, offsets, ranges, npts, batch)
        half = view(reshape(A, ns..., ntrans), 1:pms[1], ntuple(_ -> Colon(), D - 1)..., Colon())
        out = CUDA.zeros(Complex{T}, pms..., ntrans)
        if iflag < 0
            out .= conj.(half .* phase)
        else
            out .= half .* phase
        end
        _gather_twins!(twins, A, prod(ns), ntrans, iflag < 0)
        return reshape(Array(out), pms..., batch...), ks
    end
    A .*= CUDA.CuArray(FFS.Packing.offset_phase(T, msn, offsets, ranges, npts, Val(false), iflag))
    return Array(A), FFS.Grids.physical_wavenumbers(ranges, msn, Val(false))
end

# Axis `d` moved to the front. A device pass permutes through `permutedims!` into preallocated buffers;
# cuFINUFFT reads the transform axis contiguously, and the permutation runs as a device kernel.
@inline _axis_perm(nd::Int, d::Int) = (d, ntuple(i -> i < d ? i : i + 1, nd - 1)...)

# 1-D type-1 guru plan over one grid axis, `ntrans` lines per execution, points staged to the device.
function _gpu_axis_guru(::Type{T}, axis::AbstractVector{T}, m::Int, rng::T, off::T, iflag::Int,
        eps::T, ntrans::Int) where {T}
    x = CUDA.CuArray(T(2π) .* (axis .- off) ./ rng)
    guru = FINUFFT.cufinufft_makeplan(1, [m], -iflag, ntrans, eps; dtype = T, modeord = 1)
    FINUFFT.cufinufft_setpts!(guru, x)
    return guru
end

# Transform axis `d` of `A` into `out` (dim `d` length `m`), staging through `pin`/`pfk`, whose shapes are
# the axis-to-front permutations of `A` and `out`. Shared by the one-shot and the reusable plan.
function _gpu_axis_pass!(out::CUDA.CuArray{Complex{T}}, A::CUDA.CuArray{Complex{T}}, d::Int, guru,
        pin::CUDA.CuArray{Complex{T}}, pfk::CUDA.CuArray{Complex{T}}) where {T}
    nd = ndims(A)
    perm = _axis_perm(nd, d)
    Nd = size(A, d)
    m = size(out, d)
    ntrans = length(A) ÷ Nd
    permutedims!(pin, A, perm)                                   # device (Nd, rest…), contiguous
    FINUFFT.cufinufft_exec!(guru, reshape(pin, Nd, ntrans), reshape(pfk, m, ntrans))
    permutedims!(out, pfk, invperm(collect(perm)))               # restore dim order
    return out
end

# One 1-D cuFINUFFT type-1 transform along dim `d` (points = `axis`), modes = `m` in native order; other
# dims → ntrans. Raw output: the offset phase and normalization are applied once by the caller. Returns a
# new device array whose dim `d` now has length `m`.
function _gpu_nufft_axis(A::CUDA.CuArray{Complex{T}}, d::Int, axis::AbstractVector{T}, m::Int,
        rng::T, off::T, iflag::Int, eps::T) where {T}
    nd = ndims(A)
    Nd = length(axis)
    size(A, d) == Nd ||
        throw(DimensionMismatch("axis $d: field length $(size(A, d)) ≠ grid axis length $Nd"))
    sz = size(A)
    outsz = FFS.Packing.axis_out_size(sz, d, m)
    perm = _axis_perm(nd, d)
    ntrans = length(A) ÷ Nd
    guru = _gpu_axis_guru(T, axis, m, rng, off, iflag, eps, ntrans)
    out = CUDA.zeros(Complex{T}, outsz...)
    pin = CUDA.zeros(Complex{T}, ntuple(i -> sz[perm[i]], nd)...)
    pfk = CUDA.zeros(Complex{T}, ntuple(i -> outsz[perm[i]], nd)...)
    _gpu_axis_pass!(out, A, d, guru, pin, pfk)
    FINUFFT.cufinufft_destroy!(guru)
    return out
end

# ---- reusable separable device plan: per-axis guru plans and device buffers built once ----

# Device counterpart of the host `FINUFFTSeparablePlan`. `R` marks a real field, so the publish branch
# folds. An axis pass carries every line of the axis in a single execution.
FFS.Plans.coefficient_size(p::cuFINUFFTComplexPlan) = (p.ms..., p.batch...)
FFS.Plans.coefficient_type(::cuFINUFFTComplexPlan{T}) where {T} = Complex{T}
FFS.Plans.wavenumbers(p::cuFINUFFTComplexPlan) = p.ks_phys

struct cuFINUFFTSeparablePlan{T, D, R, NB, G, PI, PF, W, OB, PH, KS, TW, QW} <: FFS.AbstractSpectralPlan
    batch::NTuple{NB, Int}
    gurus::G                         # D × 1-D type-1 guru plan (C resource; self-finalizing)
    pins::PI                         # D × permuted input staging
    pfks::PF                         # D × permuted spectrum staging
    work::W                          # D+1 × device working arrays; `work[1]` takes the widened field
    out::OB                          # device (pms…, ntrans) packed output
    ms::NTuple{D, Int}
    ns::NTuple{D, Int}               # requested mode counts (axis 1 extended at even N₁)
    pms::NTuple{D, Int}              # packed output size
    ntrans::Int
    npts::Int
    neg::Bool                        # caller asked for iflag < 0
    phase::PH                        # device offset phase × 1/∏N_d
    ks_phys::KS
    twins::TW
    qw::QW                           # device (npts,) grid quadrature factor, or `nothing`
end

Base.show(io::IO, ::cuFINUFFTSeparablePlan{T, D, R}) where {T, D, R} =
    print(io, "cuFINUFFTSeparablePlan{$T, $D}(cuFINUFFT, ", R ? "real" : "complex", ")")

function FFS.plan_spectrum(::FFS.FINUFFTBackend, ::ComputationalBackends.GPUBackend{<:CUDA.CUDABackend},
        g::FlowGeometries.Grids.AbstractStructuredGrid{<:FlowGeometries.Geometry.AbstractCartesianGeometry},
        ::Type{T}, ms::NTuple{D, Int}; batch::Tuple = (), iflag::Int = 1,
        eps::Real = _default_eps(real(float(T)))) where {T, D}
    CUDA.functional() || throw(ArgumentError("cuFINUFFT requires a functional CUDA device."))
    Tr = real(float(T))
    R = T <: Real
    ndims(g) == D || throw(DimensionMismatch("grid has $(ndims(g)) dims; asked for $D mode counts"))
    axs = ntuple(d -> Tr.(collect(FlowGeometries.Grids.coordinates(g, d))), D)
    offsets, ranges = FFS.Grids.axis_geometry(Tr, g, D)
    Ns = ntuple(d -> length(axs[d]), D)
    npts = prod(Ns)
    ntrans = prod(batch; init = 1)
    sgn = R ? 1 : iflag                # a real field's tables are built for +1, conjugated on publish
    ns = R ? FFS.Packing.hermitian_request_size(ms) : ms
    epsv = Tr(eps)

    work = ntuple(d -> CUDA.zeros(Complex{Tr}, FFS.Packing.axis_work_shape(Ns, ns, d, batch)...), D + 1)
    nd = D + length(batch)
    pins = ntuple(D) do d
        perm = _axis_perm(nd, d)
        sz = size(work[d])
        CUDA.zeros(Complex{Tr}, ntuple(i -> sz[perm[i]], nd)...)
    end
    pfks = ntuple(D) do d
        perm = _axis_perm(nd, d)
        sz = size(work[d + 1])
        CUDA.zeros(Complex{Tr}, ntuple(i -> sz[perm[i]], nd)...)
    end
    gurus = ntuple(D) do d
        guru = _gpu_axis_guru(Tr, axs[d], ns[d], ranges[d], offsets[d], sgn, epsv,
            length(work[d]) ÷ Ns[d])
        finalizer(FINUFFT.cufinufft_destroy!, guru)   # free the C plan when the plan is GC'd
    end

    pms = FFS.Packing.packed_size(ms, Val(R))
    phase = CUDA.CuArray(FFS.Packing.offset_phase(Tr, ms, offsets, ranges, npts, Val(R), R ? 1 : iflag))
    ks_phys = FFS.Grids.physical_wavenumbers(ranges, ms, Val(R))
    ks_out, twins = R ? _with_twins(Tr, ks_phys, ms, ns, offsets, ranges, npts, batch) : (ks_phys, ())
    out = R ? CUDA.zeros(Complex{Tr}, pms..., ntrans) : CUDA.zeros(Complex{Tr}, 1)
    qwh = FFS.Grids.quadrature_scale(g, Tr, npts)
    qw = qwh === nothing ? nothing : CUDA.CuArray(Complex{Tr}.(qwh))
    bt = NTuple{length(batch), Int}(batch)
    return cuFINUFFTSeparablePlan{Tr, D, R, length(bt), typeof(gurus), typeof(pins), typeof(pfks),
            typeof(work), typeof(out), typeof(phase), typeof(ks_out), typeof(twins), typeof(qw)}(
        bt, gurus, pins, pfks, work, out, ms, ns, pms, ntrans, npts, iflag < 0, phase, ks_out, twins, qw,
    )
end

FFS.Plans.coefficient_size(p::cuFINUFFTSeparablePlan) = (p.pms..., p.batch...)
FFS.Plans.coefficient_type(::cuFINUFFTSeparablePlan{T}) where {T} = Complex{T}
FFS.Plans.wavenumbers(p::cuFINUFFTSeparablePlan) = p.ks_phys

"""
    calculate_spectrum!(coeffs, plan::cuFINUFFTSeparablePlan, field) -> ks_phys

Execute a prebuilt separable structured device plan in place. `field` is `(N₁…N_D, batch…)`; `coeffs`
receives the packed half for a real field and the full native spectrum for a complex one. The per-axis
guru plans, point sortings and device buffers are reused across calls.
"""
function FFS.calculate_spectrum!(coeffs::AbstractArray{Complex{T}},
        plan::cuFINUFFTSeparablePlan{T, D, R}, field) where {T, D, R}
    A = plan.work[1]
    length(field) == length(A) || throw(DimensionMismatch(
        "field holds $(length(field)) values; this plan was built for $(length(A)) — pass the " *
        "matching `batch=` to plan_spectrum"))
    copyto!(A, field)                                            # host/device → device, widen
    if plan.qw !== nothing
        # `A` carries the grid shape, so the factor reshapes to it and broadcasts over the batch.
        A .*= reshape(plan.qw, ntuple(i -> size(A, i), Val(D))...,
            ntuple(_ -> 1, ndims(A) - D)...)                     # grid quadrature factor
    end
    for d in 1:D
        _gpu_axis_pass!(plan.work[d + 1], plan.work[d], d, plan.gurus[d], plan.pins[d], plan.pfks[d])
    end
    F = plan.work[D + 1]
    if R
        half = view(reshape(F, plan.ns..., plan.ntrans), 1:plan.pms[1],
            ntuple(_ -> Colon(), D - 1)..., Colon())
        if plan.neg
            plan.out .= conj.(half .* plan.phase)
        else
            plan.out .= half .* plan.phase
        end
        _gather_twins!(plan.twins, F, prod(plan.ns), plan.ntrans, plan.neg)
        Ph = prod(plan.pms) * plan.ntrans
        copyto!(view(reshape(coeffs, :), 1:Ph), view(reshape(plan.out, :), 1:Ph))
    else
        F .*= plan.phase                                          # (ms…) broadcasts over the batch
        Pm = prod(plan.ms) * plan.ntrans
        copyto!(view(reshape(coeffs, :), 1:Pm), view(reshape(F, :), 1:Pm))
    end
    return plan.ks_phys
end

# =============================================================================
# Inverse (synthesis) via a type-2 device guru plan. cuFINUFFT has no real-input transform, so a real
# field's packed half is first completed to the full native spectrum by `Packing.unpacked` WITH the
# halved axis's Nyquist twin — the `k₁ < 0` rows need `+N_d/2` wherever an even axis `d ≥ 2` sits at
# `−N_d/2`, and index negation alone lands off the native axis there. The forward publishes
# `C = fk · p / M` with `|p| = 1` the grid-offset phase, and a type-2 wants `û = C / p`; its `iflag` is
# the forward's, sign flipped, so `Σ û e^{+iflag·i·k·x}` comes back. Coefficients arrive on the host,
# matching what the forward returns, and the spectrum is staged to the device for the execution.
# =============================================================================

function _guru2(::Type{T}, scaled::Tuple, ms::NTuple{D, Int}, ntrans::Int, iflag::Int,
        eps::Real) where {T, D}
    CUDA.functional() || throw(ArgumentError("cuFINUFFT requires a functional CUDA device."))
    dev = ntuple(d -> CUDA.CuArray(scaled[d]), D)
    guru = FINUFFT.cufinufft_makeplan(2, collect(ms), iflag, ntrans, T(eps); dtype = T, modeord = 1)
    if D == 1
        FINUFFT.cufinufft_setpts!(guru, dev[1])
    elseif D == 2
        FINUFFT.cufinufft_setpts!(guru, dev[1], dev[2])
    elseif D == 3
        FINUFFT.cufinufft_setpts!(guru, dev[1], dev[2], dev[3])
    else
        FINUFFT.cufinufft_destroy!(guru)
        throw(ArgumentError("cuFINUFFT supports up to 3 dimensions; got $D"))
    end
    return guru
end

function FFS._synthesize(::FFS.FINUFFTBackend, ::ComputationalBackends.GPUBackend{<:CUDA.CUDABackend},
        g::Union{FlowGeometries.Grids.AbstractStructuredGrid{<:FlowGeometries.Geometry.AbstractCartesianGeometry},
                 FFS.Grids.PointwiseCartesian},
        coeffs::AbstractArray{Complex{T}}, ms::NTuple{D, Int}; real_output::Bool = true, iflag::Int = 1,
        ks = nothing, eps::Union{Nothing, Real} = nothing, kwargs...) where {T <: Real, D}
    coords, spatial = FFS.Grids.point_coordinates(T, g, D)
    Ls = ntuple(d -> FFS.Grids.axis_range(T, g, d), D)
    scaled, offsets, ranges, M = _scaled_points(T, coords, Ls)
    epsv = eps === nothing ? _default_eps(T) : eps
    sz = FFS.Packing.packed_size(ms, Val(real_output))
    size(coeffs)[1:D] == sz || throw(DimensionMismatch(
        "real_output=$(real_output) expects $(sz) on the spectral dims; got $(size(coeffs)[1:D])."))
    batch = ntuple(i -> size(coeffs, D + i), ndims(coeffs) - D)
    ntrans = prod(batch; init = 1)
    # A real field's coefficients are the packed half; complete them to the native cube with the twin.
    full = real_output ? FFS.Packing.unpacked(reshape(coeffs, sz..., ntrans), ms, ks) :
                         reshape(coeffs, ms..., ntrans)
    p = FFS.Packing.offset_phase(T, ms, offsets, ranges, M, Val(false), iflag) .* M
    fk = CUDA.CuArray(full .* conj.(p))                          # (ms…) broadcasts over the batch
    cj = CUDA.zeros(Complex{T}, M, ntrans)
    guru = _guru2(T, scaled, ms, ntrans, iflag, epsv)
    FINUFFT.cufinufft_exec!(guru, fk, cj)
    FINUFFT.cufinufft_destroy!(guru)
    host = Array(cj)
    real_output || return reshape(host, spatial..., batch...)
    out = Array{T}(undef, spatial..., batch...)
    @inbounds for i in eachindex(host)
        out[i] = real(host[i])
    end
    return out
end

end # module FlowFieldSpectracuFINUFFTExt
