module FlowFieldSpectraNonuniformFFTsExt

using NonuniformFFTs: NonuniformFFTs
using FlowFieldSpectra: FlowFieldSpectra as FFS
using ComputationalBackends: ComputationalBackends
using SpectralBackends: SpectralBackends
using FlowGeometries: FlowGeometries

# =============================================================================
# NUFFT (type-1) on the packed-native model, via NonuniformFFTs.jl — a portable provider for
# `NUFFTSpectralBackend` with the same convention as the other NUFFT provider (`Σⱼ vⱼ exp(-i k·xⱼ)`,
# ×1/M, offset-correction phase; `iflag = 1` is the `Σ exp(-i k·x)` sign). Load one NUFFT provider —
# both exts define the same `_calculate_spectrum_nufft`/`plan_spectrum` methods (last-one-wins).
#
# The plan is built with `fftshift = false`, so the transform output is already in native order: a real
# field takes the real-input plan (`PlanNUFFT(T<:Real, …)`, a real-to-complex FFT — half the memory, no
# widen) whose output is the packed half `(N₁÷2+1, N₂…)` (axis 1 rfftfreq, axes ≥2 native fftfreq); a
# complex field gives the full native spectrum. Both are published directly with the offset phase. A real
# field's halved axis also carries its Nyquist twins, gathered from the oversampled grid (see below). A
# scattered grid maps directly through the guru plan; a nonuniform tensor grid takes the separable
# per-axis path (bottom of file), which needs no point cloud. Points are fixed by the grid, so plan +
# `set_points!` run once. NonuniformFFTs' `exec_type1!` allocates internal buffers each call; the
# wrapper adds none.
# =============================================================================

_default_eps(::Type{Float32}) = 1.0f-6
_default_eps(::Type{T}) where {T <: Real} = 1.0e-8
_default_eps(::Type{Complex{T}}) where {T} = _default_eps(T)

# Half-support + oversampling for a target tolerance. NonuniformFFTs needs `σ·N ≥ 2m` per axis, so for a
# grid too small at the default σ = 2 we raise σ (honoring eps), keeping m at the accuracy target.
function _plan_accuracy(eps::Real, ms::NTuple)
    m = clamp(ceil(Int, -log10(eps)) + 1, 1, 16)
    σ = float(max(2, cld(2 * m, minimum(ms))))
    return NonuniformFFTs.HalfSupport(m), σ
end

# Native (unshifted) fftfreq integer frequencies of a full axis of length N.
_fftfreq_ints(N::Int) = Int[j <= (N - 1) ÷ 2 ? j : j - N for j in 0:(N - 1)]

# Scaled points (→ [0, 2π)), the packed offset-correction phase (× 1/M; built for `iflag = +1`, the real
# path applies the sign by a final conjugate), and packed physical wavenumbers. `R` marks a real field
# (axis 1 halved to rfftfreq `0:N₁÷2`); the other axes are native fftfreq — matching the `fftshift=false`
# transform output.
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

# =============================================================================
# Nyquist twins. A packed half's index negation sends an even axis at `−N_d/2` to `+N_d/2`, which is off
# the native axis, so `transect_spectrum` needs those coefficients supplied. The transform already
# computes them: `exec_type1!` FFTs onto the oversampled grid `data.ûs` and its deconvolution step only
# reads that array while truncating to the native box, so `+N_d/2` is present and merely dropped. Each
# twin value is therefore one oversampled entry times the same deconvolution `normfactor / ∏ϕ̂` the
# stored entries get — `ϕ̂` is even in `k`, so `ϕ̂(+N_d/2)` is the coefficient held at the `−N_d/2` output
# index. Plan time folds that factor together with the offset phase, leaving one multiply per value.
# =============================================================================

struct TwinPeek{SL, IX, FC}
    slice::SL        # (sl…, batch…) published twin values
    src::IX          # slice spatial index → linear index into the oversampled `ûs`
    fac::FC          # (S,) deconvolution × offset phase × 1/M
end

function _twin_peek(::Type{Tr}, p, ms::NTuple{D, Int}, offsets, ranges, M::Int, mask::Int,
        batch::Tuple) where {Tr <: Real, D}
    ovs = size(first(p.data.ûs))
    normfactor = prod(Ñ -> 2π / Ñ, size(first(p.data.us)))
    phis = map(NonuniformFFTs.fourier_coefficients, p.kernels)
    shape, src, fac = FFS.Packing.twin_table(Tr, ms, mask, ovs, offsets, ranges, M;
        phis = phis, normfactor = normfactor)
    return TwinPeek(zeros(Complex{Tr}, shape..., batch...), src, fac)
end

# One peek per nonempty subset of the axes `2:D`, and the `ks` whose halved axis carries their slices.
function _with_twins(::Type{Tr}, ks_phys::Tuple, p, ms::NTuple{D, Int}, offsets, ranges, M::Int,
        batch::Tuple) where {Tr <: Real, D}
    D >= 2 || return ks_phys, ()
    peeks = ntuple(FFS.Packing.n_twin_slices(Val(D))) do mask
        _twin_peek(Tr, p, ms, offsets, ranges, M, mask, batch)
    end
    twin = FFS.Packing.NyquistTwin(map(t -> t.slice, peeks))
    return (FFS.Packing.with_twin(ks_phys[1], twin), Base.tail(ks_phys)...), peeks
end

# Gather every transform's twins out of the batch of oversampled arrays. `ûs` is overwritten by the next
# `exec_type1!`, so this runs before the next execution.
@inline _gather_twins!(::Tuple{}, ûs::Tuple, base::Int, nvalid::Int, neg::Bool) = nothing
@inline function _gather_twins!(peeks::Tuple, ûs::Tuple, base::Int, nvalid::Int, neg::Bool)
    _gather_twin!(first(peeks), ûs, base, nvalid, neg)
    return _gather_twins!(Base.tail(peeks), ûs, base, nvalid, neg)
end

# Chunk-local transform `c` is the caller's transform `base + c`; a partial final chunk gathers only its
# `nvalid` real transforms.
function _gather_twin!(tp::TwinPeek, ûs::Tuple, base::Int, nvalid::Int, neg::Bool)
    S = length(tp.src)
    @inbounds for c in 1:nvalid
        u = ûs[c]
        soff = (base + c - 1) * S
        if neg
            for s in 1:S
                tp.slice[soff + s] = conj(u[tp.src[s]] * tp.fac[s])
            end
        else
            for s in 1:S
                tp.slice[soff + s] = u[tp.src[s]] * tp.fac[s]
            end
        end
    end
    return nothing
end

# ---- immutable plans (no C resource → no finalizer) ----

"""
    DEFAULT_BATCH_CHUNK

Default transforms per `exec_type1!` on this CPU path, overridable per call as `batch_chunk=` on
`plan_spectrum` and `calculate_spectrum`. One execution spreads that many values per point and evaluates
the point's `(2m)^D` kernel weights once for all of them, so batching amortizes the dominant cost of a
type-1 transform. NonuniformFFTs holds one oversampled grid *and* one padded per-thread spreading buffer
per transform, so the count also sets the plan's memory; the spreading buffer is what throughput is
sensitive to, and its size follows the kernel half-support and the plan's block size. A batch larger
than this runs as successive chunks, so plan memory does not follow the caller's batch shape.
`batch_chunk <= 0` runs the whole batch in one execution.

The value applies to this CPU spreading path at its default block size. A different half-support, block
size, or device shifts the best count, so callers set it per call.
"""
const DEFAULT_BATCH_CHUNK = 4

# Transforms a field carries, and the native chunk a plan executes.
@inline _nbatch(batch::Tuple) = prod(batch; init = 1)
@inline _chunk(B::Int, batch_chunk::Int) = batch_chunk <= 0 ? max(B, 1) : clamp(batch_chunk, 1, max(B, 1))

struct NUFFTNonuniformComplexPlan{T, D, NB, P, CJ, FK, PH, KS, QW} <: FFS.AbstractSpectralPlan
    batch::NTuple{NB, Int}
    plan::P
    cj::CJ                           # C × (M,) reused strengths
    fk::FK                           # C × (ms…) reused full native spectra
    ms::NTuple{D, Int}
    M::Int
    nbatch::Int                      # transforms the caller's field must carry
    iflag::Int
    phase::PH
    ks_phys::KS
    qw::QW                           # (M,) grid quadrature factor, or `nothing` for a constant measure
end

struct NUFFTNonuniformRealPlan{T, D, NB, P, CJ, FK, PH, KS, TW, QW} <: FFS.AbstractSpectralPlan
    batch::NTuple{NB, Int}
    plan::P                          # NonuniformFFTs.PlanNUFFT{T<:Real}, packed half
    cj::CJ                           # C × (M,) reused real strengths (no widen)
    fk_half::FK                      # C × (N₁÷2+1, N₂…) reused packed half-spectra
    ms::NTuple{D, Int}
    M::Int
    nbatch::Int
    iflag::Int
    phase::PH                        # (N₁÷2+1, N₂…) offset phase × 1/M (built for iflag=+1)
    ks_phys::KS                      # halved axis carries the twin whose slices `twins` refill
    twins::TW
    qw::QW                           # (M,) grid quadrature factor, or `nothing` for a constant measure
end

Base.show(io::IO, ::NUFFTNonuniformComplexPlan{T, D}) where {T, D} = print(io, "NUFFTNonuniformComplexPlan{$T, $D}(NonuniformFFTs)")
Base.show(io::IO, ::NUFFTNonuniformRealPlan{T, D}) where {T, D} = print(io, "NUFFTNonuniformRealPlan{$T, $D}(NonuniformFFTs)")

function _nu_plan(::Type{Complex{Tr}}, coords::Tuple, ms::NTuple{D, Int}, Ls::NTuple{D},
        iflag::Int, eps::Real, batch::Tuple, batch_chunk::Int, qw = nothing) where {Tr <: Real, D}
    scaled, phase, ks_phys, M, _, _ = _scaled_phase_ks(Tr, coords, ms, Ls, iflag, false)
    hs, σ = _plan_accuracy(eps, ms)
    B = _nbatch(batch)
    C = _chunk(B, batch_chunk)
    plan = NonuniformFFTs.PlanNUFFT(Complex{Tr}, ms; ntransforms = Val(C), fftshift = false, m = hs, σ = σ)
    NonuniformFFTs.set_points!(plan, scaled)
    cj = ntuple(_ -> Vector{Complex{Tr}}(undef, M), C)
    fk = ntuple(_ -> Array{Complex{Tr}, D}(undef, ms...), C)
    bt = NTuple{length(batch), Int}(batch)
    return NUFFTNonuniformComplexPlan{Tr, D, length(bt), typeof(plan), typeof(cj), typeof(fk), typeof(phase), typeof(ks_phys), typeof(qw)}(
        bt, plan, cj, fk, ms, M, B, iflag, phase, ks_phys, qw,
    )
end

function _nu_plan(::Type{Tr}, coords::Tuple, ms::NTuple{D, Int}, Ls::NTuple{D},
        iflag::Int, eps::Real, batch::Tuple, batch_chunk::Int, qw = nothing) where {Tr <: Real, D}
    scaled, phase, ks_phys, M, offsets, ranges = _scaled_phase_ks(Tr, coords, ms, Ls, 1, true)   # sign applied via final conj
    hs, σ = _plan_accuracy(eps, ms)
    B = _nbatch(batch)
    C = _chunk(B, batch_chunk)
    plan = NonuniformFFTs.PlanNUFFT(Tr, ms; ntransforms = Val(C), fftshift = false, m = hs, σ = σ)
    NonuniformFFTs.set_points!(plan, scaled)
    cj = ntuple(_ -> Vector{Tr}(undef, M), C)
    fk_half = ntuple(_ -> Array{Complex{Tr}, D}(undef, size(plan)...), C)
    ks_twin, twins = _with_twins(Tr, ks_phys, plan, ms, offsets, ranges, M, batch)
    bt = NTuple{length(batch), Int}(batch)
    return NUFFTNonuniformRealPlan{Tr, D, length(bt), typeof(plan), typeof(cj), typeof(fk_half), typeof(phase), typeof(ks_twin), typeof(twins), typeof(qw)}(
        bt, plan, cj, fk_half, ms, M, B, iflag, phase, ks_twin, twins, qw,
    )
end

# ---- complex path: full native spectrum ----

function FFS.calculate_spectrum!(coeffs::AbstractArray{Complex{T}}, plan::NUFFTNonuniformComplexPlan{T, D},
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
            cj = plan.cj[c]
            if c > nvalid
                fill!(cj, zero(Complex{T}))          # pad the final chunk to the plan's transform count
                continue
            end
            foff = (base + c - 1) * M
            qw = plan.qw
            for i in 1:M
                z = Complex{T}(field[foff + i])
                qw === nothing || (z *= qw[i])                   # grid quadrature factor
                cj[i] = neg ? conj(z) : z
            end
        end
        NonuniformFFTs.exec_type1!(plan.fk, plan.plan, plan.cj)
        for c in 1:nvalid
            fk = plan.fk[c]
            coff = (base + c - 1) * Pm
            if neg
                for i in 1:Pm
                    coeffs[coff + i] = conj(fk[i]) * plan.phase[i]
                end
            else
                for i in 1:Pm
                    coeffs[coff + i] = fk[i] * plan.phase[i]
                end
            end
        end
    end
    return plan.ks_phys
end

function _check_ntrans(len::Int, M::Int, B::Int)
    len == M * B || throw(DimensionMismatch(
        "field carries $(len ÷ M) transforms; this plan was built for $B — pass the matching `batch=` " *
        "to plan_spectrum (`batch_chunk=` sets how many run per execution, independently of this)"))
    return nothing
end

# ---- real path: publish the packed half directly ----

function FFS.calculate_spectrum!(coeffs::AbstractArray{Complex{T}}, plan::NUFFTNonuniformRealPlan{T, D},
        field) where {T, D}
    M = plan.M
    Ph = length(first(plan.fk_half))                              # packed mode count
    B = plan.nbatch
    C = length(plan.cj)
    _check_ntrans(length(field), M, B)
    neg = plan.iflag < 0
    # real field: the iflag=-1 result is the conjugate of the iflag=+1 result (for which `phase` is built)
    @inbounds for base in 0:C:(B - 1)
        nvalid = min(C, B - base)
        for c in 1:C
            cj = plan.cj[c]
            if c > nvalid
                fill!(cj, zero(T))                    # pad the final chunk to the plan's transform count
                continue
            end
            foff = (base + c - 1) * M
            qw = plan.qw
            if qw === nothing
                for i in 1:M
                    cj[i] = field[foff + i]                       # real strengths — no widen
                end
            else
                for i in 1:M
                    cj[i] = field[foff + i] * qw[i]               # grid quadrature factor
                end
            end
        end
        NonuniformFFTs.exec_type1!(plan.fk_half, plan.plan, plan.cj)
        _gather_twins!(plan.twins, plan.plan.data.ûs, base, nvalid, neg)
        for c in 1:nvalid
            fk = plan.fk_half[c]
            coff = (base + c - 1) * Ph
            if neg
                for i in 1:Ph
                    coeffs[coff + i] = conj(fk[i] * plan.phase[i])
                end
            else
                for i in 1:Ph
                    coeffs[coff + i] = fk[i] * plan.phase[i]
                end
            end
        end
    end
    return plan.ks_phys
end

# ---- plan_spectrum + one-shot entries; dispatch the real/complex path on the field eltype ----

function FFS.plan_spectrum(::FFS.NonuniformFFTsBackend, ::ComputationalBackends.AbstractExecutionBackend,
        g::FFS.Grids.PointwiseCartesian, ::Type{T}, ms::NTuple{D, Int};
        batch::Tuple = (), iflag::Int = 1, eps::Real = _default_eps(T),
        batch_chunk::Int = DEFAULT_BATCH_CHUNK) where {T, D}
    coords, _ = FFS.Grids.point_coordinates(real(float(T)), g, D)
    Ls = ntuple(d -> FFS.Grids.axis_range(real(float(T)), g, d), D)
    qw = FFS.Grids.quadrature_scale(g, real(float(T)), length(coords[1]))
    return _nu_plan(T, coords, ms, Ls, iflag, eps, batch, batch_chunk, qw)
end

function FFS._calculate_spectrum_nufft(::FFS.NonuniformFFTsBackend, ::ComputationalBackends.AbstractExecutionBackend,
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
    plan = _nu_plan(E, coords, NTuple{D, Int}(ms), Ls, iflag, epsv, batch, batch_chunk, qw)
    pms = FFS.Packing.packed_size(NTuple{D, Int}(ms), Val(eltype(field) <: Real))
    coeffs = zeros(Complex{Tr}, pms..., batch...)
    ks = FFS.calculate_spectrum!(coeffs, plan, field)
    return coeffs, ks
end

# =============================================================================
# Separable transform for a nonuniform tensor-product (structured) Cartesian grid. The `D`-dimensional
# Fourier kernel factorizes over a tensor grid, so the transform is `D` successive 1-D type-1 NUFFTs —
# one per axis, every OTHER spatial and batch dim carried as the transform batch. Each 1-D pass needs
# only its own length-`N_d` axis, so no `∏N_d` coordinate cloud is built, and spreading costs `D·2m` per
# point in place of `(2m)^D`.
#
# Every pass is complex and unshifted, so the composed result is the full native spectrum, from which a
# real field's packed half and its Nyquist twins come out exactly as on the scattered path: axis 1 asks
# for `hermitian_request_size` modes (putting `±N₁/2` on the axis) and the twins are conjugate reads,
# whose negated frequency the native set holds. The grid-offset phase and `1/∏N_d` are applied once at
# the end. NonuniformFFTs' type-1 is fixed at `Σ v e^{-ikx}`, so an `iflag = -1` complex field is
# conjugated on the way in and out, which composes across the axes as a single conjugation of the whole
# product.
# =============================================================================

# One 1-D type-1 NUFFT along dim `d` (points = `axis`), modes = `m` in native order; every other dim of
# `A` rides as the transform batch, chunked so the plan holds a bounded number of oversampled grids. The
# lines are gathered in place through `Packing.axis_layout`, so the only array this allocates is the
# result. Raw output: the caller applies the offset phase and normalization.
"""
    AxisPlan

One axis's reusable 1-D type-1 plan: the `PlanNUFFT` over that axis's points, its `C` strength/spectrum
vectors, and the chunk offset scratch. Built once per axis and executed by `_nu_axis_pass!`.
"""
struct AxisPlan{P, CJ, FK, IX}
    plan::P
    cjs::CJ                          # C × (N_d,) strengths
    fks::FK                          # C × (m,) spectra
    m::Int
    Nd::Int
    C::Int
    inoff::IX
    outoff::IX
end

Base.show(io::IO, p::AxisPlan) = print(io, "AxisPlan(N=", p.Nd, " → m=", p.m, ", chunk=", p.C, ")")

# `rest` is how many lines the pass will carry, which sets the chunk this plan executes.
function _nu_axis_plan(::Type{Tr}, axis::AbstractVector{Tr}, m::Int, rng::Tr, off::Tr, eps::Real,
        rest::Int, batch_chunk::Int) where {Tr}
    Nd = length(axis)
    C = _chunk(rest, batch_chunk)
    hs, σ = _plan_accuracy(eps, (m,))
    plan = NonuniformFFTs.PlanNUFFT(Complex{Tr}, (m,); ntransforms = Val(C), fftshift = false, m = hs, σ = σ)
    NonuniformFFTs.set_points!(plan, (Tr(2π) .* (axis .- off) ./ rng,))   # points → the transform's period
    cjs = ntuple(_ -> Vector{Complex{Tr}}(undef, Nd), C)
    fks = ntuple(_ -> Vector{Complex{Tr}}(undef, m), C)
    inoff = Vector{Int}(undef, C)
    outoff = Vector{Int}(undef, C)
    return AxisPlan{typeof(plan), typeof(cjs), typeof(fks), typeof(inoff)}(
        plan, cjs, fks, m, Nd, C, inoff, outoff)
end

# Transform axis `d` of `A` into `out`, whose dim `d` holds `ap.m` modes. Lines are gathered straight out
# of `A` and scattered straight into `out`, so neither is permuted.
function _nu_axis_pass!(out::AbstractArray{Complex{Tr}}, A::AbstractArray{Complex{Tr}}, d::Int,
        ap::AxisPlan) where {Tr}
    pre, Nd, post = FFS.Packing.axis_layout(size(A), d)
    Nd == ap.Nd ||
        throw(DimensionMismatch("axis $d: field length $Nd ≠ this plan's grid axis length $(ap.Nd)"))
    m = ap.m
    C = ap.C
    rest = pre * post
    for base in 0:C:(rest - 1)
        nvalid = min(C, rest - base)
        FFS.Packing.axis_chunk_offsets!(ap.inoff, ap.outoff, base, nvalid, pre, Nd, m)
        FFS.Packing.gather_axis_block!(ap.cjs, A, ap.inoff, nvalid, pre, Nd)
        NonuniformFFTs.exec_type1!(ap.fks, ap.plan, ap.cjs)
        FFS.Packing.scatter_axis_block!(out, ap.fks, ap.outoff, nvalid, pre, m)
    end
    return out
end

function _nu_axis(A::AbstractArray{Complex{Tr}}, d::Int, axis::AbstractVector{Tr}, m::Int,
        rng::Tr, off::Tr, eps::Real, batch_chunk::Int) where {Tr}
    sz = size(A)
    sz[d] == length(axis) ||
        throw(DimensionMismatch("axis $d: field length $(sz[d]) ≠ grid axis length $(length(axis))"))
    pre, _, post = FFS.Packing.axis_layout(sz, d)
    ap = _nu_axis_plan(Tr, axis, m, rng, off, eps, pre * post, batch_chunk)
    return _nu_axis_pass!(Array{Complex{Tr}}(undef, FFS.Packing.axis_out_size(sz, d, m)), A, d, ap)
end

# Hybrid composite: one stretched axis of a working array whose other axes are already transformed.
function FFS._axis_nufft(::FFS.NonuniformFFTsBackend, ::ComputationalBackends.AbstractExecutionBackend,
        A::AbstractArray{Complex{Tr}}, d::Int, axis::AbstractVector{Tr}, m::Int, rng::Tr, off::Tr,
        eps::Real; batch_chunk::Int = DEFAULT_BATCH_CHUNK, kwargs...) where {Tr}
    return _nu_axis(A, d, axis, m, rng, off, eps, batch_chunk)
end

# The same pass held for reuse: the hybrid PLAN builds one of these per stretched axis and executes it
# with `_axis_nufft_exec!`, so a repeated call rebuilds no `PlanNUFFT` and re-sorts no points.
function FFS._axis_nufft_plan(::FFS.NonuniformFFTsBackend, ::ComputationalBackends.AbstractExecutionBackend,
        ::Type{Tr}, insize::Tuple, d::Int, axis::AbstractVector{Tr}, m::Int, rng::Tr, off::Tr,
        eps::Real; batch_chunk::Int = DEFAULT_BATCH_CHUNK, kwargs...) where {Tr}
    insize[d] == length(axis) || throw(DimensionMismatch(
        "axis $d: working length $(insize[d]) ≠ grid axis length $(length(axis))"))
    pre, _, post = FFS.Packing.axis_layout(insize, d)
    return _nu_axis_plan(Tr, axis, m, rng, off, eps, pre * post, batch_chunk)
end

FFS._axis_nufft_exec!(out::AbstractArray, ap::AxisPlan, A::AbstractArray, d::Int) =
    _nu_axis_pass!(out, A, d, ap)

function FFS._calculate_spectrum_nufft(::FFS.NonuniformFFTsBackend, ::ComputationalBackends.AbstractExecutionBackend,
        g::FlowGeometries.Grids.AbstractStructuredGrid{<:FlowGeometries.Geometry.AbstractCartesianGeometry},
        field::AbstractArray, ms::Tuple; iflag::Int = 1, eps::Union{Nothing, Real} = nothing,
        batch_chunk::Int = DEFAULT_BATCH_CHUNK, kwargs...)
    D = ndims(g)
    Tr = real(float(eltype(g)))
    R = eltype(field) <: Real
    msn = NTuple{D, Int}(ms)
    axs = ntuple(d -> Tr.(FlowGeometries.Grids.coordinates(g, d)), D)
    offsets, ranges = FFS.Grids.axis_geometry(Tr, g, D)
    npts = prod(size(g))
    batch = ntuple(i -> size(field, D + i), ndims(field) - D)
    ntrans = prod(batch; init = 1)
    epsv = eps === nothing ? _default_eps(float(eltype(field))) : eps
    ns = R ? FFS.Packing.hermitian_request_size(msn) : msn
    neg = !R && iflag < 0
    A = Array{Complex{Tr}}(undef, size(field)...)
    copyto!(A, field)                                          # widen real→complex working copy
    qw = FFS.Grids.quadrature_scale(g, Tr, npts)               # grid quadrature factor
    if qw !== nothing
        @inbounds for b in 1:ntrans, j in 1:npts
            A[j + (b - 1) * npts] *= qw[j]
        end
    end
    neg && (A .= conj.(A))                                     # one conjugation covers all D passes
    for d in 1:D
        A = _nu_axis(A, d, axs[d], ns[d], ranges[d], offsets[d], epsv, batch_chunk)
    end
    if R
        pms = FFS.Packing.packed_size(msn, Val(true))
        phase = FFS.Packing.offset_phase(Tr, msn, offsets, ranges, npts, Val(true))
        ks_phys = FFS.Grids.physical_wavenumbers(ranges, msn, Val(true))
        ks, twins = FFS.Packing.conj_twins(Tr, ks_phys, msn, ns, offsets, ranges, npts, batch)
        coeffs = Array{Complex{Tr}}(undef, pms..., batch...)
        FFS.Packing.publish_packed!(coeffs, A, phase, ns, pms, ntrans, iflag < 0)
        FFS.Packing.gather_conj_twins!(twins, A, prod(ns), ntrans, iflag < 0)
        return coeffs, ks
    end
    neg && (A .= conj.(A))                                     # closes the input conjugation
    A .*= FFS.Packing.offset_phase(Tr, msn, offsets, ranges, npts, Val(false), iflag)
    return A, FFS.Grids.physical_wavenumbers(ranges, msn, Val(false))
end

# ---- reusable separable plan: the per-axis 1-D plans and working buffers built once ----

# `R` marks a real field, so the publish branch folds. Every working array shares rank and element type,
# so indexing them at a runtime axis stays type-stable.
FFS.Plans.coefficient_size(p::NUFFTNonuniformComplexPlan) = (p.ms..., p.batch...)
FFS.Plans.coefficient_type(::NUFFTNonuniformComplexPlan{T}) where {T} = Complex{T}
FFS.Plans.wavenumbers(p::NUFFTNonuniformComplexPlan) = p.ks_phys

FFS.Plans.coefficient_size(p::NUFFTNonuniformRealPlan) =
    (FFS.Packing.packed_size(p.ms, Val(true))..., p.batch...)
FFS.Plans.coefficient_type(::NUFFTNonuniformRealPlan{T}) where {T} = Complex{T}
FFS.Plans.wavenumbers(p::NUFFTNonuniformRealPlan) = p.ks_phys

struct NUFFTSeparablePlan{T, D, R, NB, AP, W, PH, KS, TW, QW} <: FFS.AbstractSpectralPlan
    batch::NTuple{NB, Int}
    axes::AP                         # D × AxisPlan, one per grid axis
    work::W                          # D+1 × working arrays; `work[1]` takes the widened field
    ms::NTuple{D, Int}
    ns::NTuple{D, Int}               # requested mode counts (axis 1 extended at even N₁)
    pms::NTuple{D, Int}              # packed output size
    ntrans::Int
    npts::Int
    neg::Bool                        # caller asked for iflag < 0
    phase::PH
    ks_phys::KS                      # halved axis carries the twin whose slices `twins` refill
    twins::TW
    qw::QW                           # (npts,) grid quadrature factor, or `nothing`
end

Base.show(io::IO, ::NUFFTSeparablePlan{T, D, R}) where {T, D, R} =
    print(io, "NUFFTSeparablePlan{$T, $D}(NonuniformFFTs, ", R ? "real" : "complex", ")")

# `pms` is the packed mode count for the field's realness, already the full native size when complex.
FFS.Plans.coefficient_size(p::NUFFTSeparablePlan) = (p.pms..., p.batch...)
FFS.Plans.coefficient_type(::NUFFTSeparablePlan{T}) where {T} = Complex{T}
FFS.Plans.wavenumbers(p::NUFFTSeparablePlan) = p.ks_phys

function FFS.plan_spectrum(::FFS.NonuniformFFTsBackend, ::ComputationalBackends.AbstractExecutionBackend,
        g::FlowGeometries.Grids.AbstractStructuredGrid{<:FlowGeometries.Geometry.AbstractCartesianGeometry},
        ::Type{T}, ms::NTuple{D, Int}; batch::Tuple = (), iflag::Int = 1,
        eps::Real = _default_eps(T), batch_chunk::Int = DEFAULT_BATCH_CHUNK) where {T, D}
    Tr = real(float(T))
    R = T <: Real
    ndims(g) == D || throw(DimensionMismatch("grid has $(ndims(g)) dims; asked for $D mode counts"))
    axs = ntuple(d -> Tr.(FlowGeometries.Grids.coordinates(g, d)), D)
    offsets, ranges = FFS.Grids.axis_geometry(Tr, g, D)
    Ns = ntuple(d -> length(axs[d]), D)
    npts = prod(Ns)
    ntrans = prod(batch; init = 1)
    ns = R ? FFS.Packing.hermitian_request_size(ms) : ms

    work = ntuple(d -> Array{Complex{Tr}}(undef, FFS.Packing.axis_work_shape(Ns, ns, d, batch)...), D + 1)
    axes = ntuple(D) do d
        pre, _, post = FFS.Packing.axis_layout(size(work[d]), d)
        _nu_axis_plan(Tr, axs[d], ns[d], ranges[d], offsets[d], eps, pre * post, batch_chunk)
    end

    pms = FFS.Packing.packed_size(ms, Val(R))
    phase = FFS.Packing.offset_phase(Tr, ms, offsets, ranges, npts, Val(R), R ? 1 : iflag)
    ks_phys = FFS.Grids.physical_wavenumbers(ranges, ms, Val(R))
    ks_out, twins = R ?
        FFS.Packing.conj_twins(Tr, ks_phys, ms, ns, offsets, ranges, npts, batch) : (ks_phys, ())
    qw = FFS.Grids.quadrature_scale(g, Tr, npts)
    bt = NTuple{length(batch), Int}(batch)
    return NUFFTSeparablePlan{Tr, D, R, length(bt), typeof(axes), typeof(work), typeof(phase),
            typeof(ks_out), typeof(twins), typeof(qw)}(
        bt, axes, work, ms, ns, pms, ntrans, npts, iflag < 0, phase, ks_out, twins, qw,
    )
end

"""
    calculate_spectrum!(coeffs, plan::NUFFTSeparablePlan, field) -> ks_phys

Execute a prebuilt separable structured plan in place. `field` is `(N₁…N_D, batch…)`; `coeffs` is the
packed half for a real field and the full native spectrum for a complex one. The per-axis 1-D plans,
their point sortings and the working buffers are reused across calls.
"""
function FFS.calculate_spectrum!(coeffs::AbstractArray{Complex{T}},
        plan::NUFFTSeparablePlan{T, D, R}, field) where {T, D, R}
    A = plan.work[1]
    length(field) == length(A) || throw(DimensionMismatch(
        "field holds $(length(field)) values; this plan was built for $(length(A)) — pass the " *
        "matching `batch=` to plan_spectrum"))
    copyto!(A, field)                                          # widen real→complex working copy
    qw = plan.qw
    if qw !== nothing
        npts = plan.npts
        @inbounds for b in 1:plan.ntrans, j in 1:npts
            A[j + (b - 1) * npts] *= qw[j]                      # grid quadrature factor
        end
    end
    # The 1-D type-1 is fixed at `Σ v e^{-ikx}`, so a complex field's `iflag = -1` is conjugated in and
    # out; a real field's tables are built for `+1` and conjugated on publish.
    !R && plan.neg && (A .= conj.(A))
    for d in 1:D
        _nu_axis_pass!(plan.work[d + 1], plan.work[d], d, plan.axes[d])
    end
    F = plan.work[D + 1]
    if R
        FFS.Packing.publish_packed!(coeffs, F, plan.phase, plan.ns, plan.pms, plan.ntrans, plan.neg)
        FFS.Packing.gather_conj_twins!(plan.twins, F, prod(plan.ns), plan.ntrans, plan.neg)
    else
        plan.neg && (F .= conj.(F))                             # closes the input conjugation
        Pm = prod(plan.ms)
        @inbounds for t in 1:plan.ntrans
            o = (t - 1) * Pm
            for i in 1:Pm
                coeffs[o + i] = F[o + i] * plan.phase[i]
            end
        end
    end
    return plan.ks_phys
end

# =============================================================================
# Inverse (synthesis) via `exec_type2!`. The forward publishes `C = fk · p / M` with `|p| = 1` the
# grid-offset phase, and a type-2 returns `Σ_k û_k e^{+i k x̃}` against its own scaled points, so
# `û = C / p = C · conj(p)`.
#
# A real field goes through the COMPLEX type-2 on the full native spectrum (completed from the packed
# half by `Packing.unpacked` with the halved axis's Nyquist twin), taking the real part. The real-input
# plan's type-2 cannot serve here: it deconvolves the half into the OVERSAMPLED half-spectrum, where the
# native `k₁ = N₁/2` mode sits in the interior, so its c2r step emits that mode as a conjugate pair at
# both `±N₁/2` while the native set holds only `−N₁/2`. On a nonuniformly sampled grid those are
# different functions of `x`, so no rescaling of the half reconciles them.
# =============================================================================

# Reusable synthesis: the type-2 plan with its point sorting, the per-chunk spectrum and strength
# buffers, the offset phase, and the native cube a packed half expands into — all built once.
struct NUFFTSynthesisPlan{T, D, R, NB, P, FK, CJ, PH, FB, SP} <: FFS.AbstractSynthesisPlan
    plan::P
    fks::FK                          # `C` native-cube buffers the type-2 reads
    cjs::CJ                          # `C` strength vectors it writes
    phase::PH
    full::FB                         # the native cube the packed half expands into
    ms::NTuple{D, Int}
    spatial::SP
    batch::NTuple{NB, Int}
    ntrans::Int
    M::Int
    C::Int
    neg::Bool
end

Base.show(io::IO, ::NUFFTSynthesisPlan{T, D, R}) where {T, D, R} =
    print(io, "NUFFTSynthesisPlan{$T, $D}(NonuniformFFTs, ", R ? "real" : "complex", ")")

FFS.Plans.field_size(p::NUFFTSynthesisPlan) = (p.spatial..., p.batch...)
FFS.Plans.field_type(::NUFFTSynthesisPlan{T, D, R}) where {T, D, R} = R ? T : Complex{T}

function FFS.Plans.plan_synthesis(::FFS.NonuniformFFTsBackend,
        exec::ComputationalBackends.AbstractExecutionBackend,
        g::Union{FlowGeometries.Grids.AbstractStructuredGrid{<:FlowGeometries.Geometry.AbstractCartesianGeometry},
                 FFS.Grids.PointwiseCartesian},
        ::Type{T}, ms::NTuple{D, Int}; batch::Tuple = (), iflag::Int = 1,
        eps::Union{Nothing, Real} = nothing, batch_chunk::Int = DEFAULT_BATCH_CHUNK,
        kwargs...) where {T, D}
    Tr = real(float(T))
    R = T <: Real
    coords, spatial = FFS.Grids.point_coordinates(Tr, g, D)
    Ls = ntuple(d -> FFS.Grids.axis_range(Tr, g, d), D)
    epsv = eps === nothing ? _default_eps(Tr) : eps
    bt = NTuple{length(batch), Int}(batch)
    ntrans = prod(bt; init = 1)
    scaled, offsets, ranges, M = _scaled_points(Tr, coords, Ls)
    phase = FFS.Packing.offset_phase(Tr, ms, offsets, ranges, M, Val(false), iflag) .* M
    hs, σ = _plan_accuracy(epsv, ms)
    C = _chunk(ntrans, batch_chunk)
    plan = NonuniformFFTs.PlanNUFFT(Complex{Tr}, ms; ntransforms = Val(C), fftshift = false, m = hs, σ = σ)
    NonuniformFFTs.set_points!(plan, scaled)
    fks = ntuple(_ -> Array{Complex{Tr}, D}(undef, ms...), C)
    cjs = ntuple(_ -> Vector{Complex{Tr}}(undef, M), C)
    full = Array{Complex{Tr}}(undef, ms..., ntrans)
    return NUFFTSynthesisPlan{Tr, D, R, length(bt), typeof(plan), typeof(fks), typeof(cjs),
            typeof(phase), typeof(full), typeof(spatial)}(
        plan, fks, cjs, phase, full, ms, spatial, bt, ntrans, M, C, iflag < 0)
end

function FFS.Plans.synthesize!(out::AbstractArray, plan::NUFFTSynthesisPlan{T, D, R},
        coeffs::AbstractArray; ks = nothing) where {T, D, R}
    size(out) == FFS.Plans.field_size(plan) || throw(DimensionMismatch(
        "out is $(size(out)); this plan writes $(FFS.Plans.field_size(plan))"))
    sz = FFS.Packing.packed_size(plan.ms, Val(R))
    size(coeffs)[1:D] == sz || throw(DimensionMismatch(
        "this plan expects $(sz) on the spectral dims; got $(size(coeffs)[1:D])"))
    # A packed half expands into the held native cube; a full spectrum is already one.
    full = if R
        FFS.Packing.unpacked!(plan.full, reshape(coeffs, sz..., plan.ntrans), plan.ms, ks)
        plan.full
    else
        reshape(coeffs, plan.ms..., plan.ntrans)
    end
    _synth_type2!(out, full, plan.phase, plan.plan, plan.fks, plan.cjs, plan.ms, plan.M,
        plan.ntrans, plan.C, plan.neg)
    return out
end

function FFS._synthesize(::FFS.NonuniformFFTsBackend, exec::ComputationalBackends.AbstractExecutionBackend,
        g::Union{FlowGeometries.Grids.AbstractStructuredGrid{<:FlowGeometries.Geometry.AbstractCartesianGeometry},
                 FFS.Grids.PointwiseCartesian},
        coeffs::AbstractArray{Complex{Tr}}, ms::NTuple{D, Int}; real_output::Bool = true, iflag::Int = 1,
        ks = nothing, eps::Union{Nothing, Real} = nothing,
        batch_chunk::Int = DEFAULT_BATCH_CHUNK, kwargs...) where {Tr <: Real, D}
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
    plan = NonuniformFFTs.PlanNUFFT(Complex{Tr}, ms; ntransforms = Val(C), fftshift = false, m = hs, σ = σ)
    NonuniformFFTs.set_points!(plan, scaled)
    fks = ntuple(_ -> Array{Complex{Tr}, D}(undef, ms...), C)
    cjs = ntuple(_ -> Vector{Complex{Tr}}(undef, M), C)
    # Branch on the output element type here, so `_synth_type2!` specializes on a concrete array.
    return real_output ?
        _synth_type2!(Array{Tr}(undef, spatial..., batch...), full, p, plan, fks, cjs, ms, M, ntrans, C, iflag < 0) :
        _synth_type2!(Array{Complex{Tr}}(undef, spatial..., batch...), full, p, plan, fks, cjs, ms, M, ntrans, C, iflag < 0)
end

# Type-2 execution loop. `Z` fixes whether the destination takes the real part or the whole value, so the
# per-element branch folds away.
function _synth_type2!(out::AbstractArray{Z}, full, p, plan, fks::Tuple, cjs::Tuple,
        ms::NTuple{D, Int}, M::Int, ntrans::Int, C::Int, neg::Bool) where {Z, D}
    Pm = prod(ms)
    @inbounds for base in 0:C:(ntrans - 1)
        nvalid = min(C, ntrans - base)
        for c in 1:C
            fk = fks[c]
            if c > nvalid
                fill!(fk, zero(eltype(fk)))          # pad the final chunk to the plan's transform count
                continue
            end
            coff = (base + c - 1) * Pm
            for i in 1:Pm
                z = full[coff + i]
                fk[i] = (neg ? conj(z) : z) * conj(p[i])
            end
        end
        NonuniformFFTs.exec_type2!(cjs, plan, fks)
        for c in 1:nvalid
            cj = cjs[c]
            foff = (base + c - 1) * M
            for j in 1:M
                v = neg ? conj(cj[j]) : cj[j]
                out[foff + j] = Z <: Real ? real(v) : v
            end
        end
    end
    return out
end

# Points folded to [0, 2π) plus the per-axis offset and period, as `_scaled_phase_ks` computes them.
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

end # module FlowFieldSpectraNonuniformFFTsExt
