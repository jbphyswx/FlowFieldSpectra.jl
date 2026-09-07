module FlowFieldSpectraFINUFFTExt

using FINUFFT: FINUFFT
using FlowFieldSpectra: FlowFieldSpectra as FFS
using ComputationalBackends: ComputationalBackends
using SpectralBackends: SpectralBackends
using FlowGeometries: FlowGeometries

# =============================================================================
# NUFFT (FINUFFT type-1) on the packed-native model. A scattered (unstructured) Cartesian grid uses the
# guru plan directly; a nonuniform tensor-product (structured) grid uses the separable per-axis 1-D NUFFT
# (bottom of file). Plans pass `modeord = 1`, so the transform output is already in native (unshifted)
# order. The nonuniform points are fixed by the grid, so `finufft_makeplan` + `finufft_setpts!` run once;
# `finufft_exec!` runs per call with `ntrans = prod(batch)`. The field `(N, batch…)` reshapes to a
# contiguous `(N, ntrans)` strengths view.
#
# FINUFFT takes complex strengths and returns the full native spectrum for a real field too, so a real
# field is widened into the plan's `cj` buffer once per call (a bandwidth copy; FINUFFT has no
# counterpart to FFT's rfft real-input path). The published `coeffs` is still the packed half:
# `Packing.hermitian_request_size` asks for one extra axis-1 mode at even `N₁`, putting `±N₁/2` on that
# axis, and `Packing.publish_packed!` takes the half as the leading axis-1 entries. The Nyquist twins are
# conjugate reads of the same spectrum (`Packing.conj_twins` / `gather_conj_twins!`) — a real field's
# output is Hermitian, so the negated frequency each twin needs is native. Those helpers are shared with
# the NonuniformFFTs ext, which reaches the same native spectrum by a separable route.
# Steady-state execution allocates nothing. Physical wavenumbers / point scaling use the grid's
# periodic length.
# =============================================================================

# Default tolerance: above the float type's machine epsilon.
_default_eps(::Type{T}) where {T} = T === Float32 ? 1.0e-6 : 1.0e-8

# Fill the `(M, ntrans)` strengths from a `(N, batch…)` field, applying the grid's quadrature factor
# (`nothing` for a constant measure); see `FFS.Grids.quadrature_scale`. Column-major orders coincide, so
# the unweighted case is a single linear copy.
function _fill_strengths!(cj::AbstractMatrix, field, qw)
    qw === nothing && return copyto!(cj, field)
    M = size(cj, 1)
    @inbounds for t in axes(cj, 2), i in 1:M
        cj[i, t] = field[i + (t - 1) * M] * qw[i]
    end
    return cj
end

# A real field's plan carries the packed gather table + twins; a complex field's carries the native-order phase.
struct FINUFFTRealPlan{T, D, NB, G, CJ, FK, PH, KS, TW, QW} <: FFS.AbstractSpectralPlan
    batch::NTuple{NB, Int}
    guru::G                          # guru plan (C resource; self-finalizing, see _guru).
    cj::CJ                           # (M, ntrans) complex strengths buffer
    fk::FK                           # (ns…, ntrans) requested-size native spectrum buffer
    ms::NTuple{D, Int}
    ns::NTuple{D, Int}               # requested mode counts (axis 1 extended at even N₁)
    pms::NTuple{D, Int}              # packed output size
    ntrans::Int
    M::Int                           # number of nonuniform points
    neg::Bool                        # caller asked for iflag < 0; conjugate the published half
    phase::PH                        # (pms…) offset phase × 1/M
    ks_phys::KS                      # halved axis carries the twin whose slices `twins` refill
    twins::TW
    qw::QW                           # grid quadrature factor, or `nothing` for a constant measure
end

struct FINUFFTComplexPlan{T, D, NB, G, CJ, FK, PH, KS, QW} <: FFS.AbstractSpectralPlan
    batch::NTuple{NB, Int}
    guru::G
    cj::CJ                           # (M, ntrans) complex strengths buffer
    fk::FK                           # (ms…, ntrans) full native spectrum buffer
    ms::NTuple{D, Int}
    ntrans::Int
    M::Int
    phase::PH                        # (ms…, 1) translation-correction phase × (1/M)
    ks_phys::KS
    qw::QW                           # grid quadrature factor, or `nothing` for a constant measure
end

Base.show(io::IO, ::FINUFFTRealPlan{T, D}) where {T, D} = print(io, "FINUFFTRealPlan{$T, $D}(FINUFFT)")
Base.show(io::IO, ::FINUFFTComplexPlan{T, D}) where {T, D} = print(io, "FINUFFTComplexPlan{$T, $D}(FINUFFT)")

FFS.Plans.coefficient_size(p::FINUFFTRealPlan) = (p.pms..., p.batch...)
FFS.Plans.coefficient_type(::FINUFFTRealPlan{T}) where {T} = Complex{T}
FFS.Plans.wavenumbers(p::FINUFFTRealPlan) = p.ks_phys

FFS.Plans.coefficient_size(p::FINUFFTComplexPlan) = (p.ms..., p.batch...)
FFS.Plans.coefficient_type(::FINUFFTComplexPlan{T}) where {T} = Complex{T}
FFS.Plans.wavenumbers(p::FINUFFTComplexPlan) = p.ks_phys

# Point scaling shared by both paths: points folded to [0, 2π) plus the per-axis offset and period.
function _scaled_points(::Type{T}, coords::Tuple, Ls::NTuple{D}) where {T, D}
    M = length(coords[1])
    for d in 1:D
        length(coords[d]) == M || throw(DimensionMismatch("coordinate $d length mismatch"))
    end
    offsets = ntuple(d -> T(minimum(coords[d])), D)
    ranges = ntuple(d -> T(Ls[d]), D)    # already resolved by `Grids.axis_range`
    scaled = ntuple(d -> T(2π) .* (T.(coords[d]) .- offsets[d]) ./ ranges[d], D)
    return scaled, offsets, ranges, M
end

# Native-order guru type-1 plan over fixed points; `modeord = 1` puts the output in fftfreq order.
function _guru(::Type{T}, scaled::Tuple, ms::NTuple{D, Int}, ntrans::Int, iflag::Int, eps::Real,
        nthreads::Int) where {T, D}
    guru = FINUFFT.finufft_makeplan(1, collect(ms), -iflag, ntrans, T(eps);
        dtype = T, nthreads = nthreads, modeord = 1)
    if D == 1
        FINUFFT.finufft_setpts!(guru, scaled[1])
    elseif D == 2
        FINUFFT.finufft_setpts!(guru, scaled[1], scaled[2])
    elseif D == 3
        FINUFFT.finufft_setpts!(guru, scaled[1], scaled[2], scaled[3])
    else
        FINUFFT.finufft_destroy!(guru)
        throw(ArgumentError("the guru NUFFT supports up to 3 dimensions; got $D"))
    end
    finalizer(FINUFFT.finufft_destroy!, guru)   # free the C plan when the guru is GC'd
    return guru
end

# A real field's Hermitian output makes the `iflag = -1` result the conjugate of the `iflag = +1` one,
# so the tables are built once for `+1` and the published half is conjugated when asked.
function _nufft_real_plan(::Type{T}, coords::Tuple, ms::NTuple{D, Int}, Ls::NTuple{D},
        batch::Tuple, iflag::Int, eps::Real, nthreads::Int, qw = nothing) where {T, D}
    scaled, offsets, ranges, M = _scaled_points(T, coords, Ls)
    ntrans = prod(batch; init = 1)
    ns = FFS.Packing.hermitian_request_size(ms)
    guru = _guru(T, scaled, ns, ntrans, 1, eps, nthreads)
    pms = FFS.Packing.packed_size(ms, Val(true))
    phase = FFS.Packing.offset_phase(T, ms, offsets, ranges, M, Val(true))
    ks_phys = FFS.Grids.physical_wavenumbers(ranges, ms, Val(true))
    ks_twin, twins = FFS.Packing.conj_twins(T, ks_phys, ms, ns, offsets, ranges, M, batch)
    cj = Matrix{Complex{T}}(undef, M, ntrans)
    fk = Array{Complex{T}, D + 1}(undef, ns..., ntrans)
    bt = NTuple{length(batch), Int}(batch)
    return FINUFFTRealPlan{T, D, length(bt), typeof(guru), typeof(cj), typeof(fk), typeof(phase), typeof(ks_twin), typeof(twins), typeof(qw)}(
        bt, guru, cj, fk, ms, ns, pms, ntrans, M, iflag < 0, phase, ks_twin, twins, qw,
    )
end

function _nufft_complex_plan(::Type{T}, coords::Tuple, ms::NTuple{D, Int}, Ls::NTuple{D},
        batch::Tuple, iflag::Int, eps::Real, nthreads::Int, qw = nothing) where {T, D}
    scaled, offsets, ranges, M = _scaled_points(T, coords, Ls)
    ntrans = prod(batch; init = 1)
    guru = _guru(T, scaled, ms, ntrans, iflag, eps, nthreads)
    phase = FFS.Packing.offset_phase(T, ms, offsets, ranges, M, Val(false), iflag)
    ks_phys = FFS.Grids.physical_wavenumbers(ranges, ms, Val(false))
    cj = Matrix{Complex{T}}(undef, M, ntrans)
    fk = Array{Complex{T}, D + 1}(undef, ms..., ntrans)
    bt = NTuple{length(batch), Int}(batch)
    return FINUFFTComplexPlan{T, D, length(bt), typeof(guru), typeof(cj), typeof(fk), typeof(phase), typeof(ks_phys), typeof(qw)}(
        bt, guru, cj, fk, ms, ntrans, M, phase, ks_phys, qw,
    )
end

"""
    calculate_spectrum!(coeffs, plan::FINUFFTRealPlan, field) -> ks_phys

Execute a prebuilt guru plan in place for a real field. `field` is `(N, batch…)` (widened into the
plan's complex strengths buffer); `coeffs` is the packed half `(N₁÷2+1, N₂…, batch…)`. Plan and point
sorting reused across calls; zero heap allocation in steady state.
"""
function FFS.calculate_spectrum!(coeffs::AbstractArray{Complex{T}}, plan::FINUFFTRealPlan{T, D},
        field) where {T, D}
    _fill_strengths!(plan.cj, field, plan.qw)                 # widen real→complex into the reused buffer
    FINUFFT.finufft_exec!(plan.guru, plan.cj, plan.fk)
    FFS.Packing.publish_packed!(coeffs, plan.fk, plan.phase, plan.ns, plan.pms, plan.ntrans, plan.neg)
    FFS.Packing.gather_conj_twins!(plan.twins, plan.fk, prod(plan.ns), plan.ntrans, plan.neg)
    return plan.ks_phys
end

"""
    calculate_spectrum!(coeffs, plan::FINUFFTComplexPlan, field) -> ks_phys

Execute a prebuilt guru plan in place for a complex field; `coeffs` is the full native spectrum
`(ms…, batch…)`.
"""
function FFS.calculate_spectrum!(coeffs::AbstractArray{Complex{T}}, plan::FINUFFTComplexPlan{T, D},
        field) where {T, D}
    _fill_strengths!(plan.cj, field, plan.qw)
    FINUFFT.finufft_exec!(plan.guru, plan.cj, plan.fk)
    plan.fk .*= plan.phase                                    # phase (ms…, 1) broadcasts over ntrans, in place
    copyto!(coeffs, plan.fk)                                  # linear copy → any coeffs shape (no reshape)
    return plan.ks_phys
end

function FFS.plan_spectrum(::FFS.FINUFFTBackend, exec::ComputationalBackends.AbstractExecutionBackend,
        g::FFS.Grids.PointwiseCartesian, ::Type{T}, ms::NTuple{D, Int};
        batch::Tuple = (), iflag::Int = 1, eps::Real = _default_eps(real(float(T)))) where {T, D}
    Tr = real(float(T))
    coords, _ = FFS.Grids.point_coordinates(Tr, g, D)
    Ls = ntuple(d -> FFS.Grids.axis_range(Tr, g, d), D)
    nth = FFS._backend_nthreads(exec)
    qw = FFS.Grids.quadrature_scale(g, Tr, length(coords[1]))
    return T <: Real ? _nufft_real_plan(Tr, coords, ms, Ls, batch, iflag, eps, nth, qw) :
           _nufft_complex_plan(Tr, coords, ms, Ls, batch, iflag, eps, nth, qw)
end

# One-shot allocating entry — scattered (unstructured) Cartesian grid → guru NUFFT.
function FFS._calculate_spectrum_nufft(::FFS.FINUFFTBackend, exec::ComputationalBackends.AbstractExecutionBackend,
        g::FFS.Grids.PointwiseCartesian,
        field::AbstractArray, ms::Tuple; iflag::Int = 1, eps::Union{Nothing, Real} = nothing, kwargs...)
    D = length(ms)
    R = eltype(field) <: Real
    T = float(real(eltype(g)))
    coords, _ = FFS.Grids.point_coordinates(T, g, D)
    Ls = ntuple(d -> FFS.Grids.axis_range(T, g, d), D)
    batch = FFS.Grids.field_batch_shape(g, field)
    epsv = eps === nothing ? _default_eps(T) : eps
    nth = FFS._backend_nthreads(exec)
    msn = NTuple{D, Int}(ms)
    qw = FFS.Grids.quadrature_scale(g, T, length(coords[1]))
    plan = R ? _nufft_real_plan(T, coords, msn, Ls, batch, iflag, epsv, nth, qw) :
               _nufft_complex_plan(T, coords, msn, Ls, batch, iflag, epsv, nth, qw)
    coeffs = zeros(Complex{T}, FFS.Packing.packed_size(msn, Val(R))..., batch...)
    ks = FFS.calculate_spectrum!(coeffs, plan, field)
    return coeffs, ks
end

# =============================================================================
# Separable NUFFT for a nonuniform tensor-product (structured) Cartesian grid. The D-dim Fourier kernel
# factorizes over a tensor-product grid, so the transform is a sequence of 1-D type-1 NUFFTs — one per
# axis — with every OTHER spatial + batch dim carried as the FINUFFT `ntrans` batch. Each 1-D transform
# needs only its own length-N_d axis: NO ∏N_d coordinate materialization (the point of a
# gridded-but-nonuniform representation). Along axis `d` the working array's dim `d` shrinks from N_d to
# ms[d]. Because every transform is 1-D, this path supports any D (FINUFFT's D≤3 cap is per-call and
# never reached here). Each axis pass leaves the raw transform in native order; the grid-offset phase and
# the `1/∏N_d` normalization are applied once at the end, through the same `Packing.offset_phase` /
# `Packing.publish_packed!` / `Packing.conj_twins` the scattered path uses, so both paths publish an
# identical layout.
#
# An axis pass carries `batch_chunk` lines per execution and gathers them in place through
# `Packing.axis_layout`, so the strengths and spectrum buffers are sized by the chunk and only the result
# is a full array. Executing the whole axis at once is both the slowest and the largest setting.
# =============================================================================
function FFS._calculate_spectrum_nufft(::FFS.FINUFFTBackend, exec::ComputationalBackends.AbstractExecutionBackend,
        g::FlowGeometries.Grids.AbstractStructuredGrid{<:FlowGeometries.Geometry.AbstractCartesianGeometry},
        field::AbstractArray, ms::Tuple; iflag::Int = 1, eps::Union{Nothing, Real} = nothing,
        batch_chunk::Union{Nothing, Int} = nothing, kwargs...)
    D = ndims(g)
    T = float(real(eltype(g)))
    R = eltype(field) <: Real
    epsv = T(eps === nothing ? _default_eps(T) : eps)
    nth = FFS._backend_nthreads(exec)
    msn = NTuple{D, Int}(ms)
    axs = ntuple(d -> T.(FlowGeometries.Grids.coordinates(g, d)), D)
    offsets, ranges = FFS.Grids.axis_geometry(T, g, D)
    npts = length(g)
    batch = ntuple(i -> size(field, D + i), ndims(field) - D)
    sgn = R ? 1 : iflag                    # a real field's tables are built for +1, conjugated on publish
    ns = R ? FFS.Packing.hermitian_request_size(msn) : msn
    A = Array{Complex{T}}(undef, size(field)...)
    copyto!(A, field)                                          # widen real→complex working copy
    qw = FFS.Grids.quadrature_scale(g, T, npts)                # grid quadrature factor
    if qw !== nothing
        nb = length(A) ÷ npts
        @inbounds for b in 1:nb, j in 1:npts
            A[j + (b - 1) * npts] *= qw[j]
        end
    end
    bc = batch_chunk === nothing ? _default_batch_chunk(nth) : batch_chunk
    for d in 1:D
        A = _nufft_axis(A, d, axs[d], ns[d], ranges[d], offsets[d], sgn, epsv, nth, bc)
    end
    if R
        phase = FFS.Packing.offset_phase(T, msn, offsets, ranges, npts, Val(true))
        ks_phys = FFS.Grids.physical_wavenumbers(ranges, msn, Val(true))
        ks, twins = FFS.Packing.conj_twins(T, ks_phys, msn, ns, offsets, ranges, npts, batch)
        pms = FFS.Packing.packed_size(msn, Val(true))
        coeffs = Array{Complex{T}}(undef, pms..., batch...)
        nt = prod(batch; init = 1)
        FFS.Packing.publish_packed!(coeffs, A, phase, ns, pms, nt, iflag < 0)
        FFS.Packing.gather_conj_twins!(twins, A, prod(ns), nt, iflag < 0)
        return coeffs, ks
    end
    A .*= FFS.Packing.offset_phase(T, msn, offsets, ranges, npts, Val(false), iflag)   # broadcasts over the batch
    return A, FFS.Grids.physical_wavenumbers(ranges, msn, Val(false))
end

"""
    BATCH_CHUNK_FLOOR

Fewest lines per `finufft_exec!` the separable structured path will choose on its own, overridable as
`batch_chunk=` on `calculate_spectrum`. One execution transforms that many lines of the axis against one
set of sorted points; the strengths and spectrum buffers are sized by the chunk (`C·(N_d + m)` values,
independent of the thread count) while the result array stays full size. Transforming the whole axis in
one execution is both the largest and the slowest setting.
"""
const BATCH_CHUNK_FLOOR = 4

# Lines per execution when the caller does not choose. `ntrans` is a parallel axis for FINUFFT, so the
# chunk must give every thread at least one line; the buffers it sizes do not grow with the thread count,
# so covering the threads is close to free. `batch_chunk <= 0` transforms the whole axis at once.
@inline _default_batch_chunk(nthreads::Int) = max(BATCH_CHUNK_FLOOR, nthreads)

@inline _chunk(B::Int, batch_chunk::Int) = batch_chunk <= 0 ? max(B, 1) : clamp(batch_chunk, 1, max(B, 1))

# One 1-D type-1 NUFFT along dim `d` (points = `axis`, length N_d), modes = `m` in native order; every
# other dim of `A` rides as an `ntrans` column, `batch_chunk` lines per execution against one reused
# guru plan. The lines are gathered in place through `Packing.axis_layout`, so the only array this
# allocates at the size of `A` is the result. Raw output: the offset phase and normalization are applied
# once by the caller.
function _nufft_axis(A::AbstractArray{Complex{T}}, d::Int, axis::AbstractVector{T}, m::Int,
        rng::T, off::T, iflag::Int, eps::T, nthreads::Int, batch_chunk::Int) where {T}
    sz = size(A)
    Nd = length(axis)
    sz[d] == Nd ||
        throw(DimensionMismatch("axis $d: field length $(sz[d]) ≠ grid axis length $Nd"))
    pre, _, post = FFS.Packing.axis_layout(sz, d)
    C = _chunk(pre * post, batch_chunk)
    guru = _axis_guru(T, axis, m, rng, off, iflag, eps, nthreads, C)
    out = Array{Complex{T}}(undef, FFS.Packing.axis_out_size(sz, d, m))
    _axis_pass!(out, A, d, guru, Matrix{Complex{T}}(undef, Nd, C), Matrix{Complex{T}}(undef, m, C),
        Vector{Int}(undef, C), Vector{Int}(undef, C))
    FINUFFT.finufft_destroy!(guru)
    return out
end

# 1-D type-1 guru plan over one grid axis, `C` lines per execution, in native mode order.
function _axis_guru(::Type{T}, axis::AbstractVector{T}, m::Int, rng::T, off::T, iflag::Int, eps::T,
        nthreads::Int, C::Int) where {T}
    x = T(2π) .* (axis .- off) ./ rng                          # scale points into FINUFFT's period
    guru = FINUFFT.finufft_makeplan(1, [m], -iflag, C, eps;
        dtype = T, nthreads = nthreads, modeord = 1)
    FINUFFT.finufft_setpts!(guru, x)
    return guru
end

# Transform axis `d` of `A` into `out`, whose dim `d` holds `size(out, d)` modes. Lines are gathered
# straight out of `A` and scattered straight into `out`, so neither is permuted; the chunk width is
# `size(cj, 2)`. Shared by the one-shot and the reusable plan.
function _axis_pass!(out::AbstractArray{Complex{T}}, A::AbstractArray{Complex{T}}, d::Int, guru,
        cj::AbstractMatrix{Complex{T}}, fk::AbstractMatrix{Complex{T}},
        inoff::Vector{Int}, outoff::Vector{Int}) where {T}
    sz = size(A)
    pre, Nd, post = FFS.Packing.axis_layout(sz, d)
    m = size(out, d)
    C = size(cj, 2)
    rest = pre * post
    for base in 0:C:(rest - 1)
        nvalid = min(C, rest - base)
        FFS.Packing.axis_chunk_offsets!(inoff, outoff, base, nvalid, pre, Nd, m)
        @inbounds for i in 1:Nd
            s = (i - 1) * pre
            for t in 1:nvalid
                cj[i, t] = A[inoff[t] + s]
            end
            for t in (nvalid + 1):C
                cj[i, t] = zero(Complex{T})    # pad the final chunk to the plan's transform count
            end
        end
        FINUFFT.finufft_exec!(guru, cj, fk)
        @inbounds for j in 1:m
            s = (j - 1) * pre
            for t in 1:nvalid
                out[outoff[t] + s] = fk[j, t]
            end
        end
    end
    return out
end

# Hybrid composite: one stretched axis of a working array whose other axes are already transformed. The
# `+1` sign matches the FFT pass's convention; the composite conjugates once for `iflag = -1`.
function FFS._axis_nufft(::FFS.FINUFFTBackend, exec::ComputationalBackends.AbstractExecutionBackend,
        A::AbstractArray{Complex{T}}, d::Int, axis::AbstractVector{T}, m::Int, rng::T, off::T,
        eps::Real; batch_chunk::Union{Nothing, Int} = nothing, kwargs...) where {T}
    nth = FFS._backend_nthreads(exec)
    bc = batch_chunk === nothing ? _default_batch_chunk(nth) : batch_chunk
    return _nufft_axis(A, d, axis, m, rng, off, 1, T(eps), nth, bc)
end

"""
    AxisGuru{G,M,IX}

One axis's reusable 1-D type-1 guru plan with the strengths / spectrum / offset buffers `_axis_pass!`
reads. The hybrid PLAN holds one per stretched axis, so a repeated call rebuilds no guru and re-sorts no
points. The guru is a C resource, so it carries a finalizer and a one-line `show` (default printing of a
struct holding a FINUFFT plan can segfault).
"""
struct AxisGuru{G, M, IX}
    guru::G
    cj::M
    fk::M
    inoff::IX
    outoff::IX
end

Base.show(io::IO, a::AxisGuru) = print(io, "AxisGuru(N=", size(a.cj, 1), " → m=", size(a.fk, 1),
    ", chunk=", size(a.cj, 2), ")")

function FFS._axis_nufft_plan(::FFS.FINUFFTBackend, exec::ComputationalBackends.AbstractExecutionBackend,
        ::Type{T}, insize::Tuple, d::Int, axis::AbstractVector{T}, m::Int, rng::T, off::T,
        eps::Real; batch_chunk::Union{Nothing, Int} = nothing, kwargs...) where {T}
    Nd = length(axis)
    insize[d] == Nd || throw(DimensionMismatch(
        "axis $d: working length $(insize[d]) ≠ grid axis length $Nd"))
    nth = FFS._backend_nthreads(exec)
    bc = batch_chunk === nothing ? _default_batch_chunk(nth) : batch_chunk
    pre, _, post = FFS.Packing.axis_layout(insize, d)
    C = _chunk(pre * post, bc)
    guru = _axis_guru(T, axis, m, rng, off, 1, T(eps), nth, C)
    finalizer(FINUFFT.finufft_destroy!, guru)
    return AxisGuru{typeof(guru), Matrix{Complex{T}}, Vector{Int}}(
        guru, Matrix{Complex{T}}(undef, Nd, C), Matrix{Complex{T}}(undef, m, C),
        Vector{Int}(undef, C), Vector{Int}(undef, C))
end

FFS._axis_nufft_exec!(out::AbstractArray, a::AxisGuru, A::AbstractArray, d::Int) =
    _axis_pass!(out, A, d, a.guru, a.cj, a.fk, a.inoff, a.outoff)

# ---- reusable separable plan: the per-axis guru plans and working buffers built once ----

# `R` marks a real field, so the publish branch folds. Every working array shares rank and element type,
# and so does every per-axis buffer, so indexing them at a runtime axis stays type-stable.
struct FINUFFTSeparablePlan{T, D, R, NB, G, CJ, FK, W, IX, PH, KS, TW, QW} <: FFS.AbstractSpectralPlan
    batch::NTuple{NB, Int}
    gurus::G                         # D × 1-D type-1 guru plan (C resource; self-finalizing)
    cjs::CJ                          # D × (N_d, C_d) strengths
    fks::FK                          # D × (ns_d, C_d) spectra
    work::W                          # D+1 × working arrays; `work[1]` takes the widened field
    inoff::IX
    outoff::IX
    ms::NTuple{D, Int}
    ns::NTuple{D, Int}               # requested mode counts (axis 1 extended at even N₁)
    pms::NTuple{D, Int}              # packed output size
    ntrans::Int
    npts::Int
    neg::Bool                        # caller asked for iflag < 0; conjugate the published half
    phase::PH
    ks_phys::KS
    twins::TW
    qw::QW                           # grid quadrature factor, or `nothing` for a constant measure
end

Base.show(io::IO, ::FINUFFTSeparablePlan{T, D, R}) where {T, D, R} =
    print(io, "FINUFFTSeparablePlan{$T, $D}(FINUFFT, ", R ? "real" : "complex", ")")

FFS.Plans.coefficient_size(p::FINUFFTSeparablePlan) = (p.pms..., p.batch...)
FFS.Plans.coefficient_type(::FINUFFTSeparablePlan{T}) where {T} = Complex{T}
FFS.Plans.wavenumbers(p::FINUFFTSeparablePlan) = p.ks_phys

function FFS.plan_spectrum(::FFS.FINUFFTBackend, exec::ComputationalBackends.AbstractExecutionBackend,
        g::FlowGeometries.Grids.AbstractStructuredGrid{<:FlowGeometries.Geometry.AbstractCartesianGeometry},
        ::Type{T}, ms::NTuple{D, Int}; batch::Tuple = (), iflag::Int = 1,
        eps::Real = _default_eps(real(float(T))),
        batch_chunk::Union{Nothing, Int} = nothing) where {T, D}
    Tr = real(float(T))
    R = T <: Real
    ndims(g) == D || throw(DimensionMismatch("grid has $(ndims(g)) dims; asked for $D mode counts"))
    axs = ntuple(d -> Tr.(FlowGeometries.Grids.coordinates(g, d)), D)
    offsets, ranges = FFS.Grids.axis_geometry(Tr, g, D)
    Ns = ntuple(d -> length(axs[d]), D)
    npts = prod(Ns)
    ntrans = prod(batch; init = 1)
    nth = FFS._backend_nthreads(exec)
    bc = batch_chunk === nothing ? _default_batch_chunk(nth) : batch_chunk
    sgn = R ? 1 : iflag                # a real field's tables are built for +1, conjugated on publish
    ns = R ? FFS.Packing.hermitian_request_size(ms) : ms
    epsv = Tr(eps)

    work = ntuple(d -> Array{Complex{Tr}}(undef, FFS.Packing.axis_work_shape(Ns, ns, d, batch)...), D + 1)
    Cs = ntuple(D) do d
        pre, _, post = FFS.Packing.axis_layout(size(work[d]), d)
        _chunk(pre * post, bc)
    end
    gurus = ntuple(D) do d
        guru = _axis_guru(Tr, axs[d], ns[d], ranges[d], offsets[d], sgn, epsv, nth, Cs[d])
        finalizer(FINUFFT.finufft_destroy!, guru)      # free the C plan when the plan is GC'd
    end
    cjs = ntuple(d -> Matrix{Complex{Tr}}(undef, Ns[d], Cs[d]), D)
    fks = ntuple(d -> Matrix{Complex{Tr}}(undef, ns[d], Cs[d]), D)
    Cmax = maximum(Cs)

    pms = FFS.Packing.packed_size(ms, Val(R))
    phase = FFS.Packing.offset_phase(Tr, ms, offsets, ranges, npts, Val(R), R ? 1 : iflag)
    ks_phys = FFS.Grids.physical_wavenumbers(ranges, ms, Val(R))
    ks_out, twins = R ?
        FFS.Packing.conj_twins(Tr, ks_phys, ms, ns, offsets, ranges, npts, batch) :
        (ks_phys, ())
    qw = FFS.Grids.quadrature_scale(g, Tr, npts)
    inoff = Vector{Int}(undef, Cmax)
    outoff = Vector{Int}(undef, Cmax)
    bt = NTuple{length(batch), Int}(batch)
    return FINUFFTSeparablePlan{Tr, D, R, length(bt), typeof(gurus), typeof(cjs), typeof(fks), typeof(work),
            typeof(inoff), typeof(phase), typeof(ks_out), typeof(twins), typeof(qw)}(
        bt, gurus, cjs, fks, work, inoff, outoff,
        ms, ns, pms, ntrans, npts, iflag < 0, phase, ks_out, twins, qw,
    )
end

"""
    calculate_spectrum!(coeffs, plan::FINUFFTSeparablePlan, field) -> ks_phys

Execute a prebuilt separable structured plan in place. `field` is `(N₁…N_D, batch…)`; `coeffs` is the
packed half `(N₁÷2+1, N₂…, batch…)` for a real field and the full native spectrum for a complex one. The
per-axis guru plans, point sortings and working buffers are reused across calls.
"""
function FFS.calculate_spectrum!(coeffs::AbstractArray{Complex{T}},
        plan::FINUFFTSeparablePlan{T, D, R}, field) where {T, D, R}
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
    for d in 1:D
        _axis_pass!(plan.work[d + 1], plan.work[d], d, plan.gurus[d], plan.cjs[d], plan.fks[d],
            plan.inoff, plan.outoff)
    end
    F = plan.work[D + 1]
    if R
        FFS.Packing.publish_packed!(coeffs, F, plan.phase, plan.ns, plan.pms, plan.ntrans, plan.neg)
        FFS.Packing.gather_conj_twins!(plan.twins, F, prod(plan.ns), plan.ntrans, plan.neg)
    else
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
# Inverse (synthesis) via a type-2 guru plan. FINUFFT has no real-input transform, so a real field's
# packed half is first completed to the full native spectrum by `Packing.unpacked` WITH the halved axis's
# Nyquist twin — the `k₁ < 0` rows need `+N_d/2` wherever an even axis `d ≥ 2` sits at `−N_d/2`, and
# index negation alone lands off the native axis there. The forward publishes `C = fk · p / M` with
# `|p| = 1` the grid-offset phase, and a type-2 wants `û = C / p`; its `iflag` is the forward's, sign
# flipped, so `Σ û e^{+iflag·i·k·x}` comes back.
# =============================================================================

function _guru2(::Type{T}, scaled::Tuple, ms::NTuple{D, Int}, ntrans::Int, iflag::Int, eps::Real,
        nthreads::Int) where {T, D}
    guru = FINUFFT.finufft_makeplan(2, collect(ms), iflag, ntrans, T(eps);
        dtype = T, nthreads = nthreads, modeord = 1)
    if D == 1
        FINUFFT.finufft_setpts!(guru, scaled[1])
    elseif D == 2
        FINUFFT.finufft_setpts!(guru, scaled[1], scaled[2])
    elseif D == 3
        FINUFFT.finufft_setpts!(guru, scaled[1], scaled[2], scaled[3])
    else
        FINUFFT.finufft_destroy!(guru)
        throw(ArgumentError("the guru NUFFT supports up to 3 dimensions; got $D"))
    end
    finalizer(FINUFFT.finufft_destroy!, guru)
    return guru
end

function FFS._synthesize(::FFS.FINUFFTBackend, exec::ComputationalBackends.AbstractExecutionBackend,
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
    fk = Array{Complex{T}}(undef, ms..., ntrans)
    Pm = prod(ms)
    @inbounds for t in 1:ntrans, i in 1:Pm
        fk[i + (t - 1) * Pm] = full[i + (t - 1) * Pm] * conj(p[i])
    end
    cj = Matrix{Complex{T}}(undef, M, ntrans)
    FINUFFT.finufft_exec!(_guru2(T, scaled, ms, ntrans, iflag, epsv, FFS._backend_nthreads(exec)), fk, cj)
    if real_output
        out = Array{T}(undef, spatial..., batch...)
        @inbounds for i in eachindex(cj)
            out[i] = real(cj[i])
        end
        return out
    end
    return reshape(copy(cj), spatial..., batch...)
end

# =============================================================================
# Reusable synthesis: the type-2 guru plan with its points preset, the phase, and every buffer the
# execution walks — the native cube a packed half expands into, the phased spectrum the guru reads, and
# the strengths it writes.
# =============================================================================

struct FINUFFTSynthesisPlan{T, D, R, NB, G, FK, CJ, PH, FB, SP} <: FFS.AbstractSynthesisPlan
    guru::G
    fk::FK
    cj::CJ
    phase::PH
    full::FB
    ms::NTuple{D, Int}
    spatial::SP
    batch::NTuple{NB, Int}
    ntrans::Int
    M::Int
end

# A default show of a struct holding FINUFFT plans can segfault.
Base.show(io::IO, ::FINUFFTSynthesisPlan{T, D, R}) where {T, D, R} =
    print(io, "FINUFFTSynthesisPlan{$T, $D}(FINUFFT, ", R ? "real" : "complex", ")")

FFS.Plans.field_size(p::FINUFFTSynthesisPlan) = (p.spatial..., p.batch...)
FFS.Plans.field_type(::FINUFFTSynthesisPlan{T, D, R}) where {T, D, R} = R ? T : Complex{T}

function FFS.Plans.plan_synthesis(::FFS.FINUFFTBackend, exec::ComputationalBackends.AbstractExecutionBackend,
        g::Union{FlowGeometries.Grids.AbstractStructuredGrid{<:FlowGeometries.Geometry.AbstractCartesianGeometry},
                 FFS.Grids.PointwiseCartesian},
        ::Type{TT}, ms::NTuple{D, Int}; batch::Tuple = (), iflag::Int = 1,
        eps::Union{Nothing, Real} = nothing, kwargs...) where {TT, D}
    T = real(float(TT))
    R = TT <: Real
    coords, spatial = FFS.Grids.point_coordinates(T, g, D)
    Ls = ntuple(d -> FFS.Grids.axis_range(T, g, d), D)
    scaled, offsets, ranges, M = _scaled_points(T, coords, Ls)
    epsv = eps === nothing ? _default_eps(T) : eps
    bt = NTuple{length(batch), Int}(batch)
    ntrans = prod(bt; init = 1)
    guru = _guru2(T, scaled, ms, ntrans, iflag, epsv, FFS._backend_nthreads(exec))
    phase = FFS.Packing.offset_phase(T, ms, offsets, ranges, M, Val(false), iflag) .* M
    return FINUFFTSynthesisPlan{T, D, R, length(bt), typeof(guru), Array{Complex{T}, D + 1},
            Matrix{Complex{T}}, typeof(phase), Array{Complex{T}, D + 1}, typeof(spatial)}(
        guru, Array{Complex{T}}(undef, ms..., ntrans), Matrix{Complex{T}}(undef, M, ntrans),
        phase, Array{Complex{T}}(undef, ms..., ntrans), ms, spatial, bt, ntrans, M)
end

function FFS.Plans.synthesize!(out::AbstractArray, plan::FINUFFTSynthesisPlan{T, D, R},
        coeffs::AbstractArray; ks = nothing) where {T, D, R}
    size(out) == FFS.Plans.field_size(plan) || throw(DimensionMismatch(
        "out is $(size(out)); this plan writes $(FFS.Plans.field_size(plan))"))
    sz = FFS.Packing.packed_size(plan.ms, Val(R))
    size(coeffs)[1:D] == sz || throw(DimensionMismatch(
        "this plan expects $(sz) on the spectral dims; got $(size(coeffs)[1:D])"))
    full = if R
        FFS.Packing.unpacked!(plan.full, reshape(coeffs, sz..., plan.ntrans), plan.ms, ks)
        plan.full
    else
        reshape(coeffs, plan.ms..., plan.ntrans)
    end
    Pm = prod(plan.ms)
    @inbounds for t in 1:plan.ntrans, i in 1:Pm
        plan.fk[i + (t - 1) * Pm] = full[i + (t - 1) * Pm] * conj(plan.phase[i])
    end
    FINUFFT.finufft_exec!(plan.guru, plan.fk, plan.cj)
    @inbounds for i in eachindex(plan.cj)
        out[i] = R ? real(plan.cj[i]) : plan.cj[i]
    end
    return out
end

end # module FlowFieldSpectraFINUFFTExt
