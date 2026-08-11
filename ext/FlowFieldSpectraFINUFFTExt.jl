module FlowFieldSpectraFINUFFTExt

using FINUFFT: FINUFFT
using FlowFieldSpectra: FlowFieldSpectra as FFS
using ComputationalBackends: ComputationalBackends
using SpectralBackends: SpectralBackends
using FlowGeometries: FlowGeometries

# =============================================================================
# NUFFT (FINUFFT type-1) on the tensor-native model. A scattered (unstructured) Cartesian grid uses the
# guru plan directly; a nonuniform tensor-product (structured) grid uses the separable per-axis 1-D
# NUFFT (bottom of file). The nonuniform points are FIXED by the grid, so `finufft_makeplan` +
# `finufft_setpts!` run ONCE; `finufft_exec!` runs per call with `ntrans = prod(batch)`. The field
# `(N, batch…)` reshapes to a contiguous `(N, ntrans)` strengths view. FINUFFT requires COMPLEX
# strengths, so a real field is widened into the plan's `cj` buffer once per call (a bandwidth copy,
# not a heap allocation — there is no real-input NUFFT fast path, unlike FFT's rfft). Steady-state
# execution allocates nothing. Physical wavenumbers / point scaling use the grid's periodic length.
# =============================================================================

_exec_nthreads(::ComputationalBackends.AbstractThreadedBackend) = Threads.nthreads()
_exec_nthreads(::ComputationalBackends.AbstractExecutionBackend) = 1

# Default FINUFFT tolerance: above the float type's machine epsilon.
_default_eps(::Type{T}) where {T} = T === Float32 ? 1.0e-6 : 1.0e-8

# Immutable: the C `guru` resource is freed by a finalizer attached to the guru handle itself
# (FINUFFT's `finufft_plan` is a mutable object) rather than to this wrapper, so the wrapper needs no
# mutability. `cj`/`fk` are reused buffers whose contents are mutated in place — never reassigned.
struct NUFFTCartesianPlan{T, D, NM, G, CJ, FK, PH, KS} <: FFS.AbstractSpectralPlan
    guru::G                          # FINUFFT guru plan (C resource; self-finalizing, see _nufft_plan).
    cj::CJ                           # (M, ntrans) complex strengths buffer
    fk::FK                           # (ms…, ntrans) complex modes buffer
    ms::NTuple{D, Int}
    ntrans::Int
    M::Int                           # number of nonuniform points
    phase::PH                        # (ms…, 1) translation-correction phase × (1/M)
    ks_phys::KS
end

function _nufft_plan(::Type{T}, coords::Tuple, ms::NTuple{D, Int}, Ls::NTuple{D},
        ntrans::Int, iflag::Int, eps::Real, nthreads::Int) where {T, D}
    M = length(coords[1])
    for d in 1:D
        length(coords[d]) == M || throw(DimensionMismatch("coordinate $d length mismatch"))
    end
    offsets = ntuple(d -> T(minimum(coords[d])), D)
    ranges = ntuple(d -> (r = T(Ls[d]); r == 0 ? one(T) : r), D)
    scaled = ntuple(d -> T(2π) .* (T.(coords[d]) .- offsets[d]) ./ ranges[d], D)

    guru = FINUFFT.finufft_makeplan(1, collect(ms), -iflag, ntrans, T(eps); dtype = T, nthreads = nthreads)
    if D == 1
        FINUFFT.finufft_setpts!(guru, scaled[1])
    elseif D == 2
        FINUFFT.finufft_setpts!(guru, scaled[1], scaled[2])
    elseif D == 3
        FINUFFT.finufft_setpts!(guru, scaled[1], scaled[2], scaled[3])
    else
        FINUFFT.finufft_destroy!(guru)
        throw(ArgumentError("FINUFFT supports up to 3 dimensions; got $D"))
    end
    finalizer(FINUFFT.finufft_destroy!, guru)   # free the C plan when the guru (held by the returned plan) is GC'd

    # Centered integer modes (modeord=0) → per-axis translation-correction phase, folding in 1/M.
    k_ints = ntuple(d -> collect(-(ms[d] ÷ 2):((ms[d] - 1) ÷ 2)), D)
    inv_M = one(T) / M
    phase = Array{Complex{T}, D + 1}(undef, ms..., 1)
    @inbounds for I in CartesianIndices(ms)
        p = one(Complex{T})
        for d in 1:D
            p *= cis(-iflag * k_ints[d][I[d]] * (offsets[d] * T(2π) / ranges[d]))
        end
        phase[I, 1] = p * inv_M
    end

    ks_phys = FFS.Grids.physical_wavenumbers(ranges, ms, T)
    cj = Matrix{Complex{T}}(undef, M, ntrans)
    fk = Array{Complex{T}, D + 1}(undef, ms..., ntrans)
    plan = NUFFTCartesianPlan{T, D, D + 1, typeof(guru), typeof(cj), typeof(fk), typeof(phase), typeof(ks_phys)}(
        guru, cj, fk, ms, ntrans, M, phase, ks_phys,
    )
    return plan
end

"""
    calculate_spectrum!(coeffs, plan::NUFFTCartesianPlan, field) -> ks_phys

Execute a prebuilt FINUFFT guru plan in place. `field` is `(N, batch…)` (widened into the plan's
complex strengths buffer); `coeffs` is `(ms…, batch…)`. Plan and point sorting reused across calls;
zero heap allocation in steady state.
"""
function FFS.calculate_spectrum!(coeffs::AbstractArray{Complex{T}}, plan::NUFFTCartesianPlan{T, D},
        field) where {T, D}
    # `copyto!` copies by linear index, so the (N, batch…) field needs no reshape to fill the
    # (M, ntrans) strengths buffer (M = N, ntrans = ∏batch, column-major orders coincide).
    copyto!(plan.cj, field)                                   # widen real→complex into the reused buffer
    FINUFFT.finufft_exec!(plan.guru, plan.cj, plan.fk)
    plan.fk .*= plan.phase                                    # phase (ms…, 1) broadcasts over ntrans, in place
    copyto!(coeffs, plan.fk)                                  # linear copy → any coeffs shape (no reshape)
    return plan.ks_phys
end

function FFS.plan_spectrum(::FFS.FINUFFTBackend, exec::ComputationalBackends.AbstractExecutionBackend,
        g::FlowGeometries.Grids.AbstractUnstructuredGrid{<:FlowGeometries.Geometry.AbstractCartesianGeometry}, ::Type{T}, ms::NTuple{D, Int};
        batch::Tuple = (), iflag::Int = 1, eps::Real = _default_eps(T)) where {T, D}
    coords = FlowGeometries.Grids.coordinates(g)
    Ls = ntuple(d -> T(FlowGeometries.Grids.period(g, d)), D)
    return _nufft_plan(T, coords, ms, Ls, prod(batch; init = 1), iflag, eps, _exec_nthreads(exec))
end

# One-shot allocating entry — scattered (unstructured) Cartesian grid → guru NUFFT.
function FFS._calculate_spectrum_nufft(::FFS.FINUFFTBackend, exec::ComputationalBackends.AbstractExecutionBackend,
        g::FlowGeometries.Grids.AbstractUnstructuredGrid{<:FlowGeometries.Geometry.AbstractCartesianGeometry},
        field::AbstractArray, ms::Tuple; iflag::Int = 1, eps::Union{Nothing, Real} = nothing, kwargs...)
    D = length(ms)
    coords = FlowGeometries.Grids.coordinates(g)
    T = float(real(eltype(g)))
    Ls = ntuple(d -> T(FlowGeometries.Grids.period(g, d)), D)
    batch = ntuple(i -> size(field, 1 + i), ndims(field) - 1)
    ntrans = prod(batch; init = 1)
    epsv = eps === nothing ? _default_eps(T) : eps
    plan = _nufft_plan(T, coords, NTuple{D, Int}(ms), Ls, ntrans, iflag, epsv, _exec_nthreads(exec))
    coeffs = zeros(Complex{T}, ms..., batch...)
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
# never reached here). Divide by ∏N_d once at the end — matches the DirectSum / scattered-NUFFT result.
# =============================================================================
function FFS._calculate_spectrum_nufft(::FFS.FINUFFTBackend, exec::ComputationalBackends.AbstractExecutionBackend,
        g::FlowGeometries.Grids.AbstractStructuredGrid{<:FlowGeometries.Geometry.AbstractCartesianGeometry},
        field::AbstractArray, ms::Tuple; iflag::Int = 1, eps::Union{Nothing, Real} = nothing, kwargs...)
    D = ndims(g)
    T = float(real(eltype(field)))
    epsv = T(eps === nothing ? _default_eps(T) : eps)
    nth = _exec_nthreads(exec)
    npts = length(g)
    A = Array{Complex{T}}(undef, size(field)...)
    copyto!(A, field)                                          # widen real→complex working copy
    for d in 1:D
        A = _nufft_axis(A, d, T.(FlowGeometries.Grids.coordinates(g, d)), Int(ms[d]), T(FlowGeometries.Grids.period(g, d)), iflag, epsv, nth)
    end
    A ./= npts
    ks = FFS.Grids.physical_wavenumbers(g, NTuple{D, Int}(ms))
    return A, ks
end

# One 1-D type-1 NUFFT along dim `d` (points = `axis`, length N_d), modes = `m`; every other dim of `A`
# is carried as an `ntrans` column. Returns a new array whose dim `d` now has length `m`.
function _nufft_axis(A::AbstractArray{Complex{T}}, d::Int, axis::AbstractVector{T}, m::Int,
        L::T, iflag::Int, eps::T, nthreads::Int) where {T}
    nd = ndims(A)
    Nd = length(axis)
    size(A, d) == Nd ||
        throw(DimensionMismatch("axis $d: field length $(size(A, d)) ≠ grid axis length $Nd"))
    perm = (d, ntuple(i -> i < d ? i : i + 1, nd - 1)...)      # move transform axis to front
    Ap = permutedims(A, perm)                                  # (Nd, rest…), dense
    restshape = size(Ap)[2:end]
    ntrans = prod(restshape; init = 1)
    cj = reshape(Ap, Nd, ntrans)                               # (Nd, ntrans) contiguous strengths
    off = T(minimum(axis))
    rng = L == 0 ? one(T) : L
    x = T(2π) .* (axis .- off) ./ rng                          # scale points into FINUFFT's period
    guru = FINUFFT.finufft_makeplan(1, [m], -iflag, ntrans, eps; dtype = T, nthreads = nthreads)
    FINUFFT.finufft_setpts!(guru, x)
    fk = Array{Complex{T}}(undef, m, ntrans)
    FINUFFT.finufft_exec!(guru, cj, fk)
    FINUFFT.finufft_destroy!(guru)
    ophase = T(2π) * off / rng                                 # per-mode translation-correction phase
    m_ints = -(m ÷ 2):((m - 1) ÷ 2)
    @inbounds for t in 1:ntrans, (mi, mm) in enumerate(m_ints)
        fk[mi, t] *= cis(-iflag * mm * ophase)
    end
    Fr = reshape(fk, m, restshape...)                          # (m, rest…)
    return permutedims(Fr, invperm(collect(perm)))             # restore dim order; dim d now length m
end

end # module FlowFieldSpectraFINUFFTExt
