module FlowFieldSpectraFINUFFTExt

using FINUFFT: FINUFFT
using FlowFieldSpectra: FlowFieldSpectra as FFS

# =============================================================================
# NUFFT (FINUFFT type-1) for scattered Cartesian grids on the tensor-native model. The nonuniform
# points are FIXED by the grid, so `finufft_makeplan` + `finufft_setpts!` run ONCE; `finufft_exec!`
# runs per call with `ntrans = prod(batch)`. The field `(N, batch…)` reshapes to a contiguous
# `(N, ntrans)` strengths view. FINUFFT requires COMPLEX strengths, so a real field is widened into the
# plan's `cj` buffer once per call (a bandwidth copy, not a heap allocation — there is no real-input
# NUFFT fast path, unlike FFT's rfft). Steady-state execution allocates nothing.
# =============================================================================

_exec_nthreads(::FFS.ThreadedBackend) = Threads.nthreads()
_exec_nthreads(::FFS.AbstractExecutionBackend) = 1

# Default FINUFFT tolerance: above the float type's machine epsilon.
_default_eps(::Type{T}) where {T} = T === Float32 ? 1.0e-6 : 1.0e-8

mutable struct NUFFTCartesianPlan{T, D, NM, CJ, FK, PH, KS} <: FFS.AbstractSpectralPlan
    guru::Any                        # FINUFFT guru plan (C resource)
    cj::CJ                           # (M, ntrans) complex strengths buffer
    fk::FK                           # (ms…, ntrans) complex modes buffer
    ms::NTuple{D, Int}
    ntrans::Int
    M::Int                           # number of nonuniform points
    phase::PH                        # (ms…, 1) translation-correction phase × (1/M)
    ks_phys::KS
end

function _nufft_plan(::Type{T}, coords::Tuple, ms::NTuple{D, Int}, domain_size::NTuple{D},
        ntrans::Int, iflag::Int, eps::Real, nthreads::Int) where {T, D}
    M = length(coords[1])
    for d in 1:D
        length(coords[d]) == M || throw(DimensionMismatch("coordinate $d length mismatch"))
    end
    offsets = ntuple(d -> T(minimum(coords[d])), D)
    ranges = ntuple(d -> (r = T(domain_size[d]); r == 0 ? one(T) : r), D)
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
    plan = NUFFTCartesianPlan{T, D, D + 1, typeof(cj), typeof(fk), typeof(phase), typeof(ks_phys)}(
        guru, cj, fk, ms, ntrans, M, phase, ks_phys,
    )
    finalizer(p -> FINUFFT.finufft_destroy!(p.guru), plan)
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

function FFS.plan_spectrum(::FFS.NUFFTBackend, exec::FFS.AbstractExecutionBackend,
        g::FFS.ScatteredCartesianGrid, ::Type{T}, ms::NTuple{D, Int};
        batch::Tuple = (), iflag::Int = 1, eps::Real = _default_eps(T)) where {T, D}
    return _nufft_plan(T, g.coords, ms, g.domain_size, prod(batch; init = 1), iflag, eps, _exec_nthreads(exec))
end

# One-shot allocating entry (routed from the (transform, execution, grid) dispatch).
function FFS._calculate_spectrum_nufft(exec::FFS.AbstractExecutionBackend, g::FFS.ScatteredCartesianGrid,
        field::AbstractArray, ms::Tuple; iflag::Int = 1, eps::Union{Nothing, Real} = nothing, kwargs...)
    D = length(ms)
    T = float(real(eltype(g.coords[1])))
    batch = ntuple(i -> size(field, 1 + i), ndims(field) - 1)
    ntrans = prod(batch; init = 1)
    epsv = eps === nothing ? _default_eps(T) : eps
    plan = _nufft_plan(T, g.coords, NTuple{D, Int}(ms), g.domain_size, ntrans, iflag, epsv, _exec_nthreads(exec))
    coeffs = zeros(Complex{T}, ms..., batch...)
    ks = FFS.calculate_spectrum!(coeffs, plan, field)
    return coeffs, ks
end

end # module FlowFieldSpectraFINUFFTExt
