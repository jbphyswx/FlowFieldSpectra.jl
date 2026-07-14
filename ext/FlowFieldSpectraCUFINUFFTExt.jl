module FlowFieldSpectraCUFINUFFTExt

using FINUFFT: FINUFFT
using CUDA: CUDA
using FlowFieldSpectra: FlowFieldSpectra as FFS

# =============================================================================
# Fast GPU NUFFT for scattered Cartesian grids: NUFFTBackend × GPUBackend{<:CUDA.CUDABackend}. This
# is the fast device path that replaces the O(N·M) direct-sum GPU kernel for scattered grids
# (issue #4). It uses cuFINUFFT via FINUFFT.jl's CUDA extension (the distinct `cufinufft_*` guru
# functions, active when FINUFFT + CUDA are both loaded). cuFINUFFT uses the same centered mode
# order (modeord = 0) as the CPU FINUFFT path, so the translation-correction phase is identical.
#
# The nonuniform points are FIXED by the grid, so `cufinufft_makeplan` + `cufinufft_setpts!` run
# once and `cufinufft_exec!` runs per call with `ntrans = n_transf`, reusing device buffers — no
# per-call device allocation in steady state. cuFINUFFT is CUDA-only (no `KA.CPU()` path); a
# non-CUDA execution backend hits the core error stub (no silent fallback). Verified only on real
# hardware via the `gpu/` project, never on CI.
# =============================================================================

# Reusable cuFINUFFT (guru) plan for scattered Cartesian grids on the GPU. `phase` already folds in
# the 1/M normalization so steady-state execution is a single in-place device broadcast + copy.
mutable struct CUFINUFFTCartesianPlan{T, D, NM, PH, KS} <: FFS.AbstractSpectralPlan
    guru::Any                            # cuFINUFFT guru plan (C resource)
    cj::CUDA.CuMatrix{Complex{T}}        # device strengths buffer (M, n_transf)
    fk::CUDA.CuArray{Complex{T}, NM}     # device modes buffer (ms..., n_transf)
    ms::NTuple{D, Int}
    n_transf::Int
    M::Int                               # number of nonuniform points
    phase::PH                            # device (ms...) translation-correction phase, ×(1/M)
    ks_phys::KS
end

# Default cuFINUFFT tolerance: precision-aware, above machine epsilon of the float type.
_default_eps(::Type{T}) where {T} = T === Float32 ? 1.0e-6 : 1.0e-8

function _gpu_nufft_plan(::Type{T}, coords::Tuple, ms::NTuple{D, Int}, domain_size::NTuple{D},
        n_transf::Int, iflag::Int, eps::Real) where {T, D}
    CUDA.functional() || throw(ArgumentError("cuFINUFFT requires a functional CUDA device."))
    M = length(coords[1])
    for d in 1:D
        length(coords[d]) == M || throw(DimensionMismatch("coordinate $d length mismatch"))
    end

    # Per-axis offset (min) and physical period; scale points to FINUFFT's radian convention and
    # stage the (fixed) points to the device once.
    offsets = ntuple(d -> T(minimum(coords[d])), D)
    ranges = ntuple(d -> (r = T(domain_size[d]); r == 0 ? one(T) : r), D)
    scaled = ntuple(d -> CUDA.CuArray(T(2π) .* (T.(collect(coords[d])) .- offsets[d]) ./ ranges[d]), D)

    # cuFINUFFT type-1 guru plan; sign is -iflag to match the e^{-ik·x} analysis convention.
    guru = FINUFFT.cufinufft_makeplan(1, collect(ms), -iflag, n_transf, T(eps); dtype = T)
    if D == 1
        FINUFFT.cufinufft_setpts!(guru, scaled[1])
    elseif D == 2
        FINUFFT.cufinufft_setpts!(guru, scaled[1], scaled[2])
    elseif D == 3
        FINUFFT.cufinufft_setpts!(guru, scaled[1], scaled[2], scaled[3])
    else
        FINUFFT.cufinufft_destroy!(guru)
        throw(ArgumentError("cuFINUFFT supports up to 3 dimensions; got $D"))
    end

    # Centered integer modes (modeord = 0) → translation-correction phase per axis, folding in 1/M.
    k_ints = ntuple(d -> collect(-(ms[d] ÷ 2):((ms[d] - 1) ÷ 2)), D)
    inv_M = one(T) / M
    phase_h = Array{Complex{T}, D}(undef, ms...)
    @inbounds for I in CartesianIndices(ms)
        p = one(Complex{T})
        for d in 1:D
            p *= cis(-iflag * k_ints[d][I[d]] * (offsets[d] * T(2π) / ranges[d]))
        end
        phase_h[I] = p * inv_M
    end
    phase = CUDA.CuArray(phase_h)

    ks_phys = FFS.Grids.physical_wavenumbers(ranges, ms, T)
    cj = CUDA.zeros(Complex{T}, M, n_transf)
    fk = CUDA.zeros(Complex{T}, ms..., n_transf)

    plan = CUFINUFFTCartesianPlan{T, D, D + 1, typeof(phase), typeof(ks_phys)}(
        guru, cj, fk, ms, n_transf, M, phase, ks_phys,
    )
    finalizer(p -> FINUFFT.cufinufft_destroy!(p.guru), plan)
    return plan
end

function _load_strengths!(plan::CUFINUFFTCartesianPlan{T}, fields_vecs::Tuple) where {T}
    length(fields_vecs) == plan.n_transf ||
        throw(DimensionMismatch("expected $(plan.n_transf) fields, got $(length(fields_vecs))"))
    @inbounds for u in 1:length(fields_vecs)
        length(fields_vecs[u]) == plan.M ||
            throw(DimensionMismatch("field $u length $(length(fields_vecs[u])) != npoints=$(plan.M)"))
        copyto!(view(plan.cj, :, u), fields_vecs[u])   # host- or device-source copy, no per-call device alloc
    end
    return plan
end
_load_strengths!(plan::CUFINUFFTCartesianPlan, field::AbstractArray) = (copyto!(plan.cj, field); plan)

"""
    calculate_spectrum!(coeffs, plan::CUFINUFFTCartesianPlan, fields) -> ks_phys

Execute a prebuilt cuFINUFFT guru plan in place on the GPU. `coeffs` may be a device (`CuArray`) or
host array of shape `(ms..., n_transf)`; the modes buffer is phase-corrected + normalized on device
then copied into `coeffs`. The plan and point sorting are reused across calls.
"""
function FFS.calculate_spectrum!(coeffs::AbstractArray{Complex{T}}, plan::CUFINUFFTCartesianPlan{T, D},
        fields) where {T, D}
    size(coeffs) == (plan.ms..., plan.n_transf) ||
        throw(DimensionMismatch("coeffs size $(size(coeffs)) != $((plan.ms..., plan.n_transf))"))
    _load_strengths!(plan, fields)
    FINUFFT.cufinufft_exec!(plan.guru, plan.cj, plan.fk)          # device, in-place
    plan.fk .*= reshape(plan.phase, plan.ms..., 1)                # fused device broadcast (phase folds 1/M)
    copyto!(coeffs, plan.fk)                                      # device→device or device→host
    return plan.ks_phys
end

# One-shot allocating entry routed from calculate_spectrum(::NUFFTBackend, ::GPUBackend, cart-grid).
# Returns a host `Array` (consistent with every other backend; reductions run on host).
function FFS._calculate_spectrum_gpu_nufft(::FFS.GPUBackend{<:CUDA.CUDABackend},
        coords_vecs::Tuple, fields_vecs::Tuple, ms::Tuple;
        iflag::Int = 1, eps::Union{Nothing, Real} = nothing,
        domain_size::Union{Nothing, Tuple} = nothing, kwargs...)
    D = length(ms)
    NU = length(fields_vecs)
    T = float(real(eltype(coords_vecs[1])))
    epsv = eps === nothing ? _default_eps(T) : eps
    ds = domain_size === nothing ?
         ntuple(d -> (e = extrema(coords_vecs[d]); T(e[2] - e[1])), D) :
         ntuple(d -> T(domain_size[d]), D)
    plan = _gpu_nufft_plan(T, coords_vecs, NTuple{D, Int}(ms), ds, NU, iflag, epsv)
    coeffs_dev = CUDA.zeros(Complex{T}, ms..., NU)
    ks = FFS.calculate_spectrum!(coeffs_dev, plan, fields_vecs)
    return Array(coeffs_dev), ks
end

function FFS.plan_spectrum(::FFS.NUFFTBackend, ::FFS.GPUBackend{<:CUDA.CUDABackend},
        g::FFS.AbstractCartesianGrid, ::Type{T}, ms::NTuple{D, Int};
        n_transf::Int = 1, iflag::Int = 1, eps::Real = _default_eps(T)) where {T, D}
    return _gpu_nufft_plan(T, g.coords, ms, g.domain_size, n_transf, iflag, eps)
end

end # module FlowFieldSpectraCUFINUFFTExt
