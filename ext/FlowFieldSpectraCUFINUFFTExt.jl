module FlowFieldSpectraCUFINUFFTExt

using FINUFFT: FINUFFT
using CUDA: CUDA
using FlowFieldSpectra: FlowFieldSpectra as FFS

# =============================================================================
# Fast GPU NUFFT for scattered Cartesian grids: NUFFTBackend × GPUBackend{<:CUDABackend}, tensor-native
# + batched. cuFINUFFT (FINUFFT.jl's CUDA extension) with `ntrans = ∏batch`; nonuniform points fixed by
# the grid ⇒ `cufinufft_makeplan` + `cufinufft_setpts!` once, `cufinufft_exec!` per call reusing device
# buffers. Same centered mode order (modeord=0) as the CPU FINUFFT path, so the translation-correction
# phase is identical. CUDA-only (no portable GPU NUFFT); a non-CUDA device hits the core error stub.
# Validated only on real NVIDIA hardware via `gpu/`, never on CI.
# =============================================================================

_default_eps(::Type{T}) where {T} = T === Float32 ? 1.0e-6 : 1.0e-8

mutable struct CUFINUFFTCartesianPlan{T, D, NM, CJ, FK, PH, KS} <: FFS.AbstractSpectralPlan
    guru::Any                            # cuFINUFFT guru plan (C resource)
    cj::CJ                               # device (M, ntrans) strengths buffer
    fk::FK                               # device (ms…, ntrans) modes buffer
    ms::NTuple{D, Int}
    ntrans::Int
    M::Int
    phase::PH                            # device (ms…, 1) translation-correction phase × (1/M)
    ks_phys::KS
end

function _gpu_nufft_plan(::Type{T}, coords::Tuple, ms::NTuple{D, Int}, domain_size::NTuple{D},
        ntrans::Int, iflag::Int, eps::Real) where {T, D}
    CUDA.functional() || throw(ArgumentError("cuFINUFFT requires a functional CUDA device."))
    M = length(coords[1])
    for d in 1:D
        length(coords[d]) == M || throw(DimensionMismatch("coordinate $d length mismatch"))
    end
    offsets = ntuple(d -> T(minimum(coords[d])), D)
    ranges = ntuple(d -> (r = T(domain_size[d]); r == 0 ? one(T) : r), D)
    scaled = ntuple(d -> CUDA.CuArray(T(2π) .* (T.(collect(coords[d])) .- offsets[d]) ./ ranges[d]), D)

    guru = FINUFFT.cufinufft_makeplan(1, collect(ms), -iflag, ntrans, T(eps); dtype = T)
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

    k_ints = ntuple(d -> collect(-(ms[d] ÷ 2):((ms[d] - 1) ÷ 2)), D)
    inv_M = one(T) / M
    phase_h = Array{Complex{T}, D + 1}(undef, ms..., 1)
    @inbounds for I in CartesianIndices(ms)
        p = one(Complex{T})
        for d in 1:D
            p *= cis(-iflag * k_ints[d][I[d]] * (offsets[d] * T(2π) / ranges[d]))
        end
        phase_h[I, 1] = p * inv_M
    end
    phase = CUDA.CuArray(phase_h)

    ks_phys = FFS.Grids.physical_wavenumbers(ranges, ms, T)
    cj = CUDA.zeros(Complex{T}, M, ntrans)
    fk = CUDA.zeros(Complex{T}, ms..., ntrans)
    plan = CUFINUFFTCartesianPlan{T, D, D + 1, typeof(cj), typeof(fk), typeof(phase), typeof(ks_phys)}(
        guru, cj, fk, ms, ntrans, M, phase, ks_phys,
    )
    finalizer(p -> FINUFFT.cufinufft_destroy!(p.guru), plan)
    return plan
end

"""
    calculate_spectrum!(coeffs, plan::CUFINUFFTCartesianPlan, field) -> ks_phys

Execute a prebuilt cuFINUFFT guru plan in place on the GPU. `field` `(N, batch…)` fills the device
strengths buffer; `coeffs` `(ms…, batch…)` (device or host) receives the phase-corrected modes.
"""
function FFS.calculate_spectrum!(coeffs::AbstractArray{Complex{T}}, plan::CUFINUFFTCartesianPlan{T, D},
        field) where {T, D}
    copyto!(plan.cj, field)                                      # host/device → device, linear (widen)
    FINUFFT.cufinufft_exec!(plan.guru, plan.cj, plan.fk)
    plan.fk .*= plan.phase                                       # phase (ms…, 1) broadcasts over ntrans, folds 1/M
    if size(coeffs) == size(plan.fk)
        copyto!(coeffs, plan.fk)
    else
        copyto!(reshape(coeffs, plan.ms..., plan.ntrans), plan.fk)
    end
    return plan.ks_phys
end

function FFS.plan_spectrum(::FFS.NUFFTBackend, ::FFS.GPUBackend{<:CUDA.CUDABackend},
        g::FFS.ScatteredCartesianGrid, ::Type{T}, ms::NTuple{D, Int};
        batch::Tuple = (), iflag::Int = 1, eps::Real = _default_eps(T)) where {T, D}
    return _gpu_nufft_plan(T, g.coords, ms, g.domain_size, prod(batch; init = 1), iflag, eps)
end

function FFS._calculate_spectrum_gpu_nufft(::FFS.GPUBackend{<:CUDA.CUDABackend},
        g::FFS.ScatteredCartesianGrid, field::AbstractArray, ms::Tuple;
        iflag::Int = 1, eps::Union{Nothing, Real} = nothing, kwargs...)
    D = length(ms)
    T = float(real(eltype(g.coords[1])))
    batch = ntuple(i -> size(field, 1 + i), ndims(field) - 1)
    epsv = eps === nothing ? _default_eps(T) : eps
    plan = _gpu_nufft_plan(T, g.coords, NTuple{D, Int}(ms), g.domain_size, prod(batch; init = 1), iflag, epsv)
    coeffs_dev = CUDA.zeros(Complex{T}, ms..., batch...)
    ks = FFS.calculate_spectrum!(coeffs_dev, plan, field)
    return Array(coeffs_dev), ks
end

# =============================================================================
# Separable GPU NUFFT for a nonuniform tensor-product grid (NonuniformCartesianGrid): the D-dim kernel
# factorizes, so it is a sequence of 1-D cuFINUFFT type-1 transforms — one per axis — with every other
# spatial + batch dim carried as the `ntrans` batch. Device-side analog of the CPU separable path
# (FINUFFTExt._nufft_axis); each 1-D transform needs only its own length-N_d axis (no ∏N_d coords).
# Because every transform is 1-D, this supports any D. CUDA-only; validated only on `gpu/`, never CI.
# =============================================================================
function FFS._calculate_spectrum_gpu_nufft(::FFS.GPUBackend{<:CUDA.CUDABackend},
        g::FFS.NonuniformCartesianGrid{FT0, D}, field::AbstractArray, ms::Tuple;
        iflag::Int = 1, eps::Union{Nothing, Real} = nothing, kwargs...) where {FT0, D}
    CUDA.functional() || throw(ArgumentError("cuFINUFFT requires a functional CUDA device."))
    T = float(real(eltype(field)))
    epsv = T(eps === nothing ? _default_eps(T) : eps)
    npts = prod(ntuple(d -> length(g.axes[d]), D))
    A = CUDA.CuArray{Complex{T}}(undef, size(field)...)
    copyto!(A, field)                                            # host/device → device, widen real→complex
    for d in 1:D
        A = _gpu_nufft_axis(A, d, g.axes[d], Int(ms[d]), T(g.domain_size[d]), iflag, epsv)
    end
    A ./= npts
    ks = FFS.Grids.physical_wavenumbers(ntuple(d -> T(g.domain_size[d]), D), NTuple{D, Int}(ms), T)
    return Array(A), ks
end

# One 1-D cuFINUFFT type-1 transform along dim `d` (points = `axis`), other dims → ntrans; × per-mode
# translation-correction phase. Returns a new device array whose dim `d` now has length `m`.
function _gpu_nufft_axis(A::CUDA.CuArray{Complex{T}}, d::Int, axis::AbstractVector, m::Int,
        L::T, iflag::Int, eps::T) where {T}
    nd = ndims(A)
    Nd = length(axis)
    size(A, d) == Nd ||
        throw(DimensionMismatch("axis $d: field length $(size(A, d)) ≠ grid axis length $Nd"))
    perm = (d, ntuple(i -> i < d ? i : i + 1, nd - 1)...)        # move transform axis to front
    Ap = permutedims(A, perm)                                    # device (Nd, rest…), contiguous
    restshape = size(Ap)[2:end]
    ntrans = prod(restshape; init = 1)
    cj = reshape(Ap, Nd, ntrans)                                 # device (Nd, ntrans)
    off = T(minimum(axis))
    rng = L == 0 ? one(T) : L
    x = CUDA.CuArray(T(2π) .* (T.(collect(axis)) .- off) ./ rng)
    guru = FINUFFT.cufinufft_makeplan(1, [m], -iflag, ntrans, eps; dtype = T)
    FINUFFT.cufinufft_setpts!(guru, x)
    fk = CUDA.zeros(Complex{T}, m, ntrans)
    FINUFFT.cufinufft_exec!(guru, cj, fk)
    FINUFFT.cufinufft_destroy!(guru)
    m_ints = -(m ÷ 2):((m - 1) ÷ 2)
    ophase = CUDA.CuArray(Complex{T}[cis(-iflag * mm * (off * T(2π) / rng)) for mm in m_ints])  # (m,)
    fk .*= reshape(ophase, m, 1)                                 # broadcast over ntrans
    Fr = reshape(fk, m, restshape...)
    return permutedims(Fr, invperm(collect(perm)))               # restore dim order; dim d now length m
end

end # module FlowFieldSpectraCUFINUFFTExt
