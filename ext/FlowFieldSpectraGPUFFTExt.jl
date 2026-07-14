module FlowFieldSpectraGPUFFTExt

using KernelAbstractions: KernelAbstractions as KA
using AbstractFFTs: AbstractFFTs
using LinearAlgebra: LinearAlgebra as LA
using FlowFieldSpectra: FlowFieldSpectra as FFS

# =============================================================================
# GPU FFT for uniform Cartesian grids: FFTBackend × GPUBackend{B} on ANY KernelAbstractions device.
# This is device-GENERIC, not CUDA-specific. Data is staged to the KA device `exec.backend`, and the
# transform is reached through `AbstractFFTs.plan_fft`, which dispatches on the *device array type*:
#   • `KA.CPU()`         → plain `Array`   → FFTW      (requires `using FFTW`)
#   • `CUDA.CUDABackend()`→ `CuArray`      → CUFFT     (requires `using CUDA`)
#   • `ROCBackend()`      → `ROCArray`     → rocFFT    (requires `using AMDGPU`)
# i.e. the FFT provider is whatever the loaded vendor package registers for that array type — the
# whole point of KernelAbstractions. Triggered by `KernelAbstractions` + `AbstractFFTs` (the
# device's FFT provider — FFTW / CUDA / AMDGPU — must also be loaded for its array type).
#
# A single planned C2C transform over the spectral dims `1:D` is applied to all `n_transf` trailing
# batch slices and reused across calls; device buffers are owned by the plan. C2C (not R2C) is the
# workhorse: correct for real and complex inputs and free of GPU-fatal scalar indexing.
# =============================================================================

struct GPUFFTCartesianPlan{T, D, N, P, AT, KS} <: FFS.AbstractSpectralPlan
    fwd::P                        # AbstractFFTs plan (FFTW / CUFFT / rocFFT) over the device array, dims 1:D
    inbuf::AT                     # device input buffer (ms..., n_transf); also fftshift scratch
    outbuf::AT                    # device transform output buffer
    ms::NTuple{D, Int}
    n_transf::Int
    shifts::NTuple{N, Int}        # fftshift = circshift by div(m,2) on spectral dims, 0 on batch
    norm::T                       # 1/prod(ms)
    ks_phys::KS
end

# Materialize a device array to a host `Array` (no-op if already one; `collect` covers CuArray/etc.).
_to_host(a::Array) = a
_to_host(a::AbstractArray) = collect(a)

function _gpu_fft_plan(dev, ::Type{T}, ms::NTuple{D, Int}, domain_size::NTuple{D},
        n_transf::Int, iflag::Int) where {T, D}
    inbuf = KA.allocate(dev, Complex{T}, ms..., n_transf)
    fill!(inbuf, zero(Complex{T}))
    outbuf = similar(inbuf)
    fwd = iflag == 1 ? AbstractFFTs.plan_fft(inbuf, 1:D) : AbstractFFTs.plan_bfft(inbuf, 1:D)
    shifts = ntuple(i -> i <= D ? div(ms[i], 2) : 0, D + 1)
    ds = ntuple(d -> T(domain_size[d]), D)
    ks_phys = FFS.Grids.physical_wavenumbers(ds, ms, T)
    return GPUFFTCartesianPlan{T, D, D + 1, typeof(fwd), typeof(inbuf), typeof(ks_phys)}(
        fwd, inbuf, outbuf, ms, n_transf, shifts, one(T) / prod(ms), ks_phys,
    )
end

function _load_input!(plan::GPUFFTCartesianPlan{T, D}, fields_vecs::Tuple) where {T, D}
    length(fields_vecs) == plan.n_transf ||
        throw(DimensionMismatch("expected $(plan.n_transf) fields, got $(length(fields_vecs))"))
    M = prod(plan.ms)
    @inbounds for (u, fu) in enumerate(fields_vecs)
        length(fu) == M || throw(DimensionMismatch("field $u length $(length(fu)) != prod(ms)=$M"))
        copyto!(plan.inbuf, (u - 1) * M + 1, fu, 1, M)   # host- or device-source copy into batch slice
    end
    return plan
end
_load_input!(plan::GPUFFTCartesianPlan, field::AbstractArray) = (copyto!(plan.inbuf, field); plan)

"""
    calculate_spectrum!(coeffs, plan::GPUFFTCartesianPlan, fields) -> ks_phys

Execute a prebuilt device FFT plan in place. `coeffs` may be a device or host array of shape
`(ms..., n_transf)`; the fftshift + normalization run on the device and the result is copied into
`coeffs`. The fftshift uses in-place `circshift!` (a device kernel on a GPU array, a plain shift on
`KA.CPU()`), so there is no GPU-fatal scalar indexing.
"""
function FFS.calculate_spectrum!(coeffs::AbstractArray{Complex{T}}, plan::GPUFFTCartesianPlan{T, D},
        fields) where {T, D}
    size(coeffs) == (plan.ms..., plan.n_transf) ||
        throw(DimensionMismatch("coeffs size $(size(coeffs)) != $((plan.ms..., plan.n_transf))"))
    _load_input!(plan, fields)
    LA.mul!(plan.outbuf, plan.fwd, plan.inbuf)          # device FFT (FFTW / CUFFT / rocFFT)
    circshift!(plan.inbuf, plan.outbuf, plan.shifts)    # device fftshift into the (now-free) input buffer
    plan.inbuf .*= plan.norm
    copyto!(coeffs, plan.inbuf)                          # device→device or device→host
    return plan.ks_phys
end

# One-shot allocating entry routed from calculate_spectrum(::FFTBackend, ::GPUBackend, cart-grid) for
# ANY KA device. Returns a host `Array` (consistent with every other backend; reductions run on host).
function FFS._calculate_spectrum_gpu_fft(exec::FFS.GPUBackend,
        coords_vecs::Tuple, fields_vecs::Tuple, ms::Tuple;
        iflag::Int = 1, domain_size::Union{Nothing, Tuple} = nothing, kwargs...)
    D = length(ms)
    NU = length(fields_vecs)
    T = float(real(eltype(fields_vecs[1])))
    ds = domain_size === nothing ?
         ntuple(d -> (e = extrema(coords_vecs[d]); T(e[2] - e[1])), D) :
         ntuple(d -> T(domain_size[d]), D)
    plan = _gpu_fft_plan(exec.backend, T, NTuple{D, Int}(ms), ds, NU, iflag)
    coeffs_dev = KA.allocate(exec.backend, Complex{T}, ms..., NU)
    ks = FFS.calculate_spectrum!(coeffs_dev, plan, fields_vecs)
    return _to_host(coeffs_dev), ks
end

function FFS.plan_spectrum(::FFS.FFTBackend, exec::FFS.GPUBackend,
        g::FFS.AbstractCartesianGrid, ::Type{T}, ms::NTuple{D, Int};
        n_transf::Int = 1, iflag::Int = 1) where {T, D}
    return _gpu_fft_plan(exec.backend, T, ms, g.domain_size, n_transf, iflag)
end

end # module FlowFieldSpectraGPUFFTExt
