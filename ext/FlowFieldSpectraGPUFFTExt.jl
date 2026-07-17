module FlowFieldSpectraGPUFFTExt

using KernelAbstractions: KernelAbstractions as KA
using AbstractFFTs: AbstractFFTs
using LinearAlgebra: LinearAlgebra as LA
using FlowFieldSpectra: FlowFieldSpectra as FFS

# =============================================================================
# GPU FFT for uniform Cartesian grids on ANY KernelAbstractions device, tensor-native. The transform
# is reached through `AbstractFFTs.plan_fft`, which dispatches on the DEVICE ARRAY TYPE — FFTW on
# `KA.CPU()` (`Array`), CUFFT on `CUDABackend` (`CuArray`), rocFFT on AMDGPU — so this is device-
# GENERIC, not CUDA-specific. C2C is the workhorse (correct for real & complex, and free of the
# GPU-fatal scalar indexing an rfft Hermitian-fill would need); the field `(N…, batch…)` is staged to
# device once, transformed over dims 1:D (batch rides along), fftshifted with a device `circshift!`,
# scaled, and returned. FFT is a full transform, so `ms == spatial_size(grid)`.
# =============================================================================

struct GPUFFTCartesianPlan{RT, D, NT, P, AT, KS} <: FFS.AbstractSpectralPlan
    fwd::P                        # AbstractFFTs plan over dims 1:D of the device array
    inbuf::AT                     # device (N…, batch…) complex input buffer (also fftshift scratch)
    outbuf::AT                    # device (N…, batch…) complex transform buffer
    ns::NTuple{D, Int}
    shifts::NTuple{NT, Int}
    norm::RT
    ks_phys::KS
end

_to_host(a::Array) = a
_to_host(a::AbstractArray) = collect(a)

_shifts(ns::NTuple{D, Int}, nbatch::Int) where {D} = ntuple(i -> i <= D ? div(ns[i], 2) : 0, D + nbatch)

function _gpu_fft_plan(dev, ::Type{RT}, ns::NTuple{D, Int}, domain_size, batch::Tuple, iflag::Int) where {RT<:Real, D}
    inbuf = KA.allocate(dev, Complex{RT}, ns..., batch...)
    fill!(inbuf, zero(Complex{RT}))
    outbuf = similar(inbuf)
    fwd = iflag == 1 ? AbstractFFTs.plan_fft(inbuf, 1:D) : AbstractFFTs.plan_bfft(inbuf, 1:D)
    ks = FFS.Grids.physical_wavenumbers(ntuple(d -> RT(domain_size[d]), D), ns, RT)
    return GPUFFTCartesianPlan{RT, D, D + length(batch), typeof(fwd), typeof(inbuf), typeof(ks)}(
        fwd, inbuf, outbuf, ns, _shifts(ns, length(batch)), one(RT) / prod(ns), ks,
    )
end

"""
    calculate_spectrum!(coeffs, plan::GPUFFTCartesianPlan, field) -> ks_phys

Execute a prebuilt device FFT plan in place. `field` (host or device, real or complex) is staged into
the device complex buffer; the fftshift + scale run on device; the result is copied into `coeffs`
(device or host). No GPU-fatal scalar indexing.
"""
function FFS.calculate_spectrum!(coeffs::AbstractArray{Complex{RT}}, plan::GPUFFTCartesianPlan{RT, D}, field) where {RT, D}
    copyto!(plan.inbuf, field)                          # host/device, widen real→complex
    LA.mul!(plan.outbuf, plan.fwd, plan.inbuf)          # device FFT (FFTW / CUFFT / rocFFT)
    circshift!(plan.inbuf, plan.outbuf, plan.shifts)    # device fftshift into the (now-free) input buffer
    plan.inbuf .*= plan.norm
    copyto!(coeffs, plan.inbuf)                          # device→device or device→host
    return plan.ks_phys
end

@inline function _gpu_fft_setup(g::FFS.UniformCartesianGrid, field::AbstractArray, ms::Tuple)
    ns = FFS.spatial_size(g)
    Tuple(ms) == ns || throw(ArgumentError("FFTBackend requires ms == spatial_size(grid) = $ns (got $(Tuple(ms)))"))
    D = length(ns)
    batch = ntuple(i -> size(field, D + i), ndims(field) - D)
    return ns, batch
end

function FFS._calculate_spectrum_gpu_fft(exec::FFS.GPUBackend, g::FFS.UniformCartesianGrid,
        field::AbstractArray, ms::Tuple; iflag::Int = 1, kwargs...)
    ns, batch = _gpu_fft_setup(g, field, ms)
    RT = real(float(eltype(field)))
    plan = _gpu_fft_plan(exec.backend, RT, ns, g.domain_size, batch, iflag)
    coeffs_dev = KA.allocate(exec.backend, Complex{RT}, ns..., batch...)
    ks = FFS.calculate_spectrum!(coeffs_dev, plan, field)
    return _to_host(coeffs_dev), ks
end

function FFS.plan_spectrum(::FFS.FFTBackend, exec::FFS.GPUBackend, g::FFS.UniformCartesianGrid,
        ::Type{T}, ms::NTuple{D, Int}; batch::Tuple = (), iflag::Int = 1) where {T, D}
    ns = FFS.spatial_size(g)
    Tuple(ms) == ns || throw(ArgumentError("FFTBackend requires ms == spatial_size(grid) = $ns (got $(Tuple(ms)))"))
    return _gpu_fft_plan(exec.backend, real(float(T)), ns, g.domain_size, batch, iflag)
end

end # module FlowFieldSpectraGPUFFTExt
