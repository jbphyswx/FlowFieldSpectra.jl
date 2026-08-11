module FlowFieldSpectraGPUFFTExt

using KernelAbstractions: KernelAbstractions as KA, @kernel, @index, @Const
using AbstractFFTs: AbstractFFTs
using LinearAlgebra: LinearAlgebra as LA
using FlowFieldSpectra: FlowFieldSpectra as FFS
using ComputationalBackends: ComputationalBackends
using SpectralBackends: SpectralBackends
using FlowGeometries: FlowGeometries

# =============================================================================
# GPU FFT for uniform structured Cartesian grids on ANY KernelAbstractions device, tensor-native. The
# transform is reached through `AbstractFFTs.plan_fft`, which dispatches on the DEVICE ARRAY TYPE —
# FFTW on `KA.CPU()` (`Array`), CUFFT on `CUDABackend` (`CuArray`), rocFFT on AMDGPU — so this is
# device-GENERIC, not CUDA-specific. C2C is the workhorse (correct for real & complex, and free of the
# GPU-fatal scalar indexing an rfft Hermitian-fill would need); the field `(N…, batch…)` is staged to
# device once, transformed over dims 1:D (batch rides along), fftshifted with a device `circshift!`,
# scaled, and returned. FFT is a full transform, so `ms == size(grid)`.
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

function _gpu_fft_plan(dev, ::Type{RT}, g, ns::NTuple{D, Int}, batch::Tuple, iflag::Int) where {RT<:Real, D}
    inbuf = KA.allocate(dev, Complex{RT}, ns..., batch...)
    fill!(inbuf, zero(Complex{RT}))
    outbuf = similar(inbuf)
    fwd = iflag == 1 ? AbstractFFTs.plan_fft(inbuf, 1:D) : AbstractFFTs.plan_bfft(inbuf, 1:D)
    ks = FFS.Grids.physical_wavenumbers(g, ns)
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

@inline function _gpu_fft_setup(g::FlowGeometries.Grids.AbstractStructuredGrid{<:FlowGeometries.Geometry.AbstractCartesianGeometry}, field::AbstractArray, ms::Tuple)
    ns = size(g)
    Tuple(ms) == ns || throw(ArgumentError("FFTSpectralBackend requires ms == size(grid) = $ns (got $(Tuple(ms)))"))
    D = length(ns)
    batch = ntuple(i -> size(field, D + i), ndims(field) - D)
    return ns, batch
end

function FFS._calculate_spectrum_gpu_fft(exec::ComputationalBackends.AbstractGPUBackend,
        g::FlowGeometries.Grids.AbstractStructuredGrid{<:FlowGeometries.Geometry.AbstractCartesianGeometry},
        field::AbstractArray, ms::Tuple; iflag::Int = 1, kwargs...)
    ns, batch = _gpu_fft_setup(g, field, ms)
    RT = real(float(eltype(field)))
    plan = _gpu_fft_plan(exec.backend, RT, g, ns, batch, iflag)
    coeffs_dev = KA.allocate(exec.backend, Complex{RT}, ns..., batch...)
    ks = FFS.calculate_spectrum!(coeffs_dev, plan, field)
    return _to_host(coeffs_dev), ks
end

function FFS.plan_spectrum(::SpectralBackends.AbstractFFTSpectralBackend, exec::ComputationalBackends.AbstractGPUBackend,
        g::FlowGeometries.Grids.AbstractStructuredGrid{<:FlowGeometries.Geometry.AbstractCartesianGeometry},
        ::Type{T}, ms::NTuple{D, Int}; batch::Tuple = (), iflag::Int = 1) where {T, D}
    FlowGeometries.Grids.isuniform(g) || throw(ArgumentError(
        "FFTSpectralBackend needs a uniform axis in every direction; got a stretched axis."))
    ns = size(g)
    Tuple(ms) == ns || throw(ArgumentError("FFTSpectralBackend requires ms == size(grid) = $ns (got $(Tuple(ms)))"))
    return _gpu_fft_plan(exec.backend, real(float(T)), g, ns, batch, iflag)
end

# =============================================================================
# Fast device-generic STRUCTURED spherical harmonic transform (FSHTSpectralBackend × GPUBackend). The
# direct projection C[l,m] = Σ_p conj(Y_lm) w_p f_p factorizes on a tensor-product (φ-axis × latitude)
# grid with uniform longitude: C[l,m] = factor(m)·Σ_iθ w_θ[iθ]·P̄_l|m|(θ_iθ)·f̂[iθ, m], where
# f̂[iθ, m] = Σ_iφ f[iφ,iθ] e^{-im(iφ-1)Δφ} is a genuine FFT over longitude (device-generic via
# AbstractFFTs — CUFFT / FFTW / rocFFT), computed ONCE per (θ,m) and reused across degrees l. Like the
# direct-sum reference it uses the grid's real colatitudes + quadrature weights (so it is exact on a
# Gauss–Legendre grid with `sampling=GaussLegendreSampling()`), NOT FastTransforms' position-based
# analysis. The field is `(nlon, nlat, batch…)`; longitude is dim 1. Same normalized-Legendre / factor /
# m↔col convention as the direct-sum kernel.
# =============================================================================
function FFS._calculate_spectrum_gpu_sht(exec::ComputationalBackends.AbstractGPUBackend,
        g::FlowGeometries.Grids.AbstractStructuredGrid{<:FlowGeometries.Geometry.AbstractSphericalGeometry},
        field::AbstractArray, ms::Tuple; sampling = nothing, weights = nothing, kwargs...)
    FT = eltype(g)
    dev = exec.backend
    lmax = ms[1] - 1
    λ = collect(FT, FlowGeometries.Grids.coordinates(g, 1))    # longitude, nlon
    φlat = collect(FT, FlowGeometries.Grids.coordinates(g, 2)) # geographic latitude, nlat
    nlon = length(λ)
    nlat = length(φlat)
    Nθc = lmax + 1
    Nφc = 2 * lmax + 1
    N = nlon * nlat
    B = length(field) ÷ N
    dφ = FT(2π) / nlon
    # The real-SH longitude transform (cos/sin from the FFT) assumes longitudes uniform from 0.
    (isapprox(λ[begin], zero(FT); atol = sqrt(eps(FT))) &&
     (nlon < 2 || isapprox(λ[begin + 1] - λ[begin], dφ; atol = sqrt(eps(FT))))) ||
        throw(ArgumentError("GPU SHT needs longitudes uniform from 0 with step 2π/nlon."))
    θ = (FT(π) / 2) .- φlat                                    # colatitude per latitude index
    wlat = FFS.Grids._sht_weights(g, nlat; sampling = sampling, weights = weights)
    wθ = wlat === nothing ? fill(FT(4π) / N, nlat) : (FT.(wlat) .* dφ)
    fld = KA.allocate(dev, Complex{FT}, nlon, nlat, B)
    copyto!(fld, field)                                        # host/device → device, widen, linear
    fhat = AbstractFFTs.fft(fld, 1)                            # longitude-FFT (device-generic), (nlon, nlat, B)
    θd = KA.allocate(dev, FT, nlat); copyto!(θd, θ)
    wd = KA.allocate(dev, FT, nlat); copyto!(wd, wθ)
    coeffs = KA.zeros(dev, Complex{FT}, Nθc, Nφc, B)
    _gpu_sht_legendre_kernel!(dev)(coeffs, fhat, θd, wd, lmax, nlat, B; ndrange = Nθc * Nφc * B)
    KA.synchronize(dev)
    batch = ntuple(i -> size(field, 2 + i), ndims(field) - 2)  # structured spherical: 2 spatial dims
    return reshape(_to_host(coeffs), Nθc, Nφc, batch...), (0:lmax, -lmax:lmax)
end

@kernel function _gpu_sht_legendre_kernel!(coeffs, @Const(fhat), @Const(θ), @Const(w),
        lmax::Int, Nθ::Int, B::Int)
    idx = @index(Global)
    Nθc = lmax + 1
    Nφc = 2 * lmax + 1
    if idx <= Nθc * Nφc * B
        row = (idx - 1) % Nθc + 1
        rest = (idx - 1) ÷ Nθc
        col = rest % Nφc + 1
        b = rest ÷ Nφc + 1
        m = col == 1 ? 0 : (iseven(col) ? -(col ÷ 2) : (col - 1) ÷ 2)
        abs_m = abs(m)
        l = row - 1 + abs_m
        if l <= lmax
            FT = eltype(θ)
            CT = eltype(coeffs)
            # Real SH (FSH convention): Y_lm = s(m)·P̄·trig, and the longitude-sum trig-projection reads
            # the FFT bin of frequency |m| — Re for cos (m ≥ 0), −Im for sin (m < 0).
            sfac = m == 0 ? one(FT) : (isodd(abs_m) ? -sqrt(FT(2)) : sqrt(FT(2)))
            bin = abs_m + 1
            acc = zero(FT)
            @inbounds for iθ in 1:Nθ
                xj = cos(θ[iθ])
                sj = sin(θ[iθ])
                P = _sht_normalized_legendre(l, abs_m, xj, sj)
                fv = fhat[bin, iθ, b]                          # longitude-FFT: bin along dim 1, latitude dim 2
                φc = m == 0 ? real(fv) : (m > 0 ? real(fv) : -imag(fv))
                acc += w[iθ] * P * φc
            end
            @inbounds coeffs[row, col, b] = CT(sfac * acc)
        end
    end
end

# Normalized associated Legendre P̄_l^m(cosθ) via a self-contained scalar recurrence (device-safe;
# same convention as the direct-sum spherical kernel).
@inline function _sht_normalized_legendre(l::Int, m::Int, x::FT, s::FT)::FT where {FT}
    m > l && return zero(FT)
    P_mm = one(FT) / sqrt(FT(4π))
    for mm in 1:m
        P_mm *= -sqrt(FT(2mm + 1) / (2mm)) * s
    end
    l == m && return P_mm
    P_lm = x * sqrt(FT(2m + 3)) * P_mm
    P_lminus1_m = P_mm
    l == m + 1 && return P_lm
    for ll in (m+2):l
        coeff1 = sqrt(FT(4ll^2 - 1) / (ll^2 - m^2))
        coeff2 = sqrt(FT(2ll + 1) * ((ll - 1)^2 - m^2) / ((2ll - 3) * (ll^2 - m^2)))
        P_lminus1_m, P_lm = P_lm, x * coeff1 * P_lm - coeff2 * P_lminus1_m
    end
    return P_lm
end

end # module FlowFieldSpectraGPUFFTExt
