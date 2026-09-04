module FlowFieldSpectraGPUFFTExt

using KernelAbstractions: KernelAbstractions as KA, @kernel, @index, @Const
using AbstractFFTs: AbstractFFTs
using LinearAlgebra: LinearAlgebra as LA
using FlowFieldSpectra: FlowFieldSpectra as FFS
using ComputationalBackends: ComputationalBackends
using SpectralBackends: SpectralBackends
using FlowGeometries: FlowGeometries

# =============================================================================
# GPU FFT for uniform structured Cartesian grids on any KernelAbstractions device, packed-native. The
# transform is reached through `AbstractFFTs.plan_rfft`/`plan_fft`, which dispatches on the device array
# type (FFTW-backed on `KA.CPU()`, the vendor library on a device array), so this is device-generic. A
# real field takes the device r2c fast path and its coefficients are the packed half
# `(N_1÷2+1, N_2…, batch…)` in native order; a complex field gives the full native spectrum. Writing the
# packed half needs no Hermitian fill and no fftshift, so there is no device scalar indexing. The field
# is staged once, transformed over dims 1:D (batch rides along), scaled, and copied out. FFT is a full
# transform, so `ms == size(grid)`.
# =============================================================================

struct GPUFFTRealPlan{RT, D, P, IN, HB, KS} <: FFS.AbstractSpectralPlan
    fwd::P                        # device rfft plan over dims 1:D of a real (N…, batch…)
    inbuf::IN                     # device real (N…, batch…) input buffer
    half::HB                      # device complex packed half (N_1÷2+1, N_2…, batch…)
    ns::NTuple{D, Int}
    norm::RT
    neg::Bool                     # `rfft` carries the +1 sign; a real field's −1 transform is its conjugate
    ks_phys::KS
end

struct GPUFFTComplexPlan{RT, D, P, CB, KS} <: FFS.AbstractSpectralPlan
    fwd::P                        # device fft/bfft plan over dims 1:D of a complex (N…, batch…)
    inbuf::CB                     # device complex input buffer
    outbuf::CB                    # device complex output buffer
    ns::NTuple{D, Int}
    norm::RT
    ks_phys::KS
end


# Real field: device r2c writing the packed half.
function _gpu_fft_plan(dev, ::Type{T}, g, ns::NTuple{D, Int}, batch::Tuple, iflag::Int) where {T<:Real, D}
    RT = float(T)
    inbuf = KA.allocate(dev, RT, ns..., batch...)
    fill!(inbuf, zero(RT))
    fwd = AbstractFFTs.plan_rfft(inbuf, 1:D)
    half = KA.allocate(dev, Complex{RT}, ns[1] ÷ 2 + 1, ns[2:D]..., batch...)
    ks = FFS.Grids.physical_wavenumbers(g, ns, Val(true))
    return GPUFFTRealPlan{RT, D, typeof(fwd), typeof(inbuf), typeof(half), typeof(ks)}(
        fwd, inbuf, half, ns, one(RT) / prod(ns), iflag < 0, ks)
end

# Complex field: device c2c, full native order.
function _gpu_fft_plan(dev, ::Type{Complex{RT}}, g, ns::NTuple{D, Int}, batch::Tuple, iflag::Int) where {RT<:Real, D}
    inbuf = KA.allocate(dev, Complex{RT}, ns..., batch...)
    fill!(inbuf, zero(Complex{RT}))
    outbuf = similar(inbuf)
    fwd = iflag == 1 ? AbstractFFTs.plan_fft(inbuf, 1:D) : AbstractFFTs.plan_bfft(inbuf, 1:D)
    ks = FFS.Grids.physical_wavenumbers(g, ns, Val(false))
    return GPUFFTComplexPlan{RT, D, typeof(fwd), typeof(inbuf), typeof(ks)}(
        fwd, inbuf, outbuf, ns, one(RT) / prod(ns), ks)
end

"""
    calculate_spectrum!(coeffs, plan, field) -> ks_phys

Execute a prebuilt device FFT plan in place. `field` (host or device) is staged into the device buffer,
transformed (`GPUFFTRealPlan` → packed half, `GPUFFTComplexPlan` → full native), scaled by `1/∏N`, and
copied into `coeffs` (device or host). No device scalar indexing.
"""
function FFS.calculate_spectrum!(coeffs::AbstractArray{Complex{RT}}, plan::GPUFFTRealPlan{RT, D}, field) where {RT, D}
    copyto!(plan.inbuf, field)                          # host/device real input
    LA.mul!(plan.half, plan.fwd, plan.inbuf)            # device rfft → packed half
    if plan.neg
        plan.half .= conj.(plan.half) .* plan.norm
    else
        plan.half .*= plan.norm
    end
    copyto!(coeffs, plan.half)                          # device→device or device→host
    return plan.ks_phys
end

function FFS.calculate_spectrum!(coeffs::AbstractArray{Complex{RT}}, plan::GPUFFTComplexPlan{RT, D}, field) where {RT, D}
    copyto!(plan.inbuf, field)
    LA.mul!(plan.outbuf, plan.fwd, plan.inbuf)          # device c2c, full native
    plan.outbuf .*= plan.norm
    copyto!(coeffs, plan.outbuf)
    return plan.ks_phys
end

@inline function _gpu_fft_setup(g::FlowGeometries.Grids.AbstractStructuredGrid{<:FlowGeometries.Geometry.AbstractCartesianGeometry}, field::AbstractArray, ms::Tuple)
    ns = size(g)
    Tuple(ms) == ns || throw(ArgumentError("FFTSpectralBackend requires ms == size(grid) = $ns (got $(Tuple(ms)))"))
    D = length(ns)
    batch = ntuple(i -> size(field, D + i), ndims(field) - D)
    return ns, batch
end

# =============================================================================
# The hybrid composite's FFT pass on a device. `AbstractFFTs.rfft`/`fft` dispatch on the device array
# type, so this is device-generic like the plans above; the field is staged once and the result stays on
# the device for the NUFFT passes that follow.
# =============================================================================

function FFS._region_fft(exec::ComputationalBackends.AbstractGPUBackend, field::AbstractArray,
        dims::Tuple, halve::Bool, conj_in::Bool)
    dev = exec.backend
    RT = real(float(eltype(field)))
    if halve
        inb = KA.allocate(dev, RT, size(field)...)
        copyto!(inb, field)
        return AbstractFFTs.rfft(inb, dims)
    end
    inb = KA.allocate(dev, Complex{RT}, size(field)...)
    copyto!(inb, field)
    conj_in && (inb .= conj.(inb))
    return AbstractFFTs.fft(inb, dims)
end

function FFS._calculate_spectrum_gpu_fft(exec::ComputationalBackends.AbstractGPUBackend,
        g::FlowGeometries.Grids.AbstractStructuredGrid{<:FlowGeometries.Geometry.AbstractCartesianGeometry},
        field::AbstractArray, ms::Tuple; iflag::Int = 1, kwargs...)
    ns, batch = _gpu_fft_setup(g, field, ms)
    T = eltype(field)
    RT = real(float(T))
    ET = T <: Real ? float(T) : Complex{RT}
    plan = _gpu_fft_plan(exec.backend, ET, g, ns, batch, iflag)
    pms = FFS.Packing.packed_size(ns, Val(T <: Real))
    coeffs_dev = KA.allocate(exec.backend, Complex{RT}, pms..., batch...)
    ks = FFS.calculate_spectrum!(coeffs_dev, plan, field)
    return FFS._to_host(coeffs_dev), ks
end

function FFS.plan_spectrum(::SpectralBackends.AbstractFFTSpectralBackend, exec::ComputationalBackends.AbstractGPUBackend,
        g::FlowGeometries.Grids.AbstractStructuredGrid{<:FlowGeometries.Geometry.AbstractCartesianGeometry},
        ::Type{T}, ms::NTuple{D, Int}; batch::Tuple = (), iflag::Int = 1) where {T, D}
    FlowGeometries.Grids.isuniform(g) || throw(ArgumentError(
        "FFTSpectralBackend needs a uniform axis in every direction; got a stretched axis."))
    ns = size(g)
    Tuple(ms) == ns || throw(ArgumentError("FFTSpectralBackend requires ms == size(grid) = $ns (got $(Tuple(ms)))"))
    ET = T <: Real ? float(T) : Complex{real(float(T))}
    return _gpu_fft_plan(exec.backend, ET, g, ns, batch, iflag)
end

# =============================================================================
# Inverse (synthesis) on the device, reached through `AbstractFFTs.plan_brfft`/`plan_bfft` so it stays
# device-generic. The forward's `1/∏N` cancels the unnormalized backward transform, so neither direction
# rescales here. A real field's half is consumed by the c2r transform, so it is staged into a scratch
# buffer and `coeffs` is left intact. This method is reached for a GPU execution backend, ahead of the
# host FFTW method, which accepts any execution backend.
# =============================================================================
function FFS._synthesize(::SpectralBackends.AbstractFFTSpectralBackend,
        exec::ComputationalBackends.AbstractGPUBackend,
        g::FlowGeometries.Grids.AbstractStructuredGrid{<:FlowGeometries.Geometry.AbstractCartesianGeometry},
        coeffs::AbstractArray{Complex{RT}}, ms::NTuple{D, Int};
        real_output::Bool = true, iflag::Int = 1, ks = nothing, kwargs...) where {RT, D}
    FlowGeometries.Grids.isuniform(g) || throw(ArgumentError(
        "FFTSpectralBackend needs a uniform axis in every direction; got a stretched axis."))
    ns = size(g)
    Tuple(ms) == ns || throw(ArgumentError(
        "FFTSpectralBackend requires ms == size(grid) = $ns (got $(Tuple(ms))); FFT is a full transform."))
    dev = exec.backend
    batch = ntuple(i -> size(coeffs, D + i), ndims(coeffs) - D)
    if real_output
        pms = FFS.Packing.packed_size(ns, Val(true))
        size(coeffs)[1:D] == pms || throw(DimensionMismatch(
            "real_output=true expects the packed half $(pms) on the spectral dims; got $(size(coeffs)[1:D]). " *
            "Pass real_output=false for a full native spectrum $(ns)."))
        scratch = KA.allocate(dev, Complex{RT}, pms..., batch...)
        copyto!(scratch, coeffs)
        iflag < 0 && (scratch .= conj.(scratch))       # a real field's iflag=-1 half is conjugated
        outd = KA.allocate(dev, RT, ns..., batch...)
        LA.mul!(outd, AbstractFFTs.plan_brfft(scratch, ns[1], 1:D), scratch)
        return FFS._to_host(outd)
    end
    size(coeffs)[1:D] == ns || throw(DimensionMismatch(
        "real_output=false expects the full native spectrum $(ns) on the spectral dims; got $(size(coeffs)[1:D])."))
    inb = KA.allocate(dev, Complex{RT}, ns..., batch...)
    copyto!(inb, coeffs)
    outb = similar(inb)
    bwd = iflag == 1 ? AbstractFFTs.plan_bfft(inb, 1:D) : AbstractFFTs.plan_fft(inb, 1:D)
    LA.mul!(outb, bwd, inb)
    return FFS._to_host(outb)
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
    wlat = FFS.Grids._sht_weights(g, nlat; sampling = sampling, weights = weights, lmax = lmax)
    wθ = FT.(wlat) .* dφ
    # The longitude transform consumes complete rows and is linear in the field, so an inactive cell is
    # left out by zeroing its datum before it is staged. A real field needs only the nonnegative bins
    # `m = 0:nlon÷2`, its `-m` bins being their conjugates; a complex field's are independent, so it takes
    # the full native set.
    CT = FFS.sph_coeff_type(eltype(field), FT)
    fld = KA.allocate(dev, CT <: Complex ? Complex{FT} : FT, nlon, nlat, B)
    copyto!(fld, FFS.Grids._zeroed_inactive(field, g))
    fhat = KA.allocate(dev, Complex{FT}, CT <: Complex ? nlon : nlon ÷ 2 + 1, nlat, B)
    LA.mul!(fhat, CT <: Complex ? AbstractFFTs.plan_fft(fld, 1) : AbstractFFTs.plan_rfft(fld, 1), fld)
    θd = KA.allocate(dev, FT, nlat); copyto!(θd, θ)
    wd = KA.allocate(dev, FT, nlat); copyto!(wd, wθ)
    coeffs = KA.zeros(dev, CT, Nθc, Nφc, B)
    _gpu_sht_legendre_kernel!(dev)(coeffs, fhat, θd, wd, lmax, nlat, B; ndrange = Nθc * Nφc * B)
    KA.synchronize(dev)
    batch = ntuple(i -> size(field, 2 + i), ndims(field) - 2)  # structured spherical: 2 spatial dims
    return reshape(FFS._to_host(coeffs), Nθc, Nφc, batch...), (0:lmax, -lmax:lmax)
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
            # the FFT bins of frequency ±|m|. For a real field the `-|m|` bin is the conjugate of the
            # `+|m|` one, so `Σ f cos(mλ)` is `Re` of a single bin and `Σ f sin(mλ)` is `−Im`; a complex
            # field's two bins are independent, giving `(g+ĝ)/2` and `i(g-ĝ)/2` with `ĝ` the `-|m|` bin,
            # which native order holds at `nbins − |m| + 1`.
            sfac = m == 0 ? one(FT) : (isodd(abs_m) ? -sqrt(FT(2)) : sqrt(FT(2)))
            bin = abs_m + 1
            nbins = size(fhat, 1)
            acc = zero(CT)
            @inbounds for iθ in 1:Nθ
                xj = cos(θ[iθ])
                sj = sin(θ[iθ])
                P = _sht_normalized_legendre(l, abs_m, xj, sj)
                fv = fhat[bin, iθ, b]                          # longitude-FFT: bin along dim 1, latitude dim 2
                if CT <: Complex
                    ĝ = fhat[abs_m == 0 ? 1 : nbins - abs_m + 1, iθ, b]
                    φc = m >= 0 ? (fv + ĝ) / 2 : im * (fv - ĝ) / 2
                else
                    φc = m >= 0 ? real(fv) : -imag(fv)
                end
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
