module FlowFieldSpectraKernelAbstractionsAbstractFFTsExt

using KernelAbstractions: KernelAbstractions as KA, @kernel, @index, @Const
using AbstractFFTs: AbstractFFTs
using LinearAlgebra: LinearAlgebra as LA
using FlowFieldSpectra: FlowFieldSpectra as FFS
using FlowTransformBindings: FlowTransformBindings as FTB
using ComputationalBackends: ComputationalBackends
using SpectralBackends: SpectralBackends
using FlowGeometries: FlowGeometries

# =============================================================================
# GPU FFT for uniform structured Cartesian grids on any KernelAbstractions device, packed-native. The
# transform is reached through `AbstractFFTs.plan_rfft`/`plan_fft`, which dispatches on the device array
# type (FFTW-backed on `KA.CPU()`, the vendor library on a device array), so this is device-generic. A
# real field takes the device r2c path and its coefficients are the packed half `(N_1÷2+1, N_2…, batch…)`
# in native order; a complex field gives the full native spectrum. Neither needs device scalar indexing.
# The field is staged once, transformed over dims 1:D (batch rides along), scaled, and copied out. FFT is
# a full transform, so `ms == size(grid)`.
# =============================================================================

struct GPUFFTRealPlan{RT, D, NB, P, IN, HB, KS} <: FFS.AbstractSpectralPlan
    batch::NTuple{NB, Int}
    fwd::P                        # device rfft plan over dims 1:D of a real (N…, batch…)
    inbuf::IN                     # device real (N…, batch…) input buffer
    half::HB                      # device complex packed half (N_1÷2+1, N_2…, batch…)
    ns::NTuple{D, Int}
    norm::RT
    neg::Bool                     # `rfft` carries the +1 sign; a real field's −1 transform is its conjugate
    ks_phys::KS
end

struct GPUFFTComplexPlan{RT, D, NB, P, CB, KS} <: FFS.AbstractSpectralPlan
    batch::NTuple{NB, Int}
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
    bt = NTuple{length(batch), Int}(batch)
    return GPUFFTRealPlan{RT, D, length(bt), typeof(fwd), typeof(inbuf), typeof(half), typeof(ks)}(
        bt, fwd, inbuf, half, ns, one(RT) / prod(ns), iflag < 0, ks)
end

# Complex field: device c2c, full native order.
function _gpu_fft_plan(dev, ::Type{Complex{RT}}, g, ns::NTuple{D, Int}, batch::Tuple, iflag::Int) where {RT<:Real, D}
    inbuf = KA.allocate(dev, Complex{RT}, ns..., batch...)
    fill!(inbuf, zero(Complex{RT}))
    outbuf = similar(inbuf)
    fwd = iflag == 1 ? AbstractFFTs.plan_fft(inbuf, 1:D) : AbstractFFTs.plan_bfft(inbuf, 1:D)
    ks = FFS.Grids.physical_wavenumbers(g, ns, Val(false))
    bt = NTuple{length(batch), Int}(batch)
    return GPUFFTComplexPlan{RT, D, length(bt), typeof(fwd), typeof(inbuf), typeof(ks)}(
        bt, fwd, inbuf, outbuf, ns, one(RT) / prod(ns), ks)
end

Base.show(io::IO, p::GPUFFTRealPlan{RT, D}) where {RT, D} =
    print(io, "GPUFFTRealPlan{", RT, ", ", D, "}(", p.ns, ", batch=", p.batch, ")")
Base.show(io::IO, p::GPUFFTComplexPlan{RT, D}) where {RT, D} =
    print(io, "GPUFFTComplexPlan{", RT, ", ", D, "}(", p.ns, ", batch=", p.batch, ")")

FFS.Plans.coefficient_size(p::GPUFFTRealPlan) =
    (FTB.packed_size(p.ns, Val(true))..., p.batch...)
FFS.Plans.coefficient_type(::GPUFFTRealPlan{RT}) where {RT} = Complex{RT}
FFS.Plans.wavenumbers(p::GPUFFTRealPlan) = p.ks_phys

FFS.Plans.coefficient_size(p::GPUFFTComplexPlan) = (p.ns..., p.batch...)
FFS.Plans.coefficient_type(::GPUFFTComplexPlan{RT}) where {RT} = Complex{RT}
FFS.Plans.wavenumbers(p::GPUFFTComplexPlan) = p.ks_phys

"""
    calculate_spectrum!(coeffs, plan, field) -> ks_phys

Execute a prebuilt device FFT plan in place. `field` (host or device) is staged into the device buffer,
transformed (`GPUFFTRealPlan` → packed half, `GPUFFTComplexPlan` → full native), scaled by `1/∏N`, and
copied into `coeffs` (device or host). No device scalar indexing.
"""
function FFS.calculate_spectrum!(coeffs::AbstractArray{Complex{RT}}, plan::GPUFFTRealPlan{RT, D}, field) where {RT, D}
    FFS.Plans._check_coefficients(coeffs, plan)
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
    FFS.Plans._check_coefficients(coeffs, plan)
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
# The hybrid composite's FFT over the uniform axes on a device. `AbstractFFTs.plan_rfft`/`plan_fft`
# dispatch on the device array type, so the pass is device-generic like the plans above, and its output
# stays on the device for the NUFFT passes that follow.
# =============================================================================

function FFS._region_fft_plan(exec::ComputationalBackends.AbstractGPUBackend, ::Type{T}, insize::Tuple,
        dims::Tuple, halve::Bool, conj_in::Bool, measure::Bool) where {T}
    RT = real(float(T))
    # `rfft` halves the first transformed dim, which the composite requires to be axis 1.
    outsize = halve ? (insize[1] ÷ 2 + 1, Base.tail(insize)...) : insize
    scratch = KA.zeros(exec.backend, halve ? RT : Complex{RT}, insize...)
    fwd = halve ? AbstractFFTs.plan_rfft(scratch, dims) : AbstractFFTs.plan_fft(scratch, dims)
    return FFS.RegionFFT(fwd, scratch, outsize, halve, conj_in)
end

# The hybrid synthesis's backward FFT over the uniform axes `dims` of the device native spectrum `A`, in
# place: `bfft` for `iflag = 1` and `fft` for `-1`.
FFS._region_bfft_plan(::ComputationalBackends.AbstractGPUBackend, A::AbstractArray, dims::Tuple, iflag::Int,
        measure::Bool) =
    iflag == 1 ? AbstractFFTs.plan_bfft!(A, dims) : AbstractFFTs.plan_fft!(A, dims)

function FFS._calculate_spectrum_gpu_fft(exec::ComputationalBackends.AbstractGPUBackend,
        g::FlowGeometries.Grids.AbstractStructuredGrid{<:FlowGeometries.Geometry.AbstractCartesianGeometry},
        field::AbstractArray, ms::Tuple; iflag::Int = 1, kwargs...)
    ns, batch = _gpu_fft_setup(g, field, ms)
    T = eltype(field)
    RT = real(float(T))
    ET = T <: Real ? float(T) : Complex{RT}
    plan = _gpu_fft_plan(exec.backend, ET, g, ns, batch, iflag)
    pms = FTB.packed_size(ns, Val(T <: Real))
    coeffs_dev = KA.allocate(exec.backend, Complex{RT}, pms..., batch...)
    ks = FFS.calculate_spectrum!(coeffs_dev, plan, field)
    return FFS._to_host(coeffs_dev), ks
end

# The forward's split on a device: all-uniform takes the device FFT plan, uniform-in-some-directions the
# hybrid composite on device arrays.
function FFS.plan_spectrum(::SpectralBackends.AbstractFFTSpectralBackend, exec::ComputationalBackends.AbstractGPUBackend,
        g::FlowGeometries.Grids.AbstractStructuredGrid{<:FlowGeometries.Geometry.AbstractCartesianGeometry},
        ::Type{T}, ms::NTuple{D, Int}; batch::Tuple = (), iflag::Int = 1,
        nufft::SpectralBackends.AbstractSpectralBackend = SpectralBackends.AutoSpectralBackend(),
        kwargs...) where {T, D}
    umask = ntuple(d -> FlowGeometries.Grids.isuniform(g, d), Val(D))
    if !all(umask)
        any(umask) || FFS._all_stretched_fft()
        return FFS._hybrid_plan(nufft, exec, g, T, ms, umask; batch = batch, iflag = iflag, kwargs...)
    end
    ns = size(g)
    Tuple(ms) == ns || throw(ArgumentError("FFTSpectralBackend requires ms == size(grid) = $ns (got $(Tuple(ms)))"))
    ET = T <: Real ? float(T) : Complex{real(float(T))}
    return _gpu_fft_plan(exec.backend, ET, g, ns, batch, iflag)
end

# =============================================================================
# Inverse (synthesis) on the device, reached through `AbstractFFTs.plan_brfft`/`plan_bfft` so it stays
# device-generic. The forward's `1/∏N` cancels the unnormalized backward transform, so neither direction
# rescales here. A real field's half is consumed by the c2r transform, so it is staged into a scratch
# buffer and `coeffs` is left intact. The FFTW extension serves the host execution backends.
# =============================================================================

# Reusable device synthesis: the backward plan and both device buffers, built once.
struct GPUFFTSynthesisPlan{RT, D, NB, R, P, SC, OB} <: FFS.AbstractSynthesisPlan
    bwd::P
    scratch::SC                   # the coefficients staged on device; the c2r consumes them
    outd::OB                      # the device field the transform writes
    ns::NTuple{D, Int}
    batch::NTuple{NB, Int}
    neg::Bool
end

Base.show(io::IO, ::GPUFFTSynthesisPlan{RT, D, NB, R}) where {RT, D, NB, R} =
    print(io, "GPUFFTSynthesisPlan{$RT, $D}(", R ? "real" : "complex", ")")

FFS.Plans.field_size(p::GPUFFTSynthesisPlan) = (p.ns..., p.batch...)
FFS.Plans.field_type(::GPUFFTSynthesisPlan{RT, D, NB, R}) where {RT, D, NB, R} = R ? RT : Complex{RT}

# A grid uniform in some directions and stretched in others takes the hybrid composite's synthesis.
function FFS.Plans.plan_synthesis(::SpectralBackends.AbstractFFTSpectralBackend,
        exec::ComputationalBackends.AbstractGPUBackend,
        g::FlowGeometries.Grids.AbstractStructuredGrid{<:FlowGeometries.Geometry.AbstractCartesianGeometry},
        ::Type{T}, ms::NTuple{D, Int}; batch::Tuple = (), iflag::Int = 1,
        nufft::SpectralBackends.AbstractSpectralBackend = SpectralBackends.AutoSpectralBackend(),
        kwargs...) where {T, D}
    umask = ntuple(d -> FlowGeometries.Grids.isuniform(g, d), Val(D))
    if !all(umask)
        any(umask) || FFS._all_stretched_fft()
        return FFS._hybrid_synthesis_plan(nufft, exec, g, T, ms, umask; batch = batch, iflag = iflag, kwargs...)
    end
    ns = size(g)
    Tuple(ms) == ns || throw(ArgumentError(
        "FFTSpectralBackend requires ms == size(grid) = $ns (got $(Tuple(ms))); FFT is a full transform."))
    dev = exec.backend
    RT = real(float(T))
    R = T <: Real
    bt = NTuple{length(batch), Int}(batch)
    if R
        scratch = KA.allocate(dev, Complex{RT}, FTB.packed_size(ns, Val(true))..., bt...)
        outd = KA.allocate(dev, RT, ns..., bt...)
        bwd = AbstractFFTs.plan_brfft(scratch, ns[1], 1:D)
    else
        scratch = KA.allocate(dev, Complex{RT}, ns..., bt...)
        outd = KA.allocate(dev, Complex{RT}, ns..., bt...)
        bwd = iflag == 1 ? AbstractFFTs.plan_bfft(scratch, 1:D) : AbstractFFTs.plan_fft(scratch, 1:D)
    end
    return GPUFFTSynthesisPlan{RT, D, length(bt), R, typeof(bwd), typeof(scratch), typeof(outd)}(
        bwd, scratch, outd, ns, bt, iflag < 0)
end

function FFS.Plans.synthesize!(out::AbstractArray, plan::GPUFFTSynthesisPlan{RT, D, NB, R},
        coeffs::AbstractArray) where {RT, D, NB, R}
    size(out) == FFS.Plans.field_size(plan) || throw(DimensionMismatch(
        "out is $(size(out)); this plan writes $(FFS.Plans.field_size(plan))"))
    size(coeffs) == size(plan.scratch) || throw(DimensionMismatch(
        "coeffs is $(size(coeffs)); this plan was built for $(size(plan.scratch))"))
    copyto!(plan.scratch, coeffs)
    R && plan.neg && (plan.scratch .= conj.(plan.scratch))   # a real field's iflag=-1 half is conjugated
    LA.mul!(plan.outd, plan.bwd, plan.scratch)
    copyto!(out, plan.outd)
    return out
end

# =============================================================================
# Device-generic spherical harmonic transform on a structured sphere (FSHTSpectralBackend × GPUBackend).
# The projection C[l,m] = Σ_p Y_lm w_p f_p factorizes on a tensor-product (φ-axis × latitude) grid with
# uniform longitude: C[l,m] = factor(m)·Σ_iθ w_θ[iθ]·P̄_l|m|(θ_iθ)·f̂[iθ, m], where
# f̂[iθ, m] = Σ_iφ f[iφ,iθ] e^{-im(iφ-1)Δφ} is an FFT over longitude (device-generic via AbstractFFTs —
# CUFFT / FFTW / rocFFT), computed once per (θ, m) and reused across degrees l. The values P̄_l|m|(θ_iθ)
# are a table the plan fills on the device. It uses the grid's colatitudes and quadrature weights, as the
# direct sum does, so it is exact on a Gauss–Legendre grid with `sampling=GaussLegendreSampling()`. The
# field is `(nlon, nlat, batch…)`; longitude is dim 1. Same normalized-Legendre / factor / m↔col
# convention as the direct-sum kernel.
# =============================================================================

"""
    GPUSHTPlan{FT,CT}

Reusable device spherical-harmonic transform on a structured `(nlon, nlat)` sphere whose longitudes are
uniform from 0: the device field and longitude-bin buffers with the longitude FFT planned on them, the
Legendre values at every latitude row (`SphericalKernels.device_legendre_table`) and the quadrature
weights, the activity mask, and the coefficient buffer, built once for a fixed `lmax` and batch shape.
`CT` is the coefficient element type, real for a real field. Execute with
`calculate_spectrum!(coeffs, plan, field)`.
"""
struct GPUSHTPlan{FT, CT, NB, FB, HB, P, LT, V, MK, CB, KS} <: FFS.AbstractSpectralPlan
    fld::FB            # device (nlon, nlat, B) field, real for a real field
    fhat::HB           # device longitude bins: `nlon ÷ 2 + 1` for a real field, `nlon` for a complex one
    fwd::P             # the longitude FFT, planned on `fld`
    leg::LT            # device packed Legendre table `(legendre_count(lmax), nlat)`
    w::V               # device quadrature weight per row, times 2π/nlon
    mask::MK           # device activity mask `(nlon, nlat)`, or `nothing` with every cell active
    cd::CB             # device (lmax + 1, 2lmax + 1, B) coefficients
    lmax::Int
    nlon::Int
    nlat::Int
    batch::NTuple{NB, Int}
    B::Int
    ks::KS
end

Base.show(io::IO, p::GPUSHTPlan{FT, CT}) where {FT, CT} =
    print(io, "GPUSHTPlan{", FT, "}(lmax=", p.lmax, ", ", CT <: Real ? "real" : "complex", ", B=", p.B,
        ")")

FFS.Plans.coefficient_size(p::GPUSHTPlan) = (p.lmax + 1, 2 * p.lmax + 1, p.batch...)
FFS.Plans.coefficient_type(::GPUSHTPlan{FT, CT}) where {FT, CT} = CT
FFS.Plans.wavenumbers(p::GPUSHTPlan) = p.ks

_device_mask(dev, ::FlowGeometries.Grids.AllActive) = nothing
function _device_mask(dev, m::AbstractArray{Bool})
    d = KA.allocate(dev, Bool, size(m)...)
    copyto!(d, Array{Bool}(m))
    return d
end

function FFS.plan_spectrum(::SpectralBackends.AbstractFSHTSpectralBackend,
        exec::ComputationalBackends.AbstractGPUBackend,
        g::FlowGeometries.Grids.AbstractStructuredGrid{<:FlowGeometries.Geometry.AbstractSphericalGeometry},
        ::Type{T}, ms::Tuple; batch::Tuple = (), sampling = nothing, weights = nothing,
        inactive::FFS.AbstractInactiveCells = FFS.ZeroFill(), kwargs...) where {T}
    FT = real(float(T))
    dev = exec.backend
    (length(ms) == 2 && ms[1] >= 1) || throw(ArgumentError(
        "the spherical-harmonic transform takes ms = (lmax + 1, 2lmax + 1); got $ms"))
    weights = FFS.Grids._inactive_weights(inactive, g, FT; sampling, weights)
    lmax = ms[1] - 1
    λ = collect(FT, FlowGeometries.Grids.coordinates(g, 1))    # longitude, nlon
    φlat = collect(FT, FlowGeometries.Grids.coordinates(g, 2)) # geographic latitude, nlat
    nlon = length(λ)
    nlat = length(φlat)
    dφ = FT(2π) / nlon
    # The real-SH longitude transform (cos/sin from the FFT) assumes longitudes uniform from 0.
    (isapprox(λ[begin], zero(FT); atol = sqrt(eps(FT))) &&
     (nlon < 2 || isapprox(λ[begin + 1] - λ[begin], dφ; atol = sqrt(eps(FT))))) ||
        throw(ArgumentError("GPU SHT needs longitudes uniform from 0 with step 2π/nlon."))
    FFS.Grids._warn_longitude_bandlimit(nlon, lmax)
    θ = (FT(π) / 2) .- φlat                                    # colatitude per latitude index
    wlat = FFS.Grids._sht_weights(g, nlat; sampling = sampling, weights = weights, lmax = lmax)
    wθ = FT.(wlat) .* dφ
    bt = NTuple{length(batch), Int}(batch)
    B = prod(bt; init = 1)
    # A real field needs only the nonnegative bins `m = 0:nlon÷2`, its `-m` bins being their conjugates;
    # a complex field's are independent, so it takes the full native set.
    CT = FFS.sph_coeff_type(T, FT)
    R = CT <: Real
    fld = KA.allocate(dev, R ? FT : Complex{FT}, nlon, nlat, B)
    fill!(fld, zero(eltype(fld)))
    fwd = R ? AbstractFFTs.plan_rfft(fld, 1) : AbstractFFTs.plan_fft(fld, 1)
    fhat = KA.allocate(dev, Complex{FT}, R ? nlon ÷ 2 + 1 : nlon, nlat, B)
    leg = FFS.SphericalKernels.device_legendre_table(dev, FT, lmax, θ)
    wd = KA.allocate(dev, FT, nlat)
    copyto!(wd, wθ)
    mask = _device_mask(dev, FlowGeometries.Grids.mask(g))
    cd = KA.zeros(dev, CT, lmax + 1, 2 * lmax + 1, B)
    ks = (0:lmax, -lmax:lmax)
    return GPUSHTPlan{FT, CT, length(bt), typeof(fld), typeof(fhat), typeof(fwd), typeof(leg), typeof(wd),
            typeof(mask), typeof(cd), typeof(ks)}(fld, fhat, fwd, leg, wd, mask, cd, lmax, nlon, nlat, bt, B, ks)
end

"""
    calculate_spectrum!(coeffs, plan::GPUSHTPlan, field) -> ks

Fill `coeffs` `(lmax + 1, 2lmax + 1, batch…)` (host or device) with the spherical-harmonic projection of
`field` `(nlon, nlat, batch…)` (host or device), reusing the plan's device buffers and FFT plan.
"""
function FFS.calculate_spectrum!(coeffs::AbstractArray{<:Number}, plan::GPUSHTPlan{FT, CT},
        field) where {FT, CT}
    FFS.Plans._check_coefficients(coeffs, plan)
    length(field) == length(plan.fld) || throw(DimensionMismatch(
        "field holds $(length(field)) values; this plan transforms $(plan.B) field(s) of " *
        "$(plan.nlon * plan.nlat) points — pass the matching `batch=` to plan_spectrum"))
    (CT <: Real && eltype(field) <: Complex) && throw(ArgumentError(
        "this plan was built for a real field, whose coefficients are real; build one for " *
        "$(eltype(field)) to transform a complex field"))
    copyto!(plan.fld, field)
    # The longitude transform consumes complete rows and is linear in the field, so an inactive cell is
    # left out by zeroing its datum; `ifelse` also clears the `NaN` masked data commonly holds.
    plan.mask === nothing || (plan.fld .= ifelse.(plan.mask, plan.fld, zero(eltype(plan.fld))))
    LA.mul!(plan.fhat, plan.fwd, plan.fld)
    dev = KA.get_backend(plan.cd)
    _gpu_sht_legendre_kernel!(dev)(plan.cd, plan.fhat, plan.leg, plan.w, plan.lmax, plan.nlat, plan.nlon,
        plan.B; ndrange = length(plan.cd))
    KA.synchronize(dev)
    copyto!(coeffs, plan.cd)
    return plan.ks
end

function FFS._calculate_spectrum_gpu_sht(exec::ComputationalBackends.AbstractGPUBackend,
        g::FlowGeometries.Grids.AbstractStructuredGrid{<:FlowGeometries.Geometry.AbstractSphericalGeometry},
        field::AbstractArray, ms::Tuple; sampling = nothing, weights = nothing,
        inactive::FFS.AbstractInactiveCells = FFS.ZeroFill(), kwargs...)
    FT = eltype(g)
    plan = FFS.plan_spectrum(SpectralBackends.FSHTSpectralBackend(), exec, g,
        eltype(field) <: Real ? FT : Complex{FT}, ms; batch = FFS.Grids.field_batch_shape(g, field),
        sampling, weights, inactive)
    coeffs = FFS.allocate_coefficients(plan)
    FFS.calculate_spectrum!(coeffs, plan, field)
    return coeffs, plan.ks
end

# =============================================================================
# The device synthesis on the same grids. Real spherical harmonics make each latitude row a trigonometric
# series in longitude, `f(λ) = A₀ + Σ_{M≥1} [A_M cos(Mλ) + B_M sin(Mλ)]` with `A_M = s(M) Σ_l C_{l,M} P̄_lM`
# and `B_M = s(M) Σ_l C_{l,−M} P̄_lM`. On longitudes `λ_j = 2πj/nlon` that series is the backward DFT of
# the bins `F_k = Σ_{M ≡ k} (A_M − iB_M)/2 + Σ_{−M ≡ k} (A_M + iB_M)/2` (`A₀` once at `k = 0`), the
# congruences mod `nlon`, so every degree evaluates exactly whatever `nlon` is.
# =============================================================================

"""
    GPUSHTSynthesisPlan{FT,R}

Reusable device spherical-harmonic synthesis on a structured `(nlon, nlat)` sphere whose longitudes are
uniform from 0: the device coefficients, the longitude bins each latitude row reduces to with the
backward longitude FFT planned on them, the device field, and the Legendre values at every latitude row
(`SphericalKernels.device_legendre_table`). `R` marks a real field. Execute with
`synthesize!(out, plan, coeffs)`.
"""
struct GPUSHTSynthesisPlan{FT, R, NB, CB, HB, FB, P, LT, SP <: Tuple} <: FFS.AbstractSynthesisPlan
    cd::CB             # device (lmax + 1, 2lmax + 1, B) coefficients
    ghat::HB           # device longitude bins: `nlon ÷ 2 + 1` for a real field, `nlon` for a complex one
    fld::FB            # device (nlon, nlat, B) field
    bwd::P             # the backward longitude FFT, `ghat` → `fld`
    leg::LT            # device packed Legendre table `(legendre_count(lmax), nlat)`
    lmax::Int
    nlon::Int
    nlat::Int
    spatial::SP
    batch::NTuple{NB, Int}
    B::Int
end

Base.show(io::IO, p::GPUSHTSynthesisPlan{FT, R}) where {FT, R} =
    print(io, "GPUSHTSynthesisPlan{", FT, "}(lmax=", p.lmax, ", ", R ? "real" : "complex", ", B=", p.B, ")")

FFS.Plans.field_size(p::GPUSHTSynthesisPlan) = (p.spatial..., p.batch...)
FFS.Plans.field_type(::GPUSHTSynthesisPlan{FT, R}) where {FT, R} = R ? FT : Complex{FT}

function FFS.Plans.plan_synthesis(::SpectralBackends.AbstractFSHTSpectralBackend,
        exec::ComputationalBackends.AbstractGPUBackend,
        g::FlowGeometries.Grids.AbstractStructuredGrid{<:FlowGeometries.Geometry.AbstractSphericalGeometry},
        ::Type{T}, ms::Tuple; batch::Tuple = (), kwargs...) where {T}
    FT = real(float(eltype(g)))
    dev = exec.backend
    (length(ms) == 2 && ms[1] >= 1) || throw(ArgumentError(
        "the spherical-harmonic synthesis takes ms = (lmax + 1, 2lmax + 1); got $ms"))
    lmax = ms[1] - 1
    λ = collect(FT, FlowGeometries.Grids.coordinates(g, 1))
    φlat = collect(FT, FlowGeometries.Grids.coordinates(g, 2))
    nlon = length(λ)
    nlat = length(φlat)
    (isapprox(λ[begin], zero(FT); atol = sqrt(eps(FT))) &&
     (nlon < 2 || isapprox(λ[begin + 1] - λ[begin], FT(2π) / nlon; atol = sqrt(eps(FT))))) ||
        throw(ArgumentError("GPU SHT needs longitudes uniform from 0 with step 2π/nlon."))
    bt = NTuple{length(batch), Int}(batch)
    B = prod(bt; init = 1)
    R = T <: Real
    cd = KA.zeros(dev, R ? FT : Complex{FT}, lmax + 1, 2 * lmax + 1, B)
    ghat = KA.zeros(dev, Complex{FT}, R ? nlon ÷ 2 + 1 : nlon, nlat, B)
    fld = KA.allocate(dev, R ? FT : Complex{FT}, nlon, nlat, B)
    bwd = R ? AbstractFFTs.plan_brfft(ghat, nlon, 1) : AbstractFFTs.plan_bfft(ghat, 1)
    leg = FFS.SphericalKernels.device_legendre_table(dev, FT, lmax, (FT(π) / 2) .- φlat)
    sp = size(g)
    return GPUSHTSynthesisPlan{FT, R, length(bt), typeof(cd), typeof(ghat), typeof(fld), typeof(bwd),
            typeof(leg), typeof(sp)}(cd, ghat, fld, bwd, leg, lmax, nlon, nlat, sp, bt, B)
end

"""
    synthesize!(out, plan::GPUSHTSynthesisPlan, coeffs) -> out

Evaluate the real spherical-harmonic series `coeffs` `(lmax + 1, 2lmax + 1, batch…)` (host or device) at
the grid's nodes on the device, into `out` (host or device). A real-field plan reads the real part of
complex coefficients.
"""
function FFS.Plans.synthesize!(out::AbstractArray, p::GPUSHTSynthesisPlan{FT, R},
        coeffs::AbstractArray) where {FT, R}
    size(out) == FFS.Plans.field_size(p) || throw(DimensionMismatch(
        "out is $(size(out)); this plan writes $(FFS.Plans.field_size(p))"))
    size(coeffs) == (p.lmax + 1, 2 * p.lmax + 1, p.batch...) || throw(DimensionMismatch(
        "coeffs is $(size(coeffs)); this plan reads $((p.lmax + 1, 2 * p.lmax + 1, p.batch...))"))
    copyto!(p.cd, (R && eltype(coeffs) <: Complex) ? real.(coeffs) : coeffs)
    dev = KA.get_backend(p.ghat)
    _gpu_sht_synth_kernel!(dev)(p.ghat, p.cd, p.leg, p.lmax, p.nlat, p.nlon, p.B; ndrange = length(p.ghat))
    KA.synchronize(dev)
    LA.mul!(p.fld, p.bwd, p.ghat)
    copyto!(out, p.fld)
    return out
end

# Bin `j` of latitude row `iθ`, slice `b`: every order whose positive or negative frequency is `j` modulo
# `nlon`, each order's degree sum read against the Legendre table.
@kernel function _gpu_sht_synth_kernel!(ghat, @Const(cd), @Const(leg), lmax::Int, Nθ::Int, nlon::Int,
        B::Int)
    idx = @index(Global)
    nbins = size(ghat, 1)
    if idx <= nbins * Nθ * B
        j = (idx - 1) % nbins
        rest = (idx - 1) ÷ nbins
        iθ = rest % Nθ + 1
        b = rest ÷ Nθ + 1
        acc = zero(eltype(ghat))
        for M in 0:lmax
            pos = mod(M, nlon) == j
            neg = M > 0 && mod(-M, nlon) == j
            (pos || neg) || continue
            A, Bm = _order_sums(cd, leg, M, iθ, lmax, b)
            if M == 0
                acc += A
            else
                pos && (acc += (A - im * Bm) / 2)
                neg && (acc += (A + im * Bm) / 2)
            end
        end
        @inbounds ghat[j + 1, iθ, b] = acc
    end
end

# `(A_M, B_M)` at latitude row `iθ`: `s(M) Σ_l C_{l,±M} P̄_lM` over `l = M … lmax`, the Legendre values
# read from the packed table, where an order's degrees are consecutive rows.
@inline function _order_sums(cd, leg, M::Int, iθ::Int, lmax::Int, b::Int)
    FT = eltype(leg)
    sfac = M == 0 ? one(FT) : (isodd(M) ? -sqrt(FT(2)) : sqrt(FT(2)))
    colp = M == 0 ? 1 : 2M + 1
    coln = 2M
    r0 = FFS.SphericalKernels.legendre_index(M, M, lmax) - M
    A = zero(eltype(cd))
    Bm = zero(eltype(cd))
    @inbounds for l in M:lmax
        P = leg[r0 + l, iθ]
        row = l - M + 1
        A += cd[row, colp, b] * P
        M > 0 && (Bm += cd[row, coln, b] * P)
    end
    return sfac * A, sfac * Bm
end

# The longitude DFT `Σ_λ f e^{-imλ}` at any integer frequency `m`, read from `fhat`: on longitudes uniform
# from 0, `m` and `m mod nlon` take the same samples. `fhat` holds all `nlon` bins in FFT order, or a real
# field's `nlon ÷ 2 + 1` nonnegative ones, whose bin `-q` is the conjugate of bin `q`.
@inline function _lon_bin(fhat, m::Int, nlon::Int, iθ::Int, b::Int)
    r = mod(m, nlon)
    size(fhat, 1) == nlon && return @inbounds fhat[r + 1, iθ, b]
    return r < size(fhat, 1) ? (@inbounds fhat[r + 1, iθ, b]) : conj(@inbounds fhat[nlon - r + 1, iθ, b])
end

@kernel function _gpu_sht_legendre_kernel!(coeffs, @Const(fhat), @Const(leg), @Const(w),
        lmax::Int, Nθ::Int, nlon::Int, B::Int)
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
            FT = eltype(leg)
            CT = eltype(coeffs)
            # Real SH (FSH convention): Y_lm = s(m)·P̄·trig, and the longitude-sum trig-projection reads
            # the FFT bins of frequency ±|m|. For a real field the `-|m|` bin is the conjugate of the
            # `+|m|` one, so `Σ f cos(mλ)` is `Re` of a single bin and `Σ f sin(mλ)` is `−Im`; a complex
            # field's two bins are independent, giving `(g+ĝ)/2` and `i(g-ĝ)/2` with `ĝ` the `-|m|` bin.
            sfac = m == 0 ? one(FT) : (isodd(abs_m) ? -sqrt(FT(2)) : sqrt(FT(2)))
            r = FFS.SphericalKernels.legendre_index(l, abs_m, lmax)
            acc = zero(CT)
            @inbounds for iθ in 1:Nθ
                P = leg[r, iθ]
                fv = _lon_bin(fhat, abs_m, nlon, iθ, b)       # longitude-FFT: bin along dim 1, latitude dim 2
                if CT <: Complex
                    ĝ = _lon_bin(fhat, -abs_m, nlon, iθ, b)
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

end # module FlowFieldSpectraKernelAbstractionsAbstractFFTsExt
