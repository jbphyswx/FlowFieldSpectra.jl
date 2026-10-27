module FlowFieldSpectraKernelAbstractionsAbstractFFTsExt

using KernelAbstractions: KernelAbstractions as KA, @kernel, @index, @Const
using AbstractFFTs: AbstractFFTs
using LinearAlgebra: LinearAlgebra as LA
using FlowFieldSpectra: FlowFieldSpectra as FFS
using ComputationalBackends: ComputationalBackends
using SpectralBackends: SpectralBackends
using FlowGeometries: FlowGeometries

# Device arrays take FFS's FFT plans (`FlowFieldSpectraAbstractFFTsExt`) through the AbstractFFTs
# implementation of their array type (CUDA's CUFFT for a `CuArray`; `KA.CPU()` allocates host arrays).
FFS._available(::FFS._DeviceFFT) = true

# =============================================================================
# Device-generic spherical harmonic transform on a structured sphere (FSHTSpectralBackend × GPUBackend).
# The projection C[l,m] = Σ_p Y_lm w_p f_p factorizes on a tensor-product (φ-axis × latitude) grid with
# longitudes λ_j = λ₀ + sλ·2πj/nlon: C[l,m] = factor(m)·Σ_iθ w_θ[iθ]·P̄_l|m|(θ_iθ)·f̂[iθ, m], where
# f̂[iθ, m] = Σ_j f[j,iθ] e^{-imλ_j} is e^{-imλ₀} times bin sλ·m of an FFT over longitude (through the
# array type's AbstractFFTs implementation), computed once per (θ, m) and reused across degrees l. The
# values P̄_l|m|(θ_iθ) are a table the plan fills on the device. It uses the grid's colatitudes and
# quadrature weights, as the direct sum does, so it is exact on a Gauss–Legendre grid with
# `sampling=GaussLegendreSampling()`. The field is `(nlon, nlat, batch…)`; longitude is dim 1. Same
# normalized-Legendre / factor / m↔col convention as the direct-sum kernel.
# =============================================================================

"""
    GPUSHTPlan{FT,CT}

Reusable device spherical-harmonic transform on a structured `(nlon, nlat)` sphere whose longitude axis
is a `FlowGeometries.Axes.PeriodicAxis` (`Grids._longitude_axis`): the device field and longitude-bin
buffers with the longitude FFT planned on them, the Legendre values at every latitude row
(`SphericalKernels.device_legendre_table`) and the quadrature weights, the activity mask, the coefficient
buffer, and the longitude origin `λ0` and step sign `sλ`, built once for a fixed `lmax` and batch shape.
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
    λ0::FT
    sλ::Int
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
    λ0, sλ = FFS.Grids._longitude_axis(g, FT)
    nlon = length(FlowGeometries.Grids.coordinates(g, 1))
    φlat = collect(FT, FlowGeometries.Grids.coordinates(g, 2)) # geographic latitude, nlat
    nlat = length(φlat)
    dφ = FT(2π) / nlon
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
            typeof(mask), typeof(cd), typeof(ks)}(fld, fhat, fwd, leg, wd, mask, cd, lmax, nlon, nlat, λ0, sλ, bt,
            B, ks)
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
        plan.B, plan.λ0, plan.sλ; ndrange = length(plan.cd))
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
    coeffs = FFS._allocate_coefficients_like(field, plan)
    FFS.calculate_spectrum!(coeffs, plan, field)
    return coeffs, plan.ks
end

# =============================================================================
# The device synthesis on the same grids. Real spherical harmonics make each latitude row a trigonometric
# series in longitude, `f(λ) = A₀ + Σ_{M≥1} [A_M cos(Mλ) + B_M sin(Mλ)]` with `A_M = s(M) Σ_l C_{l,M} P̄_lM`
# and `B_M = s(M) Σ_l C_{l,−M} P̄_lM`. On longitudes `λ_j = λ₀ + sλ·2πj/nlon` that series is the backward
# DFT of the bins `F_k = Σ_{sλM ≡ k} (A_M − iB_M)/2 · e^{iMλ₀} + Σ_{−sλM ≡ k} (A_M + iB_M)/2 · e^{−iMλ₀}`
# (`A₀` once at `k = 0`), the congruences mod `nlon`, so every degree evaluates exactly whatever `nlon` is.
# =============================================================================

"""
    GPUSHTSynthesisPlan{FT,R}

Reusable device spherical-harmonic synthesis on a structured `(nlon, nlat)` sphere whose longitude axis
is a `FlowGeometries.Axes.PeriodicAxis` (`Grids._longitude_axis`): the device coefficients, the longitude
bins each latitude row reduces to with the backward longitude FFT planned on them, the device field, the
Legendre values at every latitude row (`SphericalKernels.device_legendre_table`), and the longitude origin
`λ0` and step sign `sλ`. `R` marks a real field. Execute with `synthesize!(out, plan, coeffs)`.
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
    λ0::FT
    sλ::Int
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
    λ0, sλ = FFS.Grids._longitude_axis(g, FT)
    nlon = length(FlowGeometries.Grids.coordinates(g, 1))
    φlat = collect(FT, FlowGeometries.Grids.coordinates(g, 2))
    nlat = length(φlat)
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
            typeof(leg), typeof(sp)}(cd, ghat, fld, bwd, leg, lmax, nlon, nlat, λ0, sλ, sp, bt, B)
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
    _gpu_sht_synth_kernel!(dev)(p.ghat, p.cd, p.leg, p.lmax, p.nlat, p.nlon, p.B, p.λ0, p.sλ;
        ndrange = length(p.ghat))
    KA.synchronize(dev)
    LA.mul!(p.fld, p.bwd, p.ghat)
    copyto!(out, p.fld)
    return out
end

# Bin `j` of latitude row `iθ`, slice `b`: every order whose positive or negative frequency falls in bin
# `j` (frequency `±M` at bin `±sλ·M` modulo `nlon`), each order's degree sum read against the Legendre table
# and its longitude origin `e^{±iMλ₀}` applied.
@kernel function _gpu_sht_synth_kernel!(ghat, @Const(cd), @Const(leg), lmax::Int, Nθ::Int, nlon::Int,
        B::Int, λ0, sλ::Int)
    idx = @index(Global)
    nbins = size(ghat, 1)
    if idx <= nbins * Nθ * B
        j = (idx - 1) % nbins
        rest = (idx - 1) ÷ nbins
        iθ = rest % Nθ + 1
        b = rest ÷ Nθ + 1
        acc = zero(eltype(ghat))
        for M in 0:lmax
            pos = mod(sλ * M, nlon) == j
            neg = M > 0 && mod(-sλ * M, nlon) == j
            (pos || neg) || continue
            A, Bm = _order_sums(cd, leg, M, iθ, lmax, b)
            if M == 0
                acc += A
            else
                pos && (acc += (A - im * Bm) / 2 * cis(M * λ0))
                neg && (acc += (A + im * Bm) / 2 * cis(-M * λ0))
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

# FFT bin `m` of a row, `Σ_j f_j e^{-2πimj/nlon}`, read from `fhat` at any integer `m`, `m` and `m mod nlon`
# taking the same samples. `fhat` holds all `nlon` bins in FFT order, or a real field's `nlon ÷ 2 + 1`
# nonnegative ones, whose bin `-q` is the conjugate of bin `q`.
@inline function _lon_bin(fhat, m::Int, nlon::Int, iθ::Int, b::Int)
    r = mod(m, nlon)
    size(fhat, 1) == nlon && return @inbounds fhat[r + 1, iθ, b]
    return r < size(fhat, 1) ? (@inbounds fhat[r + 1, iθ, b]) : conj(@inbounds fhat[nlon - r + 1, iθ, b])
end

# The longitude sum `Σ_j f_j e^{-imλ_j}` of a row whose longitudes are `λ₀ + sλ·2πj/nlon`.
@inline _lon_sum(fhat, m::Int, nlon::Int, iθ::Int, b::Int, λ0, sλ::Int) =
    _lon_bin(fhat, sλ * m, nlon, iθ, b) * cis(-m * λ0)

@kernel function _gpu_sht_legendre_kernel!(coeffs, @Const(fhat), @Const(leg), @Const(w),
        lmax::Int, Nθ::Int, nlon::Int, B::Int, λ0, sλ::Int)
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
                fv = _lon_sum(fhat, abs_m, nlon, iθ, b, λ0, sλ)
                if CT <: Complex
                    ĝ = _lon_sum(fhat, -abs_m, nlon, iθ, b, λ0, sλ)
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
