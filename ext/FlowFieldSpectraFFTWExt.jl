module FlowFieldSpectraFFTWExt

using FFTW: FFTW
using LinearAlgebra: LinearAlgebra as LA
using FlowFieldSpectra: FlowFieldSpectra as FFS
using ComputationalBackends: ComputationalBackends
using SpectralBackends: SpectralBackends
using FlowGeometries: FlowGeometries

# =============================================================================
# FFT for uniform structured Cartesian grids on the packed-native model. The field is
# `(N_1,…,N_D, batch…)`; FFTW transforms the spectral dims `1:D` and batches over the trailing dims
# natively. A real field takes the `rfft` fast path and its coefficients are the packed half
# `(N_1÷2+1, N_2…N_D, batch…)` in native (unshifted) order; a complex field gives the full native
# spectrum `(N…, batch…)`. `mul!` writes the caller's `coeffs` directly and `rmul!` applies the `1/∏N`
# normalization in place, so steady-state `calculate_spectrum!` allocates nothing and copies the input
# zero times. `FFS.unpacked` reconstructs the full centered cube when one is needed.
#
# FFT is a full transform, so `ms == size(grid)`. Physical wavenumbers use the grid's periodic length.
# =============================================================================


# Real-input plan: rfft over dims 1:D, writing the packed half directly into the caller buffer. `rfft`
# computes the `iflag = +1` sign (`Σ f e^{-ikx}`); a real field's `iflag = -1` transform is its
# conjugate, so `neg` applies that in place.
struct RFFTPlan{RT, D, NB, P, KS} <: FFS.AbstractSpectralPlan
    fwd::P                        # rfft plan over dims 1:D of a real (N…, batch…) array
    ns::NTuple{D, Int}
    batch::NTuple{NB, Int}
    norm::RT                      # 1/prod(ns)
    neg::Bool
    ks_phys::KS
end

# Complex-input plan: C2C fft (or bfft) over dims 1:D, writing the full native spectrum.
struct CFFTPlan{RT, D, NB, P, KS} <: FFS.AbstractSpectralPlan
    fwd::P                        # fft/bfft plan over dims 1:D of a complex (N…, batch…) array
    ns::NTuple{D, Int}
    batch::NTuple{NB, Int}
    norm::RT
    ks_phys::KS
end

FFS.Plans.coefficient_size(p::RFFTPlan{RT, D}) where {RT, D} =
    (FFS.Packing.packed_size(p.ns, Val(true))..., p.batch...)
FFS.Plans.coefficient_type(::RFFTPlan{RT}) where {RT} = Complex{RT}
FFS.Plans.wavenumbers(p::RFFTPlan) = p.ks_phys

FFS.Plans.coefficient_size(p::CFFTPlan) = (p.ns..., p.batch...)
FFS.Plans.coefficient_type(::CFFTPlan{RT}) where {RT} = Complex{RT}
FFS.Plans.wavenumbers(p::CFFTPlan) = p.ks_phys

# Plan builders (dispatch on real vs complex element type). `a` is the array to plan on — a fresh sample
# for a reusable plan (`MEASURE` clobbers it during planning), or the field itself for a one-shot
# (`ESTIMATE` leaves it untouched). `ks_phys` follows the packed native order.
function _fftw_plan(::Type{T}, g, ns::NTuple{D, Int}, a::AbstractArray, nthreads::Int, flags;
        iflag::Int = 1) where {T<:Real, D}
    FFTW.set_num_threads(nthreads)
    fwd = FFTW.plan_rfft(a, 1:D; flags = flags)
    ks = FFS.Grids.physical_wavenumbers(g, ns, Val(true))
    bt = _batch_of(a, Val(D))
    return RFFTPlan{T, D, length(bt), typeof(fwd), typeof(ks)}(
        fwd, ns, bt, one(T) / prod(ns), iflag < 0, ks)
end

# The batch shape the plan was built over: whatever trails the `D` spatial dims of the planning array.
_batch_of(a::AbstractArray, ::Val{D}) where {D} = ntuple(i -> size(a, D + i), max(ndims(a) - D, 0))

function _fftw_plan(::Type{Complex{RT}}, g, ns::NTuple{D, Int}, a::AbstractArray, nthreads::Int, flags;
        iflag::Int = 1) where {RT<:Real, D}
    FFTW.set_num_threads(nthreads)
    fwd = iflag == 1 ? FFTW.plan_fft(a, 1:D; flags = flags) : FFTW.plan_bfft(a, 1:D; flags = flags)
    ks = FFS.Grids.physical_wavenumbers(g, ns, Val(false))
    bt = _batch_of(a, Val(D))
    return CFFTPlan{RT, D, length(bt), typeof(fwd), typeof(ks)}(fwd, ns, bt, one(RT) / prod(ns), ks)
end

"""
    calculate_spectrum!(coeffs, plan, field) -> ks_phys

Execute a prebuilt real-input `RFFTPlan` in place: `rfft` reads the real `field` directly and writes the
packed half `(N_1÷2+1, N_2…N_D, batch…)` into `coeffs`, scaled by `1/∏N`. Allocation-free and copy-free
in steady state.
"""
function FFS.calculate_spectrum!(coeffs::AbstractArray{Complex{RT}}, plan::RFFTPlan{RT, D}, field) where {RT, D}
    LA.mul!(coeffs, plan.fwd, field)
    LA.rmul!(coeffs, plan.norm)
    plan.neg && (coeffs .= conj.(coeffs))
    return plan.ks_phys
end

"""
    calculate_spectrum!(coeffs, plan, field) -> ks_phys

Execute a prebuilt complex-input `CFFTPlan` in place: writes the full native spectrum `(N…, batch…)`
into `coeffs`, scaled by `1/∏N`.
"""
function FFS.calculate_spectrum!(coeffs::AbstractArray{Complex{RT}}, plan::CFFTPlan{RT, D}, field) where {RT, D}
    LA.mul!(coeffs, plan.fwd, field)
    LA.rmul!(coeffs, plan.norm)
    return plan.ks_phys
end

# Validate FFT applicability + split off the batch shape.
@inline function _fft_setup(g::FlowGeometries.Grids.AbstractStructuredGrid{<:FlowGeometries.Geometry.AbstractCartesianGeometry}, field::AbstractArray, ms::Tuple)
    ns = size(g)
    Tuple(ms) == ns || throw(ArgumentError(
        "FFTSpectralBackend requires ms == size(grid) = $ns (got $(Tuple(ms))); FFT is a full transform. " *
        "Use NUFFT/DirectSum for a different mode count."))
    D = length(ns)
    ndims(field) >= D || throw(DimensionMismatch("field has $(ndims(field)) dims, grid needs $D spatial"))
    batch = ntuple(i -> size(field, D + i), ndims(field) - D)
    return ns, batch
end

# Same split the forward makes: all-uniform takes the pure FFTW plan, uniform-in-some-directions takes
# the hybrid composite plan, all-stretched raises pointing at NUFFT.
function FFS.plan_spectrum(::SpectralBackends.AbstractFFTSpectralBackend, exec::ComputationalBackends.AbstractExecutionBackend,
        g::FlowGeometries.Grids.AbstractStructuredGrid{<:FlowGeometries.Geometry.AbstractCartesianGeometry}, ::Type{T}, ms::NTuple{D, Int};
        batch::Tuple = (), iflag::Int = 1,
        nufft::SpectralBackends.AbstractSpectralBackend = SpectralBackends.AutoSpectralBackend(),
        kwargs...) where {T, D}
    umask = ntuple(d -> FlowGeometries.Grids.isuniform(g, d), Val(D))
    if !all(umask)
        any(umask) || throw(ArgumentError(
            "FFTSpectralBackend needs a uniform axis in at least one direction (an AbstractRange); " *
            "every axis of this grid is stretched. Use NUFFTSpectralBackend."))
        return FFS._hybrid_plan(nufft, exec, g, T, ms, umask; batch = batch, iflag = iflag, kwargs...)
    end
    ns = size(g)
    Tuple(ms) == ns || throw(ArgumentError("FFTSpectralBackend requires ms == size(grid) = $ns (got $(Tuple(ms)))"))
    nth = FFS._backend_nthreads(exec)
    sample = zeros(T, ns..., batch...)
    return _fftw_plan(T, g, ns, sample, nth, FFTW.MEASURE; iflag = iflag)
end

# One-shot allocating entry (routed from the (transform, execution, grid) dispatch; the hub already
# checked `isuniform`). Plans on the field itself with `ESTIMATE`, so there is no dummy planning array;
# the output is allocated at the packed size.
function FFS._calculate_spectrum_fft(exec::ComputationalBackends.AbstractExecutionBackend,
        g::FlowGeometries.Grids.AbstractStructuredGrid{<:FlowGeometries.Geometry.AbstractCartesianGeometry},
        field::AbstractArray, ms::Tuple; iflag::Int = 1, kwargs...)
    ns, batch = _fft_setup(g, field, ms)
    T = eltype(field)
    RT = real(float(T))
    nth = FFS._backend_nthreads(exec)
    if T <: Real
        f = eltype(field) === float(T) ? field : float.(field)
        plan = _fftw_plan(float(T), g, ns, f, nth, FFTW.ESTIMATE; iflag = iflag)
        coeffs = Array{Complex{RT}}(undef, FFS.Packing.packed_size(ns, Val(true))..., batch...)
    else
        f = eltype(field) === Complex{RT} ? field : Complex{RT}.(field)
        plan = _fftw_plan(Complex{RT}, g, ns, f, nth, FFTW.ESTIMATE; iflag = iflag)
        coeffs = Array{Complex{RT}}(undef, ns..., batch...)
    end
    ks = FFS.calculate_spectrum!(coeffs, plan, f)
    return coeffs, ks
end

# =============================================================================
# Inverse (synthesis). The forward folds `1/∏N` into its output, and FFTW's backward transforms are
# unnormalized, so the two scalings cancel exactly: `f == brfft(C, N_1)` for a real field and
# `f == bfft(C)` for a complex one, with no rescaling. FFTW's multi-dimensional c2r overwrites its input
# (it admits no `PRESERVE_INPUT`), so the real path transforms a copy and leaves the caller's `coeffs`
# intact.
# =============================================================================

# Hybrid composite: transform only the uniform axes `dims`, leaving the stretched axes for the NUFFT
# passes. `halve` takes the real field's `rfft` (which halves `dims[1] == 1`); otherwise the field is
# widened and transformed in full, so the whole `k₁` range stays available for a conjugate twin read.
# `conj_in` conjugates a complex input, which carries the `iflag = -1` sign through the composite.
# Raw output: the composite applies the offset phase and the `1/∏N_d` normalization once at the end.
function FFS._region_fft(exec::ComputationalBackends.AbstractExecutionBackend, field::AbstractArray,
        dims::Tuple, halve::Bool, conj_in::Bool)
    FFTW.set_num_threads(FFS._backend_nthreads(exec))
    halve && return FFTW.rfft(field, dims)
    A = Array{Complex{real(float(eltype(field)))}}(undef, size(field)...)
    copyto!(A, field)
    conj_in && (A .= conj.(A))
    FFTW.fft!(A, dims)
    return A
end

# ---- the same pass, held for reuse ----

"""
    RegionFFT{P,S}

The uniform-region FFT of the hybrid composite, planned once. `halve` marks the real path, whose `rfft`
shortens axis 1 to `n₁÷2+1`; `conj_in` conjugates a complex input, carrying `iflag = -1` through the
composite. `scratch` takes the (possibly conjugated) input, so the caller's field is never mutated and
the plan is built with `MEASURE`.
"""
struct RegionFFT{P, S}
    fwd::P
    scratch::S
    outsize::Tuple
    halve::Bool
    conj_in::Bool
end

Base.show(io::IO, r::RegionFFT) = print(io, "RegionFFT(", r.halve ? "rfft" : "fft", " → ", r.outsize, ")")

function FFS._region_fft_plan(exec::ComputationalBackends.AbstractExecutionBackend, ::Type{T},
        insize::Tuple, dims::Tuple, halve::Bool, conj_in::Bool) where {T}
    FFTW.set_num_threads(FFS._backend_nthreads(exec))
    RT = real(float(T))
    # `rfft` halves the FIRST transformed dim, which the composite requires to be axis 1.
    outsize = halve ? (insize[1] ÷ 2 + 1, Base.tail(insize)...) : insize
    scratch = halve ? Array{RT}(undef, insize...) : Array{Complex{RT}}(undef, insize...)
    fwd = halve ? FFTW.plan_rfft(scratch, dims; flags = FFTW.MEASURE) :
                  FFTW.plan_fft(scratch, dims; flags = FFTW.MEASURE)
    return RegionFFT{typeof(fwd), typeof(scratch)}(fwd, scratch, outsize, halve, conj_in)
end

# Output shape of the held region pass, so the composite sizes the working array that follows it.
FFS._region_fft_size(r::RegionFFT) = r.outsize

function FFS._region_fft_exec!(out::AbstractArray, r::RegionFFT, field::AbstractArray)
    copyto!(r.scratch, field)
    r.conj_in && !r.halve && (r.scratch .= conj.(r.scratch))
    LA.mul!(out, r.fwd, r.scratch)
    return out
end

# =============================================================================
# Reusable synthesis: the backward FFTW plan and the buffer its c2r consumes, built once.
#
# FFTW's multi-dimensional c2r has no `PRESERVE_INPUT`, so the real path transforms a COPY and the
# caller's `coeffs` survives. `brfft` is unnormalized and the forward carried `1/∏N`, so neither
# direction rescales.
# =============================================================================

struct FFTWSynthesisPlan{RT, D, NB, R, P, SC} <: FFS.AbstractSynthesisPlan
    bwd::P
    scratch::SC                   # the c2r's input copy; the complex path reads `coeffs` directly
    ns::NTuple{D, Int}
    batch::NTuple{NB, Int}
    neg::Bool
end

Base.show(io::IO, ::FFTWSynthesisPlan{RT, D, NB, R}) where {RT, D, NB, R} =
    print(io, "FFTWSynthesisPlan{$RT, $D}(", R ? "real" : "complex", ")")

FFS.Plans.field_size(p::FFTWSynthesisPlan) = (p.ns..., p.batch...)
FFS.Plans.field_type(::FFTWSynthesisPlan{RT, D, NB, R}) where {RT, D, NB, R} = R ? RT : Complex{RT}

function FFS.Plans.plan_synthesis(::SpectralBackends.AbstractFFTSpectralBackend,
        exec::ComputationalBackends.AbstractExecutionBackend,
        g::FlowGeometries.Grids.AbstractStructuredGrid{<:FlowGeometries.Geometry.AbstractCartesianGeometry},
        ::Type{T}, ms::NTuple{D, Int}; batch::Tuple = (), iflag::Int = 1, kwargs...) where {T, D}
    FlowGeometries.Grids.isuniform(g) || throw(ArgumentError(
        "FFTSpectralBackend needs a uniform axis in every direction; got a stretched axis."))
    ns = size(g)
    Tuple(ms) == ns || throw(ArgumentError(
        "FFTSpectralBackend requires ms == size(grid) = $ns (got $(Tuple(ms))); FFT is a full transform."))
    FFTW.set_num_threads(FFS._backend_nthreads(exec))
    RT = real(float(T))
    R = T <: Real
    bt = NTuple{length(batch), Int}(batch)
    if R
        scratch = Array{Complex{RT}}(undef, FFS.Packing.packed_size(ns, Val(true))..., bt...)
        bwd = FFTW.plan_brfft(scratch, ns[1], 1:D; flags = FFTW.MEASURE)
    else
        scratch = Array{Complex{RT}}(undef, ns..., bt...)
        bwd = iflag == 1 ? FFTW.plan_bfft(scratch, 1:D; flags = FFTW.MEASURE) :
                           FFTW.plan_fft(scratch, 1:D; flags = FFTW.MEASURE)
    end
    return FFTWSynthesisPlan{RT, D, length(bt), R, typeof(bwd), typeof(scratch)}(
        bwd, scratch, ns, bt, iflag < 0)
end

# A uniform grid's twins equal their negated-index partners, and FFT requires a uniform grid, so `ks`
# carries nothing this inverse needs.
function FFS.Plans.synthesize!(out::AbstractArray, plan::FFTWSynthesisPlan{RT, D, NB, R},
        coeffs::AbstractArray; ks = nothing) where {RT, D, NB, R}
    size(coeffs) == size(plan.scratch) || throw(DimensionMismatch(
        "coeffs is $(size(coeffs)); this plan was built for $(size(plan.scratch)) — pass the matching " *
        "`batch=` and element type to plan_synthesis"))
    size(out) == FFS.Plans.field_size(plan) || throw(DimensionMismatch(
        "out is $(size(out)); this plan writes $(FFS.Plans.field_size(plan))"))
    # A real field's `iflag = -1` half is the conjugate of its `+1` half.
    if R
        plan.neg ? (plan.scratch .= conj.(coeffs)) : copyto!(plan.scratch, coeffs)
        LA.mul!(out, plan.bwd, plan.scratch)
    else
        LA.mul!(out, plan.bwd, coeffs)
    end
    return out
end

function FFS._synthesize(::SpectralBackends.AbstractFFTSpectralBackend,
        exec::ComputationalBackends.AbstractExecutionBackend,
        g::FlowGeometries.Grids.AbstractStructuredGrid{<:FlowGeometries.Geometry.AbstractCartesianGeometry},
        coeffs::AbstractArray{Complex{RT}}, ms::NTuple{D, Int};
        real_output::Bool = true, iflag::Int = 1, ks = nothing, kwargs...) where {RT, D}
    FlowGeometries.Grids.isuniform(g) || throw(ArgumentError(
        "FFTSpectralBackend needs a uniform axis in every direction; got a stretched axis."))
    ns = size(g)
    Tuple(ms) == ns || throw(ArgumentError(
        "FFTSpectralBackend requires ms == size(grid) = $ns (got $(Tuple(ms))); FFT is a full transform."))
    FFTW.set_num_threads(FFS._backend_nthreads(exec))
    batch = ntuple(i -> size(coeffs, D + i), ndims(coeffs) - D)
    if real_output
        pms = FFS.Packing.packed_size(ns, Val(true))
        size(coeffs)[1:D] == pms || throw(DimensionMismatch(
            "real_output=true expects the packed half $(pms) on the spectral dims; got $(size(coeffs)[1:D]). " *
            "Pass real_output=false for a full native spectrum $(ns)."))
        scratch = iflag < 0 ? conj.(coeffs) : copy(coeffs)   # a real field's iflag=-1 half is conjugated
        out = Array{RT}(undef, ns..., batch...)
        LA.mul!(out, FFTW.plan_brfft(scratch, ns[1], 1:D; flags = FFTW.ESTIMATE), scratch)
        return out
    end
    size(coeffs)[1:D] == ns || throw(DimensionMismatch(
        "real_output=false expects the full native spectrum $(ns) on the spectral dims; got $(size(coeffs)[1:D])."))
    out = Array{Complex{RT}}(undef, ns..., batch...)
    bwd = iflag == 1 ? FFTW.plan_bfft(coeffs, 1:D; flags = FFTW.ESTIMATE) :
                       FFTW.plan_fft(coeffs, 1:D; flags = FFTW.ESTIMATE)
    LA.mul!(out, bwd, coeffs)
    return out
end

end # module FlowFieldSpectraFFTWExt
