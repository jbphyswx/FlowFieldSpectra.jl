module FlowFieldSpectraFFTWExt

using FFTW: FFTW
using LinearAlgebra: LinearAlgebra as LA
using FlowFieldSpectra: FlowFieldSpectra as FFS
using FlowTransformBindings: FlowTransformBindings as FTB
using SpectralBackends: SpectralBackends
using FlowGeometries: FlowGeometries

# =============================================================================
# FFT for uniform structured Cartesian grids on the packed-native model. The field is
# `(N_1,…,N_D, batch…)`; FFTW transforms the spectral dims `1:D` and batches over the trailing dims
# natively. A real field takes the `rfft` fast path and its coefficients are the packed half
# `(N_1÷2+1, N_2…N_D, batch…)` in native (unshifted) order; a complex field gives the full native
# spectrum `(N…, batch…)`. `mul!` writes the caller's `coeffs` directly and `rmul!` applies the `1/∏N`
# normalization in place, so steady-state `calculate_spectrum!` allocates nothing and copies the input
# zero times. `FlowTransformBindings.unpacked` reconstructs the full native cube when one is needed.
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

# One line each: a default `show` prints the FFTW plan, whose own `show` spans several lines.
Base.show(io::IO, p::RFFTPlan{RT, D}) where {RT, D} =
    print(io, "RFFTPlan{", RT, ", ", D, "}(", p.ns, ", batch=", p.batch, ")")
Base.show(io::IO, p::CFFTPlan{RT, D}) where {RT, D} =
    print(io, "CFFTPlan{", RT, ", ", D, "}(", p.ns, ", batch=", p.batch, ")")

FFS.Plans.coefficient_size(p::RFFTPlan{RT, D}) where {RT, D} =
    (FTB.packed_size(p.ns, Val(true))..., p.batch...)
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
    fwd = FFTW.plan_rfft(a, 1:D; flags = flags, num_threads = nthreads)
    ks = FFS.Grids.physical_wavenumbers(g, ns, Val(true))
    bt = _batch_of(a, Val(D))
    return RFFTPlan{T, D, length(bt), typeof(fwd), typeof(ks)}(
        fwd, ns, bt, one(T) / prod(ns), iflag < 0, ks)
end

# The batch shape the plan was built over: whatever trails the `D` spatial dims of the planning array.
_batch_of(a::AbstractArray, ::Val{D}) where {D} = ntuple(i -> size(a, D + i), max(ndims(a) - D, 0))

function _fftw_plan(::Type{Complex{RT}}, g, ns::NTuple{D, Int}, a::AbstractArray, nthreads::Int, flags;
        iflag::Int = 1) where {RT<:Real, D}
    fwd = iflag == 1 ? FFTW.plan_fft(a, 1:D; flags = flags, num_threads = nthreads) :
                       FFTW.plan_bfft(a, 1:D; flags = flags, num_threads = nthreads)
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
    FFS.Plans._check_coefficients(coeffs, plan)
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
    FFS.Plans._check_coefficients(coeffs, plan)
    LA.mul!(coeffs, plan.fwd, field)
    LA.rmul!(coeffs, plan.norm)
    return plan.ks_phys
end

# Validate FFT applicability + split off the batch shape.
@inline function _fft_setup(g::FlowGeometries.Grids.AbstractStructuredGrid{<:FlowGeometries.Geometry.AbstractCartesianGeometry}, field::AbstractArray, ms::Tuple)
    ns = size(g)
    Tuple(ms) == ns || throw(ArgumentError(
        "FFTSpectralBackend requires ms == size(grid) = $ns (got $(Tuple(ms))); FFT is a full transform. " *
        "Use a NUFFT or DirectSumSpectralBackend() for a different mode count."))
    D = length(ns)
    ndims(field) >= D || throw(DimensionMismatch("field has $(ndims(field)) dims, grid needs $D spatial"))
    batch = ntuple(i -> size(field, D + i), ndims(field) - D)
    return ns, batch
end

# Same split the forward makes: all-uniform takes the pure FFTW plan, uniform-in-some-directions takes
# the hybrid composite plan, all-stretched raises pointing at NUFFT. FFTW plans run on the host, so these
# methods take the host execution backends; a device plan is the KernelAbstractions extension's.
# `measure` plans with FFTW's `MEASURE`, for a plan held across executions, and otherwise with `ESTIMATE`,
# which a plan built for one call takes.
_flags(measure::Bool) = measure ? FFTW.MEASURE : FFTW.ESTIMATE

function FFS.plan_spectrum(::SpectralBackends.AbstractFFTSpectralBackend, exec::FFS._HostExec,
        g::FlowGeometries.Grids.AbstractStructuredGrid{<:FlowGeometries.Geometry.AbstractCartesianGeometry}, ::Type{T}, ms::NTuple{D, Int};
        batch::Tuple = (), iflag::Int = 1,
        nufft::SpectralBackends.AbstractSpectralBackend = SpectralBackends.AutoSpectralBackend(),
        measure::Bool = true, kwargs...) where {T, D}
    umask = ntuple(d -> FlowGeometries.Grids.isuniform(g, d), Val(D))
    if !all(umask)
        any(umask) || FFS._all_stretched_fft()
        return FFS._hybrid_plan(nufft, exec, g, T, ms, umask; batch = batch, iflag = iflag, measure = measure,
            kwargs...)
    end
    ns = size(g)
    Tuple(ms) == ns || throw(ArgumentError("FFTSpectralBackend requires ms == size(grid) = $ns (got $(Tuple(ms)))"))
    nth = FFS._backend_nthreads(exec)
    sample = zeros(T, ns..., batch...)
    return _fftw_plan(T, g, ns, sample, nth, _flags(measure); iflag = iflag)
end

# One-shot allocating entry (routed from the (transform, execution, grid) dispatch; the hub already
# checked `isuniform`). Plans on the field itself with `ESTIMATE`, so there is no dummy planning array;
# the output is allocated at the packed size.
function FFS._calculate_spectrum_fft(exec::FFS._HostExec,
        g::FlowGeometries.Grids.AbstractStructuredGrid{<:FlowGeometries.Geometry.AbstractCartesianGeometry},
        field::AbstractArray, ms::Tuple; iflag::Int = 1, kwargs...)
    ns, batch = _fft_setup(g, field, ms)
    T = eltype(field)
    RT = real(float(T))
    nth = FFS._backend_nthreads(exec)
    if T <: Real
        f = eltype(field) === float(T) ? field : float.(field)
        plan = _fftw_plan(float(T), g, ns, f, nth, FFTW.ESTIMATE; iflag = iflag)
        coeffs = Array{Complex{RT}}(undef, FTB.packed_size(ns, Val(true))..., batch...)
    else
        f = eltype(field) === Complex{RT} ? field : Complex{RT}.(field)
        plan = _fftw_plan(Complex{RT}, g, ns, f, nth, FFTW.ESTIMATE; iflag = iflag)
        coeffs = Array{Complex{RT}}(undef, ns..., batch...)
    end
    ks = FFS.calculate_spectrum!(coeffs, plan, f)
    return coeffs, ks
end

# The hybrid composite's FFT over the uniform axes `dims`, on host arrays, leaving the stretched axes for
# the NUFFT passes. `halve` takes the real field's `rfft` (which halves `dims[1] == 1`); otherwise the field
# is converted to complex and transformed in full, so the whole `k₁` range stays available for the
# conjugate reads a widened coefficient box takes.
function FFS._region_fft_plan(exec::FFS._HostExec, ::Type{T}, insize::Tuple, dims::Tuple, halve::Bool,
        conj_in::Bool, measure::Bool) where {T}
    nth = FFS._backend_nthreads(exec)
    RT = real(float(T))
    # `rfft` halves the first transformed dim, which the composite requires to be axis 1.
    outsize = halve ? (insize[1] ÷ 2 + 1, Base.tail(insize)...) : insize
    scratch = halve ? Array{RT}(undef, insize...) : Array{Complex{RT}}(undef, insize...)
    fwd = halve ? FFTW.plan_rfft(scratch, dims; flags = _flags(measure), num_threads = nth) :
                  FFTW.plan_fft(scratch, dims; flags = _flags(measure), num_threads = nth)
    return FFS.RegionFFT(fwd, scratch, outsize, halve, conj_in)
end

# The hybrid synthesis's backward FFT over the uniform axes `dims` of the native spectrum `A`, in place:
# `Σ_κ C_κ e^{+iflag·iκx}` unnormalized, `bfft` for `iflag = 1` and `fft` for `-1`.
FFS._region_bfft_plan(exec::FFS._HostExec, A::Array, dims::Tuple, iflag::Int, measure::Bool) =
    iflag == 1 ?
        FFTW.plan_bfft!(A, dims; flags = _flags(measure), num_threads = FFS._backend_nthreads(exec)) :
        FFTW.plan_fft!(A, dims; flags = _flags(measure), num_threads = FFS._backend_nthreads(exec))

# =============================================================================
# Inverse (synthesis). The forward folds `1/∏N` into its output, and FFTW's backward transforms are
# unnormalized, so the two scalings cancel exactly: `f == brfft(C, N_1)` for a real field and
# `f == bfft(C)` for a complex one, with no rescaling. FFTW's multi-dimensional c2r overwrites its input
# (it admits no `PRESERVE_INPUT`), so the real path transforms a copy and leaves the caller's `coeffs`
# intact. The reusable plan holds the backward FFTW plan and the buffer its c2r consumes.
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

# The forward's split again: a grid uniform in some directions and stretched in others takes the hybrid
# composite's synthesis plan.
function FFS.Plans.plan_synthesis(::SpectralBackends.AbstractFFTSpectralBackend, exec::FFS._HostExec,
        g::FlowGeometries.Grids.AbstractStructuredGrid{<:FlowGeometries.Geometry.AbstractCartesianGeometry},
        ::Type{T}, ms::NTuple{D, Int}; batch::Tuple = (), iflag::Int = 1,
        nufft::SpectralBackends.AbstractSpectralBackend = SpectralBackends.AutoSpectralBackend(),
        measure::Bool = true, kwargs...) where {T, D}
    umask = ntuple(d -> FlowGeometries.Grids.isuniform(g, d), Val(D))
    if !all(umask)
        any(umask) || FFS._all_stretched_fft()
        return FFS._hybrid_synthesis_plan(nufft, exec, g, T, ms, umask; batch = batch, iflag = iflag,
            measure = measure, kwargs...)
    end
    ns = size(g)
    Tuple(ms) == ns || throw(ArgumentError(
        "FFTSpectralBackend requires ms == size(grid) = $ns (got $(Tuple(ms))); FFT is a full transform."))
    nth = FFS._backend_nthreads(exec)
    RT = real(float(T))
    R = T <: Real
    bt = NTuple{length(batch), Int}(batch)
    flags = _flags(measure)
    if R
        scratch = Array{Complex{RT}}(undef, FTB.packed_size(ns, Val(true))..., bt...)
        bwd = FFTW.plan_brfft(scratch, ns[1], 1:D; flags = flags, num_threads = nth)
    else
        scratch = Array{Complex{RT}}(undef, ns..., bt...)
        bwd = iflag == 1 ? FFTW.plan_bfft(scratch, 1:D; flags = flags, num_threads = nth) :
                           FFTW.plan_fft(scratch, 1:D; flags = flags, num_threads = nth)
    end
    return FFTWSynthesisPlan{RT, D, length(bt), R, typeof(bwd), typeof(scratch)}(
        bwd, scratch, ns, bt, iflag < 0)
end

function FFS.Plans.synthesize!(out::AbstractArray, plan::FFTWSynthesisPlan{RT, D, NB, R},
        coeffs::AbstractArray) where {RT, D, NB, R}
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

end # module FlowFieldSpectraFFTWExt
