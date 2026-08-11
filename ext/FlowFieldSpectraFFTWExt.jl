module FlowFieldSpectraFFTWExt

using FFTW: FFTW
using LinearAlgebra: LinearAlgebra as LA
using FlowFieldSpectra: FlowFieldSpectra as FFS
using ComputationalBackends: ComputationalBackends
using SpectralBackends: SpectralBackends
using FlowGeometries: FlowGeometries

# =============================================================================
# FFT for uniform structured Cartesian grids on the tensor-native model. The field is
# `(N_1,…,N_D, batch…)`; FFTW transforms the spectral dims `1:D` and batches over the trailing dims
# natively. Real input takes the `rfft` fast path — read DIRECTLY, no widen-copy into a Complex buffer
# — reconstructed to the identical full centered complex spectrum. Buffers are owned by the plan, so
# steady-state `calculate_spectrum!` allocates nothing and copies the input zero times.
#
# FFT is a full transform, so `ms == size(grid)`. Physical wavenumbers use the grid's periodic length.
# =============================================================================

_exec_nthreads(::ComputationalBackends.AbstractThreadedBackend) = Threads.nthreads()
_exec_nthreads(::ComputationalBackends.AbstractExecutionBackend) = 1

# Real-input plan: rfft over dims 1:D (reads the real field directly), + full-spectrum reconstruction.
struct RFFTPlan{RT, D, NT, P, HB, FB, KS} <: FFS.AbstractSpectralPlan
    fwd::P                        # rfft plan over dims 1:D of a real (N…, batch…) array
    half::HB                      # (h1, N_2…N_D, batch…) complex half-spectrum buffer
    full::FB                      # (N…, batch…) complex full-spectrum buffer
    ns::NTuple{D, Int}
    shifts::NTuple{NT, Int}       # fftshift = circshift by div(N,2) on spectral dims, 0 on batch
    norm::RT                      # 1/prod(ns)
    ks_phys::KS
end

# Complex-input plan: in/out C2C fft (or bfft) over dims 1:D.
struct CFFTPlan{RT, D, NT, P, FB, KS} <: FFS.AbstractSpectralPlan
    fwd::P                        # fft/bfft plan over dims 1:D of a complex (N…, batch…) array
    out::FB                       # (N…, batch…) complex output buffer
    ns::NTuple{D, Int}
    shifts::NTuple{NT, Int}
    norm::RT
    ks_phys::KS
end

# Reconstruct the full (N…, batch…) complex spectrum from the rfft half (h1, N_2…N_D, batch…) via
# Hermitian symmetry F[k] = conj(F[-k]) over the spectral dims 1:D; the batch rides along.
function _full_from_rfft!(full::AbstractArray{Complex{T}, NT}, half::AbstractArray{Complex{T}, NT},
        ns::NTuple{D, Int}) where {T, D, NT}
    h1 = ns[1] ÷ 2 + 1
    # `full` is (ns…, batch…) and `half` is (h1, ns[2:D]…, batch…), both rank NT. Iterate `full`
    # directly (no reshape → no per-call header allocation): each mode is the stored lower half
    # (I[1] ≤ h1) or the Hermitian conjugate of its negated-frequency partner (batch dims copied
    # through). The negated per-axis index is j ↦ 1 if j==1 else n-j+2.
    @inbounds for I in CartesianIndices(full)
        if I[1] <= h1
            full[I] = half[I]
        else
            src = CartesianIndex(ntuple(
                d -> d == 1 ? (ns[1] - I[1] + 2) : (d <= D ? (I[d] == 1 ? 1 : ns[d] - I[d] + 2) : I[d]),
                Val(NT)))
            full[I] = conj(half[src])
        end
    end
    return full
end

# Fused fftshift (circshift by `shifts` on spectral dims) + `norm` scale, `src` → `dst`, one pass.
function _fftshift_scale!(dst::AbstractArray{Complex{T}, NT}, src::AbstractArray{Complex{T}, NT},
        shifts::NTuple{NT, Int}, norm::T) where {T, NT}
    sz = size(src)
    @inbounds for I in CartesianIndices(src)
        J = CartesianIndex(ntuple(d -> mod(I[d] - 1 - shifts[d], sz[d]) + 1, Val(NT)))
        dst[I] = src[J] * norm
    end
    return dst
end

_shifts(ns::NTuple{D, Int}, nbatch::Int) where {D} =
    ntuple(i -> i <= D ? div(ns[i], 2) : 0, D + nbatch)

# Plan builders (dispatch on real vs complex element type). `ks_phys` is taken from the grid's periodic
# domain length via `FFS.Grids.physical_wavenumbers(g, ns)`.
function _fftw_plan(::Type{T}, g, ns::NTuple{D, Int}, batch::Tuple, nthreads::Int) where {T<:Real, D}
    FFTW.set_num_threads(nthreads)
    sample = zeros(T, ns..., batch...)
    fwd = FFTW.plan_rfft(sample, 1:D)
    h1 = ns[1] ÷ 2 + 1
    half = Array{Complex{T}}(undef, h1, ns[2:D]..., batch...)
    full = Array{Complex{T}}(undef, ns..., batch...)
    shifts = _shifts(ns, length(batch))
    ks = FFS.Grids.physical_wavenumbers(g, ns)
    return RFFTPlan{T, D, D + length(batch), typeof(fwd), typeof(half), typeof(full), typeof(ks)}(
        fwd, half, full, ns, shifts, one(T) / prod(ns), ks,
    )
end

function _fftw_plan(::Type{Complex{RT}}, g, ns::NTuple{D, Int}, batch::Tuple,
        nthreads::Int; iflag::Int = 1) where {RT<:Real, D}
    FFTW.set_num_threads(nthreads)
    sample = zeros(Complex{RT}, ns..., batch...)
    fwd = iflag == 1 ? FFTW.plan_fft(sample, 1:D) : FFTW.plan_bfft(sample, 1:D)
    out = similar(sample)
    shifts = _shifts(ns, length(batch))
    ks = FFS.Grids.physical_wavenumbers(g, ns)
    return CFFTPlan{RT, D, D + length(batch), typeof(fwd), typeof(out), typeof(ks)}(
        fwd, out, ns, shifts, one(RT) / prod(ns), ks,
    )
end

"""
    calculate_spectrum!(coeffs, plan::RFFTPlan, field) -> ks_phys

Execute a prebuilt real-input FFT plan in place: `rfft` reads the real `field` directly (no widen
copy), reconstructs the full centered spectrum, and writes `(N…, batch…)` into `coeffs`. Allocation-
free and copy-free in steady state.
"""
function FFS.calculate_spectrum!(coeffs::AbstractArray{Complex{RT}}, plan::RFFTPlan{RT, D}, field) where {RT, D}
    LA.mul!(plan.half, plan.fwd, field)
    _full_from_rfft!(plan.full, plan.half, plan.ns)
    _fftshift_scale!(coeffs, plan.full, plan.shifts, plan.norm)
    return plan.ks_phys
end

"""
    calculate_spectrum!(coeffs, plan::CFFTPlan, field) -> ks_phys

Execute a prebuilt complex-input FFT plan in place.
"""
function FFS.calculate_spectrum!(coeffs::AbstractArray{Complex{RT}}, plan::CFFTPlan{RT, D}, field) where {RT, D}
    LA.mul!(plan.out, plan.fwd, field)
    _fftshift_scale!(coeffs, plan.out, plan.shifts, plan.norm)
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

function FFS.plan_spectrum(::SpectralBackends.AbstractFFTSpectralBackend, exec::ComputationalBackends.AbstractExecutionBackend,
        g::FlowGeometries.Grids.AbstractStructuredGrid{<:FlowGeometries.Geometry.AbstractCartesianGeometry}, ::Type{T}, ms::NTuple{D, Int};
        batch::Tuple = (), iflag::Int = 1) where {T, D}
    FlowGeometries.Grids.isuniform(g) || throw(ArgumentError(
        "FFTSpectralBackend needs a uniform axis in every direction; got a stretched axis. Use NUFFTSpectralBackend."))
    ns = size(g)
    Tuple(ms) == ns || throw(ArgumentError("FFTSpectralBackend requires ms == size(grid) = $ns (got $(Tuple(ms)))"))
    return T <: Real ?
        _fftw_plan(T, g, ns, batch, _exec_nthreads(exec)) :
        _fftw_plan(T, g, ns, batch, _exec_nthreads(exec); iflag = iflag)
end

# One-shot allocating entry (routed from the (transform, execution, grid) dispatch; the hub already
# checked `isuniform`).
function FFS._calculate_spectrum_fft(exec::ComputationalBackends.AbstractExecutionBackend,
        g::FlowGeometries.Grids.AbstractStructuredGrid{<:FlowGeometries.Geometry.AbstractCartesianGeometry},
        field::AbstractArray, ms::Tuple; iflag::Int = 1, kwargs...)
    ns, batch = _fft_setup(g, field, ms)
    T = eltype(field)
    RT = real(float(T))
    plan = T <: Real ?
        _fftw_plan(float(T), g, ns, batch, _exec_nthreads(exec)) :
        _fftw_plan(Complex{RT}, g, ns, batch, _exec_nthreads(exec); iflag = iflag)
    coeffs = Array{Complex{RT}}(undef, ns..., batch...)
    # If the field's float type differs from the plan's, convert once (rare; keeps the hot path exact).
    f = eltype(field) === (T <: Real ? float(T) : Complex{RT}) ? field : (T <: Real ? float.(field) : Complex{RT}.(field))
    ks = FFS.calculate_spectrum!(coeffs, plan, f)
    return coeffs, ks
end

end # module FlowFieldSpectraFFTWExt
