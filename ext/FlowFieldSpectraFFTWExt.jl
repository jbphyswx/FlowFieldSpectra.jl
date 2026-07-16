module FlowFieldSpectraFFTWExt

using FFTW: FFTW
using LinearAlgebra: LinearAlgebra as LA
using FlowFieldSpectra: FlowFieldSpectra as FFS

# =============================================================================
# FFT for uniform Cartesian grids on the tensor-native model. The field is `(N_1,…,N_D, batch…)`;
# FFTW transforms the spectral dims `1:D` and batches over the trailing dims natively. Real input
# takes the `rfft` fast path — read DIRECTLY, no widen-copy into a Complex buffer — reconstructed to
# the identical full centered complex spectrum. Buffers are owned by the plan, so steady-state
# `calculate_spectrum!` allocates nothing and copies the input zero times.
#
# FFT is a full transform, so `ms == spatial_size(grid)`.
# =============================================================================

_exec_nthreads(::FFS.ThreadedBackend) = Threads.nthreads()
_exec_nthreads(::FFS.AbstractExecutionBackend) = 1

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
function _full_from_rfft!(full::AbstractArray{Complex{T}}, half::AbstractArray{Complex{T}},
        ns::NTuple{D, Int}) where {T, D}
    h1 = ns[1] ÷ 2 + 1
    B = length(full) ÷ prod(ns)
    Ff = reshape(full, ns..., B)
    Hf = reshape(half, h1, ns[2:D]..., B)
    # Fill each full mode from the stored lower half (J[1] ≤ h1) or its Hermitian conjugate partner.
    # The negated per-axis index (j ↦ 1 if j==1 else n-j+2) is computed inline — no per-call vectors.
    @inbounds for b in 1:B
        for J in CartesianIndices(ns)
            if J[1] <= h1
                Ff[J, b] = Hf[J, b]
            else
                src = CartesianIndex(ns[1] - J[1] + 2,
                    ntuple(d -> (J[d + 1] == 1 ? 1 : ns[d + 1] - J[d + 1] + 2), D - 1)...)
                Ff[J, b] = conj(Hf[src, b])
            end
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

# Plan builders (dispatch on real vs complex element type).
function _fftw_plan(::Type{T}, ns::NTuple{D, Int}, domain_size, batch::Tuple, nthreads::Int) where {T<:Real, D}
    FFTW.set_num_threads(nthreads)
    sample = zeros(T, ns..., batch...)
    fwd = FFTW.plan_rfft(sample, 1:D)
    h1 = ns[1] ÷ 2 + 1
    half = Array{Complex{T}}(undef, h1, ns[2:D]..., batch...)
    full = Array{Complex{T}}(undef, ns..., batch...)
    shifts = _shifts(ns, length(batch))
    ks = FFS.Grids.physical_wavenumbers(ntuple(d -> T(domain_size[d]), D), ns, T)
    return RFFTPlan{T, D, D + length(batch), typeof(fwd), typeof(half), typeof(full), typeof(ks)}(
        fwd, half, full, ns, shifts, one(T) / prod(ns), ks,
    )
end

function _fftw_plan(::Type{Complex{RT}}, ns::NTuple{D, Int}, domain_size, batch::Tuple,
        nthreads::Int; iflag::Int = 1) where {RT<:Real, D}
    FFTW.set_num_threads(nthreads)
    sample = zeros(Complex{RT}, ns..., batch...)
    fwd = iflag == 1 ? FFTW.plan_fft(sample, 1:D) : FFTW.plan_bfft(sample, 1:D)
    out = similar(sample)
    shifts = _shifts(ns, length(batch))
    ks = FFS.Grids.physical_wavenumbers(ntuple(d -> RT(domain_size[d]), D), ns, RT)
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
@inline function _fft_setup(g::FFS.UniformCartesianGrid, field::AbstractArray, ms::Tuple)
    ns = FFS.spatial_size(g)
    Tuple(ms) == ns || throw(ArgumentError(
        "FFTBackend requires ms == spatial_size(grid) = $ns (got $(Tuple(ms))); FFT is a full transform. " *
        "Use NUFFT/DirectSum for a different mode count."))
    D = length(ns)
    ndims(field) >= D || throw(DimensionMismatch("field has $(ndims(field)) dims, grid needs $D spatial"))
    batch = ntuple(i -> size(field, D + i), ndims(field) - D)
    return ns, batch
end

function FFS.plan_spectrum(::FFS.FFTBackend, exec::FFS.AbstractExecutionBackend,
        g::FFS.UniformCartesianGrid, ::Type{T}, ms::NTuple{D, Int};
        batch::Tuple = (), iflag::Int = 1) where {T, D}
    ns = FFS.spatial_size(g)
    Tuple(ms) == ns || throw(ArgumentError("FFTBackend requires ms == spatial_size(grid) = $ns (got $(Tuple(ms)))"))
    return T <: Real ?
        _fftw_plan(T, ns, g.domain_size, batch, _exec_nthreads(exec)) :
        _fftw_plan(T, ns, g.domain_size, batch, _exec_nthreads(exec); iflag = iflag)
end

# One-shot allocating entry (routed from the (transform, execution, grid) dispatch).
function FFS._calculate_spectrum_fft(exec::FFS.AbstractExecutionBackend, g::FFS.UniformCartesianGrid,
        field::AbstractArray, ms::Tuple; iflag::Int = 1, kwargs...)
    ns, batch = _fft_setup(g, field, ms)
    T = eltype(field)
    RT = real(float(T))
    plan = T <: Real ?
        _fftw_plan(float(T), ns, g.domain_size, batch, _exec_nthreads(exec)) :
        _fftw_plan(Complex{RT}, ns, g.domain_size, batch, _exec_nthreads(exec); iflag = iflag)
    coeffs = Array{Complex{RT}}(undef, ns..., batch...)
    # If the field's float type differs from the plan's, convert once (rare; keeps the hot path exact).
    f = eltype(field) === (T <: Real ? float(T) : Complex{RT}) ? field : (T <: Real ? float.(field) : Complex{RT}.(field))
    ks = FFS.calculate_spectrum!(coeffs, plan, f)
    return coeffs, ks
end

end # module FlowFieldSpectraFFTWExt
