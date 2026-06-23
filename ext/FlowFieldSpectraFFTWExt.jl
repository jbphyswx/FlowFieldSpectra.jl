module FlowFieldSpectraFFTWExt

using FFTW: FFTW
using LinearAlgebra: mul!
using FlowFieldSpectra: FlowFieldSpectra as FFS, FFTBackend

# =============================================================================
# Reusable FFTW plan for uniform Cartesian grids.
#
# A single planned transform over the spectral dims `1:D` is applied to all `n_transf`
# trailing batch slices (field components, vertical levels, time, ...) at once, and reused
# across calls. Buffers are owned by the plan so steady-state execution allocates nothing.
# =============================================================================

struct FFTWCartesianPlan{T, D, N, P, KS} <: FFS.AbstractSpectralPlan
    fwd::P                       # planned fft/bfft over dims 1:D of an (ms..., n_transf) array
    inbuf::Array{Complex{T}, N}  # input buffer (ms..., n_transf)
    outbuf::Array{Complex{T}, N} # transform output buffer
    ms::NTuple{D, Int}
    n_transf::Int
    shifts::NTuple{N, Int}       # fftshift = circshift by div(m,2) on spectral dims, 0 on batch
    norm::T                      # 1/prod(ms)
    ks_phys::KS
end

function _fftw_plan(::Type{T}, ms::NTuple{D, Int}, domain_size::NTuple{D},
        n_transf::Int, iflag::Int) where {T, D}
    inbuf = zeros(Complex{T}, ms..., n_transf)
    outbuf = similar(inbuf)
    fwd = iflag == 1 ? FFTW.plan_fft(inbuf, 1:D) : FFTW.plan_bfft(inbuf, 1:D)
    shifts = ntuple(i -> i <= D ? div(ms[i], 2) : 0, D + 1)
    norm = one(T) / prod(ms)
    ds = ntuple(d -> T(domain_size[d]), D)
    ks_phys = FFS.Grids.physical_wavenumbers(ds, ms, T)
    return FFTWCartesianPlan{T, D, D + 1, typeof(fwd), typeof(ks_phys)}(
        fwd, inbuf, outbuf, ms, n_transf, shifts, norm, ks_phys,
    )
end

function FFS.plan_spectrum(::FFTBackend, g::FFS.AbstractCartesianGrid, ::Type{T},
        ms::NTuple{D, Int}; n_transf::Int = 1, iflag::Int = 1) where {T, D}
    return _fftw_plan(T, ms, g.domain_size, n_transf, iflag)
end

# Fill the plan's input buffer from a tuple of field vectors (each length prod(ms)).
function _load_input!(plan::FFTWCartesianPlan{T, D}, fields_vecs::Tuple) where {T, D}
    length(fields_vecs) == plan.n_transf ||
        throw(DimensionMismatch("expected $(plan.n_transf) fields, got $(length(fields_vecs))"))
    M = prod(plan.ms)
    @inbounds for u in 1:length(fields_vecs)
        length(fields_vecs[u]) == M ||
            throw(DimensionMismatch("field $u length $(length(fields_vecs[u])) != prod(ms)=$M"))
        copyto!(selectdim(plan.inbuf, D + 1, u), fields_vecs[u])
    end
    return plan
end

# Fill from a packed array of shape (ms..., batch...) (batch flattened to n_transf).
function _load_input!(plan::FFTWCartesianPlan{T, D}, field::AbstractArray) where {T, D}
    length(field) == length(plan.inbuf) ||
        throw(DimensionMismatch("field has $(length(field)) entries, expected $(length(plan.inbuf))"))
    copyto!(plan.inbuf, field)
    return plan
end

"""
    calculate_spectrum!(coeffs, plan::FFTWCartesianPlan, fields) -> ks_phys

Execute a prebuilt FFTW plan in place. `fields` is a tuple of `n_transf` field vectors (each
`prod(ms)` long) or a packed `(ms..., batch...)` array; `coeffs` must have shape
`(ms..., n_transf)`. Allocation-free in steady state.
"""
function FFS.calculate_spectrum!(coeffs::AbstractArray{Complex{T}}, plan::FFTWCartesianPlan{T, D},
        fields) where {T, D}
    size(coeffs) == (plan.ms..., plan.n_transf) ||
        throw(DimensionMismatch("coeffs size $(size(coeffs)) != $((plan.ms..., plan.n_transf))"))
    _load_input!(plan, fields)
    mul!(plan.outbuf, plan.fwd, plan.inbuf)
    circshift!(coeffs, plan.outbuf, plan.shifts)
    coeffs .*= plan.norm
    return plan.ks_phys
end

# One-shot allocating entry (called by the core (backend, grid) dispatch).
function FFS._calculate_spectrum_fft(
    coords_vecs::Tuple,
    fields_vecs::Tuple,
    ms::Tuple;
    iflag::Int = 1,
    domain_size::Union{Nothing, Tuple} = nothing,
    kwargs...,
)
    D = length(ms)
    NU = length(fields_vecs)
    T = float(real(eltype(fields_vecs[1])))
    ds = domain_size === nothing ?
         ntuple(d -> (e = extrema(coords_vecs[d]); T(e[2] - e[1])), D) :
         ntuple(d -> T(domain_size[d]), D)
    plan = _fftw_plan(T, NTuple{D, Int}(ms), ds, NU, iflag)
    coeffs = zeros(Complex{T}, ms..., NU)
    ks = FFS.calculate_spectrum!(coeffs, plan, fields_vecs)
    return coeffs, ks
end

end # module FlowFieldSpectraFFTWExt
