using FlowGeometries: FlowGeometries as FG
using SpectralBackends: SpectralBackends
using ComputationalBackends: ComputationalBackends

"""
    SplitFFTPlan

Reusable Fourier transform of one field held in FlowGeometries blocks, including trailing batch axes.
Each axis transforms on a split that holds that axis whole. The coefficient fields use a Cartesian
index grid; [`wavenumbers`](@ref) gives their physical mode coordinates.
Consecutive stages on the same block layout share a coefficient field.
"""
struct SplitFFTPlan{T,G,F,C,P,E,B,K} <: AbstractSpectralPlan
    domain::G
    field::F
    modes::C
    stages::P
    execution::E
    batch::B
    ks::K
    iflag::Int
end

Base.show(io::IO, p::SplitFFTPlan{T}) where {T} =
    print(io, "SplitFFTPlan{", T, "}(", size(FG.Decomposition.global_grid(p.domain)), ", batch=", p.batch, ")")

coefficient_type(::SplitFFTPlan{T}) where {T} = Complex{real(T)}
coefficient_size(p::SplitFFTPlan) = (map(length, p.ks)..., p.batch...)
wavenumbers(p::SplitFFTPlan) = p.ks

function _zero_split_field(split, ::Type{T}, batch) where {T}
    field = FG.Decomposition.split_field(split, T, batch...)
    _foreach_split_field(field) do _, array
        fill!(array, zero(T))
        return nothing
    end
    return field
end

function _foreach_split_field(f::F, field) where {F}
    FG.Decomposition.foreach_block(f, FG.Decomposition.split_of(field), field)
    return nothing
end

allocate_coefficients(p::SplitFFTPlan) =
    _zero_split_field(FG.Decomposition.split_of(last(p.modes)), coefficient_type(p), p.batch)

"""
    plan_spectrum(split::FlowGeometries.Decomposition.AbstractSplitGrid, T, ms; batch=(), measure=true)

Build a reusable FFT on an unmasked, uniform periodic Cartesian split grid, retaining the blocks and
local FFT plans on their processes. Analysis and synthesis consume and return split fields.
"""
function plan_spectrum(split::FG.Decomposition.AbstractSplitGrid{N}, ::Type{T},
                       ms::NTuple{N,Int} = size(FG.Decomposition.global_grid(split));
                       transform::SpectralBackends.AbstractSpectralBackend = SpectralBackends.FFTSpectralBackend(),
                       kwargs...) where {N,T}
    return plan_spectrum(transform, _split_fft_execution(split), split, T, ms; kwargs...)
end

function plan_spectrum(transform::SpectralBackends.AbstractSpectralBackend,
                       execution::ComputationalBackends.AbstractExecutionBackend,
                       split::FG.Decomposition.AbstractSplitGrid{N}, ::Type{T}, ms::NTuple{N,Int};
                       kwargs...) where {N,T}
    return _plan_split_fft(transform, execution, split, T, ms; kwargs...)
end

function _plan_split_fft end
function _split_fft_execution end
function _split_fft_stage end
function _local_fft_stage end
function _run_split_fft_stage! end
function _run_local_fft_stage! end

struct SplitFFTSynthesisPlan{P<:SplitFFTPlan} <: AbstractSynthesisPlan
    forward::P
end

Base.show(io::IO, p::SplitFFTSynthesisPlan) = print(io, "SplitFFTSynthesisPlan(", p.forward, ")")

"""
    plan_synthesis(plan::SplitFFTPlan)

Return the inverse transform sharing `plan`'s block fields and FFT plans. The forward and inverse use
the same workspace, so they execute sequentially.
"""
plan_synthesis(p::SplitFFTPlan) = SplitFFTSynthesisPlan(p)
field_type(p::SplitFFTSynthesisPlan) = _split_field_type(p.forward)
_split_field_type(::SplitFFTPlan{T}) where {T} = T
field_size(p::SplitFFTSynthesisPlan) =
    (size(FG.Decomposition.global_grid(p.forward.domain))..., p.forward.batch...)
allocate_field(p::SplitFFTSynthesisPlan) =
    _zero_split_field(p.forward.domain, field_type(p), p.forward.batch)

_forward_split_fft!(execution, ::Tuple{}, ::Tuple{}, previous) = nothing
function _forward_split_fft!(execution, fields::Tuple, stages::Tuple, previous)
    if FG.Decomposition.split_dims(FG.Decomposition.split_of(first(fields))) !=
       FG.Decomposition.split_dims(FG.Decomposition.split_of(previous))
        FG.Decomposition.redistribute!(first(fields), previous)
    end
    _run_split_fft_stage!(execution, first(stages), Val(:forward))
    _forward_split_fft!(execution, Base.tail(fields), Base.tail(stages), first(fields))
    return nothing
end

function _backward_split_fft!(execution, fields::Tuple{Any}, stages::Tuple{Any})
    _run_split_fft_stage!(execution, first(stages), Val(:backward))
    return nothing
end

function _backward_split_fft!(execution, fields::Tuple, stages::Tuple)
    _backward_split_fft!(execution, Base.tail(fields), Base.tail(stages))
    if FG.Decomposition.split_dims(FG.Decomposition.split_of(first(fields))) !=
       FG.Decomposition.split_dims(FG.Decomposition.split_of(first(Base.tail(fields))))
        FG.Decomposition.redistribute!(first(fields), first(Base.tail(fields)))
    end
    _run_split_fft_stage!(execution, first(stages), Val(:backward))
    return nothing
end
