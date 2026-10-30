module FlowFieldSpectraAbstractFFTsExt

using AbstractFFTs: AbstractFFTs
using LinearAlgebra: LinearAlgebra as LA
using FlowFieldSpectra: FlowFieldSpectra as FFS
using FlowTransformBindings: FlowTransformBindings as FTB
using ComputationalBackends: ComputationalBackends
using SpectralBackends: SpectralBackends
using FlowGeometries: FlowGeometries

# =============================================================================
# FFT of a uniform structured Cartesian grid, packed-native, through the AbstractFFTs interface: the
# implementation for the field's arrays plans the transform (FFTW for host arrays, a vendor library for
# device ones), with the planner options `FFS._fft_options` gives it. A real field takes the r2c transform
# and its coefficients are the packed half `(N_1÷2+1, N_2…, batch…)` in native order; a complex field gives
# the full native spectrum. FFT is a full transform, so `ms == size(grid)`.
#
# A host plan transforms the caller's arrays directly. A device plan holds its input and output in device
# memory, so a field or coefficient array elsewhere passes through them. The forward transform carries the
# sign of `Σ f e^{-ikx}`; `iflag = -1` is its conjugate, taken on the output of a real field's transform
# and on both sides of a complex one's.
# =============================================================================

const _FFTExec = Union{FFS._HostExec, ComputationalBackends.AbstractGPUBackend}

"""
    FFTPlan{RT,D,R}

Reusable FFT over dims `1:D` of a uniform structured Cartesian grid, `R` marking a real field: the forward
plan, and on a device the input and coefficient buffers it transforms between. Execute with
`calculate_spectrum!(coeffs, plan, field)`.
"""
struct FFTPlan{RT, D, R, NB, P, IN, OB, KS} <: FFS.AbstractSpectralPlan
    fwd::P                        # r2c (real field) or forward c2c plan over dims 1:D
    inbuf::IN                     # device (N…, batch…) field, `nothing` on the host
    outbuf::OB                    # device coefficients, `nothing` on the host
    ns::NTuple{D, Int}
    batch::NTuple{NB, Int}
    norm::RT                      # 1/∏N
    neg::Bool                     # iflag = -1
    ks_phys::KS
    nthreads::Int                 # threads of the scaling pass over a host array
end

Base.show(io::IO, p::FFTPlan{RT, D, R}) where {RT, D, R} =
    print(io, "FFTPlan{", RT, ", ", D, "}(", R ? "real" : "complex", ", ", p.ns, ", batch=", p.batch, ")")

FFS.Plans.coefficient_size(p::FFTPlan{RT, D, R}) where {RT, D, R} =
    (FTB.packed_size(FFS._layout_type(R), p.ns)..., p.batch...)
FFS.Plans.coefficient_type(::FFTPlan{RT}) where {RT} = Complex{RT}
FFS.Plans.wavenumbers(p::FFTPlan) = p.ks_phys

# The plan of a field of element type `E` on `A`, an array of the field's shape in the execution backend's
# memory: on a device it becomes the plan's input buffer.
function _fft_plan(exec::_FFTExec, ::Type{E}, g, ns::NTuple{D, Int}, A::AbstractArray, iflag::Int,
        measure::Bool, aligned::Bool = true) where {E, D}
    RT = real(E)
    R = E <: Real
    opts = FFS._fft_options(exec, measure, aligned)
    fwd = R ? AbstractFFTs.plan_rfft(A, 1:D; opts...) : AbstractFFTs.plan_fft(A, 1:D; opts...)
    bt = ntuple(i -> size(A, D + i), Val(ndims(A) - D))
    inbuf = _device_buffer(exec, A)
    outbuf = _device_alloc(exec, Complex{RT}, FTB.packed_size(E, ns)..., bt...)
    ks = FFS.Grids.physical_wavenumbers(g, ns, E)
    return FFTPlan{RT, D, R, length(bt), typeof(fwd), typeof(inbuf), typeof(outbuf), typeof(ks)}(
        fwd, inbuf, outbuf, ns, bt, one(RT) / prod(ns), iflag < 0, ks, FFS._backend_nthreads(exec))
end

_device_buffer(::FFS._HostExec, ::AbstractArray) = nothing
_device_buffer(::ComputationalBackends.AbstractGPUBackend, A::AbstractArray) = A
_device_alloc(::FFS._HostExec, ::Type, ::Integer...) = nothing
_device_alloc(exec::ComputationalBackends.AbstractGPUBackend, ::Type{T}, dims::Integer...) where {T} =
    FFS._alloc(exec, T, dims...)

# The array the transform reads: on the host the field itself, converted when its element type is not the
# plan's; on a device the plan's buffer, filled from the field unless the field is of the buffer's type.
_staged(::Nothing, field::AbstractArray, ::Type{E}) where {E} = field isa Array{E} ? field : Array{E}(field)
_staged(buf::AbstractArray, field::AbstractArray, ::Type{E}) where {E} =
    (field isa typeof(buf) && size(field) == size(buf)) ? field : FFS._stage!(buf, field)

# The array the transform writes: the caller's on the host, and on a device the caller's when it has the
# buffer's type, the buffer otherwise.
_into(out::AbstractArray, ::Nothing) = out
_into(out::AbstractArray, buf::AbstractArray) = (out isa typeof(buf) && size(out) == size(buf)) ? out : buf

"""
    calculate_spectrum!(coeffs, plan::FFTPlan, field) -> ks_phys

Execute a prebuilt FFT plan in place: the packed half for a real field and the full native spectrum for a
complex one, scaled by `1/∏N`, written into `coeffs`.
"""
function FFS.calculate_spectrum!(coeffs::AbstractArray{Complex{RT}}, p::FFTPlan{RT, D, R},
        field) where {RT, D, R}
    FFS.Plans._check_coefficients(coeffs, p)
    E = R ? RT : Complex{RT}
    src = _staged(p.inbuf, field, E)
    if !R && p.neg
        src = src === field ? _conj_copy(p.inbuf, field) : (src .= conj.(src))
    end
    dst = _into(coeffs, p.outbuf)
    LA.mul!(dst, p.fwd, src)
    p.neg ? _scale!(c -> conj(c) * p.norm, dst, p.nthreads) : _scale!(c -> c * p.norm, dst, p.nthreads)
    dst === coeffs || copyto!(coeffs, dst)
    return p.ks_phys
end

# `conj(field)` in a new host array, or in the plan's device buffer; the caller's field is left intact.
_conj_copy(::Nothing, field::AbstractArray) = conj.(field)
_conj_copy(buf::AbstractArray, field::AbstractArray) = (buf .= conj.(field))

# `c[i] = f(c[i])` over the whole array: a host array split across `nthreads` threads where it is large
# enough to pay for them, a device array by broadcast.
function _scale!(f::F, c::Union{Array, Base.FastContiguousSubArray{<:Any, <:Any, <:Array}},
        nthreads::Int) where {F}
    n = length(c)
    k = min(nthreads, cld(n, 1 << 15))
    if k <= 1
        @inbounds @simd for i in eachindex(c)
            c[i] = f(c[i])
        end
        return c
    end
    Threads.@threads for t in 1:k
        @inbounds @simd for i in (div((t - 1) * n, k) + 1):div(t * n, k)
            c[i] = f(c[i])
        end
    end
    return c
end
_scale!(f::F, c::AbstractArray, ::Int) where {F} = (c .= f.(c))

# `ms` checked as the grid's own size, and the field's trailing batch shape.
function _fft_setup(g::FlowGeometries.Grids.AbstractStructuredGrid, field::AbstractArray, ms::Tuple)
    ns = size(g)
    Tuple(ms) == ns || throw(ArgumentError(
        "FFTSpectralBackend requires ms == size(grid) = $ns (got $(Tuple(ms))); FFT is a full transform. " *
        "Use a NUFFT or DirectSumSpectralBackend() for a different mode count."))
    ndims(field) >= length(ns) ||
        throw(DimensionMismatch("field has $(ndims(field)) dims, grid needs $(length(ns)) spatial"))
    return ns, FFS.Grids.field_batch_shape(g, field)
end

function FFS.plan_spectrum(::SpectralBackends.AbstractFFTSpectralBackend, exec::_FFTExec,
        g::FlowGeometries.Grids.AbstractStructuredGrid{<:FlowGeometries.Geometry.AbstractCartesianGeometry},
        ::Type{T}, ms::NTuple{D, Int}; batch::Tuple = (), iflag::Int = 1,
        nufft::SpectralBackends.AbstractSpectralBackend = SpectralBackends.AutoSpectralBackend(),
        measure::Bool = true, kwargs...) where {T, D}
    udims, sdims = FFS._axis_split(g)
    if !isempty(sdims)
        isempty(udims) && FFS._all_stretched_fft()
        return FFS._hybrid_plan(nufft, exec, g, T, ms; batch, iflag, measure, kwargs...)
    end
    ns = size(g)
    Tuple(ms) == ns || throw(ArgumentError("FFTSpectralBackend requires ms == size(grid) = $ns (got $(Tuple(ms)))"))
    E = T <: Real ? float(T) : Complex{real(float(T))}
    A = fill!(FFS._alloc(exec, E, ns..., batch...), zero(E))
    return _fft_plan(exec, E, g, ns, A, iflag, measure)
end

# One-shot: a plan without measurement on the field itself on the host (no planning array), on a buffer on
# a device, with the coefficients returned where the field lives.
function FFS._calculate_spectrum_fft(exec::_FFTExec,
        g::FlowGeometries.Grids.AbstractStructuredGrid{<:FlowGeometries.Geometry.AbstractCartesianGeometry},
        field::AbstractArray, ms::Tuple; iflag::Int = 1, kwargs...)
    ns, _ = _fft_setup(g, field, ms)
    E = eltype(field) <: Real ? float(eltype(field)) : Complex{real(float(eltype(field)))}
    plan = _fft_plan(exec, E, g, ns, _one_shot_array(exec, E, field), iflag, false)
    coeffs = FFS._alloc(exec, Complex{real(E)}, FFS.Plans.coefficient_size(plan)...)
    ks = FFS.calculate_spectrum!(coeffs, plan, field)
    return FFS._like_field(field, coeffs), ks
end

_one_shot_array(::FFS._HostExec, ::Type{E}, field::AbstractArray) where {E} =
    field isa Array{E} ? field : Array{E}(field)
_one_shot_array(exec::ComputationalBackends.AbstractGPUBackend, ::Type{E}, field::AbstractArray) where {E} =
    FFS._alloc(exec, E, size(field)...)

# Several host fields of one shape and type, the coefficients of field `k` the slice `k` of a new trailing
# axis: one plan on the first field, executed on each into its slice, so no stacked copy is made. A slice
# past an offset that breaks the plan's alignment takes a plan that accepts any.
function FFS._calculate_spectrum_fft(exec::FFS._HostExec,
        g::FlowGeometries.Grids.AbstractStructuredGrid{<:FlowGeometries.Geometry.AbstractCartesianGeometry},
        fields::Tuple, ms::Tuple; iflag::Int = 1, kwargs...)
    f1 = first(fields)
    ns, batch = _fft_setup(g, f1, ms)
    E = eltype(f1)
    RT = real(E)
    csize = FTB.packed_size(E, ns)
    aligned = (prod(csize) * prod(batch; init = 1) * sizeof(Complex{RT})) % 16 == 0
    plan = _fft_plan(exec, E, g, ns, f1, iflag, false, aligned)
    coeffs = Array{Complex{RT}}(undef, csize..., batch..., length(fields))
    ks = plan.ks_phys
    for (k, f) in enumerate(fields)
        ks = FFS.calculate_spectrum!(selectdim(coeffs, ndims(coeffs), k), plan, f)
    end
    return coeffs, ks
end

# =============================================================================
# The hybrid composite's FFT over the uniform axes, on the execution backend's arrays. A real field takes
# an `rfft`, which halves `dims[1] == 1`; a complex field takes the full `fft`.
# =============================================================================

function FFS._region_fft_plan(exec::_FFTExec, ::Type{T}, insize::NTuple{N, Int}, dims::Tuple,
        conj_in::Bool, measure::Bool) where {T, N}
    RT = real(float(T))
    opts = FFS._fft_options(exec, measure, true)
    if T <: Real
        scratch = FFS._alloc(exec, RT, insize...)
        fwd = AbstractFFTs.plan_rfft(scratch, dims; opts...)
        return FFS.RegionFFT(fwd, scratch, (insize[1] ÷ 2 + 1, Base.tail(insize)...), conj_in)
    end
    scratch = FFS._alloc(exec, Complex{RT}, insize...)
    fwd = AbstractFFTs.plan_fft(scratch, dims; opts...)
    return FFS.RegionFFT(fwd, scratch, insize, conj_in)
end

# The hybrid synthesis's unnormalized backward FFT over the uniform axes `dims` of `A`, out of place: for a
# real field the c2r transform, `dims[1] == 1` of length `n1`.
function FFS._region_bfft_plan(exec::_FFTExec, ::Type{T}, A::AbstractArray, n1::Int, dims::Tuple,
        measure::Bool) where {T}
    opts = FFS._fft_options(exec, measure, true)
    return T <: Real ? AbstractFFTs.plan_brfft(A, n1, dims; opts...) : AbstractFFTs.plan_bfft(A, dims; opts...)
end

# =============================================================================
# Synthesis. The forward folds `1/∏N` into its output and the backward transforms are unnormalized, so the
# two scalings cancel: `f == brfft(C, N_1)` for a real field and `f == bfft(C)` for a complex one. A c2r
# transform overwrites its input, so a real field's transform over `D ≥ 2` dims runs as a c2c along `2:D`
# from the caller's coefficients into the plan's buffer, then the c2r along dim 1 on that buffer, which
# leaves the caller's coefficients intact with no copy. A real field over one dim copies its coefficients
# into the buffer first.
# =============================================================================

"""
    FFTSynthesisPlan{RT,D,R}

Reusable synthesis on a uniform structured Cartesian grid, `R` marking a real field: the unnormalized
backward plan from the coefficient buffer (for a real field over `D ≥ 2` dims the c2r along dim 1, after the
c2c along `2:D` out of place, `pre`, or in place in the buffer, `pre!`), the buffer, and on a device the
field buffer it writes. Execute with `synthesize!(out, plan, coeffs)`.
"""
struct FFTSynthesisPlan{RT, D, R, NB, P, Q, Q!, SC, OB} <: FFS.AbstractSynthesisPlan
    bwd::P
    pre::Q                        # c2c along `2:D` from the caller's coefficients, or `nothing`
    pre!::Q!                      # the same in place in `scratch`, or `nothing`
    scratch::SC                   # the coefficients in the plan's memory
    outd::OB                      # device (N…, batch…) field, `nothing` on the host
    ns::NTuple{D, Int}
    batch::NTuple{NB, Int}
    neg::Bool                     # iflag = -1
end

# `(pre, pre!, bwd)`: a real field over `D ≥ 2` dims splits off the c2c along `2:D`, `pre` taking any
# array's alignment (`uopts`); every other field takes the whole backward transform.
_synthesis_plans(::Type{<:Real}, scratch, n1::Int, ::Val{D}, opts, uopts) where {D} =
    (AbstractFFTs.plan_bfft(scratch, 2:D; uopts...), AbstractFFTs.plan_bfft!(scratch, 2:D; opts...),
        AbstractFFTs.plan_brfft(scratch, n1, 1; opts...))
_synthesis_plans(::Type{<:Real}, scratch, n1::Int, ::Val{1}, opts, uopts) =
    (nothing, nothing, AbstractFFTs.plan_brfft(scratch, n1, 1; opts...))
_synthesis_plans(::Type{<:Complex}, scratch, n1::Int, ::Val{D}, opts, uopts) where {D} =
    (nothing, nothing, AbstractFFTs.plan_bfft(scratch, 1:D; opts...))

Base.show(io::IO, ::FFTSynthesisPlan{RT, D, R}) where {RT, D, R} =
    print(io, "FFTSynthesisPlan{$RT, $D}(", R ? "real" : "complex", ")")

FFS.Plans.field_size(p::FFTSynthesisPlan) = (p.ns..., p.batch...)
FFS.Plans.field_type(::FFTSynthesisPlan{RT, D, R}) where {RT, D, R} = R ? RT : Complex{RT}

# The forward's split again: a grid uniform in some directions and stretched in others takes the hybrid
# composite's synthesis plan.
function FFS.Plans.plan_synthesis(::SpectralBackends.AbstractFFTSpectralBackend, exec::_FFTExec,
        g::FlowGeometries.Grids.AbstractStructuredGrid{<:FlowGeometries.Geometry.AbstractCartesianGeometry},
        ::Type{T}, ms::NTuple{D, Int}; batch::Tuple = (), iflag::Int = 1,
        nufft::SpectralBackends.AbstractSpectralBackend = SpectralBackends.AutoSpectralBackend(),
        measure::Bool = true, kwargs...) where {T, D}
    udims, sdims = FFS._axis_split(g)
    if !isempty(sdims)
        isempty(udims) && FFS._all_stretched_fft()
        return FFS._hybrid_synthesis_plan(nufft, exec, g, T, ms; batch, iflag, measure, kwargs...)
    end
    ns = size(g)
    Tuple(ms) == ns || throw(ArgumentError(
        "FFTSpectralBackend requires ms == size(grid) = $ns (got $(Tuple(ms))); FFT is a full transform."))
    RT = real(float(T))
    R = T <: Real
    bt = NTuple{length(batch), Int}(batch)
    scratch = FFS._alloc(exec, Complex{RT}, FTB.packed_size(FFS._layout_type(R), ns)..., bt...)
    pre, pre!, bwd = _synthesis_plans(T, scratch, ns[1], Val(D), FFS._fft_options(exec, measure, true),
        FFS._fft_options(exec, measure, false))
    outd = _device_alloc(exec, R ? RT : Complex{RT}, ns..., bt...)
    return FFTSynthesisPlan{RT, D, R, length(bt), typeof(bwd), typeof(pre), typeof(pre!), typeof(scratch),
            typeof(outd)}(bwd, pre, pre!, scratch, outd, ns, bt, iflag < 0)
end

function FFS.Plans.synthesize!(out::AbstractArray, p::FFTSynthesisPlan{RT, D, R},
        coeffs::AbstractArray) where {RT, D, R}
    size(coeffs) == size(p.scratch) || throw(DimensionMismatch(
        "coeffs is $(size(coeffs)); this plan was built for $(size(p.scratch)) — pass the matching " *
        "`batch=` and element type to plan_synthesis"))
    size(out) == FFS.Plans.field_size(p) || throw(DimensionMismatch(
        "out is $(size(out)); this plan writes $(FFS.Plans.field_size(p))"))
    dst = _into(out, p.outd)
    LA.mul!(dst, p.bwd, _backward_input!(p, coeffs, p.pre))
    !R && p.neg && (dst .= conj.(dst))
    dst === out || copyto!(out, dst)
    return out
end

# The array the plan's `bwd` reads. `Σ C e^{-ikx}` is `conj(Σ conj(C) e^{+ikx})`, and for a real field the
# conjugate of its half, so `iflag = -1` conjugates the coefficients. Whole transform: the coefficients in the
# buffer, or the caller's own for a complex field the transform does not overwrite.
function _backward_input!(p::FFTSynthesisPlan{RT, D, R}, coeffs, ::Nothing) where {RT, D, R}
    src = (R || p.neg) ? FFS._stage!(p.scratch, coeffs) : _staged(p.scratch, coeffs, Complex{RT})
    p.neg && (src .= conj.(src))
    return src
end

# A real field's split: the c2c along `2:D` into the buffer, out of place from coefficients already in the
# buffer's memory and type, in place in the buffer otherwise.
function _backward_input!(p::FFTSynthesisPlan, coeffs, pre)
    if !p.neg && coeffs isa typeof(p.scratch)
        LA.mul!(p.scratch, pre, coeffs)
    else
        FFS._stage!(p.scratch, coeffs)
        p.neg && (p.scratch .= conj.(p.scratch))
        p.pre! * p.scratch
    end
    return p.scratch
end

struct SplitFFTStage{I,O,F,B}
    input::I
    output::O
    forward::F
    backward::B
end

Base.show(io::IO, p::SplitFFTStage) = print(io, "SplitFFTStage(", size(p.input), " → ", size(p.output), ")")

function FFS.Plans._local_fft_stage(input::AbstractArray, output::AbstractArray,
                                  axis::Integer, execution, measure)
    options = FFS._fft_options(execution, measure, true)
    forward, backward = if eltype(input) <: Real
        AbstractFFTs.plan_rfft(input, axis:axis; options...),
        AbstractFFTs.plan_brfft(output, size(input, axis), axis:axis; options...)
    elseif input === output
        AbstractFFTs.plan_fft!(input, axis:axis; options...),
        AbstractFFTs.plan_bfft!(output, axis:axis; options...)
    else
        AbstractFFTs.plan_fft(input, axis:axis; options...),
        AbstractFFTs.plan_bfft(output, axis:axis; options...)
    end
    return SplitFFTStage(input, output, forward, backward)
end

function FFS.Plans._run_local_fft_stage!(stage::SplitFFTStage, ::Val{:forward})
    LA.mul!(stage.output, stage.forward, stage.input)
    return nothing
end

function FFS.Plans._run_local_fft_stage!(stage::SplitFFTStage, ::Val{:backward})
    LA.mul!(stage.input, stage.backward, stage.output)
    return nothing
end

function _fft_axis_layout(dims::NTuple{M,Int}, blocks, axis, ::Val{N}) where {M,N}
    slot = findfirst(==(axis), dims)
    slot === nothing && return dims
    blocks[slot] == 1 && return dims
    free = findfirst(d -> !(d in dims), 1:N)::Int
    return ntuple(k -> k == slot ? free : dims[k], Val(M))
end

function FFS.Plans._plan_split_fft(transform::SpectralBackends.AbstractAutoSpectralBackend,
                                 execution, split, ::Type{T}, ms; kwargs...) where {T}
    grid = FlowGeometries.Decomposition.global_grid(split)
    return FFS.Plans._plan_split_fft(FFS._resolve_transform(transform, grid, ms, execution),
        execution, split, T, ms; kwargs...)
end

function FFS.Plans._plan_split_fft(::SpectralBackends.AbstractFFTSpectralBackend, execution,
        split::FlowGeometries.Decomposition.AbstractSplitGrid{N}, ::Type{T}, ms::NTuple{N,Int};
        batch::Tuple = (), iflag::Int = 1, measure::Bool = true) where {N,T}
    grid = FlowGeometries.Decomposition.global_grid(split)
    FlowGeometries.Grids.grid_geometry(grid) isa FlowGeometries.Geometry.AbstractCartesianGeometry &&
        FlowGeometries.Grids.isuniform(grid) && all(FlowGeometries.Grids.periodic_flags(grid)) &&
        FlowGeometries.Grids.mask(grid) isa FlowGeometries.Grids.AllActive || throw(ArgumentError(
            "a block FFT requires an unmasked, uniform periodic Cartesian grid"))
    ms == size(grid) || throw(ArgumentError("a block FFT requires ms == size(grid) = $(size(grid))"))
    abs(iflag) == 1 || throw(ArgumentError("iflag must be +1 or -1"))
    return _split_fft_plan(execution, split, float(T), ms, FlowGeometries.Decomposition.split_dims(split),
        batch, iflag, measure)
end

function _split_fft_plan(execution, domain::FlowGeometries.Decomposition.AbstractSplitGrid{N},
                         ::Type{T}, ms, dims::NTuple{M,Int}, batch, iflag, measure) where {N,T,M}
    M < N || throw(ArgumentError("a block FFT needs at least one unsplit grid direction"))
    grid = FlowGeometries.Decomposition.global_grid(domain)
    ranges = FlowGeometries.Decomposition.map_blocks(b -> b.ranges, domain)
    blocks = map(d -> length(unique(map(r -> r[d], ranges))), dims)
    layouts = accumulate((layout, axis) -> _fft_axis_layout(layout, blocks, axis, Val(N)),
        ntuple(identity, Val(N)); init = dims)
    initial_dims = first(layouts)
    physical_split = FlowGeometries.Decomposition.split_grid(grid, execution;
        dims = initial_dims, blocks, halo = 0)
    field = FFS.Plans._zero_split_field(physical_split, T, batch)

    sizes = FTB.packed_size(T, ms)
    axes = map(n -> range(zero(real(T)); step = one(real(T)), length = n), sizes)
    lattice = FlowGeometries.Grids.StructuredGrid(FlowGeometries.Geometry.CartesianGeometry{real(T)}(),
        axes...; topology = ntuple(_ -> FlowGeometries.Grids.Periodic(), Val(N)))
    first_split = FlowGeometries.Decomposition.split_grid(lattice, execution;
        dims = initial_dims, blocks, halo = 0)
    first_modes = FFS.Plans._zero_split_field(first_split, Complex{real(T)}, batch)
    modes = accumulate(layouts; init = first_modes) do previous, dims
        FlowGeometries.Decomposition.split_dims(FlowGeometries.Decomposition.split_of(previous)) == dims &&
            return previous
        layout = FlowGeometries.Decomposition.split_grid(first_split; dims)
        return FFS.Plans._zero_split_field(layout, Complex{real(T)}, batch)
    end
    stages = ntuple(Val(N)) do axis
        input = axis == 1 ? field : modes[axis]
        FFS.Plans._split_fft_stage(execution, input, modes[axis], axis; measure)
    end
    ks = FFS.Grids.physical_wavenumbers(grid, ms, T)
    return FFS.Plans.SplitFFTPlan{T,typeof(domain),typeof(field),typeof(modes),typeof(stages),
        typeof(execution),typeof(batch),typeof(ks)}(domain, field, modes, stages, execution, batch, ks, iflag)
end

function FFS.calculate_spectrum!(coeffs::FlowGeometries.Decomposition.AbstractSplitField,
                               plan::FFS.Plans.SplitFFTPlan{T},
                               field::FlowGeometries.Decomposition.AbstractSplitField) where {T}
    field === plan.field || FlowGeometries.Decomposition.redistribute!(plan.field, field)
    if T <: Complex && plan.iflag < 0
        _conjugate_split!(plan.field)
    end
    FFS.Plans._run_split_fft_stage!(plan.execution, first(plan.stages), Val(:forward))
    FFS.Plans._forward_split_fft!(plan.execution, Base.tail(plan.modes), Base.tail(plan.stages), first(plan.modes))
    norm = one(real(T)) / prod(size(FlowGeometries.Decomposition.global_grid(plan.domain)))
    negative = plan.iflag < 0
    FFS.Plans._foreach_split_field(last(plan.modes)) do _, array
        if negative
            array .= conj.(array) .* norm
        else
            array .*= norm
        end
        return nothing
    end
    coeffs === last(plan.modes) || FlowGeometries.Decomposition.redistribute!(coeffs, last(plan.modes))
    return FFS.wavenumbers(plan)
end

function _conjugate_split!(field)
    FFS.Plans._foreach_split_field(field) do _, array
        array .= conj.(array)
        return nothing
    end
    return nothing
end

function FFS.Plans.synthesize!(out::FlowGeometries.Decomposition.AbstractSplitField,
                               inverse::FFS.Plans.SplitFFTSynthesisPlan,
                               coeffs::FlowGeometries.Decomposition.AbstractSplitField)
    plan = inverse.forward
    coeffs === last(plan.modes) || FlowGeometries.Decomposition.redistribute!(last(plan.modes), coeffs)
    plan.iflag < 0 && _conjugate_split!(last(plan.modes))
    FFS.Plans._backward_split_fft!(plan.execution, plan.modes, plan.stages)
    FFS.field_type(inverse) <: Complex && plan.iflag < 0 && _conjugate_split!(plan.field)
    out === plan.field || FlowGeometries.Decomposition.redistribute!(out, plan.field)
    return out
end

end # module FlowFieldSpectraAbstractFFTsExt
