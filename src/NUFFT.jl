# NUFFT transforms of Cartesian grids through FlowTransformBindings' plans. A plan's buffers are `Array`s
# under a host execution backend and device arrays under a `GPUBackend`, whose nodes FlowTransformBindings
# builds the library plan on; the KernelAbstractions extension allocates device arrays and gathers on
# them. Every transform runs in `FFTModes` order, so a spectrum arrives in native `fftfreq` order.
#
# FFS's `iflag` is the sign of `-i` in its type-1 exponent, `Σ f e^{-iflag·ik·x}`, so FlowTransformBindings
# receives `-iflag`. A real field's tables are built for `iflag = +1`: its `iflag = -1` spectrum is the
# conjugate, applied on publish.

const _FTBLibrary = Union{FTB.FINUFFTBackend, FTB.NonuniformFFTsBackend}
const _HostExec = Union{ComputationalBackends.AbstractSerialBackend, ComputationalBackends.AbstractThreadedBackend}

# ---- where a plan's arrays live ----

_alloc(::_HostExec, ::Type{T}, dims::Integer...) where {T} = Array{T}(undef, dims...)
_alloc(exec::ComputationalBackends.AbstractExecutionBackend, ::Type, ::Integer...) = throw(ArgumentError(
    "a NUFFT plan runs on SerialBackend, ThreadedBackend or GPUBackend (with `using KernelAbstractions`); " *
    "got $(nameof(typeof(exec)))"))

_to_exec(::_HostExec, a::AbstractArray) = a
_to_exec(exec, a::AbstractArray) = copyto!(_alloc(exec, eltype(a), size(a)...), a)

_on_device(::_HostExec) = false
_on_device(::ComputationalBackends.AbstractGPUBackend) = true

# ---- transform counts and tolerance when the caller names none ----

_nufft_tol(::Type{Tr}, ::Nothing) where {Tr} = FTB.default_tolerance(Tr)
_nufft_tol(::Type{Tr}, eps::Real) where {Tr} = Tr(eps)

# Transforms per execution. FINUFFT threads over the transforms of one execution, so it takes a whole
# batch. NonuniformFFTs holds an oversampled grid and a spreading buffer per transform, so on the host it
# takes four, keeping plan memory independent of the batch.
_batch_default(::FTB.FINUFFTBackend, total::Int, nth::Int, dev::Bool) = total
_batch_default(::FTB.NonuniformFFTsBackend, total::Int, nth::Int, dev::Bool) = dev ? total : 4

# Lines per execution along one axis of a separable transform. A host FINUFFT pass gives each of its
# threads at least one line.
_line_default(::FTB.FINUFFTBackend, total::Int, nth::Int, dev::Bool) = dev ? total : max(4, nth)
_line_default(::FTB.NonuniformFFTsBackend, total::Int, nth::Int, dev::Bool) = dev ? total : 4

# `batch_chunk ≤ 0` runs every transform in one execution.
_chunk(default::Int, total::Int, ::Nothing) = clamp(default, 1, max(total, 1))
_chunk(::Int, total::Int, bc::Integer) = bc <= 0 ? max(total, 1) : clamp(Int(bc), 1, max(total, 1))

# ---- layout tables ----

# Frequency at index `j` of a packed axis of `n` modes: `0 … n÷2` on the halved axis 1 of a real field,
# `fftfreq` order elsewhere.
@inline _axis_freq(j::Int, n::Int, halved::Bool) =
    halved ? j - 1 : (j - 1 <= (n - 1) ÷ 2 ? j - 1 : j - 1 - n)

"""
    _packed_src(ms, ::Val{R}, dims) -> Vector{Int}

For each packed coefficient, its linear index in one transform's spectrum of shape `dims` in `FFTModes`
order: the same frequency, on axes holding at least the packed ones.
"""
function _packed_src(ms::NTuple{D, Int}, ::Val{R}, dims::NTuple{D, Int}) where {D, R}
    pms = Packing.packed_size(ms, Val(R))
    src = Vector{Int}(undef, prod(pms))
    @inbounds for (i, I) in enumerate(CartesianIndices(pms))
        lin = 1
        stride = 1
        for d in 1:D
            k = _axis_freq(I[d], ms[d], R && d == 1)
            lin += (Packing.ovs_index(k, dims[d]) - 1) * stride
            stride *= dims[d]
        end
        src[i] = lin
    end
    return src
end

# ---- gathers ----

# `dst[doff + i] = S(src[soff + idx[i]]) · w[i]` for `i ≤ n`, `S` the conjugate where `csrc`, the product
# conjugated where `neg`. The device method is the KernelAbstractions extension's.
function _gather_scaled!(dst::AbstractArray, doff::Int, src::Array, soff::Int, idx::Array{Int}, w::Array,
        n::Int, csrc::Bool, neg::Bool)
    @inbounds for i in 1:n
        s = src[soff + idx[i]]
        v = (csrc ? conj(s) : s) * w[i]
        dst[doff + i] = neg ? conj(v) : v
    end
    return dst
end

# A host plan gathers into the caller's array; a device plan gathers into its staging buffer and copies it
# across.
@inline _publish!(dst, doff, src, soff, idx, w, n, csrc, neg, ::Nothing) =
    _gather_scaled!(dst, doff, src, soff, idx, w, n, csrc, neg)
function _publish!(dst, doff, src, soff, idx, w, n, csrc, neg, stage::AbstractArray)
    _gather_scaled!(stage, 0, src, soff, idx, w, n, csrc, neg)
    copyto!(dst, doff + 1, stage, 1, n)
    return dst
end

"""
    TwinGather

One mask's [`Packing.NyquistTwin`](@ref) slice with the [`Packing.twin_table`](@ref) that fills it from a
spectrum, a conjugate read where `csrc`.
"""
struct TwinGather{SL, IX, FC, ST}
    slice::SL        # host values, `(shape…, batch…)`
    src::IX
    fac::FC
    stage::ST        # device staging, or `nothing` on the host
    n::Int
    csrc::Bool
end

function _twin_set(exec, ::Type{Tr}, ks_phys::Tuple, ms::NTuple{D, Int}, dims::NTuple{D, Int},
        offsets, ranges, M::Int, batch::Tuple; conjugate::Bool, phis = nothing,
        normfactor::Real = 1) where {Tr, D}
    D >= 2 || return ks_phys, ()
    dev = _on_device(exec)
    gs = ntuple(Packing.n_twin_slices(Val(D))) do mask
        shape, src, fac = Packing.twin_table(Tr, ms, mask, dims, offsets, ranges, M; phis, normfactor,
            conjugate)
        S = length(src)
        TwinGather(zeros(Complex{Tr}, shape..., batch...), _to_exec(exec, S == 0 ? [1] : src),
            _to_exec(exec, S == 0 ? [zero(Complex{Tr})] : fac),
            dev ? _alloc(exec, Complex{Tr}, max(S, 1)) : nothing, S, conjugate)
    end
    twin = Packing.NyquistTwin(map(g -> g.slice, gs))
    return (Packing.with_twin(ks_phys[1], twin), Base.tail(ks_phys)...), gs
end

@inline _gather_twins!(::Tuple{}, fk, soff::Int, t::Int, neg::Bool) = nothing
@inline function _gather_twins!(gs::Tuple, fk, soff::Int, t::Int, neg::Bool)
    g = first(gs)
    g.n == 0 || _publish!(g.slice, (t - 1) * g.n, fk, soff, g.src, g.fac, g.n, g.csrc, neg, g.stage)
    return _gather_twins!(Base.tail(gs), fk, soff, t, neg)
end

# ---- field staging ----

# Column `c` of the strengths from the field's values `foff+1 … foff+M`, weighted by the grid quadrature
# `qw`. A complex strength buffer takes a real field widened.
function _fill_column!(V::Array, c::Int, field, foff::Int, M::Int, qw)
    o = (c - 1) * M
    @inbounds for i in 1:M
        z = convert(eltype(V), field[foff + i])
        V[o + i] = qw === nothing ? z : z * qw[i]
    end
    return V
end
function _fill_column!(V::AbstractArray, c::Int, field, foff::Int, M::Int, qw)
    col = view(V, :, c)
    if eltype(field) === eltype(V)
        copyto!(V, (c - 1) * M + 1, field, foff + 1, M)
    else
        copyto!(col, eltype(V).(view(vec(field), (foff + 1):(foff + M))))
    end
    qw === nothing || (col .*= qw)
    return V
end

# `A` scaled point by point by the grid quadrature, over each of its `ntrans` fields.
function _scale_points!(A::Array, qw, npts::Int, ntrans::Int)
    qw === nothing && return A
    @inbounds for b in 1:ntrans, j in 1:npts
        A[j + (b - 1) * npts] *= qw[j]
    end
    return A
end
function _scale_points!(A::AbstractArray, qw, npts::Int, ntrans::Int)
    qw === nothing || (reshape(A, npts, ntrans) .*= qw)
    return A
end

# The field copied into a working array of the transform's element type.
_stage!(A::Array, field) = copyto!(A, field)
_stage!(A::AbstractArray, field) = copyto!(A, eltype(field) === eltype(A) ? field : eltype(A).(field))

# =============================================================================
# Point sets: a node cloud, and a curvilinear grid's cells.
#
# A real field on NonuniformFFTs takes a real-data plan, whose spectrum is the packed half. Its Nyquist
# twins are computed by that transform already: the type 1 FFTs onto an oversampled grid and keeps only
# the native box, so `+N_d/2` is there before truncation, and each twin is one oversampled entry times the
# deconvolution the kept modes get (`FlowTransformBindings.oversampled_spectra`).
#
# A real field on FINUFFT takes complex strengths at `Packing.hermitian_request_size`, which puts `±N₁/2`
# on axis 1; the packed half is the leading axis-1 entries and each twin a conjugate read of the Hermitian
# spectrum.
# =============================================================================

# How a real field's point plan is built and where its twins are read.
_real_values(::FTB.NonuniformFFTsBackend) = true
_real_values(::FTB.FINUFFTBackend) = false

"""
    NUFFTPointPlan{T,D,R}

Reusable type-1 NUFFT over a Cartesian point set (a node cloud or a curvilinear grid's cells): the
FlowTransformBindings plan with `C` transforms per execution, its strength and spectrum buffers, and the
tables that publish the packed layout and its Nyquist twins. `R` marks a real field. Execute with
`calculate_spectrum!(coeffs, plan, field)`; [`close!`](@ref) releases the library plan.
"""
struct NUFFTPointPlan{T, D, R, NB, P, V, U, PH, IX, ST, KS, TW, QW} <: Plans.AbstractSpectralPlan
    plan::P
    values::V                        # (M, C) strengths
    modes::U                         # (mode_size…, C) spectra
    phase::PH                        # packed-layout offset phase × 1/M
    src::IX                          # packed index → index into one transform's spectrum
    stage::ST                        # device staging for the coefficients, or `nothing` on the host
    batch::NTuple{NB, Int}
    ms::NTuple{D, Int}
    pms::NTuple{D, Int}
    M::Int
    B::Int                           # transforms the field carries
    C::Int                           # transforms per execution
    neg::Bool                        # real field with `iflag < 0`: conjugate the published values
    ks_phys::KS
    twins::TW
    qw::QW                           # (M,) grid quadrature factor, or `nothing` for a constant measure
end

Base.show(io::IO, p::NUFFTPointPlan{T, D, R}) where {T, D, R} =
    print(io, "NUFFTPointPlan{", T, ", ", D, "}(", R ? "real" : "complex", ", ", p.plan, ")")

Plans.coefficient_size(p::NUFFTPointPlan) = (p.pms..., p.batch...)
Plans.coefficient_type(::NUFFTPointPlan{T}) where {T} = Complex{T}
Plans.wavenumbers(p::NUFFTPointPlan) = p.ks_phys
Plans.close!(p::NUFFTPointPlan) = FTB.close!(p.plan)

function _point_plan(t::_FTBLibrary, exec, g, ::Type{T}, ms::NTuple{D, Int}, batch::Tuple, iflag::Int,
        eps, batch_chunk) where {T, D}
    Tr = real(float(T))
    R = T <: Real
    coords, _ = Grids.point_coordinates(Tr, g, D)
    Ls = ntuple(d -> Grids.axis_range(Tr, g, d), Val(D))
    M = length(coords[1])
    offsets = ntuple(d -> Tr(minimum(coords[d])), Val(D))
    B = prod(batch; init = 1)
    nth = _backend_nthreads(exec)
    dev = _on_device(exec)
    C = _chunk(_batch_default(t, B, nth, dev), B, batch_chunk)
    realplan = R && _real_values(t)
    ns = (R && !realplan) ? Packing.hermitian_request_size(ms) : ms
    VT = realplan ? Tr : Complex{Tr}
    plan = FTB.plan_nufft(t, VT, ntuple(d -> _to_exec(exec, coords[d]), Val(D)), ns; ntrans = C,
        tol = _nufft_tol(Tr, eps), order = FTB.FFTModes(), period = Ls, origin = offsets, nthreads = nth,
        iflag = R ? -1 : -iflag)
    dims = FTB.mode_size(plan)
    pms = Packing.packed_size(ms, Val(R))
    phase = _to_exec(exec, Packing.offset_phase(Tr, ms, offsets, Ls, M, Val(R), R ? 1 : iflag))
    ks_phys = Grids.physical_wavenumbers(Ls, ms, Val(R))
    ks, twins = if !R
        ks_phys, ()
    elseif realplan
        us, normfactor, phis = FTB.oversampled_spectra(plan)
        _twin_set(exec, Tr, ks_phys, ms, size(first(us)), offsets, Ls, M, batch; conjugate = false,
            phis = map(collect, phis), normfactor)
    else
        _twin_set(exec, Tr, ks_phys, ms, dims, offsets, Ls, M, batch; conjugate = true)
    end
    qwh = Grids.quadrature_scale(g, Tr, M)
    qw = qwh === nothing ? nothing : _to_exec(exec, collect(Tr, qwh))
    bt = NTuple{length(batch), Int}(batch)
    values = _alloc(exec, VT, M, C)
    modes = _alloc(exec, Complex{Tr}, dims..., C)
    src = _to_exec(exec, _packed_src(ms, Val(R), dims))
    stage = dev ? _alloc(exec, Complex{Tr}, prod(pms)) : nothing
    return NUFFTPointPlan{Tr, D, R, length(bt), typeof(plan), typeof(values), typeof(modes), typeof(phase),
            typeof(src), typeof(stage), typeof(ks), typeof(twins), typeof(qw)}(
        plan, values, modes, phase, src, stage, bt, ms, pms, M, B, C, R && iflag < 0, ks, twins, qw)
end

# The twins of transform `t`, the `c`-th of its execution. A real-data plan's are read from the oversampled
# spectra, one array per transform; a complex plan's from its own spectrum.
_point_twins!(p::NUFFTPointPlan, c::Int, t::Int) = _point_twins!(p, p.plan, c, t)
function _point_twins!(p::NUFFTPointPlan, ::FTB.AbstractNUFFTPlan{<:Real}, c::Int, t::Int)
    us, _, _ = FTB.oversampled_spectra(p.plan)
    _gather_twins!(p.twins, us[c], 0, t, p.neg)
end
_point_twins!(p::NUFFTPointPlan, ::FTB.AbstractNUFFTPlan, c::Int, t::Int) =
    _gather_twins!(p.twins, p.modes, (c - 1) * (length(p.modes) ÷ p.C), t, p.neg)

"""
    calculate_spectrum!(coeffs, plan::NUFFTPointPlan, field) -> ks_phys

Execute a point-set NUFFT plan in place. `field` is `(N, batch…)`; `coeffs` is the packed half
`(ms[1]÷2+1, ms[2:D]…, batch…)` for a real field and the full native spectrum for a complex one.
"""
function calculate_spectrum!(coeffs::AbstractArray{Complex{T}}, p::NUFFTPointPlan{T, D, R},
        field) where {T, D, R}
    M, B, C = p.M, p.B, p.C
    length(field) == M * B || throw(DimensionMismatch(
        "field holds $(length(field)) values; this plan transforms $B field(s) of $M points — pass the " *
        "matching `batch=` to plan_spectrum"))
    size(coeffs) == Plans.coefficient_size(p) || throw(DimensionMismatch(
        "coeffs is $(size(coeffs)); this plan writes $(Plans.coefficient_size(p)) — allocate it with " *
        "`allocate_coefficients(plan)`"))
    Pn = length(p.modes) ÷ C
    Ph = prod(p.pms)
    for base in 0:C:(B - 1)
        nvalid = min(C, B - base)
        for c in 1:C
            if c <= nvalid
                _fill_column!(p.values, c, field, (base + c - 1) * M, M, p.qw)
            else
                fill!(view(p.values, :, c), zero(eltype(p.values)))
            end
        end
        FTB.nufft_type1!(p.modes, p.plan, p.values)
        for c in 1:nvalid
            t = base + c
            _publish!(coeffs, (t - 1) * Ph, p.modes, (c - 1) * Pn, p.src, p.phase, Ph, false, p.neg, p.stage)
            R && _point_twins!(p, c, t)
        end
    end
    return p.ks_phys
end

function Plans.plan_spectrum(t::_FTBLibrary,
        exec::Union{_HostExec, ComputationalBackends.AbstractGPUBackend}, g::Grids.PointwiseCartesian,
        ::Type{T}, ms::NTuple{D, Int}; batch::Tuple = (), iflag::Int = 1, eps = nothing,
        batch_chunk::Union{Nothing, Integer} = nothing) where {T, D}
    return _point_plan(t, exec, g, T, ms, batch, iflag, eps, batch_chunk)
end

function _nufft_one_shot(t::_FTBLibrary, exec, g, field, ms::NTuple{D, Int}; iflag::Int = 1,
        eps = nothing, batch_chunk = nothing, kwargs...) where {D}
    E = float(eltype(field))
    batch = Grids.field_batch_shape(g, field)
    plan = Plans.plan_spectrum(t, exec, g, E, ms; batch, iflag, eps, batch_chunk)
    try
        coeffs = zeros(Complex{real(E)}, Plans.coefficient_size(plan)...)
        ks = calculate_spectrum!(coeffs, plan, field)
        return coeffs, ks
    finally
        Plans.close!(plan)
    end
end

_calculate_spectrum_nufft(t::_FTBLibrary, exec::_HostExec,
        g::Union{Grids.PointwiseCartesian,
                 FlowGeometries.Grids.AbstractStructuredGrid{<:FlowGeometries.Geometry.AbstractCartesianGeometry}},
        field::AbstractArray, ms::Tuple; kwargs...) =
    _nufft_one_shot(t, exec, g, field, NTuple{length(ms), Int}(ms); kwargs...)

_calculate_spectrum_gpu_nufft(t::_FTBLibrary, exec::ComputationalBackends.AbstractGPUBackend,
        g::Union{Grids.PointwiseCartesian,
                 FlowGeometries.Grids.AbstractStructuredGrid{<:FlowGeometries.Geometry.AbstractCartesianGeometry}},
        field::AbstractArray, ms::Tuple; kwargs...) =
    _nufft_one_shot(t, exec, g, field, NTuple{length(ms), Int}(ms); kwargs...)

# =============================================================================
# One axis of a separable or hybrid transform: a 1-D type-1 NUFFT over the axis's grid points, every other
# dim of the working array a transform of the batch. `iflag` is FlowTransformBindings' sign.
# =============================================================================

"""
    NUFFTAxisPass

One grid axis's 1-D type-1 NUFFT: the plan with `C` lines per execution, the strength and spectrum
buffers, the host line offsets, and the device staging for the axis-to-front permutation.
"""
struct NUFFTAxisPass{P, V, U, ST}
    plan::P
    values::V                        # (N_d, C)
    modes::U                         # (m, C)
    inoff::Vector{Int}
    outoff::Vector{Int}
    stage::ST                        # device `(pin, pfk)`, or `nothing` on the host
    Nd::Int
    m::Int
    C::Int
end

Base.show(io::IO, a::NUFFTAxisPass) =
    print(io, "NUFFTAxisPass(N=", a.Nd, " → m=", a.m, ", chunk=", a.C, ")")

_close!(a::NUFFTAxisPass) = FTB.close!(a.plan)

@inline _axis_perm(nd::Int, d::Int) = (d, ntuple(i -> i < d ? i : i + 1, nd - 1)...)

function _axis_pass(t::_FTBLibrary, exec, ::Type{Tr}, insize::Tuple, d::Int, axis::AbstractVector,
        m::Int, rng, off, tol, batch_chunk; iflag::Int = -1) where {Tr}
    Nd = length(axis)
    insize[d] == Nd || throw(DimensionMismatch("axis $d: working length $(insize[d]) ≠ grid axis length $Nd"))
    pre, _, post = Packing.axis_layout(insize, d)
    rest = pre * post
    nth = _backend_nthreads(exec)
    dev = _on_device(exec)
    C = _chunk(_line_default(t, rest, nth, dev), rest, batch_chunk)
    plan = FTB.plan_nufft(t, Complex{Tr}, (_to_exec(exec, collect(Tr, axis)),), (m,); ntrans = C,
        tol = Tr(tol), order = FTB.FFTModes(), period = Tr(rng), origin = Tr(off), nthreads = nth, iflag)
    stage = dev ? (_alloc(exec, Complex{Tr}, Nd, rest), _alloc(exec, Complex{Tr}, m, rest)) : nothing
    return NUFFTAxisPass(plan, _alloc(exec, Complex{Tr}, Nd, C), _alloc(exec, Complex{Tr}, m, C),
        Vector{Int}(undef, C), Vector{Int}(undef, C), stage, Nd, m, C)
end

# Transform axis `d` of `A` into `out`, whose dim `d` holds `ap.m` modes. On the host the lines are
# gathered straight out of `A` and scattered straight into `out`, so neither is permuted.
function _axis_run!(out::Array, A::Array, d::Int, ap::NUFFTAxisPass)
    pre, Nd, post = Packing.axis_layout(size(A), d)
    Nd == ap.Nd || throw(DimensionMismatch("axis $d: field length $Nd ≠ this plan's grid axis length $(ap.Nd)"))
    m, C = ap.m, ap.C
    V, U = ap.values, ap.modes
    rest = pre * post
    for base in 0:C:(rest - 1)
        nvalid = min(C, rest - base)
        Packing.axis_chunk_offsets!(ap.inoff, ap.outoff, base, nvalid, pre, Nd, m)
        @inbounds for i in 1:Nd
            s = (i - 1) * pre
            for c in 1:nvalid
                V[i, c] = A[ap.inoff[c] + s]
            end
            for c in (nvalid + 1):C
                V[i, c] = zero(eltype(V))
            end
        end
        FTB.nufft_type1!(U, ap.plan, V)
        @inbounds for j in 1:m
            s = (j - 1) * pre
            for c in 1:nvalid
                out[ap.outoff[c] + s] = U[j, c]
            end
        end
    end
    return out
end

# On a device the axis moves to the front, so each line is a contiguous column of the staging array.
function _axis_run!(out::AbstractArray, A::AbstractArray, d::Int, ap::NUFFTAxisPass)
    pin, pfk = ap.stage
    nd = ndims(A)
    perm = _axis_perm(nd, d)
    Nd, m, C = ap.Nd, ap.m, ap.C
    rest = size(pin, 2)
    permutedims!(reshape(pin, ntuple(i -> size(A, perm[i]), nd)), A, perm)
    for base in 0:C:(rest - 1)
        nvalid = min(C, rest - base)
        copyto!(ap.values, 1, pin, base * Nd + 1, nvalid * Nd)
        nvalid < C && fill!(view(ap.values, :, (nvalid + 1):C), zero(eltype(ap.values)))
        FTB.nufft_type1!(ap.modes, ap.plan, ap.values)
        copyto!(pfk, base * m + 1, ap.modes, 1, nvalid * m)
    end
    permutedims!(out, reshape(pfk, ntuple(i -> size(out, perm[i]), nd)), invperm(collect(perm)))
    return out
end

# The hybrid composite's stretched axes (`_calculate_spectrum_hybrid`, `HybridPlan`): the `Σ e^{-ikx}`
# sign of its FFT pass, the composite conjugating once for `iflag = -1`.
function _axis_nufft(t::_FTBLibrary, exec::ComputationalBackends.AbstractExecutionBackend,
        A::AbstractArray{Complex{Tr}}, d::Int, axis::AbstractVector, m::Int, rng, off, eps;
        batch_chunk = nothing, kwargs...) where {Tr}
    ap = _axis_pass(t, exec, Tr, size(A), d, axis, m, rng, off, eps, batch_chunk)
    try
        return _axis_run!(_alloc(exec, Complex{Tr}, Packing.axis_out_size(size(A), d, m)...), A, d, ap)
    finally
        _close!(ap)
    end
end

_axis_nufft_plan(t::_FTBLibrary, exec::ComputationalBackends.AbstractExecutionBackend, ::Type{Tr},
        insize::Tuple, d::Int, axis::AbstractVector, m::Int, rng, off, eps; batch_chunk = nothing,
        kwargs...) where {Tr} =
    _axis_pass(t, exec, Tr, insize, d, axis, m, rng, off, eps, batch_chunk)

_axis_nufft_exec!(out::AbstractArray, ap::NUFFTAxisPass, A::AbstractArray, d::Int) = _axis_run!(out, A, d, ap)

@inline _run_axes!(work::Tuple, ::Tuple{}, d::Int) = nothing
@inline function _run_axes!(work::Tuple, axes::Tuple, d::Int)
    _axis_run!(work[d + 1], work[d], d, first(axes))
    return _run_axes!(work, Base.tail(axes), d + 1)
end

# =============================================================================
# Separable transform for a nonuniform tensor-product (structured) Cartesian grid: `D` successive 1-D
# passes, so no `∏N_d` coordinate cloud is built and spreading costs `D·2m` per point in place of `(2m)^D`.
# Every pass is complex, so a real field's result is the full spectrum at `hermitian_request_size`, whose
# packed half is its leading axis-1 entries and whose Nyquist twins are conjugate reads.
# =============================================================================

"""
    NUFFTSeparablePlan{T,D,R}

Reusable separable NUFFT over a nonuniform tensor-product Cartesian grid: one [`NUFFTAxisPass`](@ref)
per axis, the working arrays the passes write through, and the publish and twin tables. Execute with
`calculate_spectrum!(coeffs, plan, field)`; [`close!`](@ref) releases the library plans.
"""
struct NUFFTSeparablePlan{T, D, R, NB, AP, W, PH, IX, ST, KS, TW, QW} <: Plans.AbstractSpectralPlan
    axes::AP                         # D × NUFFTAxisPass
    work::W                          # D+1 working arrays; `work[1]` takes the field
    phase::PH                        # offset phase × 1/∏N_d
    src::IX
    stage::ST
    batch::NTuple{NB, Int}
    ms::NTuple{D, Int}
    ns::NTuple{D, Int}               # mode counts the passes produce
    pms::NTuple{D, Int}
    npts::Int
    ntrans::Int
    neg::Bool                        # real field with `iflag < 0`: conjugate the published values
    ks_phys::KS
    twins::TW
    qw::QW
end

Base.show(io::IO, p::NUFFTSeparablePlan{T, D, R}) where {T, D, R} =
    print(io, "NUFFTSeparablePlan{", T, ", ", D, "}(", R ? "real" : "complex", ", ", first(p.axes).plan, ")")

Plans.coefficient_size(p::NUFFTSeparablePlan) = (p.pms..., p.batch...)
Plans.coefficient_type(::NUFFTSeparablePlan{T}) where {T} = Complex{T}
Plans.wavenumbers(p::NUFFTSeparablePlan) = p.ks_phys
Plans.close!(p::NUFFTSeparablePlan) = (foreach(_close!, p.axes); nothing)

function Plans.plan_spectrum(t::_FTBLibrary,
        exec::Union{_HostExec, ComputationalBackends.AbstractGPUBackend},
        g::FlowGeometries.Grids.AbstractStructuredGrid{<:FlowGeometries.Geometry.AbstractCartesianGeometry},
        ::Type{T}, ms::NTuple{D, Int}; batch::Tuple = (), iflag::Int = 1, eps = nothing,
        batch_chunk::Union{Nothing, Integer} = nothing) where {T, D}
    Tr = real(float(T))
    R = T <: Real
    ndims(g) == D || throw(DimensionMismatch("grid has $(ndims(g)) dims; asked for $D mode counts"))
    axs = ntuple(d -> Tr.(FlowGeometries.Grids.coordinates(g, d)), Val(D))
    offsets, ranges = Grids.axis_geometry(Tr, g, D)
    Ns = ntuple(d -> length(axs[d]), Val(D))
    npts = prod(Ns)
    ntrans = prod(batch; init = 1)
    ns = R ? Packing.hermitian_request_size(ms) : ms
    tol = _nufft_tol(Tr, eps)
    sgn = R ? -1 : -iflag
    work = ntuple(d -> _alloc(exec, Complex{Tr}, Packing.axis_work_shape(Ns, ns, d, batch)...), Val(D + 1))
    axes = ntuple(d -> _axis_pass(t, exec, Tr, size(work[d]), d, axs[d], ns[d], ranges[d], offsets[d], tol,
        batch_chunk; iflag = sgn), Val(D))
    pms = Packing.packed_size(ms, Val(R))
    phase = _to_exec(exec, Packing.offset_phase(Tr, ms, offsets, ranges, npts, Val(R), R ? 1 : iflag))
    ks_phys = Grids.physical_wavenumbers(ranges, ms, Val(R))
    ks, twins = R ? _twin_set(exec, Tr, ks_phys, ms, ns, offsets, ranges, npts, batch; conjugate = true) :
                    (ks_phys, ())
    qwh = Grids.quadrature_scale(g, Tr, npts)
    qw = qwh === nothing ? nothing : _to_exec(exec, collect(Tr, qwh))
    src = _to_exec(exec, _packed_src(ms, Val(R), ns))
    stage = _on_device(exec) ? _alloc(exec, Complex{Tr}, prod(pms)) : nothing
    bt = NTuple{length(batch), Int}(batch)
    return NUFFTSeparablePlan{Tr, D, R, length(bt), typeof(axes), typeof(work), typeof(phase), typeof(src),
            typeof(stage), typeof(ks), typeof(twins), typeof(qw)}(
        axes, work, phase, src, stage, bt, ms, ns, pms, npts, ntrans, R && iflag < 0, ks, twins, qw)
end

"""
    calculate_spectrum!(coeffs, plan::NUFFTSeparablePlan, field) -> ks_phys

Execute a separable structured plan in place. `field` is `(N₁…N_D, batch…)`; `coeffs` is the packed half
for a real field and the full native spectrum for a complex one.
"""
function calculate_spectrum!(coeffs::AbstractArray{Complex{T}}, p::NUFFTSeparablePlan{T, D, R},
        field) where {T, D, R}
    A = p.work[1]
    length(field) == length(A) || throw(DimensionMismatch(
        "field holds $(length(field)) values; this plan was built for $(length(A)) — pass the matching " *
        "`batch=` to plan_spectrum"))
    size(coeffs) == Plans.coefficient_size(p) || throw(DimensionMismatch(
        "coeffs is $(size(coeffs)); this plan writes $(Plans.coefficient_size(p)) — allocate it with " *
        "`allocate_coefficients(plan)`"))
    _stage!(A, field)
    _scale_points!(A, p.qw, p.npts, p.ntrans)
    _run_axes!(p.work, p.axes, 1)
    F = p.work[D + 1]
    Pn = prod(p.ns)
    Ph = prod(p.pms)
    for t in 1:p.ntrans
        _publish!(coeffs, (t - 1) * Ph, F, (t - 1) * Pn, p.src, p.phase, Ph, false, p.neg, p.stage)
        _gather_twins!(p.twins, F, (t - 1) * Pn, t, p.neg)
    end
    return p.ks_phys
end

# =============================================================================
# The hybrid composite on a device: the FFT over the uniform axes from the GPU FFT extension, each stretched
# axis a device `NUFFTAxisPass`, and the publish and twins as device gathers. The host composite lives with the
# rest of the hybrid in the package root.
# =============================================================================

function _calculate_spectrum_hybrid(t::_FTBLibrary, exec::ComputationalBackends.AbstractGPUBackend,
        g::FlowGeometries.Grids.AbstractStructuredGrid{<:FlowGeometries.Geometry.AbstractCartesianGeometry},
        field::AbstractArray, ms::NTuple{D, Int}, umask::NTuple{D, Bool}; iflag::Int = 1, eps = nothing,
        batch_chunk = nothing, kwargs...) where {D}
    Tr = real(float(eltype(g)))
    R = eltype(field) <: Real
    batch = ntuple(i -> size(field, D + i), ndims(field) - D)
    ntrans = prod(batch; init = 1)
    h = _hybrid_derive(g, Tr, ms, umask, R, iflag, eps, batch)
    W = _region_fft(exec, quadrature_weighted(g, field), h.udims, h.halve, !R && h.neg)
    for d in h.sdims
        W = _axis_nufft(t, exec, W, d, h.axs[d], ms[d], h.ranges[d], h.offs_all[d], h.epsv;
            batch_chunk = batch_chunk)
    end
    Pn = prod(h.nsx)
    if R
        pms = h.pms
        Ph = prod(pms)
        coeffs = Array{Complex{Tr}}(undef, pms..., batch...)
        src = _to_exec(exec, _packed_src(ms, Val(true), h.nsx))
        phase = _to_exec(exec, h.phase)
        stage = _alloc(exec, Complex{Tr}, Ph)
        ks, twins = h.need_twin ?
            _twin_set(exec, Tr, h.ks_phys, ms, h.nsx, h.offs, h.ranges, h.npts, batch; conjugate = true) :
            (h.ks_phys, ())
        for k in 1:ntrans
            _publish!(coeffs, (k - 1) * Ph, W, (k - 1) * Pn, src, phase, Ph, false, h.neg, stage)
            _gather_twins!(twins, W, (k - 1) * Pn, k, h.neg)
        end
        return coeffs, ks
    end
    h.neg && (W .= conj.(W))                     # closes the conjugation applied to the input
    W .*= _to_exec(exec, h.phase)
    coeffs = Array{Complex{Tr}}(undef, ms..., batch...)
    copyto!(coeffs, W)
    return coeffs, h.ks
end

# =============================================================================
# Synthesis: a complex type 2 on the full native spectrum. A real field's packed half is first completed by
# `Packing.unpacked` with its Nyquist twin, and the real part taken. The forward publishes `C = fk · p` with
# `p` the offset phase × 1/M, so the type 2 takes `û = C · conj(p) · M` and evaluates
# `Σ û e^{+iflag·ik·y}`.
# =============================================================================

"""
    NUFFTSynthesisPlan{T,D,R}

Reusable type-2 NUFFT inverse onto a Cartesian grid's points: the plan with `C` transforms per execution,
its spectrum and strength buffers (with host staging on a device), the offset phase, and the native cube a
packed half expands into. Execute with `synthesize!(out, plan, coeffs; ks)`; [`close!`](@ref) releases
the library plan.
"""
struct NUFFTSynthesisPlan{T, D, R, NB, P, U, V, HU, HV, PH, FB, SP} <: Plans.AbstractSynthesisPlan
    plan::P
    modes::U                         # (ms…, C)
    values::V                        # (M, C)
    hmodes::HU                       # host staging for `modes`, or `modes` itself on the host
    hvalues::HV                      # host staging for `values`, or `values` itself on the host
    phase::PH                        # host (ms…) offset phase × M, built for `iflag`
    full::FB                         # host (ms…, ntrans)
    ms::NTuple{D, Int}
    spatial::SP
    batch::NTuple{NB, Int}
    ntrans::Int
    M::Int
    C::Int
end

Base.show(io::IO, p::NUFFTSynthesisPlan{T, D, R}) where {T, D, R} =
    print(io, "NUFFTSynthesisPlan{", T, ", ", D, "}(", R ? "real" : "complex", ", ", p.plan, ")")

Plans.field_size(p::NUFFTSynthesisPlan) = (p.spatial..., p.batch...)
Plans.field_type(::NUFFTSynthesisPlan{T, D, R}) where {T, D, R} = R ? T : Complex{T}
Plans.close!(p::NUFFTSynthesisPlan) = FTB.close!(p.plan)

function Plans.plan_synthesis(t::_FTBLibrary,
        exec::Union{_HostExec, ComputationalBackends.AbstractGPUBackend},
        g::Union{FlowGeometries.Grids.AbstractStructuredGrid{<:FlowGeometries.Geometry.AbstractCartesianGeometry},
                 Grids.PointwiseCartesian},
        ::Type{TT}, ms::NTuple{D, Int}; batch::Tuple = (), iflag::Int = 1, eps = nothing,
        batch_chunk::Union{Nothing, Integer} = nothing, kwargs...) where {TT, D}
    T = real(float(TT))
    R = TT <: Real
    coords, spatial = Grids.point_coordinates(T, g, D)
    Ls = ntuple(d -> Grids.axis_range(T, g, d), Val(D))
    M = length(coords[1])
    offsets = ntuple(d -> T(minimum(coords[d])), Val(D))
    bt = NTuple{length(batch), Int}(batch)
    ntrans = prod(bt; init = 1)
    nth = _backend_nthreads(exec)
    dev = _on_device(exec)
    C = _chunk(_batch_default(t, ntrans, nth, dev), ntrans, batch_chunk)
    # The type 2 evaluates `Σ û e^{+iflag·ik·y}`, FlowTransformBindings' `-iflag` convention.
    plan = FTB.plan_nufft(t, Complex{T}, ntuple(d -> _to_exec(exec, coords[d]), Val(D)), ms; ntrans = C,
        tol = _nufft_tol(T, eps), order = FTB.FFTModes(), period = Ls, origin = offsets, nthreads = nth,
        iflag = -iflag)
    modes = _alloc(exec, Complex{T}, ms..., C)
    values = _alloc(exec, Complex{T}, M, C)
    hmodes = dev ? Array{Complex{T}}(undef, ms..., C) : modes
    hvalues = dev ? Array{Complex{T}}(undef, M, C) : values
    phase = Packing.offset_phase(T, ms, offsets, Ls, M, Val(false), iflag) .* M
    full = Array{Complex{T}}(undef, ms..., ntrans)
    return NUFFTSynthesisPlan{T, D, R, length(bt), typeof(plan), typeof(modes), typeof(values), typeof(hmodes),
            typeof(hvalues), typeof(phase), typeof(full), typeof(spatial)}(
        plan, modes, values, hmodes, hvalues, phase, full, ms, spatial, bt, ntrans, M, C)
end

function Plans.synthesize!(out::AbstractArray, p::NUFFTSynthesisPlan{T, D, R}, coeffs::AbstractArray;
        ks = nothing) where {T, D, R}
    size(out) == Plans.field_size(p) || throw(DimensionMismatch(
        "out is $(size(out)); this plan writes $(Plans.field_size(p))"))
    sz = Packing.packed_size(p.ms, Val(R))
    size(coeffs)[1:D] == sz || throw(DimensionMismatch(
        "this plan expects $(sz) on the spectral dims; got $(size(coeffs)[1:D])"))
    full = if R
        Packing.unpacked!(p.full, reshape(coeffs, sz..., p.ntrans), p.ms, ks)
        p.full
    else
        reshape(coeffs, p.ms..., p.ntrans)
    end
    _synthesis_run!(out, p, full)
    return out
end

function _synthesis_run!(out::AbstractArray{Z}, p::NUFFTSynthesisPlan, full) where {Z}
    Pm = prod(p.ms)
    M, C = p.M, p.C
    fk, cj = p.hmodes, p.hvalues
    @inbounds for base in 0:C:(p.ntrans - 1)
        nvalid = min(C, p.ntrans - base)
        for c in 1:C
            o = (c - 1) * Pm
            if c > nvalid
                for i in 1:Pm
                    fk[o + i] = zero(eltype(fk))
                end
                continue
            end
            foff = (base + c - 1) * Pm
            for i in 1:Pm
                fk[o + i] = full[foff + i] * conj(p.phase[i])
            end
        end
        fk === p.modes || copyto!(p.modes, fk)
        FTB.nufft_type2!(p.values, p.plan, p.modes)
        cj === p.values || copyto!(cj, p.values)
        for c in 1:nvalid
            o = (c - 1) * M
            ooff = (base + c - 1) * M
            for j in 1:M
                out[ooff + j] = Z <: Real ? real(cj[o + j]) : cj[o + j]
            end
        end
    end
    return out
end

function _synthesize(t::_FTBLibrary, exec::Union{_HostExec, ComputationalBackends.AbstractGPUBackend},
        g::Union{FlowGeometries.Grids.AbstractStructuredGrid{<:FlowGeometries.Geometry.AbstractCartesianGeometry},
                 Grids.PointwiseCartesian},
        coeffs::AbstractArray{Complex{Tr}}, ms::NTuple{D, Int}; real_output::Bool = true, iflag::Int = 1,
        ks = nothing, eps = nothing, batch_chunk = nothing, kwargs...) where {Tr <: Real, D}
    sz = Packing.packed_size(ms, Val(real_output))
    size(coeffs)[1:D] == sz || throw(DimensionMismatch(
        "real_output=$(real_output) expects $(sz) on the spectral dims; got $(size(coeffs)[1:D])."))
    batch = ntuple(i -> size(coeffs, D + i), ndims(coeffs) - D)
    plan = Plans.plan_synthesis(t, exec, g, real_output ? Tr : Complex{Tr}, ms; batch, iflag, eps,
        batch_chunk)
    try
        return Plans.synthesize!(Plans.allocate_field(plan), plan, coeffs; ks)
    finally
        Plans.close!(plan)
    end
end
