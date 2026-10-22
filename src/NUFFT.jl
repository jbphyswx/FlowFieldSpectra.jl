# NUFFT transforms of Cartesian grids through FlowTransformBindings' plans. A plan's buffers are `Array`s
# under a host execution backend and device arrays under a `GPUBackend`, whose nodes FlowTransformBindings
# builds the library plan on; the KernelAbstractions extension allocates device arrays and gathers on
# them. Every transform runs in `FFTModes` order, so a spectrum arrives in native `fftfreq` order.
#
# FFS's `iflag` is the sign of `-i` in its type-1 exponent, `Σ f e^{-iflag·ik·x}`, so FlowTransformBindings
# receives `-iflag`. A real field's tables are built for `iflag = +1`: its `iflag = -1` spectrum is the
# conjugate, applied on publish.

const _FTBLibrary = Union{FTB.FINUFFTBackend, FTB.NonuniformFFTsBackend}

# ---- where a plan's arrays live ----

_alloc(::_HostExec, ::Type{T}, dims::Integer...) where {T} = Array{T}(undef, dims...)
_alloc(exec::ComputationalBackends.AbstractExecutionBackend, ::Type, ::Integer...) = throw(ArgumentError(
    "a NUFFT plan runs on SerialBackend, ThreadedBackend or GPUBackend (with `using KernelAbstractions`); " *
    "got $(nameof(typeof(exec)))"))

_to_exec(::_HostExec, a::AbstractArray) = a
_to_exec(exec, a::AbstractArray) = copyto!(_alloc(exec, eltype(a), size(a)...), a)

_on_device(::_HostExec) = false
_on_device(::ComputationalBackends.AbstractGPUBackend) = true

# A library tag reaches the plan catch-all only on a grid it does not apply to.
_plan_hint(::_FTBLibrary, ::ComputationalBackends.AbstractExecutionBackend) = _NO_APPLY

# ---- library plans a constructor releases when it fails ----

_close!(p::FTB.AbstractNUFFTPlan) = FTB.close!(p)

# `build()`, with every library plan in `held` closed if it throws. A cuFINUFFT plan holds device memory
# that only `close!` frees.
function _closing_on_error(build, held)
    try
        return build()
    catch
        foreach(_close!, held)
        rethrow()
    end
end

# `ntuple(make, n)` of axis passes built in turn; when one fails to build, those built before it are closed.
function _axis_passes(make, n)
    built = NUFFTAxisPass[]
    try
        return ntuple(n) do i
            a = make(i)
            push!(built, a)
            return a
        end
    catch
        foreach(_close!, built)
        rethrow()
    end
end

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
#
# A real field's transform computes the packed half of the `ms`-mode box. Its coefficient box widens an
# axis `d ≥ 2` to `ms_d + 1` modes (`Grids.mode_box`), and a mode at `+ms_d/2` there is the conjugate of
# the computed mode `-k`, or zero where `-k` sits at `+ms_e/2` on another widened axis
# (`FlowTransformBindings.outside_modes`).

"""
    _packed_src(ms, box, dims, ::Val{R}; mirror = true) -> Vector{Int}

For each coefficient of the box `box` of an `ms`-mode transform, axis 1 halved for a real field (`R`),
where one transform's spectrum of shape `dims` in `FFTModes` order holds it: `i > 0` at entry `i`, `i < 0`
as the conjugate of entry `-i`, `0` nowhere. A computed mode is read at its frequency, `dims` holding at
least the computed modes on every axis. A mode at `+ms_d/2` on a widened axis is read as the conjugate at
`-k` where `mirror` is set and `-k` is computed, `dims` then holding `-k₁` on axis 1, and is `0`
otherwise.
"""
function _packed_src(ms::NTuple{D, Int}, box::NTuple{D, Int}, dims::NTuple{D, Int}, ::Val{R};
        mirror::Bool = true) where {D, R}
    pbs = FTB.packed_size(box, Val(R))
    src = Vector{Int}(undef, prod(pbs))
    @inbounds for (i, I) in enumerate(CartesianIndices(pbs))
        k = _packed_frequencies(I, box, Val(R))
        plus, minus = _widened_edges(k, ms, box)
        src[i] = !plus ? _spectrum_index(k, dims, 1) : (mirror && !minus) ? -_spectrum_index(k, dims, -1) : 0
    end
    return src
end

# The integer frequencies of coefficient `I` of the box `box`: `0 … box₁÷2` on the halved axis 1 of a real
# field, `FFTModes` order elsewhere.
@inline _packed_frequencies(I::CartesianIndex{D}, box::NTuple{D, Int}, ::Val{R}) where {D, R} =
    ntuple(d -> R && d == 1 ? I[d] - 1 : FTB.mode_frequency(FTB.FFTModes(), I[d], box[d]), Val(D))

# Whether `k` sits at `+ms_d/2` on a widened axis `d ≥ 2`, and whether at `-ms_e/2` on one.
@inline function _widened_edges(k::NTuple{D, Int}, ms::NTuple{D, Int}, box::NTuple{D, Int}) where {D}
    plus = false
    minus = false
    for d in 2:D
        box[d] > ms[d] || continue
        plus |= k[d] == ms[d] ÷ 2
        minus |= k[d] == -(ms[d] ÷ 2)
    end
    return plus, minus
end

# The linear index of frequency `s·k` in a spectrum of shape `dims` in `FFTModes` order, `kv + 1` on a
# halved axis.
@inline function _spectrum_index(k::NTuple{D, Int}, dims::NTuple{D, Int}, s::Int) where {D}
    lin = 1
    stride = 1
    for d in 1:D
        lin += (FTB.ovs_index(s * k[d], dims[d]) - 1) * stride
        stride *= dims[d]
    end
    return lin
end

"""
    _oversampled_table(Tr, ms, box, ovdims, M, phis, normfactor) -> (pos, src, fac)

The coefficients of a real field's packed box `box` at `+ms_d/2` on a widened axis that hold a value, read
from the oversampled spectrum of shape `ovdims` a real-data NonuniformFFTs plan of `ms` modes over `M`
points computed (`FlowTransformBindings.oversampled_spectra`): coefficient `pos[i]` is entry `src[i]`
times `fac[i] = normfactor / (M Π_d phis[d])`, the factor the plan's own modes take.
"""
function _oversampled_table(::Type{Tr}, ms::NTuple{D, Int}, box::NTuple{D, Int}, ovdims::NTuple{D, Int},
        M::Int, phis::Tuple, normfactor::Real) where {Tr, D}
    pos = Int[]
    src = Int[]
    fac = Tr[]
    for (i, I) in enumerate(CartesianIndices(FTB.packed_size(box, Val(true))))
        k = _packed_frequencies(I, box, Val(true))
        plus, minus = _widened_edges(k, ms, box)
        (plus && !minus) || continue
        β = Tr(normfactor) / M
        for d in 1:D
            β /= Tr(phis[d][FTB.even_table_index(k[d], ms[d], d == 1)])
        end
        push!(pos, i)
        push!(src, _spectrum_index(k, ovdims, 1))
        push!(fac, β)
    end
    return pos, src, fac
end

# ---- gathers ----

# `dst[doff + (c-1)·n + i] = S(src[soff + (c-1)·sstride + idx[i]]) · w[i]` for `i ≤ n` and each of `ntr`
# transforms `c`, `S` the conjugate where `csrc`, the product conjugated where `neg`; a number `w` scales
# every entry alike. The device method is the KernelAbstractions extension's, one launch for all `ntr`.
@inline _weight(w::AbstractArray, i::Int) = @inbounds w[i]
@inline _weight(w::Number, ::Int) = w

function _gather_scaled!(dst::AbstractArray, doff::Int, src::Array, soff::Int, sstride::Int,
        idx::Array{Int}, w::Union{Array, Number}, n::Int, ntr::Int, csrc::Bool, neg::Bool)
    @inbounds for c in 1:ntr
        so = soff + (c - 1) * sstride
        o = doff + (c - 1) * n
        for i in 1:n
            s = src[so + idx[i]]
            v = (csrc ? conj(s) : s) * _weight(w, i)
            dst[o + i] = neg ? conj(v) : v
        end
    end
    return dst
end

"""
    _gather_packed!(dst, doff, src, soff, sstride, idx, scale, n, ntr, neg)

For `i ≤ n` and each of `ntr` transforms `c`, `dst[doff + (c-1)·n + i]` is `scale` times the entry
`idx[i]` names ([`_packed_src`](@ref)) of transform `c`'s spectrum, `soff + (c-1)·sstride` into `src`, or
zero where it names none, conjugated where `neg`. The device method is the KernelAbstractions
extension's, one launch for all `ntr`.
"""
function _gather_packed!(dst::AbstractArray, doff::Int, src::Array, soff::Int, sstride::Int,
        idx::Array{Int}, scale::Real, n::Int, ntr::Int, neg::Bool)
    @inbounds for c in 1:ntr
        so = soff + (c - 1) * sstride
        o = doff + (c - 1) * n
        for i in 1:n
            j = idx[i]
            v = j > 0 ? src[so + j] * scale : j < 0 ? conj(src[so - j]) * scale : zero(eltype(dst))
            dst[o + i] = neg ? conj(v) : v
        end
    end
    return dst
end

"""
    _scatter_each!(dst, doff, n, srcs, ntr, pos, idx, fac, neg)

For each of `ntr` transforms `c`, `n` coefficients apart in `dst` from `doff`, the coefficient `pos[i]` is
`srcs[c][idx[i]] · fac[i]`, conjugated where `neg`. The device method is the KernelAbstractions
extension's, one launch for all `ntr`.
"""
function _scatter_each!(dst::AbstractArray, doff::Int, n::Int, srcs::Tuple, ntr::Int, pos::Array{Int},
        idx::Array{Int}, fac::Array, neg::Bool)
    @inbounds for c in 1:ntr
        s = srcs[c]
        o = doff + (c - 1) * n
        for i in eachindex(pos)
            v = s[idx[i]] * fac[i]
            dst[o + pos[i]] = neg ? conj(v) : v
        end
    end
    return dst
end

# Where a plan publishes `ntr` transforms of `n` coefficients that belong at offset `doff` of `dst`: there
# on the host, and on a device the plan's staging buffer, which `_unstage!` copies across once every table
# has written it.
@inline _publish_target(dst, doff::Int, ::Nothing) = (dst, doff)
@inline _publish_target(dst, doff::Int, stage::AbstractArray) = (stage, 0)
@inline _unstage!(dst, doff::Int, ::Nothing, len::Int) = dst
@inline _unstage!(dst, doff::Int, stage::AbstractArray, len::Int) = copyto!(dst, doff + 1, stage, 1, len)

# `ntr` transforms of a spectrum, `sstride` apart, published `n` apiece into `dst` from offset `doff`
# through the table `idx`.
function _publish!(dst, doff::Int, src, soff::Int, sstride::Int, idx, scale::Real, n::Int, ntr::Int,
        neg::Bool, stage)
    t, o = _publish_target(dst, doff, stage)
    _gather_packed!(t, o, src, soff, sstride, idx, scale, n, ntr, neg)
    return _unstage!(dst, doff, stage, n * ntr)
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
# Point sets: a node cloud, and a curvilinear grid's cells. Every axis of a point set tells `+ms_d/2` from
# `-ms_d/2` apart, so a real field's coefficient box widens every even axis.
#
# A real field on NonuniformFFTs takes a real-data plan, whose spectrum is the packed half of the `ms`-mode
# box. The modes at `+ms_d/2` on an axis `d ≥ 2` are computed by that transform already: the type 1 FFTs
# onto an oversampled grid and keeps only the `ms`-mode box, so they are there before truncation, each one
# oversampled entry times the deconvolution the kept modes get (`FlowTransformBindings.oversampled_spectra`).
#
# A real field on FINUFFT takes complex strengths at `FlowTransformBindings.hermitian_request_size`, which
# puts `±ms₁/2` on axis 1; a mode at `+ms_d/2` is a conjugate read of that Hermitian spectrum.
# =============================================================================

# How a real field's point plan is built and where its modes at `+ms_d/2` are read.
_real_values(::FTB.NonuniformFFTsBackend) = true
_real_values(::FTB.FINUFFTBackend) = false

"""
    NUFFTPointPlan{T,D,R}

Reusable type-1 NUFFT over a Cartesian point set (a node cloud or a curvilinear grid's cells): the
FlowTransformBindings plan with `C` transforms per execution, its strength and spectrum buffers, and the
tables that publish the coefficient box. `R` marks a real field. Execute with
`calculate_spectrum!(coeffs, plan, field)`; [`close!`](@ref) releases the library plan.
"""
struct NUFFTPointPlan{T, D, R, NB, P, V, U, IX, OV, ST, KS, QW} <: Plans.AbstractSpectralPlan
    plan::P
    values::V                        # (M, C) strengths
    modes::U                         # (mode_size…, C) spectra
    scale::T                         # 1/M
    src::IX                          # coefficient → entry of one transform's spectrum (`_packed_src`)
    oversampled::OV                  # `_oversampled_table` on the backend, or `nothing`
    stage::ST                        # device staging for the coefficients, or `nothing` on the host
    batch::NTuple{NB, Int}
    ms::NTuple{D, Int}
    pms::NTuple{D, Int}              # packed coefficient box
    M::Int
    B::Int                           # transforms the field carries
    C::Int                           # transforms per execution
    neg::Bool                        # real field with `iflag < 0`: conjugate the published values
    ks_phys::KS
    qw::QW                           # (M,) grid quadrature factor, or `nothing` for a constant measure
end

Base.show(io::IO, p::NUFFTPointPlan{T, D, R}) where {T, D, R} =
    print(io, "NUFFTPointPlan{", T, ", ", D, "}(", R ? "real" : "complex", ", ", p.plan, ")")

Plans.coefficient_size(p::NUFFTPointPlan) = (p.pms..., p.batch...)
Plans.coefficient_type(::NUFFTPointPlan{T}) where {T} = Complex{T}
Plans.wavenumbers(p::NUFFTPointPlan) = p.ks_phys
Plans.close!(p::NUFFTPointPlan) = FTB.close!(p.plan)

function _point_plan(t::_FTBLibrary, exec, g, ::Type{T}, ms::NTuple{D, Int}, batch::Tuple, iflag::Int,
        eps, batch_chunk, box) where {T, D}
    Tr = real(float(T))
    R = T <: Real
    coords, _ = Grids.point_coordinates(Tr, g, D)
    # The points fold from each direction's origin, so the spectrum is measured from it as it comes.
    offsets, Ls = Grids.axis_geometry(Tr, g, D, box)
    M = length(coords[1])
    B = prod(batch; init = 1)
    nth = _backend_nthreads(exec)
    dev = _on_device(exec)
    C = _chunk(_batch_default(t, B, nth, dev), B, batch_chunk)
    realplan = R && _real_values(t)
    ns = (R && !realplan) ? FTB.hermitian_request_size(ms) : ms
    VT = realplan ? Tr : Complex{Tr}
    plan = FTB.plan_nufft(t, VT, ntuple(d -> _to_exec(exec, coords[d]), Val(D)), ns; ntrans = C,
        tol = _nufft_tol(Tr, eps), order = FTB.FFTModes(), period = Ls, origin = offsets, nthreads = nth,
        iflag = R ? -1 : -iflag)
    return _closing_on_error((plan,)) do
        dims = FTB.mode_size(plan)
        mb = Grids.mode_box(g, ms, Val(R))
        pms = FTB.packed_size(mb, Val(R))
        ks_phys = Grids.physical_wavenumbers(Ls, mb, Val(R))
        ovs = realplan ? _oversampled_publish(exec, Tr, ms, mb, plan, M) : nothing
        qwh = Grids.quadrature_scale(g, Tr, M)
        qw = qwh === nothing ? nothing : _to_exec(exec, collect(Tr, qwh))
        bt = NTuple{length(batch), Int}(batch)
        values = _alloc(exec, VT, M, C)
        modes = _alloc(exec, Complex{Tr}, dims..., C)
        src = _to_exec(exec, _packed_src(ms, mb, dims, Val(R); mirror = !realplan))
        stage = dev ? _alloc(exec, Complex{Tr}, prod(pms) * C) : nothing
        NUFFTPointPlan{Tr, D, R, length(bt), typeof(plan), typeof(values), typeof(modes),
                typeof(src), typeof(ovs), typeof(stage), typeof(ks_phys), typeof(qw)}(
            plan, values, modes, one(Tr) / M, src, ovs, stage, bt, ms, pms, M, B, C, R && iflag < 0,
            ks_phys, qw)
    end
end

# A real-data plan's `_oversampled_table` on the execution backend, or `nothing` where it is empty.
function _oversampled_publish(exec, ::Type{Tr}, ms::NTuple{D, Int}, mb::NTuple{D, Int}, plan,
        M::Int) where {Tr, D}
    us, normfactor, phis = FTB.oversampled_spectra(plan)
    pos, src, fac = _oversampled_table(Tr, ms, mb, size(first(us)), M, map(collect, phis), normfactor)
    isempty(pos) && return nothing
    return (; pos = _to_exec(exec, pos), src = _to_exec(exec, src), fac = _to_exec(exec, fac))
end

# The coefficients a real-data plan reads from its oversampled spectra, for the `nvalid` transforms of an
# execution published `n` apiece into `dst` from `off`.
@inline _publish_oversampled!(dst, off::Int, n::Int, ::Nothing, plan, nvalid::Int, neg::Bool) = dst
function _publish_oversampled!(dst, off::Int, n::Int, t, plan, nvalid::Int, neg::Bool)
    us, _, _ = FTB.oversampled_spectra(plan)
    return _scatter_each!(dst, off, n, us, nvalid, t.pos, t.src, t.fac, neg)
end

"""
    calculate_spectrum!(coeffs, plan::NUFFTPointPlan, field) -> ks_phys

Execute a point-set NUFFT plan in place. `field` is `(N, batch…)`; `coeffs` is the packed half of the
coefficient box (`Grids.mode_box`) for a real field and the full spectrum for a complex one.
"""
function calculate_spectrum!(coeffs::AbstractArray{Complex{T}}, p::NUFFTPointPlan{T, D, R},
        field) where {T, D, R}
    M, B, C = p.M, p.B, p.C
    length(field) == M * B || throw(DimensionMismatch(
        "field holds $(length(field)) values; this plan transforms $B field(s) of $M points — pass the " *
        "matching `batch=` to plan_spectrum"))
    Plans._check_coefficients(coeffs, p)
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
        dst, off = _publish_target(coeffs, base * Ph, p.stage)
        _gather_packed!(dst, off, p.modes, 0, Pn, p.src, p.scale, Ph, nvalid, p.neg)
        _publish_oversampled!(dst, off, Ph, p.oversampled, p.plan, nvalid, p.neg)
        _unstage!(coeffs, base * Ph, p.stage, Ph * nvalid)
    end
    return p.ks_phys
end

function Plans.plan_spectrum(t::_FTBLibrary,
        exec::Union{_HostExec, ComputationalBackends.AbstractGPUBackend}, g::Grids.PointwiseCartesian,
        ::Type{T}, ms::NTuple{D, Int}; batch::Tuple = (), iflag::Int = 1, eps = nothing,
        batch_chunk::Union{Nothing, Integer} = nothing, box = nothing) where {T, D}
    return _point_plan(t, exec, g, T, ms, batch, iflag, eps, batch_chunk, box)
end

function _nufft_one_shot(t::_FTBLibrary, exec, g, field, ms::NTuple{D, Int}; iflag::Int = 1,
        eps = nothing, batch_chunk = nothing, box = nothing, kwargs...) where {D}
    E = float(eltype(field))
    batch = Grids.field_batch_shape(g, field)
    plan = Plans.plan_spectrum(t, exec, g, E, ms; batch, iflag, eps, batch_chunk, box)
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

# An array of size `sz` read as `(pre, N_d, post)` along axis `d`: `pre` the product of the sizes below `d`,
# `post` of those above. Line `r` of the `pre·post` lines starts at `mod1(r, pre) + ((r - 1) ÷ pre)·pre·N_d`.
@inline function _axis_layout(sz::NTuple{N, Int}, d::Int) where {N}
    pre = 1
    for i in 1:(d - 1)
        pre *= sz[i]
    end
    post = 1
    for i in (d + 1):N
        post *= sz[i]
    end
    return pre, sz[d], post
end

# `sz` with axis `d` transformed to `m` modes.
@inline _axis_out_size(sz::NTuple{N, Int}, d::Int, m::Int) where {N} = ntuple(i -> i == d ? m : sz[i], Val(N))

# The working array entering pass `d`: axes below `d` at their `ns` modes, the rest at their `Ns` points,
# the batch after them.
@inline _axis_work_shape(Ns::NTuple{D, Int}, ns::NTuple{D, Int}, d::Int, batch::Tuple) where {D} =
    (ntuple(i -> i < d ? ns[i] : Ns[i], Val(D))..., batch...)

# Starting linear indices of lines `base+1 … base+nvalid` in the input (`Nd` per line) and the output (`m`
# per line).
@inline function _axis_chunk_offsets!(inoff::Vector{Int}, outoff::Vector{Int}, base::Int, nvalid::Int,
        pre::Int, Nd::Int, m::Int)
    @inbounds for t in 1:nvalid
        r = base + t
        a = mod1(r, pre)
        c = (r - 1) ÷ pre
        inoff[t] = a + c * pre * Nd
        outoff[t] = a + c * pre * m
    end
    return nothing
end

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
    pre, _, post = _axis_layout(insize, d)
    rest = pre * post
    nth = _backend_nthreads(exec)
    dev = _on_device(exec)
    C = _chunk(_line_default(t, rest, nth, dev), rest, batch_chunk)
    plan = FTB.plan_nufft(t, Complex{Tr}, (_to_exec(exec, collect(Tr, axis)),), (m,); ntrans = C,
        tol = Tr(tol), order = FTB.FFTModes(), period = Tr(rng), origin = Tr(off), nthreads = nth, iflag)
    return _closing_on_error((plan,)) do
        stage = dev ? (_alloc(exec, Complex{Tr}, Nd, rest), _alloc(exec, Complex{Tr}, m, rest)) : nothing
        NUFFTAxisPass(plan, _alloc(exec, Complex{Tr}, Nd, C), _alloc(exec, Complex{Tr}, m, C),
            Vector{Int}(undef, C), Vector{Int}(undef, C), stage, Nd, m, C)
    end
end

# Transform axis `d` of `A` into `out`, whose dim `d` holds `ap.m` modes. On the host the lines are
# gathered straight out of `A` and scattered straight into `out`, so neither is permuted.
function _axis_run!(out::Array, A::Array, d::Int, ap::NUFFTAxisPass)
    pre, Nd, post = _axis_layout(size(A), d)
    Nd == ap.Nd || throw(DimensionMismatch("axis $d: field length $Nd ≠ this plan's grid axis length $(ap.Nd)"))
    m, C = ap.m, ap.C
    V, U = ap.values, ap.modes
    rest = pre * post
    for base in 0:C:(rest - 1)
        nvalid = min(C, rest - base)
        _axis_chunk_offsets!(ap.inoff, ap.outoff, base, nvalid, pre, Nd, m)
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

# A stretched axis of the hybrid composite, `insize` the working size on its points side and `iflag`
# FlowTransformBindings' sign. The forward (`HybridPlan`) takes the `Σ e^{-ikx}` sign of its FFT pass and
# conjugates once for `iflag = -1`; the synthesis (`HybridSynthesisPlan`) passes its own.
_axis_nufft_plan(t::_FTBLibrary, exec::ComputationalBackends.AbstractExecutionBackend, ::Type{Tr},
        insize::Tuple, d::Int, axis::AbstractVector, m::Int, rng, off, eps; batch_chunk = nothing,
        iflag::Int = -1, kwargs...) where {Tr} =
    _axis_pass(t, exec, Tr, insize, d, axis, m, rng, off, eps, batch_chunk; iflag)

# The hybrid composite's stretched axes in turn, pass `i` along dim `sdims[i]` reading the previous one's
# output and writing `work[i]`: the forward's type 1, points to modes.
@inline _run_stretched!(W, ::Tuple{}, ::Tuple{}, ::Tuple{}) = W
@inline _run_stretched!(W, work::Tuple, axes::Tuple, sdims::Tuple) =
    _run_stretched!(_axis_run!(first(work), W, first(sdims), first(axes)), Base.tail(work), Base.tail(axes),
        Base.tail(sdims))

# The same walk for the synthesis's type 2, modes to points.
@inline _run_stretched_synthesis!(W, ::Tuple{}, ::Tuple{}, ::Tuple{}) = W
@inline _run_stretched_synthesis!(W, work::Tuple, axes::Tuple, sdims::Tuple) =
    _run_stretched_synthesis!(_axis_synth!(first(work), W, first(sdims), first(axes)), Base.tail(work),
        Base.tail(axes), Base.tail(sdims))

# The type 2 along axis `d`: `A` holds `ap.m` modes there and `out` its `ap.Nd` points. On the host the
# lines are gathered straight out of `A` and scattered straight into `out`. The line offsets are the
# forward's, `inoff` on the points side and `outoff` on the modes side.
function _axis_synth!(out::Array, A::Array, d::Int, ap::NUFFTAxisPass)
    pre, Nd, post = _axis_layout(size(out), d)
    Nd == ap.Nd || throw(DimensionMismatch("axis $d: field length $Nd ≠ this plan's grid axis length $(ap.Nd)"))
    m, C = ap.m, ap.C
    V, U = ap.values, ap.modes
    rest = pre * post
    for base in 0:C:(rest - 1)
        nvalid = min(C, rest - base)
        _axis_chunk_offsets!(ap.inoff, ap.outoff, base, nvalid, pre, Nd, m)
        @inbounds for j in 1:m
            s = (j - 1) * pre
            for c in 1:nvalid
                U[j, c] = A[ap.outoff[c] + s]
            end
            for c in (nvalid + 1):C
                U[j, c] = zero(eltype(U))
            end
        end
        FTB.nufft_type2!(V, ap.plan, U)
        @inbounds for i in 1:Nd
            s = (i - 1) * pre
            for c in 1:nvalid
                out[ap.inoff[c] + s] = V[i, c]
            end
        end
    end
    return out
end

# On a device the axis moves to the front, so each line is a contiguous column of the staging array.
function _axis_synth!(out::AbstractArray, A::AbstractArray, d::Int, ap::NUFFTAxisPass)
    pin, pfk = ap.stage
    nd = ndims(A)
    perm = _axis_perm(nd, d)
    Nd, m, C = ap.Nd, ap.m, ap.C
    rest = size(pfk, 2)
    permutedims!(reshape(pfk, ntuple(i -> size(A, perm[i]), nd)), A, perm)
    for base in 0:C:(rest - 1)
        nvalid = min(C, rest - base)
        copyto!(ap.modes, 1, pfk, base * m + 1, nvalid * m)
        nvalid < C && fill!(view(ap.modes, :, (nvalid + 1):C), zero(eltype(ap.modes)))
        FTB.nufft_type2!(ap.values, ap.plan, ap.modes)
        copyto!(pin, base * Nd + 1, ap.values, 1, nvalid * Nd)
    end
    permutedims!(out, reshape(pin, ntuple(i -> size(out, perm[i]), nd)), invperm(collect(perm)))
    return out
end

@inline _run_axes!(work::Tuple, ::Tuple{}, d::Int) = nothing
@inline function _run_axes!(work::Tuple, axes::Tuple, d::Int)
    _axis_run!(work[d + 1], work[d], d, first(axes))
    return _run_axes!(work, Base.tail(axes), d + 1)
end

# =============================================================================
# Separable transform for a nonuniform tensor-product (structured) Cartesian grid: `D` successive 1-D
# passes, spreading `D·2m` per point over the axes alone. Every pass is complex, so a real field's result
# is the full spectrum at `hermitian_request_size`, whose leading axis-1 entries are the computed modes
# and whose conjugate reads give the modes at `+ms_d/2` of a widened axis.
# =============================================================================

"""
    NUFFTSeparablePlan{T,D,R}

Reusable separable NUFFT over a nonuniform tensor-product Cartesian grid: one [`NUFFTAxisPass`](@ref)
per axis, the working arrays the passes write through, and the table that publishes the coefficient box.
Execute with `calculate_spectrum!(coeffs, plan, field)`; [`close!`](@ref) releases the library plans.
"""
struct NUFFTSeparablePlan{T, D, R, NB, AP, W, IX, ST, KS, QW} <: Plans.AbstractSpectralPlan
    axes::AP                         # D × NUFFTAxisPass
    work::W                          # D+1 working arrays; `work[1]` takes the field
    scale::T                         # 1/∏N_d
    src::IX                          # coefficient → entry of one transform's spectrum (`_packed_src`)
    stage::ST
    batch::NTuple{NB, Int}
    ms::NTuple{D, Int}
    ns::NTuple{D, Int}               # mode counts the passes produce
    pms::NTuple{D, Int}              # packed coefficient box
    npts::Int
    ntrans::Int
    neg::Bool                        # real field with `iflag < 0`: conjugate the published values
    ks_phys::KS
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
        batch_chunk::Union{Nothing, Integer} = nothing, box = nothing) where {T, D}
    Tr = real(float(T))
    R = T <: Real
    ndims(g) == D || throw(DimensionMismatch("grid has $(ndims(g)) dims; asked for $D mode counts"))
    axs = ntuple(d -> Tr.(FlowGeometries.Grids.coordinates(g, d)), Val(D))
    offsets, ranges = Grids.axis_geometry(Tr, g, D, box)
    Ns = ntuple(d -> length(axs[d]), Val(D))
    npts = prod(Ns)
    ntrans = prod(batch; init = 1)
    ns = R ? FTB.hermitian_request_size(ms) : ms
    tol = _nufft_tol(Tr, eps)
    sgn = R ? -1 : -iflag
    work = ntuple(d -> _alloc(exec, Complex{Tr}, _axis_work_shape(Ns, ns, d, batch)...), Val(D + 1))
    axes = _axis_passes(Val(D)) do d
        _axis_pass(t, exec, Tr, size(work[d]), d, axs[d], ns[d], ranges[d], offsets[d], tol, batch_chunk;
            iflag = sgn)
    end
    return _closing_on_error(axes) do
        mb = Grids.mode_box(g, ms, Val(R))
        pms = FTB.packed_size(mb, Val(R))
        ks_phys = Grids.physical_wavenumbers(ranges, mb, Val(R))
        qwh = Grids.quadrature_scale(g, Tr, npts)
        qw = qwh === nothing ? nothing : _to_exec(exec, collect(Tr, qwh))
        src = _to_exec(exec, _packed_src(ms, mb, ns, Val(R)))
        stage = _on_device(exec) ? _alloc(exec, Complex{Tr}, prod(pms) * ntrans) : nothing
        bt = NTuple{length(batch), Int}(batch)
        NUFFTSeparablePlan{Tr, D, R, length(bt), typeof(axes), typeof(work), typeof(src),
                typeof(stage), typeof(ks_phys), typeof(qw)}(
            axes, work, one(Tr) / npts, src, stage, bt, ms, ns, pms, npts, ntrans, R && iflag < 0, ks_phys, qw)
    end
end

"""
    calculate_spectrum!(coeffs, plan::NUFFTSeparablePlan, field) -> ks_phys

Execute a separable structured plan in place. `field` is `(N₁…N_D, batch…)`; `coeffs` is the packed half
of the coefficient box (`Grids.mode_box`) for a real field and the full spectrum for a complex one.
"""
function calculate_spectrum!(coeffs::AbstractArray{Complex{T}}, p::NUFFTSeparablePlan{T, D, R},
        field) where {T, D, R}
    A = p.work[1]
    length(field) == length(A) || throw(DimensionMismatch(
        "field holds $(length(field)) values; this plan was built for $(length(A)) — pass the matching " *
        "`batch=` to plan_spectrum"))
    Plans._check_coefficients(coeffs, p)
    _stage!(A, field)
    _scale_points!(A, p.qw, p.npts, p.ntrans)
    _run_axes!(p.work, p.axes, 1)
    F = p.work[D + 1]
    Pn = prod(p.ns)
    Ph = prod(p.pms)
    _publish!(coeffs, 0, F, 0, Pn, p.src, p.scale, Ph, p.ntrans, p.neg, p.stage)
    return p.ks_phys
end

# =============================================================================
# Synthesis: a complex type 2 on the modes of `Grids.series_modes`. A real field's packed half is first
# written as that series' coefficients (`DirectSum.native_coefficients!`), and the real part taken. The
# forward publishes `C = fk / M` measured from each direction's origin, the point positions `y` the type 2
# folds, so it evaluates `Σ C e^{+iflag·ik·y}` as given.
# =============================================================================

"""
    NUFFTSynthesisPlan{T,D,R}

Reusable type-2 NUFFT synthesis onto a Cartesian grid's points: the plan with `C` transforms per execution,
its spectrum and strength buffers (with host staging on a device), and for a real field the host series a
packed half is written into. Execute with `synthesize!(out, plan, coeffs)`; [`close!`](@ref) releases the
library plan.
"""
struct NUFFTSynthesisPlan{T, D, R, NB, P, U, V, HU, HV, FB, SP} <: Plans.AbstractSynthesisPlan
    plan::P
    modes::U                         # (ns…, C)
    values::V                        # (M, C)
    hmodes::HU                       # host staging for `modes`, or `modes` itself on the host
    hvalues::HV                      # host staging for `values`, or `values` itself on the host
    full::FB                         # host (ns…, ntrans) for a real field, `nothing` for a complex one
    ms::NTuple{D, Int}
    box::NTuple{D, Int}              # coefficient box
    ns::NTuple{D, Int}               # mode counts of the series
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
        batch_chunk::Union{Nothing, Integer} = nothing, box = nothing, kwargs...) where {TT, D}
    T = real(float(TT))
    R = TT <: Real
    coords, spatial = Grids.point_coordinates(T, g, D)
    # The forward's origins and Fourier lengths, `box`'s where `g` is part of a larger grid.
    offsets, Ls = Grids.axis_geometry(T, g, D, box)
    M = length(coords[1])
    bt = NTuple{length(batch), Int}(batch)
    ntrans = prod(bt; init = 1)
    nth = _backend_nthreads(exec)
    dev = _on_device(exec)
    C = _chunk(_batch_default(t, ntrans, nth, dev), ntrans, batch_chunk)
    mb = Grids.mode_box(g, ms, Val(R))
    ns = Grids.series_modes(g, ms, Val(R))
    # The type 2 evaluates `Σ û e^{+iflag·ik·y}`, FlowTransformBindings' `-iflag` convention.
    plan = FTB.plan_nufft(t, Complex{T}, ntuple(d -> _to_exec(exec, coords[d]), Val(D)), ns; ntrans = C,
        tol = _nufft_tol(T, eps), order = FTB.FFTModes(), period = Ls, origin = offsets, nthreads = nth,
        iflag = -iflag)
    return _closing_on_error((plan,)) do
        modes = _alloc(exec, Complex{T}, ns..., C)
        values = _alloc(exec, Complex{T}, M, C)
        hmodes = dev ? Array{Complex{T}}(undef, ns..., C) : modes
        hvalues = dev ? Array{Complex{T}}(undef, M, C) : values
        full = R ? Array{Complex{T}}(undef, ns..., ntrans) : nothing
        NUFFTSynthesisPlan{T, D, R, length(bt), typeof(plan), typeof(modes), typeof(values), typeof(hmodes),
                typeof(hvalues), typeof(full), typeof(spatial)}(
            plan, modes, values, hmodes, hvalues, full, ms, mb, ns, spatial, bt, ntrans, M, C)
    end
end

function Plans.synthesize!(out::AbstractArray, p::NUFFTSynthesisPlan{T, D, R},
        coeffs::AbstractArray) where {T, D, R}
    size(out) == Plans.field_size(p) || throw(DimensionMismatch(
        "out is $(size(out)); this plan writes $(Plans.field_size(p))"))
    sz = FTB.packed_size(p.box, Val(R))
    size(coeffs)[1:D] == sz || throw(DimensionMismatch(
        "this plan expects $(sz) on the spectral dims; got $(size(coeffs)[1:D])"))
    full = R ? DirectSum.native_coefficients!(p.full, reshape(coeffs, sz..., p.ntrans), p.ms, p.box) :
        reshape(coeffs, p.ms..., p.ntrans)
    _synthesis_run!(out, p, full)
    return out
end

function _synthesis_run!(out::AbstractArray{Z}, p::NUFFTSynthesisPlan, full) where {Z}
    Pm = prod(p.ns)
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
                fk[o + i] = full[foff + i]
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
        eps = nothing, batch_chunk = nothing, kwargs...) where {Tr <: Real, D}
    sz = FTB.packed_size(Grids.mode_box(g, ms, Val(real_output)), Val(real_output))
    size(coeffs)[1:D] == sz || throw(DimensionMismatch(
        "real_output=$(real_output) expects $(sz) on the spectral dims; got $(size(coeffs)[1:D])."))
    batch = ntuple(i -> size(coeffs, D + i), ndims(coeffs) - D)
    plan = Plans.plan_synthesis(t, exec, g, real_output ? Tr : Complex{Tr}, ms; batch, iflag, eps,
        batch_chunk)
    try
        return Plans.synthesize!(Plans.allocate_field(plan), plan, coeffs)
    finally
        Plans.close!(plan)
    end
end
