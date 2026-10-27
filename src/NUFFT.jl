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

# A device plan's buffer for a caller's array, `nothing` on the host, where the plan reads and writes the
# caller's arrays.
_device_buffer(::_HostExec, ::Type, ::Integer...) = nothing
_device_buffer(exec::ComputationalBackends.AbstractGPUBackend, ::Type{T}, dims::Integer...) where {T} =
    _alloc(exec, T, dims...)

# The array a plan reads: the caller's `a` with no buffer or where `a` has the buffer's type and shape, the
# buffer filled from `a` otherwise.
_input_in(::Nothing, a::AbstractArray) = a
_input_in(buf::AbstractArray, a::AbstractArray) =
    (a isa typeof(buf) && size(a) == size(buf)) ? a : _stage!(buf, a)

# The array a plan writes: the caller's `a` with no buffer or where `a` has the buffer's type and shape, the
# buffer otherwise, which the caller's array is then filled from.
_output_in(::Nothing, a::AbstractArray) = a
_output_in(buf::AbstractArray, a::AbstractArray) = (a isa typeof(buf) && size(a) == size(buf)) ? a : buf

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

# The axis passes `make(dims[i], axes[i], sizes[i])` built in turn; when one fails to build, those built
# before it are closed. A tuple per axis, so the result's length is the grid type's.
_axis_passes(make::F, ::Tuple{}, ::Tuple{}, ::Tuple{}) where {F} = ()
function _axis_passes(make::F, dims::Tuple, axes::Tuple, sizes::Tuple) where {F}
    a = make(first(dims), first(axes), first(sizes))
    rest = try
        _axis_passes(make, Base.tail(dims), Base.tail(axes), Base.tail(sizes))
    catch
        _close!(a)
        rethrow()
    end
    return (a, rest...)
end

# ---- transform counts and tolerance when the caller names none ----

_nufft_tol(::Type{Tr}, ::Nothing) where {Tr} = FTB.default_tolerance(Tr)
_nufft_tol(::Type{Tr}, eps::Real) where {Tr} = Tr(eps)

# The library tag a plan builds with. The FFTW extension gives a host NonuniformFFTs tag naming no planner
# flags FFTW's `MEASURE` for a held plan and `ESTIMATE` for a plan built for one call (`measure = false`),
# one tag type either way.
_planned_library(t::SpectralBackends.AbstractSpectralBackend, exec, measure::Bool) = t

# Transforms per execution. FINUFFT threads over the transforms of one execution, so it takes a whole
# batch. NonuniformFFTs holds an oversampled grid and a spreading buffer per transform, so on the host it
# takes four, keeping plan memory independent of the batch.
_batch_default(::FTB.FINUFFTBackend, total::Int, dev::Bool) = total
_batch_default(::FTB.NonuniformFFTsBackend, total::Int, dev::Bool) = dev ? total : 4

# Lines per execution along one axis: four per lane on the host, every line at once on a device.
_line_default(total::Int, dev::Bool) = dev ? total : 4

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
    _packed_src(ms, box, dims, ::Type{T}; mirror = true) -> Vector{Int}

For each coefficient of the box `box` of an `ms`-mode transform of a field of element type `T`, axis 1
halved for a real `T`, where one transform's spectrum of shape `dims` in `FFTModes` order holds it: `i > 0`
at entry `i`, `i < 0` as the conjugate of entry `-i`, `0` nowhere. A computed mode is read at its
frequency, `dims` holding at least the computed modes on every axis. A mode at `+ms_d/2` on a widened axis
is read at its frequency where `dims[d] > ms[d]` on each such axis, as the conjugate at `-k` where
`mirror` is set and `-k` is computed (`dims` then holding `-k₁` on axis 1), and is `0` otherwise; with
`-ms_e/2` on another widened axis it is `0`.
"""
function _packed_src(ms::NTuple{D, Int}, box::NTuple{D, Int}, dims::NTuple{D, Int}, ::Type{T};
        mirror::Bool = true) where {D, T}
    pbs = FTB.packed_size(T, box)
    src = Vector{Int}(undef, prod(pbs))
    @inbounds for (i, I) in enumerate(CartesianIndices(pbs))
        k = _packed_frequencies(I, box, T)
        plus, minus = _widened_edges(k, ms, box)
        src[i] = !plus ? _spectrum_index(k, dims, 1) :
                 minus ? 0 :
                 _plus_held(k, ms, box, dims) ? _spectrum_index(k, dims, 1) :
                 mirror ? -_spectrum_index(k, dims, -1) : 0
    end
    return src
end

# Whether `dims` holds every `+ms_d/2` of `k` on a widened axis, a spectrum computed over the box there.
@inline function _plus_held(k::NTuple{D, Int}, ms::NTuple{D, Int}, box::NTuple{D, Int},
        dims::NTuple{D, Int}) where {D}
    for d in 2:D
        box[d] > ms[d] && k[d] == ms[d] ÷ 2 && dims[d] <= ms[d] && return false
    end
    return true
end

# The integer frequencies of coefficient `I` of the box `box`: `0 … box₁÷2` on the halved axis 1 of a real
# field, `FFTModes` order elsewhere.
@inline _packed_frequencies(I::CartesianIndex{D}, box::NTuple{D, Int}, ::Type{T}) where {D, T} =
    ntuple(d -> T <: Real && d == 1 ? I[d] - 1 : FTB.mode_frequency(FTB.FFTModes(), I[d], box[d]), Val(D))

# The element-type stand-in a plan's realness parameter `R` gives the layout functions.
@inline _layout_type(R::Bool) = R ? Real : Complex

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
    _oversampled_table(Tr, ms, box, ovdims, phis, scale) -> (pos, src, fac)

The coefficients of a real field's packed box `box` at `+ms_d/2` on a widened axis that hold a value, and
where a real-data NonuniformFFTs plan of `ms` modes holds them in its oversampled spectrum of shape
`ovdims` (`FlowTransformBindings.oversampled_spectra`): coefficient `pos[i]` is entry `src[i]` times
`fac[i] = scale / Π_d phis[d]`. The forward reads them at `scale = normfactor / M`, the factor the plan's
own modes take with FFS's `1/M`; the synthesis writes them at `scale = 1` and divides
(`FlowTransformBindings.oversampled_type2!`).
"""
function _oversampled_table(::Type{Tr}, ms::NTuple{D, Int}, box::NTuple{D, Int}, ovdims::NTuple{D, Int},
        phis::Tuple, scale::Real) where {Tr, D}
    pos = Int[]
    src = Int[]
    fac = Tr[]
    for (i, I) in enumerate(CartesianIndices(FTB.packed_size(Real, box)))
        k = _packed_frequencies(I, box, Real)
        plus, minus = _widened_edges(k, ms, box)
        (plus && !minus) || continue
        β = Tr(scale)
        for d in 1:D
            β /= Tr(phis[d][FTB.even_table_index(k[d], ms[d], d == 1)])
        end
        push!(pos, i)
        push!(src, _spectrum_index(k, ovdims, 1))
        push!(fac, β)
    end
    return pos, src, fac
end

"""
    _mode_src(T, ms, box) -> (src, w)

For each mode of a real-data plan over `ms` modes, its packed half in `FFTModes` order, the entry `src` of
one field's packed half over the coefficient box `box` holding the same frequency, and the weight `w` it
enters with: `1/2` on the row `k₁ = ms₁/2` of an even axis 1 the box does not widen, a Nyquist row the
plan's type 2 counts with a mirror it does not have, `1` elsewhere.
"""
function _mode_src(::Type{T}, ms::NTuple{D, Int}, box::NTuple{D, Int}) where {T, D}
    pbs = FTB.packed_size(Real, box)
    pms = FTB.packed_size(Real, ms)
    src = Vector{Int}(undef, prod(pms))
    w = Vector{T}(undef, prod(pms))
    @inbounds for (i, I) in enumerate(CartesianIndices(pms))
        k = _packed_frequencies(I, ms, Real)
        src[i] = _spectrum_index(k, pbs, 1)
        w[i] = _nyquist_row(k[1], ms[1], box[1]) ? T(1 // 2) : one(T)
    end
    return src, w
end

# Whether `k₁` is the Nyquist row of an even axis 1 the box `box₁` does not widen.
@inline _nyquist_row(k1::Int, m1::Int, box1::Int) = iseven(m1) && box1 == m1 && k1 == m1 ÷ 2

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

"""
    _inject_each!(spectra, src, soff, sstride, ntr, pos, idx, fac, neg)

The reverse of [`_scatter_each!`](@ref): for each of `ntr` transforms `c`, entry `idx[i]` of `spectra[c]` is
`fac[i]` times coefficient `pos[i]` of transform `c`, `soff + (c-1)·sstride` into `src`, the coefficient
conjugated where `neg`. The device method is the KernelAbstractions extension's, one launch for all `ntr`.
"""
function _inject_each!(spectra::Tuple, src::Array, soff::Int, sstride::Int, ntr::Int, pos::Array{Int},
        idx::Array{Int}, fac::Array, neg::Bool)
    @inbounds for c in 1:ntr
        s = spectra[c]
        o = soff + (c - 1) * sstride
        for i in eachindex(pos)
            v = src[o + pos[i]]
            s[idx[i]] = (neg ? conj(v) : v) * fac[i]
        end
    end
    return spectra
end

# Where a plan publishes `ntr` transforms of `n` coefficients that belong at offset `doff` of `dst`: there
# on the host, and on a device the plan's staging buffer, which `_unstage!` copies across once every table
# has written it.
@inline _publish_target(dst, doff::Int, ::Nothing) = (dst, doff)
@inline _publish_target(dst, doff::Int, stage::AbstractArray) = (stage, 0)
@inline _unstage!(dst, doff::Int, ::Nothing, len::Int) = dst
@inline _unstage!(dst, doff::Int, stage::AbstractArray, len::Int) = copyto!(dst, doff + 1, stage, 1, len)

# The coefficients of a box `(box…, batch…)` in FFT order that hold zero, set in every field of `F`: on two
# widened axes `d ≠ e` (`box > ms`), the modes at `+ms_d/2` with `-ms_e/2`, which lie outside the computed
# modes and their mirror (`FlowTransformBindings.outside_modes`). Each pair of axes is one slab of `F`.
function _zero_corners!(F::AbstractArray{<:Any, N}, ms::NTuple{D, Int}, box::NTuple{D, Int}) where {N, D}
    for d in 2:D, e in 2:D
        (d != e && box[d] > ms[d] && box[e] > ms[e]) || continue
        id = ms[d] ÷ 2 + 1
        ie = ms[e] ÷ 2 + 2
        view(F, ntuple(k -> k == d ? (id:id) : k == e ? (ie:ie) : (1:size(F, k)), Val(N))...) .= zero(eltype(F))
    end
    return F
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

# Each point's quadrature factor over the `npts` a transform normalizes by, on the execution backend: a
# per-point vector, or one number where the measure is constant.
_point_weights(exec, ::Nothing, ::Type{T}, npts::Int) where {T} = one(T) / npts
_point_weights(exec, qw, ::Type{T}, npts::Int) where {T} = _to_exec(exec, collect(T, qw) ./ npts)

# `A[k] = f[k] · w` over each of the `ntrans` fields of `npts` points, `w` a per-point vector or one number,
# the field conjugated where `cj`: the staging and the weighting in one pass.
function _stage_weighted!(A::Array{TA}, field::Array, w::Vector, npts::Int, ntrans::Int, cj::Bool) where {TA}
    @inbounds for b in 1:ntrans
        o = (b - 1) * npts
        @simd for j in 1:npts
            v = TA(field[o + j])
            A[o + j] = (cj ? conj(v) : v) * w[j]
        end
    end
    return A
end
function _stage_weighted!(A::Array{TA}, field::Array, w::Number, npts::Int, ntrans::Int, cj::Bool) where {TA}
    @inbounds @simd for k in eachindex(A, field)
        v = TA(field[k])
        A[k] = (cj ? conj(v) : v) * w
    end
    return A
end
function _stage_weighted!(A::AbstractArray, field, w, npts::Int, ntrans::Int, cj::Bool)
    _stage!(A, field)
    cj && (A .= conj.(A))
    return _weight_points!(A, w, npts, ntrans)
end

_weight_points!(A::AbstractArray, w::Number, npts::Int, ntrans::Int) = (A .*= w; A)
_weight_points!(A::AbstractArray, w::AbstractVector, npts::Int, ntrans::Int) = (reshape(A, npts, ntrans) .*= w; A)

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
    coords, _ = Grids.point_coordinates(Tr, g, Val(D))
    # The points fold from each direction's origin, so the spectrum is measured from it as it comes.
    offsets, Ls = Grids.axis_geometry(Tr, g, Val(D), box)
    M = length(coords[1])
    B = prod(batch; init = 1)
    nth = _backend_nthreads(exec)
    dev = _on_device(exec)
    C = _chunk(_batch_default(t, B, dev), B, batch_chunk)
    realplan = R && _real_values(t)
    ns = (R && !realplan) ? FTB.hermitian_request_size(ms) : ms
    VT = realplan ? Tr : Complex{Tr}
    plan = FTB.plan_nufft(t, VT, ntuple(d -> _to_exec(exec, coords[d]), Val(D)), ns; ntrans = C,
        tol = _nufft_tol(Tr, eps), order = FTB.FFTModes(), period = Ls, origin = offsets, nthreads = nth,
        iflag = R ? -1 : -iflag)
    return _closing_on_error((plan,)) do
        dims = FTB.mode_size(plan)
        mb = Grids.mode_box(g, ms, T)
        pms = FTB.packed_size(T, mb)
        ks_phys = Grids.physical_wavenumbers(Ls, mb, T)
        ovs = _oversampled_publish(t, exec, T, ms, mb, plan, M)
        qwh = Grids.quadrature_scale(g, Tr, M)
        qw = qwh === nothing ? nothing : _to_exec(exec, collect(Tr, qwh))
        bt = NTuple{length(batch), Int}(batch)
        values = _alloc(exec, VT, M, C)
        modes = _alloc(exec, Complex{Tr}, dims..., C)
        src = _to_exec(exec, _packed_src(ms, mb, dims, T; mirror = !realplan))
        stage = _device_buffer(exec, Complex{Tr}, prod(pms) * C)
        NUFFTPointPlan{Tr, D, T <: Real, length(bt), typeof(plan), typeof(values), typeof(modes),
                typeof(src), typeof(ovs), typeof(stage), typeof(ks_phys), typeof(qw)}(
            plan, values, modes, one(Tr) / M, src, ovs, stage, bt, ms, pms, M, B, C, T <: Real && iflag < 0,
            ks_phys, qw)
    end
end

# The `_oversampled_table` on the execution backend of a real field's real-data plan (`_real_values`),
# empty where no coefficient is read there; `nothing` for every other plan.
_oversampled_publish(::_FTBLibrary, exec, ::Type, ms::Tuple, mb::Tuple, plan, M::Int) = nothing
function _oversampled_publish(::FTB.NonuniformFFTsBackend, exec, ::Type{T}, ms::NTuple{D, Int},
        mb::NTuple{D, Int}, plan, M::Int) where {T <: Real, D}
    Tr = real(float(T))
    us, normfactor, phis = FTB.oversampled_spectra(plan)
    pos, src, fac = _oversampled_table(Tr, ms, mb, size(first(us)), map(collect, phis), normfactor / M)
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
    plan = Plans.plan_spectrum(_planned_library(t, exec, false), exec, g, E, ms; batch, iflag, eps, batch_chunk,
        box)
    try
        coeffs = _allocate_coefficients_like(field, plan)
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
@inline function _axis_chunk_offsets!(inoff::AbstractVector{Int}, outoff::AbstractVector{Int}, base::Int,
        nvalid::Int, pre::Int, Nd::Int, m::Int)
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

One grid axis's 1-D NUFFT over `C` lines per execution. On the host its executions are split over lanes,
one task each, every lane with a single-threaded library plan, its own strength and spectrum buffers and
line offsets: a library plan threaded over a few short lines costs more than the lines. On a device, one
plan and the staging for the axis-to-front permutation.

A line's `m` output modes are the library's spectrum rows `1:kept` in order, rows past `gap` one row on,
and on a NonuniformFFTs pass whose output widens to `ms_d + 1` modes the row `+ms_d/2` between them, held
in each line's oversampled spectrum ([`_LinePlus`](@ref)).
"""
struct NUFFTAxisPass{PS, VS, US, IX, ST, PL}
    plans::PS                        # one library plan per lane
    values::VS                       # (N_d, C) per lane
    modes::US                        # (library spectrum rows, C) per lane
    inoff::IX                        # (C, lanes) line offsets on the points side
    outoff::IX                       # (C, lanes) line offsets on the modes side
    stage::ST                        # device `(pin, pfk)`, or `nothing` on the host
    plus::PL                         # `_LinePlus` on NonuniformFFTs' complex passes, else `nothing`
    Nd::Int
    m::Int
    kept::Int
    gap::Int
    C::Int
end

Base.show(io::IO, a::NUFFTAxisPass) =
    print(io, "NUFFTAxisPass(N=", a.Nd, " → m=", a.m, ", chunk=", a.C, ", lanes=", length(a.plans), ")")

_close!(a::NUFFTAxisPass) = (foreach(FTB.close!, a.plans); nothing)

# The permutation of a rank-`nd` array bringing axis `d` to the front.
@inline _axis_perm(::Val{nd}, d::Int) where {nd} = (d, ntuple(i -> i < d ? i : i + 1, Val(nd - 1))...)

"""
    _axis_pass(library, exec, ::Type{LT}, insize, d, axis, n, b, rng, off, tol, batch_chunk; iflag)

The [`NUFFTAxisPass`](@ref) along dim `d` of a working array of size `insize` on its points side, whose
lines have element type `LT`: an `n`-mode transform whose coefficient box holds `b` modes on this axis (`n`
or `n + 1`), written to `b` output modes for a complex line and the half `b ÷ 2 + 1` for a real one.
`iflag` is FlowTransformBindings' sign.
"""
function _axis_pass(t::_FTBLibrary, exec, ::Type{LT}, insize::Tuple, d::Int, axis::AbstractVector, n::Int,
        b::Int, rng, off, tol, batch_chunk; iflag::Int = -1) where {LT}
    Tr = real(LT)
    VT = _line_values(t, LT)
    m = LT <: Real ? b ÷ 2 + 1 : b
    Nd = length(axis)
    insize[d] == Nd || throw(DimensionMismatch("axis $d: working length $(insize[d]) ≠ grid axis length $Nd"))
    pre, _, post = _axis_layout(insize, d)
    rest = pre * post
    dev = _on_device(exec)
    C = _chunk(_line_default(rest, dev), rest, batch_chunk)
    lanes = _axis_lanes(exec, cld(rest, C))
    xs = _to_exec(exec, collect(Tr, axis))
    nth = dev ? _backend_nthreads(exec) : 1
    plans = _lane_plans(lanes) do
        FTB.plan_nufft(t, VT, (xs,), (_line_modes(t, n, b),); ntrans = C, tol = Tr(tol), order = FTB.FFTModes(),
            period = Tr(rng), origin = Tr(off), nthreads = nth, iflag)
    end
    return _closing_on_error(plans) do
        nu = first(FTB.mode_size(first(plans)))
        plus = _line_plus(t, exec, VT, first(plans), n, b)
        active = _plus_active(plus)
        # The spectrum rows past `kept` hold zero for a type 2 and are never read after a type 1.
        NUFFTAxisPass(plans, [_alloc(exec, VT, Nd, C) for _ in 1:lanes],
            [fill!(_alloc(exec, Complex{Tr}, nu, C), zero(Complex{Tr})) for _ in 1:lanes],
            Matrix{Int}(undef, C, lanes), Matrix{Int}(undef, C, lanes),
            _axis_stage(exec, LT, Complex{Tr}, Nd, m, rest), plus, Nd, m, min(nu, m - active),
            active ? nu ÷ 2 : nu, C)
    end
end

# The modes a line plan asks its library for: NonuniformFFTs the transform's `n`, its `+n/2` read from the
# oversampled spectrum; FINUFFT the box's `b`.
_line_modes(::FTB.NonuniformFFTsBackend, n::Int, b::Int) = n
_line_modes(::FTB.FINUFFTBackend, n::Int, b::Int) = b

# The strength type of a line plan for lines of element type `T`: real on NonuniformFFTs, whose real-data
# plan takes a real line; complex otherwise.
_line_values(::FTB.NonuniformFFTsBackend, ::Type{T}) where {T <: Real} = T
_line_values(::_FTBLibrary, ::Type{T}) where {T} = Complex{real(T)}

# A type 2's value written to a line of element type `T`: the real part for a real line, which a complex
# plan evaluates from the half with its rows `k₁ < 0` at zero.
@inline _line_value(::Type{<:Real}, v) = real(v)
@inline _line_value(::Type, v) = v

"""
    _LinePlus

Where a NonuniformFFTs line plan of `n` modes holds the mode `+n/2` of an output widened to `n + 1` modes
(`active`): output row `pos[1]`, entry `idx[1]` of each line's oversampled spectrum, which the type 1 reads
times `read[1] = normfactor / ϕ̂(n/2)` and the type 2 writes times `write[1] = 1 / ϕ̂(n/2)`
(`FlowTransformBindings.oversampled_spectra`). The tables live in the plan's memory, where the device
gathers read them.
"""
struct _LinePlus{I, F}
    active::Bool
    pos::I
    idx::I
    read::F
    write::F
end

_plus_active(::Nothing) = false
_plus_active(p::_LinePlus) = p.active

_line_plus(::FTB.FINUFFTBackend, exec, ::Type, plan, n::Int, b::Int) = nothing
_line_plus(::FTB.NonuniformFFTsBackend, exec, ::Type{<:Real}, plan, n::Int, b::Int) = nothing
function _line_plus(::FTB.NonuniformFFTsBackend, exec, ::Type{Complex{T}}, plan, n::Int, b::Int) where {T}
    _, normfactor, phis = FTB.oversampled_spectra(plan)
    h = n ÷ 2
    ϕ = collect(first(phis))[FTB.even_table_index(h, n, false)]
    return _LinePlus(b > n, _to_exec(exec, [h + 1]), _to_exec(exec, [h + 1]), _to_exec(exec, T[normfactor / ϕ]),
        _to_exec(exec, T[1 / ϕ]))
end

# Lanes of a pass of `nexec` executions: one per thread on the host, at most one per execution; one on a
# device.
_axis_lanes(exec::_HostExec, nexec::Int) = max(1, min(_backend_nthreads(exec), nexec))
_axis_lanes(::ComputationalBackends.AbstractGPUBackend, ::Int) = 1

# `n` library plans from `make()`; those built are closed when a later one fails.
function _lane_plans(make::F, n::Int) where {F}
    plans = fill(make(), n)
    k = 1
    try
        while k < n
            plans[k + 1] = make()
            k += 1
        end
    catch
        foreach(_close!, view(plans, 1:k))
        rethrow()
    end
    return plans
end

# Executions `1:nexec` of a pass over its lanes, lane `l` running `l, l + lanes, …` in turn.
function _over_lanes(f::F, ap::NUFFTAxisPass, nexec::Int) where {F}
    nl = length(ap.plans)
    if nl == 1
        for e in 1:nexec
            f(1, e)
        end
    else
        Threads.@threads :static for l in 1:nl
            for e in l:nl:nexec
                f(l, e)
            end
        end
    end
    return nothing
end

# A device pass's staging for its `rest` lines, points of type `LT` and modes of type `CT`, `nothing` on the
# host.
_axis_stage(::_HostExec, ::Type, ::Type, ::Int, ::Int, ::Int) = nothing
_axis_stage(exec::ComputationalBackends.AbstractGPUBackend, ::Type{LT}, ::Type{CT}, Nd::Int, m::Int,
    rest::Int) where {LT, CT} = (_alloc(exec, LT, Nd, rest), _alloc(exec, CT, m, rest))

# Transform axis `d` of `A` into `out`, whose dim `d` holds `ap.m` modes. On the host the lines are
# gathered straight out of `A` and scattered straight into `out`, so neither is permuted.
function _axis_run!(out::Array, A::Array, d::Int, ap::NUFFTAxisPass)
    pre, Nd, post = _axis_layout(size(A), d)
    Nd == ap.Nd || throw(DimensionMismatch("axis $d: field length $Nd ≠ this plan's grid axis length $(ap.Nd)"))
    rest = pre * post
    _over_lanes((l, e) -> _axis_lines!(out, A, ap, l, e, pre, rest), ap, cld(rest, ap.C))
    return out
end

# Execution `e` of the forward pass on lane `l`: lines `(e-1)C+1 …` gathered, transformed, scattered.
function _axis_lines!(out::Array, A::Array, ap::NUFFTAxisPass, l::Int, e::Int, pre::Int, rest::Int)
    Nd, m, C, kept = ap.Nd, ap.m, ap.C, ap.kept
    g = min(ap.gap, kept)
    V, U = ap.values[l], ap.modes[l]
    inoff, outoff = view(ap.inoff, :, l), view(ap.outoff, :, l)
    base = (e - 1) * C
    nvalid = min(C, rest - base)
    _axis_chunk_offsets!(inoff, outoff, base, nvalid, pre, Nd, m)
    @inbounds for i in 1:Nd
        s = (i - 1) * pre
        for c in 1:nvalid
            V[i, c] = A[inoff[c] + s]
        end
        for c in (nvalid + 1):C
            V[i, c] = zero(eltype(V))
        end
    end
    plan = ap.plans[l]
    FTB.nufft_type1!(U, plan, V)
    @inbounds for j in 1:kept
        s = (j - 1 + (j > g)) * pre
        for c in 1:nvalid
            out[outoff[c] + s] = U[j, c]
        end
    end
    return _plus_lines!(out, outoff, pre, nvalid, ap.plus, plan)
end

# The row `+n/2` of the forward's `nvalid` lines, read from the plan's oversampled spectra.
_plus_lines!(out::Array, outoff, pre::Int, nvalid::Int, ::Nothing, plan) = out
function _plus_lines!(out::Array, outoff, pre::Int, nvalid::Int, p::_LinePlus, plan)
    p.active || return out
    us, _, _ = FTB.oversampled_spectra(plan)
    s = (p.pos[1] - 1) * pre
    j, f = p.idx[1], p.read[1]
    @inbounds for c in 1:nvalid
        out[outoff[c] + s] = us[c][j] * f
    end
    return out
end

# On a device the axis moves to the front, so each line is a contiguous column of the staging array.
function _axis_run!(out::AbstractArray, A::AbstractArray, d::Int, ap::NUFFTAxisPass)
    pin, pfk = ap.stage
    perm = _axis_perm(Val(ndims(A)), d)
    Nd, m, C = ap.Nd, ap.m, ap.C
    V, U = first(ap.values), first(ap.modes)
    plan = first(ap.plans)
    rest = size(pin, 2)
    permutedims!(reshape(pin, map(p -> size(A, p), perm)), A, perm)
    for base in 0:C:(rest - 1)
        nvalid = min(C, rest - base)
        copyto!(V, 1, pin, base * Nd + 1, nvalid * Nd)
        nvalid < C && fill!(view(V, :, (nvalid + 1):C), zero(eltype(V)))
        FTB.nufft_type1!(U, plan, V)
        _rows_to_lines!(pfk, base, U, nvalid, ap.kept, ap.gap)
        _plus_columns!(pfk, base * m, m, plan, nvalid, ap.plus)
    end
    permutedims!(out, reshape(pfk, map(p -> size(out, p), perm)), invperm(collect(perm)))
    return out
end

# Rows `1:kept` of the first `nvalid` columns of `U` into columns `base+1 …` of `F`, rows past `gap` one row
# on; and back, for the type 2.
function _rows_to_lines!(F::AbstractMatrix, base::Int, U::AbstractMatrix, nvalid::Int, kept::Int, gap::Int)
    cols = (base + 1):(base + nvalid)
    g = min(gap, kept)
    view(F, 1:g, cols) .= view(U, 1:g, 1:nvalid)
    kept > g && (view(F, (g + 2):(kept + 1), cols) .= view(U, (g + 1):kept, 1:nvalid))
    return F
end
function _lines_to_rows!(U::AbstractMatrix, F::AbstractMatrix, base::Int, nvalid::Int, kept::Int, gap::Int)
    cols = (base + 1):(base + nvalid)
    g = min(gap, kept)
    view(U, 1:g, 1:nvalid) .= view(F, 1:g, cols)
    kept > g && (view(U, (g + 1):kept, 1:nvalid) .= view(F, (g + 2):(kept + 1), cols))
    return U
end

# The row `+n/2` of `nvalid` staged lines, `m` apart from `off` in `F`, read from the plan's oversampled
# spectra.
_plus_columns!(F, off::Int, m::Int, plan, nvalid::Int, ::Nothing) = F
function _plus_columns!(F, off::Int, m::Int, plan, nvalid::Int, p::_LinePlus)
    p.active || return F
    us, _, _ = FTB.oversampled_spectra(plan)
    return _scatter_each!(F, off, m, us, nvalid, p.pos, p.idx, p.read, false)
end

# A stretched axis of the hybrid composite, an `n`-mode transform onto the box's `b` modes, `insize` the
# working size on its points side and `iflag` FlowTransformBindings' sign. The forward (`HybridPlan`) takes
# the `Σ e^{-ikx}` sign of its FFT pass and conjugates once for `iflag = -1`; the synthesis
# (`HybridSynthesisPlan`) passes its own.
_axis_nufft_plan(t::_FTBLibrary, exec::ComputationalBackends.AbstractExecutionBackend, ::Type{Tr},
        insize::Tuple, d::Int, axis::AbstractVector, n::Int, b::Int, rng, off, eps; batch_chunk = nothing,
        iflag::Int = -1, kwargs...) where {Tr} =
    _axis_pass(t, exec, Complex{Tr}, insize, d, axis, n, b, rng, off, eps, batch_chunk; iflag)

# Axis passes in turn, pass `i` along dim `sdims[i]` reading the previous one's output and writing
# `work[i]`: the forward's type 1, points to modes.
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
    rest = pre * post
    _over_lanes((l, e) -> _axis_synth_lines!(out, A, ap, l, e, pre, rest), ap, cld(rest, ap.C))
    return out
end

# Execution `e` of the type 2 on lane `l`.
function _axis_synth_lines!(out::Array, A::Array, ap::NUFFTAxisPass, l::Int, e::Int, pre::Int, rest::Int)
    Nd, m, C, kept = ap.Nd, ap.m, ap.C, ap.kept
    g = min(ap.gap, kept)
    V, U = ap.values[l], ap.modes[l]
    inoff, outoff = view(ap.inoff, :, l), view(ap.outoff, :, l)
    base = (e - 1) * C
    nvalid = min(C, rest - base)
    _axis_chunk_offsets!(inoff, outoff, base, nvalid, pre, Nd, m)
    @inbounds for j in 1:kept
        s = (j - 1 + (j > g)) * pre
        for c in 1:nvalid
            U[j, c] = A[outoff[c] + s]
        end
        for c in (nvalid + 1):C
            U[j, c] = zero(eltype(U))
        end
    end
    _line_type2!(V, ap.plans[l], U, ap.plus, A, outoff, pre, nvalid)
    @inbounds for i in 1:Nd
        s = (i - 1) * pre
        for c in 1:nvalid
            out[inoff[c] + s] = _line_value(eltype(out), V[i, c])
        end
    end
    return out
end

# The type 2 of `nvalid` lines, the row `+n/2` of each written into the plan's oversampled spectra.
_line_type2!(V, plan, U, ::Nothing, A::Array, outoff, pre::Int, nvalid::Int) = FTB.nufft_type2!(V, plan, U)
function _line_type2!(V, plan, U, p::_LinePlus, A::Array, outoff, pre::Int, nvalid::Int)
    p.active || return FTB.nufft_type2!(V, plan, U)
    s = (p.pos[1] - 1) * pre
    j, f = p.idx[1], p.write[1]
    return FTB.oversampled_type2!(V, plan, U) do us
        @inbounds for c in 1:nvalid
            us[c][j] = A[outoff[c] + s] * f
        end
    end
end

# On a device the axis moves to the front, so each line is a contiguous column of the staging array.
function _axis_synth!(out::AbstractArray, A::AbstractArray, d::Int, ap::NUFFTAxisPass)
    pin, pfk = ap.stage
    perm = _axis_perm(Val(ndims(A)), d)
    Nd, m, C = ap.Nd, ap.m, ap.C
    V, U = first(ap.values), first(ap.modes)
    plan = first(ap.plans)
    rest = size(pfk, 2)
    permutedims!(reshape(pfk, map(p -> size(A, p), perm)), A, perm)
    for base in 0:C:(rest - 1)
        nvalid = min(C, rest - base)
        _lines_to_rows!(U, pfk, base, nvalid, ap.kept, ap.gap)
        nvalid < C && fill!(view(U, :, (nvalid + 1):C), zero(eltype(U)))
        _staged_type2!(V, plan, U, ap.plus, pfk, base * m, m, nvalid)
        _values_to_lines!(pin, base * Nd, V, nvalid * Nd)
    end
    permutedims!(out, reshape(pin, map(p -> size(out, p), perm)), invperm(collect(perm)))
    return out
end

# The first `n` values of `V` into `pin` from offset `off`, their real parts where the lines are real.
_values_to_lines!(pin::AbstractArray{T}, off::Int, V::AbstractArray{T}, n::Int) where {T} =
    copyto!(pin, off + 1, V, 1, n)
function _values_to_lines!(pin::AbstractArray{<:Real}, off::Int, V::AbstractArray{<:Complex}, n::Int)
    view(vec(pin), (off + 1):(off + n)) .= real.(view(vec(V), 1:n))
    return pin
end

# The device type 2 of `nvalid` staged lines, `m` apart from `off` in `F`, the row `+n/2` of each written
# into the plan's oversampled spectra.
_staged_type2!(V, plan, U, ::Nothing, F, off::Int, m::Int, nvalid::Int) = FTB.nufft_type2!(V, plan, U)
function _staged_type2!(V, plan, U, p::_LinePlus, F, off::Int, m::Int, nvalid::Int)
    p.active || return FTB.nufft_type2!(V, plan, U)
    return FTB.oversampled_type2!(V, plan, U) do us
        _inject_each!(us, F, off, m, nvalid, p.pos, p.idx, p.write, false)
    end
end

# =============================================================================
# Separable transform for a nonuniform tensor-product (structured) Cartesian grid: `D` successive 1-D
# passes, spreading `D·2m` per point over the axes alone. Each pass writes its coefficient box's modes. A
# real field's first pass writes the `k₁ ≥ 0` half of axis 1 (a real-data plan on NonuniformFFTs; FINUFFT's
# complex plan keeping those rows), and the passes after it run on those lines alone, so the last pass writes
# the coefficients in their packed layout. The field is staged with its quadrature weights over `∏N_d`,
# which carries the transform's normalization.
# =============================================================================

"""
    NUFFTSeparablePlan{T,D,R}

Reusable separable NUFFT over a nonuniform tensor-product Cartesian grid: one [`NUFFTAxisPass`](@ref)
per axis, the working arrays the passes write through, and the per-point weights the field is staged with.
Execute with `calculate_spectrum!(coeffs, plan, field)`; [`close!`](@ref) releases the library plans.
"""
struct NUFFTSeparablePlan{T, D, R, NB, AP, W, WP, KS} <: Plans.AbstractSpectralPlan
    axes::AP                         # D × NUFFTAxisPass
    work::W                          # D+1 working arrays; `work[1]` takes the field, `work[D+1]` the coefficients
    w::WP                            # quadrature factor over ∏N_d per point, one number for a constant measure
    batch::NTuple{NB, Int}
    ms::NTuple{D, Int}
    box::NTuple{D, Int}              # coefficient box
    pms::NTuple{D, Int}              # packed coefficient box
    npts::Int
    ntrans::Int
    neg::Bool                        # real field with `iflag < 0`: conjugate the coefficients
    ks_phys::KS
end

Base.show(io::IO, p::NUFFTSeparablePlan{T, D, R}) where {T, D, R} =
    print(io, "NUFFTSeparablePlan{", T, ", ", D, "}(", R ? "real" : "complex", ", ", first(first(p.axes).plans), ")")

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
    ndims(g) == D || throw(DimensionMismatch("grid has $(ndims(g)) dims; asked for $D mode counts"))
    axs = ntuple(d -> Tr.(FlowGeometries.Grids.coordinates(g, d)), Val(D))
    offsets, ranges = Grids.axis_geometry(Tr, g, Val(D), box)
    Ns = ntuple(d -> length(axs[d]), Val(D))
    npts = prod(Ns)
    ntrans = prod(batch; init = 1)
    mb = Grids.mode_box(g, ms, T)
    nsx = ntuple(d -> (d == 1 && T <: Real) ? mb[1] ÷ 2 + 1 : mb[d], Val(D))
    tol = _nufft_tol(Tr, eps)
    sgn = T <: Real ? -1 : -iflag
    W1 = T <: Real ? Tr : Complex{Tr}
    work = (_alloc(exec, W1, Ns..., batch...),
        ntuple(d -> _alloc(exec, Complex{Tr}, _axis_work_shape(Ns, nsx, d + 1, batch)...), Val(D))...)
    sizes = map(size, Base.front(work))
    pass1 = _axis_pass(t, exec, W1, sizes[1], 1, axs[1], ms[1], mb[1], ranges[1], offsets[1], tol, batch_chunk;
        iflag = sgn)
    axes = _closing_on_error((pass1,)) do
        rest = _axis_passes(Base.tail(ntuple(identity, Val(D))), Base.tail(axs), Base.tail(sizes)) do d, axis, insize
            _axis_pass(t, exec, Complex{Tr}, insize, d, axis, ms[d], mb[d], ranges[d], offsets[d], tol, batch_chunk;
                iflag = sgn)
        end
        (pass1, rest...)
    end
    return _closing_on_error(axes) do
        pms = FTB.packed_size(T, mb)
        ks_phys = Grids.physical_wavenumbers(ranges, mb, T)
        w = _point_weights(exec, Grids.quadrature_scale(g, Tr, npts), Tr, npts)
        bt = NTuple{length(batch), Int}(batch)
        NUFFTSeparablePlan{Tr, D, T <: Real, length(bt), typeof(axes), typeof(work), typeof(w), typeof(ks_phys)}(
            axes, work, w, bt, ms, mb, pms, npts, ntrans, T <: Real && iflag < 0, ks_phys)
    end
end

"""
    calculate_spectrum!(coeffs, plan::NUFFTSeparablePlan, field) -> ks_phys

Execute a separable structured plan in place. `field` is `(N₁…N_D, batch…)`; `coeffs` is the packed half
of the coefficient box (`Grids.mode_box`) for a real field and the full spectrum for a complex one.
"""
function calculate_spectrum!(coeffs::AbstractArray{Complex{T}}, p::NUFFTSeparablePlan{T, D, R},
        field) where {T, D, R}
    A = first(p.work)
    length(field) == length(A) || throw(DimensionMismatch(
        "field holds $(length(field)) values; this plan was built for $(length(A)) — pass the matching " *
        "`batch=` to plan_spectrum"))
    Plans._check_coefficients(coeffs, p)
    _stage_weighted!(A, field, p.w, p.npts, p.ntrans, false)
    F = _run_into!(coeffs, A, p.work, p.axes, ntuple(identity, Val(D)))
    _finish_coefficients!(coeffs, F, p.ms, p.box, p.neg)
    return p.ks_phys
end

# The passes from `A`, the last one writing `coeffs` where they are in the plan's memory and type, the plan's
# last working array otherwise.
@inline _run_into!(coeffs, A, work::Tuple, axes::Tuple, sdims::Tuple) =
    _run_stretched!(A, (Base.front(Base.tail(work))..., _output_in(last(work), coeffs)), axes, sdims)

# The last pass's output `F` as the coefficients: the corners that hold zero set (`_zero_corners!`), the
# conjugate for a real field's `iflag = -1`, and copied into `coeffs` where `F` is the plan's array.
function _finish_coefficients!(coeffs, F, ms::Tuple, box::Tuple, neg::Bool)
    _zero_corners!(F, ms, box)
    neg && (F .= conj.(F))
    F === coeffs || copyto!(coeffs, F)
    return coeffs
end

# =============================================================================
# Separable synthesis on a nonuniform tensor-product Cartesian grid: `D` successive 1-D type-2 passes,
# modes to points, axes `2 … D` first and axis 1 last. A real field's packed half runs the passes on its
# `k₁ ≥ 0` lines and axis 1 writes the real field: a real-data plan on NonuniformFFTs, which counts each
# row `k₁ > 0` with its mirror; on FINUFFT a complex plan over the box's axis-1 modes, its rows `k₁ < 0` at
# zero and the real part taken. The coefficients are staged with the row weights that make either sum the
# whole box ([`_half_row_weights`](@ref)), and conjugated for `iflag = -1`.
# =============================================================================

"""
    NUFFTSeparableSynthesisPlan{T,D,R}

Reusable synthesis on a nonuniform tensor-product Cartesian grid, the counterpart of
[`NUFFTSeparablePlan`](@ref): one type-2 [`NUFFTAxisPass`](@ref) per axis in `sdims` order, the working
arrays the passes write through, and a real field's axis-1 row weights. Execute with
`synthesize!(out, plan, coeffs)`; [`close!`](@ref) releases the library plans.
"""
struct NUFFTSeparableSynthesisPlan{T, D, R, NB, AP, SD, W, WV, SP} <: Plans.AbstractSynthesisPlan
    axes::AP                         # D type-2 passes, in `sdims` order
    sdims::SD                        # `(2, …, D, 1)`
    work::W                          # `work[1]` the coefficients, `work[i+1]` pass `i`'s output
    w1::WV                           # a real field's axis-1 row weights, `nothing` for a complex field
    neg::Bool                        # real field with `iflag < 0`: the coefficients conjugated
    spatial::SP
    batch::NTuple{NB, Int}
end

Base.show(io::IO, p::NUFFTSeparableSynthesisPlan{T, D, R}) where {T, D, R} =
    print(io, "NUFFTSeparableSynthesisPlan{", T, ", ", D, "}(", R ? "real" : "complex", ", ",
        first(first(p.axes).plans), ")")

Plans.field_size(p::NUFFTSeparableSynthesisPlan) = (p.spatial..., p.batch...)
Plans.field_type(::NUFFTSeparableSynthesisPlan{T, D, R}) where {T, D, R} = R ? T : Complex{T}
Plans.close!(p::NUFFTSeparableSynthesisPlan) = (foreach(_close!, p.axes); nothing)

"""
    _half_row_weights(library, ::Type{T}, m1, b1) -> Vector{T}

The weight each row `k₁ = 0 … b1÷2` of a real field's packed half enters the axis-1 type 2 with, `m1` the
transform's and `b1` the coefficient box's axis-1 mode count, so the pass sums the whole box. NonuniformFFTs'
real-data plan counts each row past `k₁ = 0` with its mirror: `1/2` on a Nyquist row the box does not
widen, `1` elsewhere. FINUFFT's complex plan reads no mirror: the row's fold weight, `2` past `k₁ = 0`
but on that Nyquist row.
"""
_half_row_weights(::FTB.NonuniformFFTsBackend, ::Type{T}, m1::Int, b1::Int) where {T} =
    [_nyquist_row(k, m1, b1) ? T(1 // 2) : one(T) for k in 0:(b1 ÷ 2)]
_half_row_weights(::FTB.FINUFFTBackend, ::Type{T}, m1::Int, b1::Int) where {T} =
    [(k == 0 || _nyquist_row(k, m1, b1)) ? one(T) : T(2) for k in 0:(b1 ÷ 2)]

# A real field's row weights on the execution backend, `nothing` for a complex field.
_staging_weights(t::_FTBLibrary, exec, ::Type{<:Real}, ::Type{T}, m1::Int, b1::Int) where {T} =
    _to_exec(exec, _half_row_weights(t, T, m1, b1))
_staging_weights(::_FTBLibrary, exec, ::Type{<:Complex}, ::Type, ::Int, ::Int) = nothing

function Plans.plan_synthesis(t::_FTBLibrary,
        exec::Union{_HostExec, ComputationalBackends.AbstractGPUBackend},
        g::FlowGeometries.Grids.AbstractStructuredGrid{<:FlowGeometries.Geometry.AbstractCartesianGeometry},
        ::Type{TT}, ms::NTuple{D, Int}; batch::Tuple = (), iflag::Int = 1, eps = nothing,
        batch_chunk::Union{Nothing, Integer} = nothing, box = nothing, kwargs...) where {TT, D}
    T = real(float(TT))
    ndims(g) == D || throw(DimensionMismatch("grid has $(ndims(g)) dims; asked for $D mode counts"))
    axs = ntuple(d -> T.(FlowGeometries.Grids.coordinates(g, d)), Val(D))
    offsets, ranges = Grids.axis_geometry(T, g, Val(D), box)
    Ns = ntuple(d -> length(axs[d]), Val(D))
    bt = NTuple{length(batch), Int}(batch)
    mb = Grids.mode_box(g, ms, TT)
    s0 = (ntuple(d -> (d == 1 && TT <: Real) ? mb[1] ÷ 2 + 1 : mb[d], Val(D))..., bt...)
    tol = _nufft_tol(T, eps)
    # The type 2 evaluates `Σ C e^{+iflag·iκx}`, FlowTransformBindings' `-iflag`; a real field's passes take
    # `iflag = +1` and the conjugate coefficients for `-1`.
    sgn = TT <: Real ? -1 : -iflag
    LT = TT <: Real ? T : Complex{T}
    sdims = (Base.tail(ntuple(identity, Val(D)))..., 1)
    # Pass `i` grows dim `sdims[i]` from its box's modes to its points: `outs[i]` is the size it writes.
    outs = accumulate((s, d) -> _axis_out_size(s, d, Ns[d]), sdims; init = s0)
    lead = _axis_passes(Base.front(sdims), Base.tail(axs), Base.front(outs)) do d, axis, outsize
        _axis_pass(t, exec, Complex{T}, outsize, d, axis, ms[d], mb[d], ranges[d], offsets[d], tol, batch_chunk;
            iflag = sgn)
    end
    axes = _closing_on_error(lead) do
        (lead..., _axis_pass(t, exec, LT, last(outs), 1, first(axs), ms[1], mb[1], ranges[1], offsets[1], tol,
            batch_chunk; iflag = sgn))
    end
    return _closing_on_error(axes) do
        work = (_alloc(exec, Complex{T}, s0...), map(s -> _alloc(exec, Complex{T}, s...), Base.front(outs))...,
            _alloc(exec, LT, last(outs)...))
        w1 = _staging_weights(t, exec, TT, T, ms[1], mb[1])
        NUFFTSeparableSynthesisPlan{T, D, TT <: Real, length(bt), typeof(axes), typeof(sdims), typeof(work),
                typeof(w1), typeof(Ns)}(axes, sdims, work, w1, TT <: Real && iflag < 0, Ns, bt)
    end
end

function Plans.synthesize!(out::AbstractArray, p::NUFFTSeparableSynthesisPlan{T, D, R},
        coeffs::AbstractArray) where {T, D, R}
    size(out) == Plans.field_size(p) || throw(DimensionMismatch(
        "out is $(size(out)); this plan writes $(Plans.field_size(p))"))
    W = first(p.work)
    size(coeffs) == size(W) || throw(DimensionMismatch(
        "coeffs is $(size(coeffs)); this plan reads $(size(W))"))
    _stage_half!(W, coeffs, p.w1, p.neg)
    dst = _output_in(last(p.work), out)
    _run_stretched_synthesis!(W, (Base.front(Base.tail(p.work))..., dst), p.axes, p.sdims)
    dst === out || copyto!(out, dst)
    return out
end

# The coefficients in the plan's first working array: a real field's half weighted along axis 1, and
# conjugated for `iflag < 0`.
_stage_half!(W::AbstractArray, coeffs, ::Nothing, neg::Bool) = _stage!(W, coeffs)
function _stage_half!(W::AbstractArray, coeffs, w1::AbstractVector, neg::Bool)
    _stage!(W, coeffs)
    neg ? (W .= conj.(W) .* w1) : (W .*= w1)
    return W
end

# =============================================================================
# Synthesis onto a point set's points: a type 2 from the coefficients. The forward publishes `C = fk / M`
# measured from each direction's origin, the point positions `y` the type 2 folds, so it evaluates
# `Σ C e^{+iflag·ik·y}` as given. A complex field's coefficients are the series. A real field's packed half
# is evaluated on NonuniformFFTs by a real-data plan at the forward's modes, its modes at `+ms_d/2` written
# into the plan's oversampled spectra (`FlowTransformBindings.oversampled_type2!`); on FINUFFT, which has
# no real transform, it is written as the complex series of `Grids.series_modes`
# (`DirectSum.native_coefficients!`) and the real part taken. Every step runs in the plan's memory.
# =============================================================================

"""
    NUFFTSynthesisPlan{T,D,R}

Reusable type-2 NUFFT synthesis onto a Cartesian grid's points: the plan with `C` transforms per execution,
its spectrum and value buffers, how it reads the coefficients ([`_synthesis_reader`](@ref)), and on a
device the coefficient and field buffers a host array passes through. Execute with
`synthesize!(out, plan, coeffs)`; [`close!`](@ref) releases the library plan.
"""
struct NUFFTSynthesisPlan{T, D, R, NB, P, U, V, RD, CB, OB, SP} <: Plans.AbstractSynthesisPlan
    plan::P
    modes::U                         # (ns…, C)
    values::V                        # (M, C)
    reader::RD                       # `nothing`, `_ComplexSeries` or `_RealHalf`
    cbuf::CB                         # device coefficients, `nothing` on the host
    outd::OB                         # device field, `nothing` on the host
    ms::NTuple{D, Int}
    box::NTuple{D, Int}              # coefficient box
    ns::NTuple{D, Int}               # mode counts of the library plan's spectrum
    spatial::SP
    batch::NTuple{NB, Int}
    ntrans::Int
    M::Int
    C::Int
end

# A real field's packed half written as the complex series: the `DirectSum.native_table`, the series, and
# the real part of the values.
struct _ComplexSeries{NT, FB, RV}
    native::NT
    full::FB
    rvals::RV
end

# A real field's packed half in a real-data plan: each plan mode's coefficient and weight (`_mode_src`),
# and the coefficients at `+ms_d/2` with their entries and factors in the plan's oversampled spectra
# (`_oversampled_table`, times the same weight).
struct _RealHalf{IX, WV, P, J, W}
    src::IX
    w::WV
    pos::P
    idx::J
    fac::W
    neg::Bool                        # `iflag = -1`: the coefficients conjugated
end

"""
    _synthesis_reader(library, exec, ::Type{TT}, ::Type{T}, plan, ms, box, ns, M, ntrans, C, iflag)

How a synthesis of a field of element type `TT` reads its coefficients: `nothing` for a complex field,
whose coefficients are the series; a `_RealHalf` for a real field on NonuniformFFTs; a `_ComplexSeries` for
a real field on FINUFFT.
"""
_synthesis_reader(::_FTBLibrary, exec, ::Type{<:Complex}, args...) = nothing
_synthesis_reader(::FTB.FINUFFTBackend, exec, ::Type{<:Real}, ::Type{T}, plan, ms::Tuple, box::Tuple, ns::Tuple,
    M::Int, ntrans::Int, C::Int, iflag::Int) where {T} =
    _ComplexSeries(map(a -> _to_exec(exec, a), DirectSum.native_table(ms, box)),
        _alloc(exec, Complex{T}, ns..., ntrans), _alloc(exec, T, M, C))
function _synthesis_reader(::FTB.NonuniformFFTsBackend, exec, ::Type{<:Real}, ::Type{T}, plan, ms::Tuple,
        box::Tuple, ns::Tuple, M::Int, ntrans::Int, C::Int, iflag::Int) where {T}
    us, _, phis = FTB.oversampled_spectra(plan)
    pos, idx, fac = _oversampled_table(T, ms, box, size(first(us)), map(collect, phis), one(T))
    pb1 = FTB.packed_size(Real, box)[1]
    for i in eachindex(pos)
        _nyquist_row(mod1(pos[i], pb1) - 1, ms[1], box[1]) && (fac[i] /= 2)
    end
    src, w = _mode_src(T, ms, box)
    return _RealHalf(_to_exec(exec, src), _to_exec(exec, w), _to_exec(exec, pos), _to_exec(exec, idx),
        _to_exec(exec, fac), iflag < 0)
end

# Whether a field of element type `TT` is synthesized by a real-data plan on `t`.
_real_half(t::_FTBLibrary, ::Type{TT}) where {TT} = TT <: Real && _real_values(t)

Base.show(io::IO, p::NUFFTSynthesisPlan{T, D, R}) where {T, D, R} =
    print(io, "NUFFTSynthesisPlan{", T, ", ", D, "}(", R ? "real" : "complex", ", ", p.plan, ")")

Plans.field_size(p::NUFFTSynthesisPlan) = (p.spatial..., p.batch...)
Plans.field_type(::NUFFTSynthesisPlan{T, D, R}) where {T, D, R} = R ? T : Complex{T}
Plans.close!(p::NUFFTSynthesisPlan) = FTB.close!(p.plan)

function Plans.plan_synthesis(t::_FTBLibrary,
        exec::Union{_HostExec, ComputationalBackends.AbstractGPUBackend}, g::Grids.PointwiseCartesian,
        ::Type{TT}, ms::NTuple{D, Int}; batch::Tuple = (), iflag::Int = 1, eps = nothing,
        batch_chunk::Union{Nothing, Integer} = nothing, box = nothing, kwargs...) where {TT, D}
    T = real(float(TT))
    coords, spatial = Grids.point_coordinates(T, g, Val(D))
    # The forward's origins and Fourier lengths, `box`'s where `g` is part of a larger grid.
    offsets, Ls = Grids.axis_geometry(T, g, Val(D), box)
    M = length(coords[1])
    bt = NTuple{length(batch), Int}(batch)
    ntrans = prod(bt; init = 1)
    nth = _backend_nthreads(exec)
    dev = _on_device(exec)
    C = _chunk(_batch_default(t, ntrans, dev), ntrans, batch_chunk)
    mb = Grids.mode_box(g, ms, TT)
    half = _real_half(t, TT)
    VT = half ? T : Complex{T}
    # The type 2 evaluates `Σ û e^{+iflag·ik·y}`, FlowTransformBindings' `-iflag` convention; a real-data
    # plan takes `iflag = +1` and the conjugate coefficients for `-1`.
    plan = FTB.plan_nufft(t, VT, ntuple(d -> _to_exec(exec, coords[d]), Val(D)),
        half ? ms : Grids.series_modes(g, ms, TT); ntrans = C, tol = _nufft_tol(T, eps), order = FTB.FFTModes(),
        period = Ls, origin = offsets, nthreads = nth, iflag = half ? -1 : -iflag)
    return _closing_on_error((plan,)) do
        ns = FTB.mode_size(plan)
        modes = _alloc(exec, Complex{T}, ns..., C)
        values = _alloc(exec, VT, M, C)
        reader = _synthesis_reader(t, exec, TT, T, plan, ms, mb, ns, M, ntrans, C, iflag)
        cbuf = _device_buffer(exec, Complex{T}, FTB.packed_size(TT, mb)..., bt...)
        outd = _device_buffer(exec, float(TT), spatial..., bt...)
        NUFFTSynthesisPlan{T, D, TT <: Real, length(bt), typeof(plan), typeof(modes), typeof(values),
                typeof(reader), typeof(cbuf), typeof(outd), typeof(spatial)}(
            plan, modes, values, reader, cbuf, outd, ms, mb, ns, spatial, bt, ntrans, M, C)
    end
end

function Plans.synthesize!(out::AbstractArray, p::NUFFTSynthesisPlan{T, D, R},
        coeffs::AbstractArray) where {T, D, R}
    size(out) == Plans.field_size(p) || throw(DimensionMismatch(
        "out is $(size(out)); this plan writes $(Plans.field_size(p))"))
    sz = FTB.packed_size(_layout_type(R), p.box)
    size(coeffs) == (sz..., p.batch...) || throw(DimensionMismatch(
        "coeffs is $(size(coeffs)); this plan reads $((sz..., p.batch...))"))
    dst = _output_in(p.outd, out)
    _synthesis_run!(dst, p, _input_in(p.cbuf, coeffs), p.reader)
    dst === out || copyto!(out, dst)
    return out
end

# A complex field: its coefficients are the series.
_synthesis_run!(O::AbstractArray, p::NUFFTSynthesisPlan, src, ::Nothing) = _series_run!(O, p, src, nothing)

# A real field on FINUFFT: the complex series, its values' real part.
_synthesis_run!(O::AbstractArray, p::NUFFTSynthesisPlan, src, r::_ComplexSeries) =
    _series_run!(O, p, DirectSum.native_coefficients!(r.full, src, r.native...), r.rvals)

# The type 2 `C` series at a time, each chunk copied into the plan's spectrum buffer (its unused columns
# zero) and its values into the field `O`, `M` per transform, through `rvals` where they are complex and
# the field real.
function _series_run!(O::AbstractArray, p::NUFFTSynthesisPlan{T, D}, full, rvals) where {T, D}
    Pm = prod(p.ns)
    C, M = p.C, p.M
    for base in 0:C:(p.ntrans - 1)
        nvalid = min(C, p.ntrans - base)
        copyto!(p.modes, 1, full, base * Pm + 1, nvalid * Pm)
        _zero_unused!(p.modes, nvalid, C, Val(D))
        FTB.nufft_type2!(p.values, p.plan, p.modes)
        _publish_values!(O, base * M, p.values, nvalid * M, rvals)
    end
    return O
end

# A real field on NonuniformFFTs: each chunk's packed half gathered into the real-data plan's modes, its
# modes at `+ms_d/2` written into the plan's oversampled spectra, the real values copied into `O`.
function _synthesis_run!(O::AbstractArray, p::NUFFTSynthesisPlan{T, D}, src, r::_RealHalf) where {T, D}
    Pm = prod(p.ns)
    Pc = prod(FTB.packed_size(Real, p.box))
    C, M = p.C, p.M
    for base in 0:C:(p.ntrans - 1)
        nvalid = min(C, p.ntrans - base)
        _gather_scaled!(p.modes, 0, src, base * Pc, Pc, r.src, r.w, Pm, nvalid, false, r.neg)
        _zero_unused!(p.modes, nvalid, C, Val(D))
        FTB.oversampled_type2!(p.values, p.plan, p.modes) do spectra
            _inject_each!(spectra, src, base * Pc, Pc, nvalid, r.pos, r.idx, r.fac, r.neg)
        end
        copyto!(O, base * M + 1, p.values, 1, nvalid * M)
    end
    return O
end

# The spectrum buffer's columns past the `nvalid` transforms of a chunk at zero.
_zero_unused!(modes::AbstractArray, nvalid::Int, C::Int, ::Val{D}) where {D} =
    nvalid < C ? fill!(view(modes, ntuple(_ -> Colon(), Val(D))..., (nvalid + 1):C), zero(eltype(modes))) : modes

_publish_values!(O, off::Int, vals, n::Int, ::Nothing) = copyto!(O, off + 1, vals, 1, n)
function _publish_values!(O, off::Int, vals, n::Int, rvals)
    rvals .= real.(vals)
    return copyto!(O, off + 1, rvals, 1, n)
end

function _synthesize(t::_FTBLibrary, exec::Union{_HostExec, ComputationalBackends.AbstractGPUBackend},
        g::Union{FlowGeometries.Grids.AbstractStructuredGrid{<:FlowGeometries.Geometry.AbstractCartesianGeometry},
                 Grids.PointwiseCartesian},
        ::Type{T}, coeffs::AbstractArray{Complex{Tr}}, ms::NTuple{D, Int}; iflag::Int = 1,
        eps = nothing, batch_chunk = nothing, kwargs...) where {T, Tr <: Real, D}
    _check_series(coeffs, FTB.packed_size(T, Grids.mode_box(g, ms, T)), T)
    batch = ntuple(i -> size(coeffs, D + i), Val(ndims(coeffs) - D))
    plan = Plans.plan_synthesis(_planned_library(t, exec, false), exec, g, T <: Real ? Tr : Complex{Tr}, ms;
        batch, iflag, eps, batch_chunk)
    try
        return Plans.synthesize!(_allocate_field_like(coeffs, plan), plan, coeffs)
    finally
        Plans.close!(plan)
    end
end
