module Packing

# Packed-native spectral layout. A real Cartesian field transforms to an rfft-packed half-spectrum
# (axis 1 holds modes 0:n₁÷2, length n₁÷2+1); a complex field to the full spectrum in native (unshifted)
# `fftfreq` order. The negative-axis-1 half of a real spectrum is recoverable as `C[k] = conj(C[-k])`,
# so it is stored once. `unpacked` reconstructs the full native-order complex array for plotting/interop;
# `fold_weight` gives the energy multiplier (2 on interior halved-axis modes, 1 at dc/Nyquist) so a
# reduction over the stored half equals the two-sided sum.

export unpacked, unpacked!

# Wavenumber axis of a halved (rfft) direction: modes 0:n÷2, value `(i-1)·2π/L`. Its type marks the
# direction as Hermitian-halved, so a reduction folds its interior modes. `twin` carries the
# [`NyquistTwin`](@ref) values a reduction over this axis needs on a nonperiodically-sampled grid, or
# `nothing` when index negation already reaches them (a uniform grid, or no even full axis).
struct RFFTAxis{FT, TW} <: AbstractVector{FT}
    scale::FT      # 2π/L
    n::Int         # full transform length; the axis stores n÷2+1 modes
    twin::TW
end
RFFTAxis(scale::FT, n::Int) where {FT} = RFFTAxis{FT, Nothing}(scale, n, nothing)
Base.size(a::RFFTAxis) = (a.n ÷ 2 + 1,)
Base.IndexStyle(::Type{<:RFFTAxis}) = Base.IndexLinear()
Base.@propagate_inbounds Base.getindex(a::RFFTAxis{FT}, i::Int) where {FT} = FT(i - 1) * a.scale

# The twin a reduction should consult for this axis, and the axis carrying one.
@inline axis_twin(a::RFFTAxis) = a.twin
@inline axis_twin(::AbstractVector) = nothing
@inline with_twin(a::RFFTAxis, tw) = RFFTAxis{eltype(a), typeof(tw)}(a.scale, a.n, tw)

# Wavenumber axis of a full direction in native order: `[0,1,…,(n-1)÷2, -(n÷2),…,-1]·2π/L`.
struct FFTAxis{FT} <: AbstractVector{FT}
    scale::FT
    n::Int
end
Base.size(a::FFTAxis) = (a.n,)
Base.IndexStyle(::Type{<:FFTAxis}) = Base.IndexLinear()
Base.@propagate_inbounds function Base.getindex(a::FFTAxis{FT}, i::Int) where {FT}
    j = i - 1
    m = j <= (a.n - 1) ÷ 2 ? j : j - a.n
    return FT(m) * a.scale
end

@inline _axis_scale(L::FT) where {FT} = FT(2π) / (L == zero(FT) ? one(FT) : L)

# Packed spectral size for a Cartesian transform of full sizes `ns`: a real input halves axis 1.
@inline packed_size(ns::NTuple{D, Int}, ::Val{true}) where {D} =
    ntuple(d -> d == 1 ? ns[d] ÷ 2 + 1 : ns[d], Val(D))
@inline packed_size(ns::NTuple{D, Int}, ::Val{false}) where {D} = ns

# Energy fold multiplier for mode index `i` of a halved axis of full length `n`: the dc mode (i==1) and,
# for even `n`, the Nyquist mode (i==n÷2+1) are their own conjugate partners; every interior mode stands
# in for a stored-once negative partner.
@inline is_halved(::RFFTAxis) = true
@inline is_halved(::AbstractVector) = false
@inline function fold_weight(a::RFFTAxis{FT}, i::Int) where {FT}
    return (i == 1 || (iseven(a.n) && i == a.n ÷ 2 + 1)) ? one(FT) : FT(2)
end

# The even-Nyquist mode of a halved axis: stored once at `k = +n/2`, aliasing the full axes' `−n/2`.
@inline is_nyquist(a::RFFTAxis, i::Int) = iseven(a.n) && i == a.n ÷ 2 + 1
@inline is_nyquist(::AbstractVector, ::Int) = false

# The `−n/2` entry of a full native axis (even `n` only), which sits at index `n÷2+1`. A mode on such an
# entry cannot have its `k₁<0` partner obtained by mirroring the stored magnitude: Hermitian sends it to
# the `+n/2` twin, a different value unless the grid is periodic. Reductions must treat it specially.
@inline is_neg_nyquist(a::FFTAxis, i::Int) = iseven(a.n) && i == a.n ÷ 2 + 1
@inline is_neg_nyquist(::AbstractVector, ::Int) = false

# Which full axes `d ≥ 2` hold mode `I` on their `−n_d/2` plane, as a bitmask with bit `d−2` set. Zero
# means index negation reaches every partner of `I`. Type-stable over the heterogeneous axis tuple.
@inline nyquist_mask(ks::Tuple, I::CartesianIndex) = _nqmask(Base.tail(ks), I, 2, 0)
@inline _nqmask(ks::Tuple, I::CartesianIndex, d::Int, acc::Int) =
    _nqmask(Base.tail(ks), I, d + 1, acc | (is_neg_nyquist(first(ks), I[d]) ? (1 << (d - 2)) : 0))
@inline _nqmask(::Tuple{}, ::CartesianIndex, ::Int, acc::Int) = acc

# Is axis `i` collapsed under `mask`? Axis 1 is never a twin axis.
@inline in_nyquist_mask(mask::Int, i::Int) = i >= 2 && ((mask >> (i - 2)) & 1) == 1

# Twin slice count for `D` spectral axes: one per nonempty subset of the axes `2:D`. `Val`-wrapped so
# the slice tuple's length is a compile-time constant.
@inline n_twin_slices(::Val{D}) where {D} = Val((1 << (D - 1)) - 1)

# Index of `−k` on a full native axis of length `n`: `1 ↦ 1`, else `n − i + 2`. At an even axis's `−n/2`
# this maps to itself, which is the aliasing identity of a periodic grid. A reduction holding a computed
# twin reads the twin at those entries.
@inline neg_index(a::FFTAxis, i::Int) = i == 1 ? 1 : a.n - i + 2
@inline neg_index(a::AbstractVector, i::Int) = i == 1 ? 1 : length(a) - i + 2

"""
    NyquistTwin(slices)

Coefficients at the modes a packed half cannot reach by index negation. Negating a mode sends an even
axis sitting at `−N_d/2` to `+N_d/2`, which is off the native axis; the two agree only when the grid
samples periodically. `slices[mask]` covers the modes whose axes-at-`−N_d/2` are exactly `mask` (bit
`d−2` per axis `d ≥ 2`): entry `I` holds `C` at `k₁` kept, `+N_d/2` on the masked axes, and `−k_e`
elsewhere. Each slice collapses its masked axes to extent 1, so the whole object holds a hyperplane's
worth of values. Index it as `t[mask, I]` with `mask = nyquist_mask(ks, I)`.
"""
struct NyquistTwin{S <: Tuple}
    slices::S
end
Base.@propagate_inbounds function Base.getindex(t::NyquistTwin, mask::Int, I::CartesianIndex{N}) where {N}
    return t.slices[mask][CartesianIndex(ntuple(i -> in_nyquist_mask(mask, i) ? 1 : I[i], Val(N)))]
end

"""
    twin_at(t, mask, I, b)

Entry of the `mask` slice at spectral index `I` and batch member `b`. `getindex(t, mask, I)` takes an
index whose rank already spans the batch dims; this form takes a purely spectral `I` of rank `D` and the
batch member separately, for a caller that walks the spectral dims and offsets the batch linearly.
"""
Base.@propagate_inbounds function twin_at(t::NyquistTwin, mask::Int, I::CartesianIndex{D}, b::Int) where {D}
    s = t.slices[mask]
    lin = 1
    stride = 1
    for d in 1:D
        idx = in_nyquist_mask(mask, d) ? 1 : I[d]
        lin += (idx - 1) * stride
        stride *= size(s, d)
    end
    return s[lin + (b - 1) * stride]   # `stride` is now the slice's spectral length
end
Base.show(io::IO, t::NyquistTwin) = print(io, "NyquistTwin(", join(size.(t.slices), ", "), ")")
Base.show(io::IO, ::MIME"text/plain", t::NyquistTwin) = show(io, t)

# Index of integer frequency `kv` on an oversampled axis of length `Ñ` in FFTW order. On a real-input
# axis 1 (`kv ≥ 0`) `Ñ` is the stored half length and this is still `kv + 1`.
@inline ovs_index(kv::Int, Ñ::Int) = kv >= 0 ? kv + 1 : Ñ + kv + 1

# Index into a per-axis coefficient table sampled at the native wavenumbers of an axis of length `N`.
# The tables here are even in `k`, and an even full axis carries that magnitude only at its `−N/2` entry.
@inline function even_table_index(kv::Int, N::Int, r2c::Bool)
    r2c && return kv + 1
    (iseven(N) && abs(kv) == N ÷ 2) && return N ÷ 2 + 1
    return kv >= 0 ? kv + 1 : N + kv + 1
end

# Native (unshifted) fftfreq integer frequencies of a full axis of length N.
fftfreq_ints(N::Int) = Int[j <= (N - 1) ÷ 2 ? j : j - N for j in 0:(N - 1)]

# Per-axis target frequencies of the `mask` twin slice at slice index `I`: `+N_d/2` on the masked axes,
# `k₁` on axis 1, `−k_d` elsewhere.
@inline _twin_freqs(ms::NTuple{D, Int}, mask::Int, kint::Tuple, I::CartesianIndex) where {D} =
    ntuple(d -> in_nyquist_mask(mask, d) ? ms[d] ÷ 2 : (d == 1 ? kint[1][I[d]] : -kint[d][I[d]]), Val(D))

"""
    twin_slice_shape(ms, mask) -> NTuple

Shape of the `mask` slice of a [`NyquistTwin`](@ref): the masked axes collapse to their single `+N_d/2`
mode, and an odd axis has none, which empties the slice.
"""
@inline twin_slice_shape(ms::NTuple{D, Int}, mask::Int) where {D} =
    ntuple(e -> in_nyquist_mask(mask, e) ? (iseven(ms[e]) ? 1 : 0) : packed_size(ms, Val(true))[e], Val(D))

"""
    twin_table(Tr, ms, mask, dims, offsets, ranges, M; phis, normfactor, conjugate) -> (shape, src, fac)

Gather table for the `mask` slice of a [`NyquistTwin`](@ref) read out of a spectrum with array
dimensions `dims`: `twin[i] = S(fk[src[i]]) * fac[i]`, where `S` is `conj` when `conjugate` and the
identity otherwise. Each entry's target frequency is `+N_d/2` on the masked axes, `k₁` on axis 1, and
`−k_d` elsewhere; `fac` carries the grid-offset phase at that target frequency together with `1/M` and,
when `phis` is given, `normfactor / ∏ phis[d]`.

`src` indexes the target frequency itself (`conjugate = false`, for a spectrum that holds `+N_d/2`
outright, such as an oversampled NUFFT grid) or its negative (`conjugate = true`, for a Hermitian
spectrum of a real field, where `fk[-k] = conj(fk[k])` puts the value within a native mode set). Both
indices come from the target frequency, so an axis already sitting at `−N_d/2` still reads its `+N_d/2`
partner. `phis` holds one per-axis coefficient table sampled at the native wavenumbers, even in `k`.
"""
function twin_table(::Type{Tr}, ms::NTuple{D, Int}, mask::Int, dims::NTuple{D, Int},
        offsets::NTuple{D}, ranges::NTuple{D}, M::Int;
        phis::Union{Nothing, Tuple} = nothing, normfactor::Real = 1, conjugate::Bool = false) where {Tr, D}
    shape = twin_slice_shape(ms, mask)
    S = prod(shape)
    kint = ntuple(d -> d == 1 ? collect(0:(ms[1] ÷ 2)) : fftfreq_ints(ms[d]), D)
    src = Vector{Int}(undef, S)
    fac = Vector{Complex{Tr}}(undef, S)
    inv_M = one(Tr) / M
    isign = conjugate ? -1 : 1
    @inbounds for (si, I) in enumerate(CartesianIndices(shape))
        kt = _twin_freqs(ms, mask, kint, I)
        lin = 1
        stride = 1
        β = Tr(normfactor) * inv_M
        ph = one(Complex{Tr})
        for d in 1:D
            lin += (ovs_index(isign * kt[d], dims[d]) - 1) * stride
            stride *= dims[d]
            phis === nothing || (β /= Tr(phis[d][even_table_index(kt[d], ms[d], d == 1)]))
            ph *= cis(-Tr(kt[d]) * (Tr(offsets[d]) * Tr(2π) / Tr(ranges[d])))
        end
        src[si] = lin
        fac[si] = β * ph
    end
    return shape, src, fac
end

"""
    hermitian_request_size(ms) -> NTuple

Mode counts a transform must produce so that every mode of a real field's packed half is present. Axis 1
asks for one extra at even `N₁`, which makes its length odd and so puts both `±N₁/2` on the native axis;
the packed half's axis-1 frequencies `0:N₁÷2` are then the leading `N₁÷2+1` entries for even and odd
`N₁` alike. Negating a packed entry whose axis 1 sits at `+N₁/2` and whose axis `d` sits at `−N_d/2`
lands on `+N_d/2`, off-axis, so a conjugate read cannot substitute.
"""
@inline hermitian_request_size(ms::NTuple{D, Int}) where {D} =
    ntuple(d -> d == 1 && iseven(ms[1]) ? ms[1] + 1 : ms[d], Val(D))

"""
    publish_packed!(coeffs, fk, phase, ns, pms, ntrans, neg) -> coeffs

Write a real field's packed half from the full native spectrum `fk` of requested size `ns` (see
[`hermitian_request_size`](@ref)). The half is the leading `pms[1]` entries of axis 1, so each axis-1 run
copies contiguously out of the longer `ns[1]` run and the axes above it match one for one. `phase` is the
packed-layout offset correction; `neg` conjugates the published values.

This is the HOST write: a linear scalar sweep, allocation-free on a host array. A device path takes
[`packed_half_view`](@ref) and broadcasts, which reads no element from the host.
"""
function publish_packed!(coeffs, fk, phase, ns::NTuple{D, Int}, pms::NTuple{D, Int}, ntrans::Int,
        neg::Bool) where {D}
    n1s = ns[1]
    n1p = pms[1]
    Ph = prod(pms)
    inner = Ph ÷ n1p                                 # entries above axis 1, shared by both layouts
    Pm = prod(ns)
    @inbounds for t in 1:ntrans
        foff = (t - 1) * Pm
        coff = (t - 1) * Ph
        for j in 0:(inner - 1)
            soff = foff + j * n1s
            poff = j * n1p
            for i in 1:n1p
                v = fk[soff + i] * phase[poff + i]
                coeffs[coff + poff + i] = neg ? conj(v) : v
            end
        end
    end
    return coeffs
end

"""
    packed_half_view(fk, ns::NTuple{D,Int}, pms::NTuple{D,Int}, ntrans) -> SubArray

The packed half of a full native spectrum as a lazy view: the leading `pms[1]` entries of axis 1 with
every other axis whole, batch axis last.

A device path publishes with `out .= view .* phase` (or its conjugate); [`publish_packed!`](@ref) is the
host form of the same write.
"""
@inline packed_half_view(fk, ns::NTuple{D, Int}, pms::NTuple{D, Int}, ntrans::Int) where {D} =
    view(reshape(fk, ns..., ntrans), 1:pms[1], ntuple(_ -> Colon(), Val(D - 1))..., Colon())

"""
    ConjTwinSlice(slice, src, fac)

One mask's [`NyquistTwin`](@ref) slice together with the table that fills it by conjugate reads of a
Hermitian native spectrum (see [`twin_table`](@ref) with `conjugate = true`).
"""
struct ConjTwinSlice{SL, IX, FC}
    slice::SL
    src::IX
    fac::FC
end

"""
    conj_twins(Tr, ks_phys, ms, ns, offsets, ranges, M, batch) -> (ks, slices)

`ks_phys` with a [`NyquistTwin`](@ref) attached to its halved axis, plus the per-mask
[`ConjTwinSlice`](@ref)s that fill it from a Hermitian native spectrum of requested size `ns`. A real
field's transform is Hermitian, so each twin value is a conjugate read at the negated frequency, which
that spectrum holds.
"""
function conj_twins(::Type{Tr}, ks_phys::Tuple, ms::NTuple{D, Int}, ns::NTuple{D, Int},
        offsets::NTuple{D}, ranges::NTuple{D}, M::Int, batch::Tuple) where {Tr, D}
    D >= 2 || return ks_phys, ()
    gs = ntuple(n_twin_slices(Val(D))) do mask
        shape, src, fac = twin_table(Tr, ms, mask, ns, offsets, ranges, M; conjugate = true)
        ConjTwinSlice(zeros(Complex{Tr}, shape..., batch...), src, fac)
    end
    return (with_twin(ks_phys[1], NyquistTwin(map(g -> g.slice, gs))), Base.tail(ks_phys)...), gs
end

"""
    gather_conj_twins!(slices, fk, Pm, ntrans, neg)

Refill every [`ConjTwinSlice`](@ref) from the native spectrum `fk`, whose transforms are `Pm` modes
apart. Entry `i` of transform `t` is `conj(fk[(t-1)Pm + src[i]]) * fac[i]`, conjugated again where `neg`.
"""
@inline gather_conj_twins!(::Tuple{}, fk, Pm::Int, ntrans::Int, neg::Bool) = nothing
@inline function gather_conj_twins!(gs::Tuple, fk, Pm::Int, ntrans::Int, neg::Bool)
    _gather_one_conj!(first(gs), fk, Pm, ntrans, neg)
    return gather_conj_twins!(Base.tail(gs), fk, Pm, ntrans, neg)
end

function _gather_one_conj!(g::ConjTwinSlice, fk, Pm::Int, ntrans::Int, neg::Bool)
    S = length(g.src)
    S == 0 && return nothing
    @inbounds for t in 1:ntrans
        foff = (t - 1) * Pm
        soff = (t - 1) * S
        for s in 1:S
            v = conj(fk[foff + g.src[s]]) * g.fac[s]
            g.slice[soff + s] = neg ? conj(v) : v
        end
    end
    return nothing
end

# =============================================================================
# Per-axis (separable) transform addressing. A tensor-product grid transforms one axis at a time, so a
# provider needs each length-`N_d` line of the working array as a contiguous strength vector, and writes
# back a line of length `m`. Treating the array as `(pre, N_d, post)` reaches line `r` by a stride-`pre`
# walk, so no permuted copy of the array is needed; `C` consecutive lines are handled per chunk with the
# stride loop OUTERMOST, so each stride step touches one contiguous run of `C` entries and the array is
# swept once per pass.
# =============================================================================

"""
    axis_layout(sz, d) -> (pre, N_d, post)

`sz` viewed as `(pre, N_d, post)` for a transform along axis `d`: `pre` is the product of the axes below
`d`, `post` the product of those above. The lines to transform number `pre · post`, and line `r` starts at
`mod1(r, pre) + ((r - 1) ÷ pre) · pre · N_d`.
"""
@inline function axis_layout(sz::NTuple{N, Int}, d::Int) where {N}
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

# Shape of the result of transforming axis `d` to `m` modes.
@inline axis_out_size(sz::NTuple{N, Int}, d::Int, m::Int) where {N} =
    ntuple(i -> i == d ? m : sz[i], Val(N))

"""
    axis_work_shape(Ns, ns, d, batch) -> NTuple

Shape a separable transform's working array carries entering pass `d`: the axes below `d` already hold
their `ns` modes, the rest still hold their `Ns` grid points, and the batch rides above. `d = 1` is the
field's own shape and `d = D+1` the finished spectrum's.
"""
@inline axis_work_shape(Ns::NTuple{D, Int}, ns::NTuple{D, Int}, d::Int, batch::Tuple) where {D} =
    (ntuple(i -> i < d ? ns[i] : Ns[i], Val(D))..., batch...)

"""
    axis_chunk_offsets!(inoff, outoff, base, nvalid, pre, Nd, m)

Starting linear indices of lines `base+1 … base+nvalid` in the input (`Nd` per line) and in the output
(`m` per line). Lines sharing a slab get consecutive offsets, so the gather reads contiguously.
"""
@inline function axis_chunk_offsets!(inoff::Vector{Int}, outoff::Vector{Int}, base::Int, nvalid::Int,
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
    gather_axis_block!(cjs, A, inoff, nvalid, pre, Nd)

Fill `cjs[t][i] = A[inoff[t] + (i-1)·pre]` for `t ≤ nvalid`, zeroing the strength vectors past `nvalid`
so a partial final chunk still presents the plan's full transform count.
"""
function gather_axis_block!(cjs::Tuple, A, inoff::Vector{Int}, nvalid::Int, pre::Int, Nd::Int)
    @inbounds for i in 1:Nd
        s = (i - 1) * pre
        for t in 1:nvalid
            cjs[t][i] = A[inoff[t] + s]
        end
    end
    @inbounds for t in (nvalid + 1):length(cjs)
        fill!(cjs[t], zero(eltype(cjs[t])))
    end
    return nothing
end

"""
    scatter_axis_block!(out, fks, outoff, nvalid, pre, m)

Write `out[outoff[t] + (j-1)·pre] = fks[t][j]` for `t ≤ nvalid`, the inverse of
[`gather_axis_block!`](@ref) on the transformed axis length `m`.
"""
function scatter_axis_block!(out, fks::Tuple, outoff::Vector{Int}, nvalid::Int, pre::Int, m::Int)
    @inbounds for j in 1:m
        s = (j - 1) * pre
        for t in 1:nvalid
            out[outoff[t] + s] = fks[t][j]
        end
    end
    return nothing
end

"""
    offset_phase(Tr, ms, offsets, ranges, M, ::Val{R}, iflag=1) -> Array

Grid-offset correction `exp(-iflag·i·k·x₀)/M` over the packed half (`R = true`) or the full native
spectrum (`R = false`), for points scaled from a domain starting at `offsets` with period `ranges`.
"""
function offset_phase(::Type{Tr}, ms::NTuple{D, Int}, offsets::NTuple{D}, ranges::NTuple{D}, M::Int,
        ::Val{R}, iflag::Int = 1) where {Tr, D, R}
    sz = packed_size(ms, Val(R))
    kint = ntuple(d -> (d == 1 && R) ? collect(0:(ms[1] ÷ 2)) : fftfreq_ints(ms[d]), D)
    inv_M = one(Tr) / M
    phase = Array{Complex{Tr}, D}(undef, sz...)
    @inbounds for I in CartesianIndices(sz)
        p = Complex{Tr}(inv_M)
        for d in 1:D
            p *= cis(-iflag * Tr(kint[d][I[d]]) * (Tr(offsets[d]) * Tr(2π) / Tr(ranges[d])))
        end
        phase[I] = p
    end
    return phase
end

# Energy fold multiplier for a mode: the product of the per-axis `fold_weight` over the halved (rfft)
# spectral axes, and 1 over full axes. `ks` is the tuple of spectral wavenumber axes; a `CartesianIndex`
# that also carries batch dims is fine (only its leading spectral entries are read). Type-stable over a
# heterogeneous `ks` (mixed `RFFTAxis`/`FFTAxis`/range) via tuple recursion. On a full centered/native
# spectrum (no `RFFTAxis`) this is one, so a reduction over it is unchanged.
@inline mode_fold(ks::Tuple, I::CartesianIndex) = _fold_prod(eltype(first(ks)), ks, I, 1)
@inline _fold_prod(::Type{FT}, ks::Tuple, I::CartesianIndex, d::Int) where {FT} =
    (is_halved(first(ks)) ? fold_weight(first(ks), I[d]) : one(FT)) * _fold_prod(FT, Base.tail(ks), I, d + 1)
@inline _fold_prod(::Type{FT}, ::Tuple{}, ::CartesianIndex, ::Int) where {FT} = one(FT)

# Squared isotropic wavenumber `Σ_d k_d(I_d)²` and the max resolved wavenumber `min_d max|k_d|`, folded
# over the heterogeneous axis tuple by recursion so `ks[d]` with a runtime `d` never boxes.
@inline ksq(::Type{T}, ks::Tuple, I::CartesianIndex) where {T} = ksq(T, ks, I, 1)
@inline function ksq(::Type{T}, ks::Tuple, I::CartesianIndex, d::Int) where {T}
    kv = T(first(ks)[I[d]])
    return kv * kv + ksq(T, Base.tail(ks), I, d + 1)
end
@inline ksq(::Type{T}, ::Tuple{}, ::CartesianIndex, ::Int) where {T} = zero(T)

@inline function _axis_absmax(::Type{T}, ax) where {T}
    m = zero(T)
    @inbounds for v in ax
        a = abs(T(v))
        m = ifelse(a > m, a, m)
    end
    return m
end
@inline kmax(::Type{T}, ks::Tuple) where {T} = min(_axis_absmax(T, first(ks)), kmax(T, Base.tail(ks)))
@inline kmax(::Type{T}, ::Tuple{}) where {T} = T(Inf)

# Product of the wavenumber spacings of the axes in `dims`, folded over the heterogeneous axis tuple by
# recursion so `ks[d]` with a runtime `d` never boxes.
@inline dk_product(::Type{T}, ks::Tuple, dims::Tuple) where {T} = _dkp(T, ks, dims, 1)
@inline _dkp(::Type{T}, ks::Tuple, dims::Tuple, d::Int) where {T} =
    (d in dims ? T(first(ks)[2] - first(ks)[1]) : one(T)) * _dkp(T, Base.tail(ks), dims, d + 1)
@inline _dkp(::Type{T}, ::Tuple{}, ::Tuple, ::Int) where {T} = one(T)

"""
    unpacked!(full, coeffs, ns::NTuple{D,Int}, ks = nothing) -> full

Fill `full` (`ns…, batch…`) with the complete native-order spectrum of the rfft-packed half `coeffs`
(`ns[1]÷2+1, ns[2:D]…, batch…`), by Hermitian symmetry over the spectral dims `1:D`.

Each mode is either stored outright or the conjugate of its negated-frequency partner, whose per-axis
index is `j ↦ 1` for `j == 1` and `n-j+2` otherwise. Where an even axis `d ≥ 2` sits at `−ns[d]/2` that
negation lands off the native axis, so the partner comes from the halved axis's [`NyquistTwin`](@ref),
carried on `ks`; on a nonuniformly sampled grid `ks` is required for an exact result. `full` and `coeffs`
share rank, and the batch dims ride along.
"""
function unpacked!(full::AbstractArray{Complex{T}, NT}, coeffs::AbstractArray{Complex{T}, NT},
        ns::NTuple{D, Int}, ks = nothing) where {T, D, NT}
    # Native axis 1 holds `k₁ ≥ 0` only up to `(n₁−1)÷2`. At even `n₁` the entry one past that is
    # `−n₁/2`, which the conjugate branch fills from the stored `+n₁/2`.
    h1 = (ns[1] - 1) ÷ 2 + 1
    twin = ks === nothing ? nothing : axis_twin(ks[1])
    @inbounds for I in CartesianIndices(full)
        if I[1] <= h1
            full[I] = coeffs[I]
            continue
        end
        # `conj(C[−k])` needs `−k_d`, which leaves the native axis where an even axis `d ≥ 2` sits at
        # `−n_d/2`; the halved axis's twin holds that value.
        q = twin === nothing ? 0 : nyquist_mask(ks, I)
        if q != 0
            src1 = CartesianIndex(ntuple(d -> d == 1 ? (ns[1] - I[1] + 2) : I[d], Val(NT)))
            full[I] = conj(twin[q, src1])
        else
            src = CartesianIndex(ntuple(
                d -> d == 1 ? (ns[1] - I[1] + 2) : (d <= D ? (I[d] == 1 ? 1 : ns[d] - I[d] + 2) : I[d]),
                Val(NT)))
            full[I] = conj(coeffs[src])
        end
    end
    return full
end

"""
    unpacked(coeffs, ns::NTuple{D,Int}, ks = nothing) -> full

Full native-order complex spectrum `(ns…, batch…)` reconstructed from an rfft-packed half `coeffs`
`(ns[1]÷2+1, ns[2:D]…, batch…)` via `C[k] = conj(C[-k])`. Passing the `ks` that `calculate_spectrum`
returned supplies the halved axis's [`NyquistTwin`](@ref), which the `k₁ < 0` rows need wherever an even
axis `d ≥ 2` sits at `−n_d/2`: negating that index lands on `+n_d/2`, off the native axis. With `ks`
omitted, those entries take the negated-index value, which matches the twin under uniform sampling.
"""
function unpacked(coeffs::AbstractArray{Complex{T}, NT}, ns::NTuple{D, Int}, ks = nothing) where {T, D, NT}
    batch = ntuple(i -> size(coeffs, D + i), NT - D)
    full = Array{Complex{T}, NT}(undef, ns..., batch...)
    return unpacked!(full, coeffs, ns, ks)
end

end # module Packing
