module DirectSum

using FlowGeometries: FlowGeometries
using LinearAlgebra: LinearAlgebra as LA
using ..Grids: Grids
using ..Packing: Packing
using ..SphericalKernels: SphericalKernels

export sph_mode_index

# The dependency-free transform selected by `DirectSumSpectralBackend()`.
# It reads a field tensor `(spatial…, batch…)` and writes coefficients in
# the packed layout `(spectral…, batch…)` (a real Cartesian field halves axis 1 to `k₁ ≥ 0`). A
# structured tensor-product grid factorizes into `D` successive 1-D DFTs (`O(D · n^{D+1})`, one dense
# matrix per axis, BLAS-backed), uniform or nonuniform. A scattered point cloud does not factorize, so
# it is the `O(∏ms · Npts)` direct sum, still computing only the packed half for a real field. Structured
# grids read coordinates through their 1-D axes (no per-point coordinate blob); unstructured grids read
# their per-node vectors.

"""
    sph_mode_index(l, m) -> CartesianIndex

`CartesianIndex` of degree `l`, order `m` in the `(lmax+1, 2lmax+1)` coefficient array.
"""
@inline function sph_mode_index(l::Int, m::Int)
    row = l - abs(m) + 1
    col = m == 0 ? 1 : (m < 0 ? 2 * abs(m) : 2 * m + 1)
    return CartesianIndex(row, col)
end

# Phase argument Σ_d ks[d][K[d]] · x_d, Val-unrolled over the type-parameter dim count so each tuple
# index is a compile-time constant. This stays type-stable and allocation-free even when `@inbounds`
# is disabled (as under `--check-bounds=yes`, how `Pkg.test` runs) — a plain `for d in 1:D` loop
# indexing the heterogeneous `ks`/`axes`/`coords` tuples at a runtime `d` can otherwise box.
@inline _phase_tensor(ks::Tuple, axes::Tuple, K::CartesianIndex{D}, P::CartesianIndex{D}, ::Type{FT}) where {D, FT} =
    sum(ntuple(d -> FT(ks[d][K[d]]) * FT(axes[d][P[d]]), Val(D)))
@inline _phase_scattered(ks::Tuple, coords::Tuple, K::CartesianIndex{D}, j::Int, ::Type{FT}) where {D, FT} =
    sum(ntuple(d -> FT(ks[d][K[d]]) * FT(coords[d][j]), Val(D)))

# One axis's dense 1-D DFT matrix `Wₐₚ = exp(−iflag·i·k_a·x_p)`, `(m × n)` with `m` the packed mode count
# along this axis and `n` the point count. Uniform or nonuniform `xaxis`.
function _dft_matrix(::Type{FT}, kaxis, xaxis, m::Int, n::Int, iflag::Int) where {FT}
    W = Matrix{Complex{FT}}(undef, m, n)
    @inbounds for p in 1:n
        xp = FT(xaxis[p])
        for a in 1:m
            W[a, p] = cis(-iflag * FT(kaxis[a]) * xp)
        end
    end
    return W
end

"""
    AxisPass{FT, ND, M, A}

One axis's contraction: its dense DFT matrix, the two matrix aliases the matmul reads and writes, and —
for an axis past the first — the staging buffers and the permutation bringing it to the front and back.

Every alias and permutation is built once, so a pass reshapes nothing and inverts no permutation per
call. Run it with `_axis_pass!`.
"""
struct AxisPass{FT, ND, M <: AbstractMatrix{Complex{FT}}, A <: AbstractArray{Complex{FT}, ND}}
    W::M
    src::M                        # `(n, rest)` view of what the matmul reads
    dst::M                        # `(m, rest)` view of what it writes
    pbuf::A                       # front-permuted input; empty on axis 1
    rbuf::A                       # its matmul result; empty on axis 1
    perm::NTuple{ND, Int}
    iperm::NTuple{ND, Int}
    permute::Bool
end

# `dest` at axis `a`'s own position, from `A` entering the pass.
@inline function _axis_pass!(dest::AbstractArray, a::AxisPass, A::AbstractArray)
    if a.permute
        permutedims!(a.pbuf, A, a.perm)
        LA.mul!(a.dst, a.W, a.src)
        permutedims!(dest, a.rbuf, a.iperm)
    else
        LA.mul!(a.dst, a.W, a.src)
    end
    return dest
end

# =============================================================================
# Cartesian setup / run split. `cart_setup` reads the grid — its per-axis DFT matrices, the contraction
# chain's working arrays, the wavenumber axes and the Nyquist-twin storage — and `cart_run!` transforms a
# field against what it returns. The one-shot composes the two and `DirectSumCartesianPlan` holds the
# setup, so the two paths cannot drift and a repeated caller rebuilds none of it: the matrices depend only
# on the grid, `ms` and `iflag`, and every buffer only on the shapes. The field arrives already scaled by
# the grid's quadrature factor, as it does on every other entry.
#
# `TensorCartesian` / `CloudCartesian` name the two algorithms, mirroring the spherical layout markers.
# =============================================================================

struct TensorCartesian end
struct CloudCartesian end

_cart_layout(::FlowGeometries.Grids.AbstractStructuredGrid{<:FlowGeometries.Geometry.AbstractCartesianGeometry}) =
    TensorCartesian()
_cart_layout(::Grids.PointwiseCartesian) = CloudCartesian()

# Shapes a contraction chain passes through: `chain[d]` enters axis `d` and `chain[D+1]` is the finished
# spectrum, axis `e < d` already at `outs[e]` and `e ≥ d` still at `ss[e]`.
_chain_shapes(ss::NTuple{D, Int}, outs::NTuple{D, Int}, batch::Tuple) where {D} =
    ntuple(d -> (ntuple(e -> e < d ? outs[e] : ss[e], Val(D))..., batch...), Val(D + 1))

# The permutation bringing axis `d` of a rank-`nd` array to the front.
_perm_front(nd::Int, d::Int) = (d, ntuple(i -> i < d ? i : i + 1, nd - 1)...)

# One axis-by-axis contraction from point counts `ss` to mode counts `outs`, the batch dims riding along:
# the per-axis dense DFT matrices and every working array it walks through. Shared by the forward and by
# the structured Nyquist twin, which is the same contraction on the masked wavenumbers.
function _contraction(::Type{FT}, kaxes::Tuple, axes::Tuple, ss::NTuple{D, Int}, outs::NTuple{D, Int},
        batch::Tuple, iflag::Int) where {FT, D}
    shapes = _chain_shapes(ss, outs, batch)
    nd = D + length(batch)
    chain = ntuple(d -> Array{Complex{FT}}(undef, shapes[d]), Val(D + 1))
    empty = Array{Complex{FT}}(undef, ntuple(_ -> 0, nd))    # axis 1 contracts in place, needing neither
    ident = ntuple(identity, nd)
    passes = ntuple(Val(D)) do d
        W = _dft_matrix(FT, kaxes[d], axes[d], outs[d], ss[d], iflag)
        if d == 1
            return AxisPass{FT, nd, Matrix{Complex{FT}}, Array{Complex{FT}, nd}}(
                W, _mat(chain[1], ss[1]), _mat(chain[2], outs[1]), empty, empty, ident, ident, false)
        end
        perm = _perm_front(nd, d)
        pbuf = Array{Complex{FT}}(undef, ntuple(i -> shapes[d][perm[i]], nd))
        rbuf = Array{Complex{FT}}(undef, (outs[d], Base.tail(size(pbuf))...))
        return AxisPass{FT, nd, Matrix{Complex{FT}}, Array{Complex{FT}, nd}}(
            W, _mat(pbuf, ss[d]), _mat(rbuf, outs[d]), pbuf, rbuf, perm,
            NTuple{nd, Int}(invperm(collect(perm))), true)
    end
    return (; passes, chain)
end

# `(n, length(a) ÷ n)` matrix alias of `a`, sharing its memory. An empty leading extent leaves an empty
# alias, which a mask touching an odd axis produces and `_run_contraction!` skips.
_mat(a::Array{T}, n::Int) where {T} = n == 0 ? reshape(a, 0, 0) : reshape(a, n, length(a) ÷ n)

# `dest = (1/npts) · W_D ⋯ W_1 · field`, through the contraction's own buffers.
function _run_contraction!(dest::AbstractArray, c, field::AbstractArray, npts::Int)
    isempty(dest) && return dest       # a mask touching an odd axis has no `+N_d/2` mode to compute
    D = length(c.passes)
    copyto!(c.chain[1], field)
    for d in 1:D
        _axis_pass!(c.chain[d + 1], c.passes[d], c.chain[d])
    end
    copyto!(dest, c.chain[D + 1])
    dest ./= npts
    return dest
end

"""
    cart_setup(layout, grid, T, ms, batch, iflag) -> NamedTuple

Everything a Cartesian direct sum reads from `grid`: the wavenumber axes, and for a tensor layout the
per-axis DFT matrices and the working arrays their contraction walks through, plus the Nyquist-twin
storage. `T` is the field's element type, whose realness fixes the packed layout.

Run it against a field with [`cart_run!`](@ref). The result depends on the grid, `ms`, `iflag` and the
batch shape alone, so `DirectSumCartesianPlan` holds it across executions.
"""
function cart_setup(::TensorCartesian, g, ::Type{T}, ms::NTuple{D, Int}, batch::Tuple,
        iflag::Int) where {T, D}
    FT = real(float(T))
    R = T <: Real
    ss = size(g)
    ks = Grids.physical_wavenumbers(g, ms, Val(R))
    pms = Packing.packed_size(ms, Val(R))
    return (; ks, pms, Npts = prod(ss), iflag,
        fwd = _contraction(FT, ks, FlowGeometries.Grids.coordinates(g), ss, pms, batch, iflag),
        twin = _twin_setup(TensorCartesian(), g, T, ms, pms, ks, batch, iflag, FT))
end

function cart_setup(::CloudCartesian, g, ::Type{T}, ms::NTuple{D, Int}, batch::Tuple,
        iflag::Int) where {T, D}
    FT = real(float(T))
    R = T <: Real
    coords = FlowGeometries.Grids.coordinates(g)
    ks = Grids.physical_wavenumbers(g, ms, Val(R))
    pms = Packing.packed_size(ms, Val(R))
    return (; ks, pms, coords, N = length(coords[1]), iflag,
        twin = _twin_setup(CloudCartesian(), g, T, ms, pms, ks, batch, iflag, FT))
end

"""
    cart_run!(coeffs, layout, setup, field) -> ks

Fill `coeffs` with the Cartesian spectrum of `field` against a [`cart_setup`](@ref) result, and return the
wavenumber axes, the halved axis carrying a Nyquist twin where one is required.

`field` arrives already scaled by the grid's quadrature factor. A tensor layout contracts axis by axis, a
point cloud sums directly, and both write only into `coeffs` and the setup's own buffers.
"""
function cart_run!(coeffs::AbstractArray{Complex{FT}}, l::TensorCartesian, s,
        field::AbstractArray) where {FT}
    _run_contraction!(coeffs, s.fwd, field, s.Npts)
    return _attach_twin(s.twin, l, s, field)
end

function cart_run!(coeffs::AbstractArray{Complex{FT}}, l::CloudCartesian, s,
        field::AbstractArray) where {FT}
    _cloud_sum!(coeffs, s.ks, s.coords, s.pms, s.N, s.iflag, field, FT)
    return _attach_twin(s.twin, l, s, field)
end

# `C[k, b] = (1/N) Σ_j f[j, b] · exp(-iflag · i · k·x_j)` over a mode set `pms` with wavenumber axes
# `kaxes`. The forward passes the packed mode set; the twin passes its masked one.
function _cloud_sum!(dest::AbstractArray{Complex{FT}}, kaxes::Tuple, coords::Tuple,
        pms::NTuple{D, Int}, N::Int, iflag::Int, field::AbstractArray, ::Type{FT}) where {FT, D}
    M = prod(pms)
    B = M == 0 ? 0 : length(dest) ÷ M
    fill!(dest, zero(Complex{FT}))
    @inbounds for (mi, I) in enumerate(CartesianIndices(pms))
        for j in 1:N
            phi = _phase_scattered(kaxes, coords, I, j, FT)
            W = cis(-iflag * phi)
            for b in 1:B
                dest[mi + (b - 1) * M] += field[j + (b - 1) * N] * W
            end
        end
    end
    B == 0 || (dest ./= N)
    return dest
end

# =============================================================================
# Cartesian forward (analysis):  C[k, b] = (1/Npts) Σ_p f[p, b] · exp(-iflag · i · k·x_p)
# =============================================================================

# A structured tensor-product grid (uniform or nonuniform) separates into `D` successive 1-D transforms,
# one dense matrix per axis; a point cloud does not separate, so the direct sum stands. A real field
# halves axis 1 to `k₁ ≥ 0` either way and the batch dims ride along. `ms` may differ from the grid size
# (the per-axis matrices are rectangular `pms[d] × ss[d]`).
#
# One-shot: set up against this grid, then run. A repeated caller holds the setup in a
# `DirectSumCartesianPlan`.
function _calculate_spectrum_cartesian_direct!(
    coeffs::AbstractArray{Complex{FT}},
    g::FlowGeometries.Grids.AbstractGrid{<:FlowGeometries.Geometry.AbstractCartesianGeometry},
    field::AbstractArray,
    ms::NTuple{D, Int},
    iflag::Int,
) where {FT, D}
    layout = _cart_layout(g)
    T = eltype(field) <: Real ? FT : Complex{FT}
    batch = Grids.field_batch_shape(g, field)
    return cart_run!(coeffs, layout, cart_setup(layout, g, T, ms, batch, iflag), field)
end

# =============================================================================
# Nyquist twins: the coefficients at `+N_d/2` that a packed half's index negation misses (it sends an
# even axis's `−N_d/2` back to itself). `transect_spectrum` integrates the halved axis with no radial
# cutoff, so it reaches those modes and reads the twin for the `k₁ < 0` rows. One slice per nonempty
# subset of the axes `2:D`, each with its subset's axes collapsed, so the whole object costs a
# hyperplane.
# =============================================================================

# Shape of the `mask` slice: masked axes collapse to their single `+N_d/2` mode, and an odd axis has
# none, which empties that slice.
@inline _twin_shape(ms::NTuple{D, Int}, pms::NTuple{D, Int}, mask::Int) where {D} =
    ntuple(e -> Packing.in_nyquist_mask(mask, e) ? (iseven(ms[e]) ? 1 : 0) : pms[e], Val(D))

# Axis `d`'s wavenumbers under `mask`, length `m`: the single `+N_d/2` when masked, `k₁` as stored on
# axis 1, and `−k_d` elsewhere. The stored `−N_d/2` sits at index `n÷2+1`, so its negative is `+N_d/2`.
# The axis is materialized once per `(mask, d)`: reading `ks[d]` per MODE instead boxes the mixed
# `RFFTAxis`/`FFTAxis` tuple at a runtime `d`, which costs more than the vector it saves.
function _twin_kaxis(::Type{FT}, ks::Tuple, ms::NTuple{D, Int}, mask::Int, d::Int, m::Int) where {FT, D}
    v = Vector{FT}(undef, m)
    kd = ks[d]
    if Packing.in_nyquist_mask(mask, d)
        m == 1 && (v[1] = -FT(kd[ms[d] ÷ 2 + 1]))
    else
        s = d == 1 ? one(FT) : -one(FT)
        @inbounds for a in 1:m
            v[a] = s * FT(kd[a])
        end
    end
    return v
end

# Twin storage, built once: one slice per mask, that mask's wavenumber axes, the `NyquistTwin` wrapper,
# the `ks` tuple carrying it, and — on a structured grid — the masked contraction that fills it. A run
# then writes into the slices and hands back the same objects.
#
# `nothing` where no twin is required: a complex field's full native spectrum stores every partner
# outright, `D == 1`'s twins are conjugates so the mirror is already exact, and uniform sampling makes
# `exp(±i(N_d/2)x_j)` the same vector, so a uniform grid's twins equal their negated-index partners.
# `isuniform` reads the axis types, so that branch resolves at compile time.
_twin_setup(l, g, ::Type{T}, ms, pms, ks, batch, iflag::Int, ::Type{FT}) where {T, FT} = nothing
function _twin_setup(l, g::FlowGeometries.Grids.AbstractGrid{<:FlowGeometries.Geometry.AbstractCartesianGeometry},
        ::Type{T}, ms::NTuple{D, Int}, pms::NTuple{D, Int}, ks::Tuple, batch::Tuple, iflag::Int,
        ::Type{FT}) where {T <: Real, FT, D}
    (D >= 2 && !FlowGeometries.Grids.isuniform(g)) || return nothing
    nm = Packing.n_twin_slices(Val(D))
    shapes = ntuple(mask -> _twin_shape(ms, pms, mask), nm)
    kax = ntuple(mask -> ntuple(d -> _twin_kaxis(FT, ks, ms, mask, d, shapes[mask][d]), Val(D)), nm)
    slices = ntuple(mask -> zeros(Complex{FT}, shapes[mask]..., batch...), nm)
    tw = Packing.NyquistTwin(slices)
    return (; slices, kax, shapes,
        ks = (Packing.with_twin(ks[1], tw), Base.tail(ks)...),
        contractions = _twin_contractions(l, g, shapes, kax, batch, iflag, FT))
end

# The structured twin separates the same way the forward does, on the masked wavenumbers. A point cloud
# sums directly and needs no contraction.
_twin_contractions(::CloudCartesian, g, shapes, kax, batch, iflag::Int, ::Type{FT}) where {FT} = nothing
_twin_contractions(::TensorCartesian, g, shapes::NTuple{NM}, kax, batch, iflag::Int,
        ::Type{FT}) where {NM, FT} =
    ntuple(mask -> _contraction(FT, kax[mask], FlowGeometries.Grids.coordinates(g), size(g),
        shapes[mask], batch, iflag), NM)

# Fill the twin from this field and hand back the `ks` carrying it; with no twin required, `ks` as built.
@inline _attach_twin(::Nothing, l, s, field) = s.ks
function _attach_twin(t, l, s, field)
    _fill_twin!(t, l, s, field)
    return t.ks
end

function _fill_twin!(t, ::TensorCartesian, s, field::AbstractArray)
    for mask in eachindex(t.slices)
        _run_contraction!(t.slices[mask], t.contractions[mask], field, s.Npts)
    end
    return t
end

function _fill_twin!(t, ::CloudCartesian, s, field::AbstractArray)
    FT = real(eltype(t.slices[1]))
    for mask in eachindex(t.slices)
        _cloud_sum!(t.slices[mask], t.kax[mask], s.coords, t.shapes[mask], s.N, s.iflag, field, FT)
    end
    return t
end

# =============================================================================
# Cartesian inverse (synthesis) from the packed half, writing a REAL field directly.
#
# The native sum is `f[p] = Σ_{κ native} C_full[κ] · exp(+iflag·i·κ·x_p)`, and the packed half stores
# `k₁ ∈ [0, N₁/2]` against every native `k_·`. Two passes over the store cover the native set exactly:
#
#   pass 1 — native `κ₁ ≥ 0` rows, which the store holds outright. At even `N₁` the stored `+N₁/2` is
#            not one of them (native carries `−N₁/2` at that magnitude), so that row is skipped here.
#   pass 2 — native `κ₁ = −k₁(I) < 0` rows at the same `κ_· = k_·(I)`, whose value is
#            `conj(C[k₁(I), −κ_·])`. Index negation supplies `−κ_·` except where an even axis `d ≥ 2`
#            sits at `−N_d/2`, since `+N_d/2` is off the native axis; the axis's `NyquistTwin` holds
#            exactly that value. This is the same two-pass structure `transect_spectrum!` uses, for the
#            same reason.
#
# `out` is real, so each term contributes its real part; the imaginary parts cancel over the native set.
# One sweep of the stored half replaces a full-cube inverse, with no unpacked buffer.
# =============================================================================

# Phase of native mode `(−k₁(I), k_·(I))` at a tensor-grid point / a scattered point: axis 1 negated,
# the rest as stored.
@inline _phase_tensor_neg(ks::Tuple, axes::Tuple, K::CartesianIndex{D}, P::CartesianIndex{D}, ::Type{FT}) where {D, FT} =
    sum(ntuple(d -> (d == 1 ? -FT(ks[1][K[1]]) : FT(ks[d][K[d]])) * FT(axes[d][P[d]]), Val(D)))
@inline _phase_scattered_neg(ks::Tuple, coords::Tuple, K::CartesianIndex{D}, j::Int, ::Type{FT}) where {D, FT} =
    sum(ntuple(d -> (d == 1 ? -FT(ks[1][K[1]]) : FT(ks[d][K[d]])) * FT(coords[d][j]), Val(D)))

# Value of native mode `(−k₁(I), k_·(I))`: `conj` of the store at `(k₁(I), −k_·)`, read from the twin
# where index negation lands off the native axis.
@inline function _neg_row_value(coeffs, ks::Tuple, twin, I::CartesianIndex{D}, off::Int,
        pms::NTuple{D, Int}, b::Int) where {D}
    q = twin === nothing ? 0 : Packing.nyquist_mask(ks, I)
    q != 0 && return conj(Packing.twin_at(twin, q, I, b))
    lin = 1
    stride = 1
    @inbounds for d in 1:D
        idx = d == 1 ? I[1] : Packing.neg_index(ks[d], I[d])
        lin += (idx - 1) * stride
        stride *= pms[d]
    end
    return conj(coeffs[lin + off])
end

function _synthesize_packed_direct!(
    out::AbstractArray{FT},
    g::FlowGeometries.Grids.AbstractStructuredGrid{<:FlowGeometries.Geometry.AbstractCartesianGeometry},
    coeffs::AbstractArray, ms::NTuple{D, Int}, iflag::Int, twin,
) where {FT, D}
    axes = FlowGeometries.Grids.coordinates(g)
    ss = size(g)
    Npts = prod(ss)
    ks = Grids.physical_wavenumbers(g, ms, Val(true))
    pms = Packing.packed_size(ms, Val(true))
    M = prod(pms)
    B = length(out) ÷ Npts
    fill!(out, zero(FT))
    @inbounds for (pj, P) in enumerate(CartesianIndices(ss))
        for (mi, I) in enumerate(CartesianIndices(pms))
            nyq = Packing.is_nyquist(ks[1], I[1])
            W = nyq ? zero(Complex{FT}) : cis(iflag * _phase_tensor(ks, axes, I, P, FT))
            Wn = I[1] > 1 ? cis(iflag * _phase_tensor_neg(ks, axes, I, P, FT)) : zero(Complex{FT})
            for b in 1:B
                off = (b - 1) * M
                acc = nyq ? zero(FT) : real(coeffs[mi + off] * W)
                I[1] > 1 && (acc += real(_neg_row_value(coeffs, ks, twin, I, off, pms, b) * Wn))
                out[pj + (b - 1) * Npts] += acc
            end
        end
    end
    return out
end

function _synthesize_packed_direct!(
    out::AbstractArray{FT},
    g::Grids.PointwiseCartesian,
    coeffs::AbstractArray, ms::NTuple{D, Int}, iflag::Int, twin,
) where {FT, D}
    coords = FlowGeometries.Grids.coordinates(g)
    N = length(coords[1])
    ks = Grids.physical_wavenumbers(g, ms, Val(true))
    pms = Packing.packed_size(ms, Val(true))
    M = prod(pms)
    B = length(out) ÷ N
    fill!(out, zero(FT))
    @inbounds for j in 1:N
        for (mi, I) in enumerate(CartesianIndices(pms))
            nyq = Packing.is_nyquist(ks[1], I[1])
            W = nyq ? zero(Complex{FT}) : cis(iflag * _phase_scattered(ks, coords, I, j, FT))
            Wn = I[1] > 1 ? cis(iflag * _phase_scattered_neg(ks, coords, I, j, FT)) : zero(Complex{FT})
            for b in 1:B
                off = (b - 1) * M
                acc = nyq ? zero(FT) : real(coeffs[mi + off] * W)
                I[1] > 1 && (acc += real(_neg_row_value(coeffs, ks, twin, I, off, pms, b) * Wn))
                out[j + (b - 1) * N] += acc
            end
        end
    end
    return out
end

# =============================================================================
# Cartesian inverse (synthesis):  f[p, b] = Σ_k C[k, b] · exp(+iflag · i · k·x_p)
# =============================================================================

function _synthesize_cartesian_direct!(
    out::AbstractArray{Complex{FT}},
    g::FlowGeometries.Grids.AbstractStructuredGrid{<:FlowGeometries.Geometry.AbstractCartesianGeometry},
    coeffs::AbstractArray,
    ms::NTuple{D, Int},
    iflag::Int,
) where {FT, D}
    axes = FlowGeometries.Grids.coordinates(g)
    ss = size(g)
    Npts = prod(ss)
    M = prod(ms)
    B = length(out) ÷ Npts
    ks = Grids.physical_wavenumbers(g, ms, Val(false))   # native order, matching the forward's output
    fill!(out, zero(Complex{FT}))
    spat = CartesianIndices(ss)
    @inbounds for (pj, P) in enumerate(spat)
        for (mi, I) in enumerate(CartesianIndices(ms))
            phi = _phase_tensor(ks, axes, I, P, FT)
            W = cis(iflag * phi)
            for b in 1:B
                out[pj + (b - 1) * Npts] += coeffs[mi + (b - 1) * M] * W
            end
        end
    end
    return out
end

function _synthesize_cartesian_direct!(
    out::AbstractArray{Complex{FT}},
    g::Grids.PointwiseCartesian,
    coeffs::AbstractArray,
    ms::NTuple{D, Int},
    iflag::Int,
) where {FT, D}
    coords = FlowGeometries.Grids.coordinates(g)
    N = length(coords[1])
    M = prod(ms)
    B = length(out) ÷ N
    ks = Grids.physical_wavenumbers(g, ms, Val(false))   # native order, matching the forward's output
    fill!(out, zero(Complex{FT}))
    @inbounds for j in 1:N
        for (mi, I) in enumerate(CartesianIndices(ms))
            phi = _phase_scattered(ks, coords, I, j, FT)
            W = cis(iflag * phi)
            for b in 1:B
                out[j + (b - 1) * N] += coeffs[mi + (b - 1) * M] * W
            end
        end
    end
    return out
end

# =============================================================================
# Spherical forward / inverse (SHT projection / synthesis).
# Spherical grids are small (N = nlon·nlat or the scattered count), so per-point (θ, φ, weight) lists
# are materialized once — this is NOT the ∏N_d Cartesian blob, just a modest point list. θ/φ come from
# the FlowGeometries adapter's convention bridge (θ = colatitude, φ = longitude).
# =============================================================================

# Structured `(nlon, nlat)` SPHERE: point p = iλ + (jφ-1)·nlon (longitude fastest, matching the field
# layout), weight w_lat[jφ]·(2π/nlon) from the sampling's latitude quadrature. A structured SPHEROID grid
# takes the per-node method below instead: a sampling states its latitude rule for a sphere's
# colatitudes, and a spheroid's directions are the geocentric ones, so its rule is its own cell measure.
function _sph_point_data(
    g::FlowGeometries.Grids.AbstractStructuredGrid{<:FlowGeometries.Geometry.AbstractSphericalGeometry},
    ::Type{FT}; sampling = nothing, weights = nothing) where {FT}
    θpt, φpt = Grids._sph_points(g)
    nlon = length(FlowGeometries.Grids.coordinates(g, 1))
    nlat = length(FlowGeometries.Grids.coordinates(g, 2))
    N = nlon * nlat
    wlat = Grids._sht_weights(g, nlat; sampling = sampling, weights = weights)
    dλ = FT(2π) / nlon
    wpt = Vector{FT}(undef, N)
    # An inactive cell gets weight zero, so a per-point consumer of this data (the device kernel reaches
    # a structured grid through here) drops it from the quadrature. Point `p` is `iλ + (jφ-1)·nlon`,
    # which is the mask's own linear order.
    mask = FlowGeometries.Grids.mask(g)
    @inbounds for p in 1:N
        jφ = ((p - 1) ÷ nlon) + 1
        wpt[p] = mask[p] ? FT(wlat[jφ]) * dλ : zero(FT)
    end
    return FT.(θpt), FT.(φpt), wpt
end

# Every other spherical architecture — scattered nodes, and the pixelizations (HEALPix, cubed-sphere,
# icosahedral, ring, Yin-Yang), which carry no separable `(nlon, nlat)` axes: per-node (θ, φ) and the
# grid's own per-node quadrature weights (see `Grids._sph_node_weights`), overridden by `weights`.
function _sph_point_data(
    g::FlowGeometries.Grids.AbstractGrid{<:Grids.SphericalHarmonicGeometry},
    ::Type{FT}; sampling = nothing, weights = nothing) where {FT}
    θpt, φpt = Grids._sph_points(g)
    N = length(θpt)
    return FT.(θpt), FT.(φpt), Grids._sph_node_weights(g, FT, N, weights)
end

# Real spherical harmonic value in the FastSphericalHarmonics convention (verified to match
# `FSH.sph_evaluate` to round-off): Y_lm = s(m)·P̄_l^|m|·trig, with s(0)=1, s(m≠0)=(-1)^|m|√2, and
# trig = cos(mφ) for m≥0, sin(|m|φ) for m<0. `Plm` holds the normalized associated Legendre
# P̄_l^|m|(cosθ). Coefficients are real (stored in the complex array with zero imaginary part).
@inline function _real_sph(Plm::AbstractMatrix{FT}, l::Int, m::Int, φ::FT) where {FT}
    abs_m = abs(m)
    P = Plm[l+1, abs_m+1]
    m == 0 && return P
    s = isodd(abs_m) ? -sqrt(FT(2)) : sqrt(FT(2))
    return s * P * (m > 0 ? cos(m * φ) : sin(abs_m * φ))
end

# Structured spherical factorization, shared by the serial and threaded paths. `_sph_structured_setup`
# reads the grid's `(λ, θ)` axes + latitude quadrature; `_sph_longitude_dft` is the once-per-`(θ, m)`
# longitude DFT (dense matmul, dependency-free); `_sph_theta_accumulate!` runs the θ-Legendre contraction
# over a set of latitudes. Same normalized-Legendre / factor / m↔col convention as `_real_sph`; `φc` reads
# the |m| longitude bin (Re for `m ≥ 0`, −Im for `m < 0`).
function _sph_structured_setup(g, ::Type{FT}; sampling, weights, lmax = nothing) where {FT}
    λ = FlowGeometries.Grids.coordinates(g, 1)
    φlat = FlowGeometries.Grids.coordinates(g, 2)
    nlon = length(λ)
    nlat = length(φlat)
    N = nlon * nlat
    wlat = Grids._sht_weights(g, nlat; sampling = sampling, weights = weights, lmax = lmax)
    wθ = FT.(wlat) .* (FT(2π) / nlon)
    # The colatitude of each latitude row's direction, which the grid's geometry defines.
    θ = [Grids._colatitude(g, FT(φ)) for φ in φlat]
    return λ, θ, wθ, nlon, nlat, N
end

# The `(lmax+1) × nlon` longitude DFT matrix, `exp(-i·m·λ)`. Point-independent given the axis, so a plan
# builds it once.
function _sph_longitude_matrix(::Type{FT}, λ, lmax::Int, nlon::Int) where {FT}
    Wφ = Matrix{Complex{FT}}(undef, lmax + 1, nlon)
    @inbounds for iφ in 1:nlon
        λi = FT(λ[iφ])
        for a in 1:(lmax + 1)
            Wφ[a, iφ] = cis(-FT(a - 1) * λi)
        end
    end
    return Wφ
end

# A real spherical harmonic reads `Σ_φ f cos(mλ)` and `Σ_φ f sin(mλ)`. Written with the longitude DFT
# `g[a] = Σ_φ f exp(-i(a-1)λ)` and its partner `ĝ[a] = Σ_φ f exp(+i(a-1)λ)`, those are `(g+ĝ)/2` and
# `i(g-ĝ)/2`. A REAL field satisfies `ĝ = conj(g)`, collapsing them to `real(g)` and `-imag(g)`, and the
# partner transform is not built. A COMPLEX field's `ĝ` is an independent sum, so it is, and `nothing`
# marks its absence throughout this file.
@inline _phi_bin(m::Int, g, ::Nothing) = m >= 0 ? real(g) : -imag(g)
@inline _phi_bin(m::Int, g, ĝ) = m >= 0 ? (g + ĝ) / 2 : im * (g - ĝ) / 2

_partner_matrix(::Type{<:Complex}, Wφ) = conj(Wφ)
_partner_matrix(::Type{<:Real}, Wφ) = nothing
_partner_buffer(::Type{<:Complex}, ::Type{FT}, dims::Int...) where {FT} = Array{Complex{FT}}(undef, dims)
_partner_buffer(::Type{<:Real}, ::Type{FT}, dims::Int...) where {FT} = nothing
@inline _fill_partner!(::Nothing) = nothing
@inline _fill_partner!(a) = fill!(a, zero(eltype(a)))
@inline _partner_at(::Nothing, a::Int, i::Int, b::Int) = nothing
Base.@propagate_inbounds _partner_at(f, a::Int, i::Int, b::Int) = f[a, i, b]

# Longitude DFT of every latitude row, and the partner transform a complex field needs. `T` is the
# coefficient element type, whose realness is the field's.
function _sph_longitude_dft(::Type{T}, λ, field, lmax::Int, nlon::Int, nlat::Int, B::Int) where {T}
    FT = real(float(T))
    Wφ = _sph_longitude_matrix(FT, λ, lmax, nlon)
    F = reshape(field, nlon, nlat * B)
    return reshape(Wφ * F, lmax + 1, nlat, B), _partner_dft(_partner_matrix(T, Wφ), F, lmax, nlat, B)
end
_partner_dft(::Nothing, F, lmax::Int, nlat::Int, B::Int) = nothing
_partner_dft(Wc, F, lmax::Int, nlat::Int, B::Int) = reshape(Wc * F, lmax + 1, nlat, B)

# The same contraction into a held buffer, so a reused plan allocates neither the matrix nor the result.
function _sph_longitude_dft!(fhat, Wφ, field, nlon::Int, nlat::Int, B::Int)
    LA.mul!(reshape(fhat, size(Wφ, 1), nlat * B), Wφ, reshape(field, nlon, nlat * B))
    return fhat
end
_sph_longitude_dft!(::Nothing, ::Nothing, field, nlon::Int, nlat::Int, B::Int) = nothing

# Longitude sum for a set of rings, accumulated into `fh[a, jr, b]` with `jr` enumerating `rows`. The exact
# discrete sum over each ring's own longitudes, so the ring transform is the per-point projection
# reassociated. Shared by the serial and threaded ring paths, which differ only in how `rows` is split.
function _sph_ring_longitude!(fh, fh2, F, rt, rows, lmax::Int, B::Int, ::Type{FT}) where {FT}
    mask = rt.mask
    @inbounds for (jr, r) in enumerate(rows)
        for p in rt.ranges[r]
            mask[p] || continue            # an inactive cell carries no data to integrate
            λp = rt.λ[p]
            for a in 1:(lmax + 1)
                e = cis(-FT(a - 1) * λp)
                for b in 1:B
                    fh[a, jr, b] += F[p, b] * e
                end
                _ring_partner!(fh2, a, jr, B, F, p, conj(e))
            end
        end
    end
    return fh
end

@inline _ring_partner!(::Nothing, a::Int, jr::Int, B::Int, F, p::Int, e) = nothing
Base.@propagate_inbounds function _ring_partner!(fh2, a::Int, jr::Int, B::Int, F, p::Int, e)
    for b in 1:B
        fh2[a, jr, b] += F[p, b] * e
    end
    return nothing
end

function _sph_theta_accumulate!(C, fhat, fhat2, θ::AbstractVector{FT}, wθ, tables, rows, lmax::Int,
        B::Int, Plm::AbstractMatrix{FT}) where {FT}
    @inbounds for iθ in rows
        SphericalKernels.fill_legendre!(Plm, tables, cos(θ[iθ]), sin(θ[iθ]), lmax)
        w = wθ[iθ]
        for l in 0:lmax
            for m in -l:l
                abs_m = abs(m)
                s = m == 0 ? one(FT) : (isodd(abs_m) ? -sqrt(FT(2)) : sqrt(FT(2)))
                fac = w * s * Plm[l + 1, abs_m + 1]
                idx = sph_mode_index(l, m)
                for b in 1:B
                    φc = _phi_bin(m, fhat[abs_m + 1, iθ, b], _partner_at(fhat2, abs_m + 1, iθ, b))
                    C[idx, b] += fac * φc
                end
            end
        end
    end
    return C
end

# The spherical forward routes on the grid's declared layout (`Grids._sph_layout`): a tensor product
# factorizes over its shared longitude axis, an iso-latitude ring layout over each ring's own longitudes,
# and a point cloud has neither.
_calculate_spectrum_spherical_direct!(
    coeffs::AbstractArray{<:Number},
    g::FlowGeometries.Grids.AbstractGrid{<:Grids.SphericalHarmonicGeometry},
    field::AbstractArray, lmax::Int; kwargs...,
) = _sph_direct!(Grids._sph_layout(g), coeffs, g, field, lmax; kwargs...)

# =============================================================================
# Setup / run split. `sph_setup` reads the grid — `materialize`, the latitude quadrature (a Gauss–Legendre
# root solve), the ring table, the longitude DFT matrix, the Legendre tables — and `sph_run!` transforms a
# field against what it returns. The one-shot below composes the two, and a reusable plan holds the setup,
# whose share of the transform is a minority of the time and ~3/4 of the allocations.
#
# Buffers sized by the batch live in the setup, so a plan built for a batch shape reuses them; the
# one-shot reads the batch from `coeffs`.
# =============================================================================

"""
    sph_setup(layout, grid, T, lmax, B; sampling, weights) -> NamedTuple

Everything a spherical direct sum reads from `grid`: its materialized nodes, or its ring table, or its
longitude DFT matrix and latitude quadrature — whichever its layout uses — plus the Legendre recurrence
tables and the longitude buffer. `T` is the coefficient element type; a complex one adds the `exp(+imλ)`
partner transform.

Run it against a field with [`sph_run!`](@ref). `DirectSumSphericalPlan` holds it across executions.
"""
function sph_setup(::Grids.TensorSphere, g, ::Type{T}, lmax::Int, B::Int;
        sampling = nothing, weights = nothing) where {T}
    FT = real(float(T))
    λ, θ, wθ, nlon, nlat, _ = _sph_structured_setup(g, FT; sampling = sampling, weights = weights,
        lmax = lmax)
    Wφ = _sph_longitude_matrix(FT, λ, lmax, nlon)
    return (; Wφ, Wφc = _partner_matrix(T, Wφ), θ, wθ, nlon, nlat,
        fhat = Array{Complex{FT}, 3}(undef, lmax + 1, nlat, B),
        fhat2 = _partner_buffer(T, FT, lmax + 1, nlat, B),
        tables = SphericalKernels.legendre_tables(FT, lmax),
        Plm = Matrix{FT}(undef, lmax + 1, lmax + 1))
end

function sph_setup(::Grids.RingSphere, g, ::Type{T}, lmax::Int, B::Int;
        sampling = nothing, weights = nothing) where {T}
    FT = real(float(T))
    rt = Grids._ring_table(g, FT; lmax = lmax)
    nr = length(rt.ranges)
    return (; rt, nr, N = length(rt.λ),
        fhat = zeros(Complex{FT}, lmax + 1, nr, B),
        fhat2 = _partner_buffer(T, FT, lmax + 1, nr, B),
        tables = SphericalKernels.legendre_tables(FT, lmax),
        Plm = Matrix{FT}(undef, lmax + 1, lmax + 1))
end

function sph_setup(::Grids.ScatteredSphere, g, ::Type{T}, lmax::Int, B::Int;
        sampling = nothing, weights = nothing) where {T}
    FT = real(float(T))
    θpt, φpt, wpt = _sph_point_data(g, FT; sampling = sampling, weights = weights)
    return (; θpt, φpt, wpt, N = length(θpt),
        tables = SphericalKernels.legendre_tables(FT, lmax),
        Plm = Matrix{FT}(undef, lmax + 1, lmax + 1))
end

# Structured `(nlon, nlat)` grid with uniform longitude: `O(nlat · L²)` in place of the per-point
# `O(N · L²)`.
"""
    sph_run!(coeffs, layout, setup, field, lmax, B) -> ks

Fill `coeffs` `(lmax+1, 2lmax+1, B)` with the spherical spectrum of `field` against a
[`sph_setup`](@ref) result, and return `(0:lmax, -lmax:lmax)`.

A tensor layout runs one longitude DFT for every latitude, a ring layout one per ring, and a point cloud
projects per node; all three then contract over θ against the Legendre tables.
"""
function sph_run!(coeffs::AbstractArray{C}, ::Grids.TensorSphere, s, field::AbstractArray,
        lmax::Int, B::Int) where {C <: Number}
    Nθc = lmax + 1
    Nφc = 2 * lmax + 1
    _sph_longitude_dft!(s.fhat, s.Wφ, field, s.nlon, s.nlat, B)
    _sph_longitude_dft!(s.fhat2, s.Wφc, field, s.nlon, s.nlat, B)
    fill!(coeffs, zero(C))
    _sph_theta_accumulate!(reshape(coeffs, Nθc, Nφc, B), s.fhat, s.fhat2, s.θ, s.wθ, s.tables,
        1:s.nlat, lmax, B, s.Plm)
    return (0:lmax, -lmax:lmax)
end

# Iso-latitude rings (HEALPix, ring / reduced-Gaussian grids): colatitude is constant along a ring, so the
# longitude sum runs once per `(ring, m)` and the θ-Legendre contraction then runs over rings —
# `O(N·lmax + nrings·lmax²)` against the per-point `O(N·lmax²)`. The longitude sum is the exact discrete
# sum over each ring's own longitudes, so this is the per-point projection reassociated and agrees with it
# to round-off. Longitude count and quadrature weight both vary by ring here, where the tensor form has a
# shared `nlon` and a single `wθ` per latitude.
function sph_run!(coeffs::AbstractArray{C}, ::Grids.RingSphere, s, field::AbstractArray,
        lmax::Int, B::Int) where {C <: Number}
    FT = real(float(C))
    Nθc = lmax + 1
    Nφc = 2 * lmax + 1
    fill!(coeffs, zero(C))
    fill!(s.fhat, zero(eltype(s.fhat)))
    _fill_partner!(s.fhat2)
    _sph_ring_longitude!(s.fhat, s.fhat2, reshape(field, s.N, B), s.rt, 1:s.nr, lmax, B, FT)
    _sph_theta_accumulate!(reshape(coeffs, Nθc, Nφc, B), s.fhat, s.fhat2, s.rt.θ, s.rt.w, s.tables,
        1:s.nr, lmax, B, s.Plm)
    return (0:lmax, -lmax:lmax)
end

# Every other spherical layout — scattered nodes, and the pixelizations with no iso-latitude structure
# (cubed-sphere, icosahedral, Yin-Yang) — so the per-point projection stands.
function sph_run!(coeffs::AbstractArray{CT}, ::Grids.ScatteredSphere, s, field::AbstractArray,
        lmax::Int, B::Int) where {CT <: Number}
    Nθc = lmax + 1
    Nφc = 2 * lmax + 1
    N = s.N
    F = reshape(field, N, B)
    C = reshape(coeffs, Nθc, Nφc, B)
    fill!(coeffs, zero(CT))
    Plm = s.Plm
    @inbounds for p in 1:N
        wp = s.wpt[p]
        # A node of zero weight contributes nothing, and skipping it keeps a masked cell's value out of
        # the sum entirely: masked data is commonly NaN, and `0 * NaN` is NaN.
        iszero(wp) && continue
        xj = cos(s.θpt[p])
        sj = sin(s.θpt[p])
        φp = s.φpt[p]
        SphericalKernels.fill_legendre!(Plm, s.tables, xj, sj, lmax)
        for l in 0:lmax
            for m in -l:l
                Ylm = _real_sph(Plm, l, m, φp)          # real SH (FSH convention)
                idx = sph_mode_index(l, m)
                gw = Ylm * wp
                for b in 1:B
                    C[idx, b] += F[p, b] * gw
                end
            end
        end
    end
    return (0:lmax, -lmax:lmax)
end

# One-shot: set up against this grid, then run. A repeated caller holds the setup in a plan.
function _sph_direct!(layout, coeffs::AbstractArray{C}, g, field::AbstractArray, lmax::Int;
        sampling = nothing, weights = nothing) where {C <: Number}
    B = length(coeffs) ÷ ((lmax + 1) * (2 * lmax + 1))
    s = sph_setup(layout, g, C, lmax, B; sampling = sampling, weights = weights)
    return sph_run!(coeffs, layout, s, field, lmax, B)
end

function _synthesize_spherical_direct!(
    out::AbstractArray{OT},
    g::FlowGeometries.Grids.AbstractGrid{<:Grids.SphericalHarmonicGeometry},
    coeffs::AbstractArray,
    lmax::Int,
) where {OT <: Number}
    FT = real(float(OT))
    θraw, φraw = Grids._sph_points(g)     # synthesis reads nodes only; it carries no quadrature
    θpt = FT.(θraw)
    φpt = FT.(φraw)
    N = length(θpt)
    Nθc = lmax + 1
    Nφc = 2 * lmax + 1
    B = length(out) ÷ N
    O = reshape(out, N, B)
    C = reshape(coeffs, Nθc, Nφc, B)
    fill!(O, zero(OT))
    tables = SphericalKernels.legendre_tables(FT, lmax)
    Plm = Matrix{FT}(undef, lmax + 1, lmax + 1)
    @inbounds for p in 1:N
        xj = cos(θpt[p])
        sj = sin(θpt[p])
        φp = φpt[p]
        SphericalKernels.fill_legendre!(Plm, tables, xj, sj, lmax)
        for l in 0:lmax
            for m in -l:l
                Ylm = _real_sph(Plm, l, m, φp)          # real SH (FSH convention)
                idx = sph_mode_index(l, m)
                for b in 1:B
                    O[p, b] += C[idx, b] * Ylm
                end
            end
        end
    end
    return out
end

end # module DirectSum
