module DirectSum

using FlowGeometries: FlowGeometries
using FlowTransformBindings: FlowTransformBindings as FTB
using LinearAlgebra: LinearAlgebra as LA
using ..Grids: Grids
using ..SphericalKernels: SphericalKernels

export sph_mode_index

# The dependency-free transform `DirectSumSpectralBackend()` selects. It reads a field `(spatial…,
# batch…)` and writes coefficients in the packed layout `(spectral…, batch…)` (a real Cartesian field
# halves axis 1 to `k₁ ≥ 0`). A tensor-product grid, uniform or not, factorizes into `D` successive 1-D
# DFTs, one dense matrix per axis (`O(D · n^{D+1})`); a point cloud is the `O(∏ms · Npts)` direct sum.

"""
    sph_mode_index(l, m) -> CartesianIndex

`CartesianIndex` of degree `l`, order `m` in the `(lmax+1, 2lmax+1)` coefficient array.
"""
@inline function sph_mode_index(l::Int, m::Int)
    row = l - abs(m) + 1
    col = m == 0 ? 1 : (m < 0 ? 2 * abs(m) : 2 * m + 1)
    return CartesianIndex(row, col)
end

# A point set's phase separates by axis, `e^{is k·x_j} = Πᵈ e^{is k_d x_{j,d}}`, so a point fills one table
# per axis, `E[d][a] = e^{is ks[d][a] x_{j,d}}` over the indices `rngs[d]`, and a mode's exponential is a
# product of one entry per axis. `ks` mixes axis types, so the tables fill by `map` over the tuples;
# indexing `ks` at a loop variable boxes the axis.
@inline function _axis_tables!(E::Tuple, ks::Tuple, coords::Tuple, j::Int, s::FT, rngs::Tuple) where {FT}
    map(E, ks, coords, rngs) do e, k, x, rng
        xj = FT(x[j])
        @inbounds for a in rng
            e[a] = cis(s * FT(k[a]) * xj)
        end
        return e
    end
    return E
end

@inline _mode_phase(E::Tuple, I::CartesianIndex) = prod(map((e, a) -> @inbounds(e[a]), E, Tuple(I)))

# One table per axis, `m[d]` entries each.
_axis_scratch(::Type{FT}, m::Tuple) where {FT} = map(n -> Vector{Complex{FT}}(undef, n), m)

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
for an axis past the first — the staging buffers and the permutation bringing it to the front and back,
all built once. Run it with `_axis_pass!`.
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
# Cartesian setup / run split. `cart_setup` reads the grid (the per-axis DFT matrices, the contraction
# chain's working arrays, the wavenumber axes and the modes held at zero) and `cart_run!` transforms a
# field against what it returns. The one-shot composes the two and `DirectSumCartesianPlan` holds the
# setup. The field arrives already scaled by the grid's quadrature factor.
#
# `TensorCartesian` and `CloudCartesian` name the two algorithms.
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
_perm_front(::Val{nd}, d::Int) where {nd} = (d, ntuple(i -> i < d ? i : i + 1, Val(nd - 1))...)

# One axis-by-axis contraction from extents `ss` to extents `outs`, the batch dims riding along: the
# per-axis dense matrices `W_d[a, p] = exp(-iflag·i·kaxes[d][a]·axes[d][p])` and every working array it
# walks through. The forward contracts points to modes (`kaxes` the wavenumbers, `axes` the positions);
# the synthesis contracts modes to points, with the two swapped and `iflag` negated.
function _contraction(::Type{FT}, kaxes::Tuple, axes::Tuple, ss::NTuple{D, Int}, outs::NTuple{D, Int},
        batch::Tuple, iflag::Int) where {FT, D}
    shapes = _chain_shapes(ss, outs, batch)
    nd = D + length(batch)
    chain = ntuple(d -> Array{Complex{FT}}(undef, shapes[d]), Val(D + 1))
    empty = Array{Complex{FT}}(undef, ntuple(_ -> 0, Val(nd)))    # axis 1 contracts in place, needing neither
    ident = ntuple(identity, Val(nd))
    passes = ntuple(Val(D)) do d
        W = _dft_matrix(FT, kaxes[d], axes[d], outs[d], ss[d], iflag)
        if d == 1
            return AxisPass{FT, nd, Matrix{Complex{FT}}, Array{Complex{FT}, nd}}(
                W, _mat(chain[1], ss[1]), _mat(chain[2], outs[1]), empty, empty, ident, ident, false)
        end
        perm = _perm_front(Val(nd), d)
        pbuf = Array{Complex{FT}}(undef, map(p -> shapes[d][p], perm))
        rbuf = Array{Complex{FT}}(undef, (outs[d], Base.tail(size(pbuf))...))
        return AxisPass{FT, nd, Matrix{Complex{FT}}, Array{Complex{FT}, nd}}(
            W, _mat(pbuf, ss[d]), _mat(rbuf, outs[d]), pbuf, rbuf, perm,
            NTuple{nd, Int}(invperm(collect(perm))), true)
    end
    return (; passes, chain)
end

# `(n, length(a) ÷ n)` matrix alias of `a`, sharing its memory; an empty leading extent leaves an empty
# alias.
_mat(a::Array{T}, n::Int) where {T} = n == 0 ? reshape(a, 0, 0) : reshape(a, n, length(a) ÷ n)

# `dest = (1/npts) · W_D ⋯ W_1 · field`, through the contraction's own buffers.
function _run_contraction!(dest::AbstractArray, c, field::AbstractArray, npts::Int)
    isempty(dest) && return dest
    copyto!(c.chain[1], field)
    _run_chain!(c)
    copyto!(dest, last(c.chain))
    dest ./= npts
    return dest
end

# `chain[D+1] = W_D ⋯ W_1 · chain[1]`.
function _run_chain!(c)
    for d in eachindex(c.passes)
        _axis_pass!(c.chain[d + 1], c.passes[d], c.chain[d])
    end
    return last(c.chain)
end

"""
    phase_geometry(grid, ::Type{FT}, ms, ::Type{T}, box = nothing) -> (xs, ks)

The positions `x - x₀` and the wavenumber axes a Cartesian direct sum over `grid` reads, over the
coefficient box of an `ms`-mode transform (`Grids.mode_box`) of a field of element type `T`, halved for a
real `T`. `box` holds a whole grid's origins and Fourier lengths when `grid` is part of it (see
`Grids.axis_geometry`).
"""
function phase_geometry(g, ::Type{FT}, ms::NTuple{D, Int}, ::Type{T}, box = nothing) where {FT, D, T}
    offs, ranges = Grids.axis_geometry(FT, g, Val(D), box)
    return Grids.relative_coordinates(g, offs),
        Grids.physical_wavenumbers(ranges, Grids.mode_box(g, ms, T), T)
end

"""
    cart_setup(layout, grid, T, ms, batch, iflag; box = nothing) -> NamedTuple

Everything a Cartesian direct sum reads from `grid`: the wavenumber axes of the coefficient box, the
modes of a real field's packed half that hold zero (`FlowTransformBindings.outside_modes`), and for a
tensor layout the per-axis DFT matrices and the working arrays their contraction walks through. `T` is the
field's element type, whose realness fixes the packed layout; `box` is as in [`phase_geometry`](@ref).

Run it against a field with [`cart_run!`](@ref), or the threaded run of the OhMyThreads extension;
`DirectSumCartesianPlan` holds it across executions.
"""
function cart_setup(::TensorCartesian, g, ::Type{T}, ms::NTuple{D, Int}, batch::Tuple,
        iflag::Int; box = nothing) where {T, D}
    FT = real(float(T))
    ss = size(g)
    xs, ks = phase_geometry(g, FT, ms, T, box)
    pms = map(length, ks)
    return (; ks, pms, Npts = prod(ss), iflag, outside = _outside(ms, Grids.mode_box(g, ms, T), T),
        fwd = _contraction(FT, ks, xs, ss, pms, batch, iflag))
end

function cart_setup(::CloudCartesian, g, ::Type{T}, ms::NTuple{D, Int}, batch::Tuple,
        iflag::Int; box = nothing) where {T, D}
    FT = real(float(T))
    coords, ks = phase_geometry(g, FT, ms, T, box)
    pms = map(length, ks)
    return (; ks, pms, coords, N = length(coords[1]), iflag, E = _axis_scratch(FT, pms),
        outside = _outside(ms, Grids.mode_box(g, ms, T), T))
end

# The packed-half modes outside the computed box and its mirror, which a real field's transform leaves at
# zero; a complex field's box is the computed one.
_outside(ms::Tuple, bs::Tuple, ::Type{<:Real}) = FTB.outside_modes(ms, bs)
_outside(ms::Tuple, bs::Tuple, ::Type{<:Complex}) = Int[]

"""
    cart_run!(coeffs, layout, setup, field) -> ks

Fill `coeffs` with the Cartesian spectrum of `field` against a [`cart_setup`](@ref) result, the modes
outside the computed box and its mirror at zero, and return the wavenumber axes.

`field` arrives already scaled by the grid's quadrature factor. A tensor layout contracts axis by axis, a
point cloud sums directly, and both write only into `coeffs` and the setup's own buffers.
"""
function cart_run!(coeffs::AbstractArray{Complex{FT}}, ::TensorCartesian, s,
        field::AbstractArray) where {FT}
    _run_contraction!(coeffs, s.fwd, field, s.Npts)
    _zero_outside!(coeffs, s.outside, prod(s.pms))
    return s.ks
end

function cart_run!(coeffs::AbstractArray{Complex{FT}}, ::CloudCartesian, s,
        field::AbstractArray) where {FT}
    _cloud_sum!(coeffs, s.ks, s.coords, s.pms, s.N, s.iflag, field, s.E)
    _zero_outside!(coeffs, s.outside, prod(s.pms))
    return s.ks
end

# Zero the modes `outside` (linear indices into one field's `P` coefficients) of every field in `coeffs`.
function _zero_outside!(coeffs::AbstractArray, outside::AbstractVector{Int}, P::Int)
    isempty(outside) && return coeffs
    @inbounds for b in 1:(length(coeffs) ÷ P), i in outside
        coeffs[i + (b - 1) * P] = zero(eltype(coeffs))
    end
    return coeffs
end

# `C[k, b] = (1/N) Σ_j f[j, b] · exp(-iflag · i · k·x_j)` over the mode set `pms` with wavenumber axes
# `kaxes`, `E` the per-axis tables (`_axis_tables!`).
function _cloud_sum!(dest::AbstractArray{Complex{FT}}, kaxes::Tuple, coords::Tuple,
        pms::NTuple{D, Int}, N::Int, iflag::Int, field::AbstractArray, E::Tuple) where {FT, D}
    fill!(dest, zero(Complex{FT}))
    prod(pms) == 0 && return dest
    _cloud_block!(dest, kaxes, coords, pms, N, FT(-iflag), field, E, 1:pms[D])
    dest ./= N
    return dest
end

# The modes whose last index lies in `blk`, each accumulating every point in order, so a split of the
# last axis into blocks gives the sums one pass does.
function _cloud_block!(dest::AbstractArray{Complex{FT}}, kaxes::Tuple, coords::Tuple,
        pms::NTuple{D, Int}, N::Int, s::FT, field::AbstractArray, E::Tuple, blk) where {FT, D}
    M = prod(pms)
    B = length(dest) ÷ M
    rngs = ntuple(d -> d == D ? blk : Base.OneTo(pms[d]), Val(D))
    lin = LinearIndices(pms)
    @inbounds for j in 1:N
        _axis_tables!(E, kaxes, coords, j, s, rngs)
        for I in CartesianIndices(rngs)
            W = _mode_phase(E, I)
            mi = lin[I]
            for b in 1:B
                dest[mi + (b - 1) * M] += field[j + (b - 1) * N] * W
            end
        end
    end
    return dest
end

# =============================================================================
# Cartesian inverse (synthesis) from the packed half, writing a real field.
#
# The series of a real field's packed half over the coefficient box is `f[p] = Σ_κ C[κ] · exp(+iflag·i·κ·x_p)`
# over the whole box, the `k₁ < 0` half read as `C[κ] = conj(C[−κ])`. Each stored row `k₁ > 0` stands for
# itself and its mirror, so the sum is the stored modes' real parts weighted by `fold_weight`; the
# imaginary parts cancel. A tensor grid contracts the complex series of `Grids.series_modes` whose real
# part this is (`native_coefficients!`) axis by axis; a point set sums the stored modes directly at each
# point.
# =============================================================================

"""
    native_table(ms, box) -> (src, wt)

How the complex series in FFT order whose real part is the series of a real field's packed half over
`box` reads that half, `box` the coefficient box of an `ms`-mode transform (`Grids.mode_box`). The series
has `(box[1], ms[2:D]…)` modes. Mode `i` (linear) is entry `src[i]` of one field's half, or the conjugate
of entry `-src[i]` where `src[i] < 0` (`k₁ < 0`), times `wt[i]`. The weight is two where `-k` is not a mode
of the series (`k_d = -ms[d]/2` on an axis `d ≥ 2` that `box` widens) and one elsewhere, so the real part
sums `k` and `-k` alike.
"""
function native_table(ms::NTuple{D, Int}, box::NTuple{D, Int}) where {D}
    ns = (box[1], Base.tail(ms)...)
    Lp = LinearIndices(FTB.packed_size(Real, box))
    src = Vector{Int}(undef, prod(ns))
    wt = Vector{Int8}(undef, prod(ns))
    @inbounds for (fi, I) in enumerate(CartesianIndices(ns))
        k = ntuple(d -> FTB.mode_frequency(FTB.FFTModes(), I[d], ns[d]), Val(D))
        neg = k[1] < 0
        j = Lp[ntuple(d -> FTB.ovs_index(neg ? -k[d] : k[d], d == 1 ? 0 : box[d]), Val(D))...]
        src[fi] = neg ? -j : j
        wt[fi] = _mirror_missing(k, ms, box) ? Int8(2) : Int8(1)
    end
    return src, wt
end

"""
    native_coefficients!(full, half, src, wt) -> full

The series `full` `(modes…, batch…)` of each field's packed half in `half` `(packed…, batch…)`, through the
[`native_table`](@ref) `(src, wt)`. The device method is the KernelAbstractions extension's.
"""
function native_coefficients!(full::AbstractArray{Complex{FT}}, half::AbstractArray, src::Vector{Int},
        wt::Vector{Int8}) where {FT}
    M = length(src)
    B = length(full) ÷ M
    Mp = length(half) ÷ B
    @inbounds for b in 1:B, i in 1:M
        j = src[i]
        v = half[abs(j) + (b - 1) * Mp]
        full[i + (b - 1) * M] = wt[i] * Complex{FT}(j < 0 ? conj(v) : v)
    end
    return full
end

# Whether `-k` is missing from the series `native_coefficients!` writes: `k` at `-ms_d/2` on an axis
# `d ≥ 2` the box widens.
@inline function _mirror_missing(k::NTuple{D, Int}, ms::NTuple{D, Int}, box::NTuple{D, Int}) where {D}
    for d in 2:D
        box[d] > ms[d] && k[d] == -(ms[d] ÷ 2) && return true
    end
    return false
end

# A tensor grid separates: the series is `f = E_D ⋯ E_1 · C`, with one `n_d × m_d` table
# `E_d[p, a] = exp(+iflag·i·k_a·x_p)` per axis, so a call is `D` matrix products. A real field's packed
# half is turned into that series first (`native_coefficients!`), and the real part of the result is the
# field.
function _synthesize_packed_direct!(
    out::AbstractArray{FT},
    g::FlowGeometries.Grids.AbstractStructuredGrid{<:FlowGeometries.Geometry.AbstractCartesianGeometry},
    coeffs::AbstractArray, ms::NTuple{D, Int}, iflag::Int, geo = nothing,
) where {FT, D}
    s = geo === nothing ?
        synth_setup(TensorCartesian(), g, FT, ms, Grids.field_batch_shape(g, out), iflag) : geo
    native_coefficients!(first(s.chain.chain), coeffs, s.native...)
    out .= real.(_run_chain!(s.chain))
    return out
end

"""
    synth_setup(layout, grid, T, ms, batch, iflag; box = nothing)

Everything a Cartesian direct-sum synthesis over `grid` reads, for a field of element type `T`. On a
tensor grid, the [`native_table`](@ref) a real field's packed half is read through, and the per-axis tables
`E_d[p, a] = exp(+iflag·i·k_a·x_p)` with the working arrays their contraction walks through from the
series of `Grids.series_modes` `(…, batch…)` to the field `(size(grid)…, batch…)`. On a point set, the
positions and the wavenumbers of the coefficient box its direct sum reads (`phase_geometry`). `box` is as
in [`phase_geometry`](@ref).
"""
function synth_setup(::TensorCartesian, g, ::Type{T}, ms::NTuple{D, Int}, batch::Tuple, iflag::Int;
        box = nothing) where {T, D}
    FT = real(float(T))
    offs, ranges = Grids.axis_geometry(FT, g, Val(D), box)
    xs = Grids.relative_coordinates(g, offs)
    ns = Grids.series_modes(g, ms, T)
    ks = Grids.physical_wavenumbers(ranges, ns, Complex{FT})
    return (; native = native_table(ms, Grids.mode_box(g, ms, T)),
        chain = _contraction(FT, xs, ks, ns, size(g), batch, -iflag))
end

function synth_setup(::CloudCartesian, g, ::Type{T}, ms::NTuple{D, Int}, batch::Tuple, iflag::Int;
        box = nothing) where {T, D}
    FT = real(float(T))
    xs, ks = phase_geometry(g, FT, ms, T, box)
    return (; xs, ks, E = _axis_scratch(FT, map(length, ks)))
end

function _synthesize_packed_direct!(
    out::AbstractArray{FT},
    g::Grids.PointwiseCartesian,
    coeffs::AbstractArray, ms::NTuple{D, Int}, iflag::Int, geo = nothing,
) where {FT, D}
    s = geo === nothing ? synth_setup(CloudCartesian(), g, FT, ms, (), iflag) : geo
    N = length(s.xs[1])
    B = length(out) ÷ N
    fill!(out, zero(FT))
    @inbounds for j in 1:N
        _packed_point!(out, coeffs, s.E, s.ks, s.xs, map(length, s.ks), j, FT(iflag), N, B)
    end
    return out
end

# Point `j` of a real field from the packed half: each stored mode's real part, weighted by the count of
# box modes it stands for, off the point's tables.
@inline function _packed_point!(out::AbstractArray{FT}, coeffs, E::Tuple, ks::Tuple, xs::Tuple,
        pms::NTuple{D, Int}, j::Int, s::FT, N::Int, B::Int) where {FT, D}
    _axis_tables!(E, ks, xs, j, s, map(Base.OneTo, pms))
    M = prod(pms)
    @inbounds for (mi, I) in enumerate(CartesianIndices(pms))
        W = FT(FTB.fold_weight(ks[1], I[1])) * _mode_phase(E, I)
        for b in 1:B
            out[j + (b - 1) * N] += real(coeffs[mi + (b - 1) * M] * W)
        end
    end
    return out
end

# =============================================================================
# Cartesian inverse (synthesis):  f[p, b] = Σ_k C[k, b] · exp(+iflag · i · k·(x_p - x₀))
# =============================================================================

# The native spectrum, in the forward's order, through the per-axis tables of `synth_setup`.
function _synthesize_cartesian_direct!(
    out::AbstractArray{Complex{FT}},
    g::FlowGeometries.Grids.AbstractStructuredGrid{<:FlowGeometries.Geometry.AbstractCartesianGeometry},
    coeffs::AbstractArray,
    ms::NTuple{D, Int},
    iflag::Int,
    geo = nothing,
) where {FT, D}
    s = geo === nothing ?
        synth_setup(TensorCartesian(), g, Complex{FT}, ms, Grids.field_batch_shape(g, out), iflag) : geo
    copyto!(first(s.chain.chain), coeffs)
    copyto!(out, _run_chain!(s.chain))
    return out
end

function _synthesize_cartesian_direct!(
    out::AbstractArray{Complex{FT}},
    g::Grids.PointwiseCartesian,
    coeffs::AbstractArray,
    ms::NTuple{D, Int},
    iflag::Int,
    geo = nothing,
) where {FT, D}
    s = geo === nothing ? synth_setup(CloudCartesian(), g, Complex{FT}, ms, (), iflag) : geo
    N = length(s.xs[1])
    B = length(out) ÷ N
    fill!(out, zero(Complex{FT}))
    @inbounds for j in 1:N
        _native_point!(out, coeffs, s.E, s.ks, s.xs, ms, j, FT(iflag), N, B)
    end
    return out
end

# Point `j` of a complex field from its native spectrum, off the point's tables.
@inline function _native_point!(out::AbstractArray{Complex{FT}}, coeffs, E::Tuple, ks::Tuple, xs::Tuple,
        ms::NTuple{D, Int}, j::Int, s::FT, N::Int, B::Int) where {FT, D}
    _axis_tables!(E, ks, xs, j, s, map(Base.OneTo, ms))
    M = prod(ms)
    @inbounds for (mi, I) in enumerate(CartesianIndices(ms))
        W = _mode_phase(E, I)
        for b in 1:B
            out[j + (b - 1) * N] += coeffs[mi + (b - 1) * M] * W
        end
    end
    return out
end

# =============================================================================
# Spherical forward / inverse (SHT projection / synthesis). θ/φ come from `Grids._sph_points`
# (θ = colatitude, φ = longitude); a per-point (θ, φ, weight) list is materialized once per setup.
# =============================================================================

# Per-point colatitude, longitude and quadrature weight (`Grids._sph_point_weights`), an inactive point
# at weight zero, in the field's column-major order.
function _sph_point_data(
    g::FlowGeometries.Grids.AbstractGrid{<:Grids.SphericalHarmonicGeometry},
    ::Type{FT}; sampling = nothing, weights = nothing) where {FT}
    θpt, φpt = Grids._sph_points(g)
    return FT.(θpt), FT.(φpt), Grids._sph_point_weights(g, FT; sampling = sampling, weights = weights)
end

# A point's `cos(mφ)` and `sin(mφ)` for `m = 0…lmax`, so its `(lmax+1)²` harmonics read `2(lmax+1)` values
# in place of a trigonometric pair each.
@inline function _lon_trig!(cs::AbstractVector{FT}, sn::AbstractVector{FT}, φ::FT, lmax::Int) where {FT}
    @inbounds for m in 0:lmax
        sn[m + 1], cs[m + 1] = sincos(m * φ)
    end
    return nothing
end

# Real spherical harmonic in the FastSphericalHarmonics convention (`FSH.sph_evaluate`), off a point's
# `_lon_trig!` tables: Y_lm = s(m)·P̄_l^|m|·trig, with s(0)=1, s(m≠0)=(-1)^|m|√2, and trig = cos(mφ) for
# m≥0, sin(|m|φ) for m<0. `Plm` holds the normalized associated Legendre P̄_l^|m|(cosθ).
@inline function _real_sph_tab(Plm::AbstractMatrix{FT}, l::Int, m::Int, cs, sn) where {FT}
    abs_m = abs(m)
    P = @inbounds Plm[l + 1, abs_m + 1]
    m == 0 && return P
    s = isodd(abs_m) ? -sqrt(FT(2)) : sqrt(FT(2))
    return s * P * @inbounds(m > 0 ? cs[m + 1] : sn[abs_m + 1])
end

_lon_scratch(::Type{FT}, lmax::Int) where {FT} = (Vector{FT}(undef, lmax + 1), Vector{FT}(undef, lmax + 1))

# Structured spherical factorization, shared by the serial and threaded runs. `_sph_structured_setup`
# reads the grid's `(λ, θ)` axes + latitude quadrature; `_sph_longitude_dft!` is the once-per-`(θ, m)`
# longitude DFT (dense matmul, dependency-free); `_sph_order_accumulate!` runs the θ-Legendre contraction
# order by order. Same normalized-Legendre / factor / m↔col convention as `_real_sph_tab`; `_phi_bin`
# reads the |m| longitude bin (Re for `m ≥ 0`, −Im for `m < 0`).
function _sph_structured_setup(g, ::Type{FT}; sampling, weights, lmax = nothing) where {FT}
    λ = FlowGeometries.Grids.coordinates(g, 1)
    φlat = FlowGeometries.Grids.coordinates(g, 2)
    nlon = length(λ)
    nlat = length(φlat)
    N = nlon * nlat
    lmax === nothing || Grids._warn_longitude_bandlimit(nlon, lmax)
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
# `i(g-ĝ)/2`. A real field has `ĝ = conj(g)`, so they collapse to `real(g)` and `-imag(g)` and the
# partner transform is not built (`nothing` throughout this file). A complex field's `ĝ` is an
# independent sum.
@inline _phi_bin(m::Int, g, ::Nothing) = m >= 0 ? real(g) : -imag(g)
@inline _phi_bin(m::Int, g, ĝ) = m >= 0 ? (g + ĝ) / 2 : im * (g - ĝ) / 2

_partner_matrix(::Type{<:Complex}, Wφ) = conj(Wφ)
_partner_matrix(::Type{<:Real}, Wφ) = nothing
_partner_buffer(::Type{<:Complex}, ::Type{FT}, dims::Int...) where {FT} = Array{Complex{FT}}(undef, dims)
_partner_buffer(::Type{<:Real}, ::Type{FT}, dims::Int...) where {FT} = nothing
@inline _partner_at(::Nothing, a::Int, i::Int, b::Int) = nothing
Base.@propagate_inbounds _partner_at(f, a::Int, i::Int, b::Int) = f[a, i, b]

# Longitude DFT of every latitude row into a held buffer, `Wφ` the setup's matrix or its partner; a real
# field's partner is `nothing`.
function _sph_longitude_dft!(fhat, Wφ, field, nlon::Int, nlat::Int, B::Int)
    LA.mul!(reshape(fhat, size(Wφ, 1), nlat * B), Wφ, reshape(field, nlon, nlat * B))
    return fhat
end
_sph_longitude_dft!(::Nothing, ::Nothing, field, nlon::Int, nlat::Int, B::Int) = nothing

# Weighted longitude sum for a set of rings, `fh[a, r, b] = Σ_p w_p f_p e^{-i(a-1)λ_p}` over the points of
# ring `r` in `rows`, `field` holding `N` points per field. The exact discrete sum over each ring's own
# longitudes and weights, so the ring transform is the per-point projection reassociated. Rings own their
# columns of `fh`, so the serial and threaded ring paths differ only in how `rows` is split.
function _sph_ring_longitude!(fh, fh2, field, N::Int, rt, rows, lmax::Int, B::Int, ::Type{FT}) where {FT}
    wpt = rt.wpt
    @inbounds for r in rows
        for a in 1:(lmax + 1), b in 1:B
            fh[a, r, b] = zero(eltype(fh))
        end
        _zero_partner!(fh2, r, lmax, B)
        for p in rt.ranges[r]
            wp = FT(wpt[p])
            # A point of zero weight is skipped, so a masked cell's datum (commonly NaN) never enters.
            iszero(wp) && continue
            λp = rt.λ[p]
            for a in 1:(lmax + 1)
                e = cis(-FT(a - 1) * λp)
                we = wp * e
                for b in 1:B
                    fh[a, r, b] += field[p + (b - 1) * N] * we
                end
                _ring_partner!(fh2, a, r, B, field, N, p, wp * conj(e))
            end
        end
    end
    return fh
end

@inline _zero_partner!(::Nothing, r::Int, lmax::Int, B::Int) = nothing
Base.@propagate_inbounds function _zero_partner!(fh2, r::Int, lmax::Int, B::Int)
    for a in 1:(lmax + 1), b in 1:B
        fh2[a, r, b] = zero(eltype(fh2))
    end
    return nothing
end

@inline _ring_partner!(::Nothing, a::Int, r::Int, B::Int, field, N::Int, p::Int, e) = nothing
Base.@propagate_inbounds function _ring_partner!(fh2, a::Int, r::Int, B::Int, field, N::Int, p::Int, e)
    for b in 1:B
        fh2[a, r, b] += field[p + (b - 1) * N] * e
    end
    return nothing
end

# The θ-Legendre contraction, each row's longitude sum scaled by `wθ[row]`. `wθ = nothing` reads a
# longitude sum that already carries its points' weights, as the ring path's does.
@inline _row_weight(::Nothing, ::Type{FT}, i::Int) where {FT} = one(FT)
Base.@propagate_inbounds _row_weight(wθ, ::Type{FT}, i::Int) where {FT} = FT(wθ[i])

"""
    _order_starts(θ, tables, lmax) -> (ys, ks)

Every order's start `d^m_{m0} = y·2^{Kk}` at every row's colatitude `θ[j]`, `ys[m+1, j]` and `ks[m+1, j]`,
by [`SphericalKernels.legendre_start_step`](@ref) from `d^0_{00} = 1`.
"""
function _order_starts(θ::AbstractVector{FT}, tables, lmax::Int) where {FT}
    ys = Matrix{FT}(undef, lmax + 1, length(θ))
    ks = Matrix{Int32}(undef, lmax + 1, length(θ))
    @inbounds for (j, t) in enumerate(θ)
        s = sin(t)
        y, k = one(FT), Int32(0)
        ys[1, j], ks[1, j] = y, k
        for m in 1:lmax
            y, k = SphericalKernels.legendre_start_step(y, k, -s * tables.f[m])
            ys[m + 1, j], ks[m + 1, j] = y, k
        end
    end
    return ys, ks
end

"""
    _sph_order_accumulate!(coeffs, fhat, fhat2, θ, wθ, ys, ks, tables, orders, lmax, B, P) -> coeffs

The θ contraction for the orders `orders`: order `m` adds to its columns `±m` of `coeffs`
`(lmax+1, 2lmax+1, B)`, over every row `j`, the row's longitude bin times `wθ[j] s(m) P̄_ℓ^m(cos θ_j)` for
`ℓ = m … lmax`, the degrees run by `SphericalKernels.legendre_order!` into `P` from the starts `ys`, `ks`.
Each order owns its columns, so orders split across tasks write disjoint coefficients.
"""
function _sph_order_accumulate!(coeffs::AbstractArray, fhat, fhat2, θ::AbstractVector{FT}, wθ, ys, ks, tables,
        orders, lmax::Int, B::Int, P::AbstractVector{FT}) where {FT}
    Nθ = lmax + 1
    NC = Nθ * (2 * lmax + 1)
    @inbounds for m in orders
        sm = m == 0 ? one(FT) : (isodd(m) ? -sqrt(FT(2)) : sqrt(FT(2)))
        op = (sph_mode_index(m, m)[2] - 1) * Nθ - m
        on = (sph_mode_index(m, -m)[2] - 1) * Nθ - m
        for j in eachindex(θ)
            x, s = cos(θ[j]), sin(θ[j])
            SphericalKernels.legendre_order!(P, ℓ -> ℓ - m + 1, tables.α, tables.c, tables.γ, tables.nrm, m,
                lmax, ys[m + 1, j], ks[m + 1, j], x >= 0, SphericalKernels._one_minus_abs(x, s))
            w = _row_weight(wθ, FT, j) * sm
            for b in 1:B
                g = fhat[m + 1, j, b]
                ĝ = _partner_at(fhat2, m + 1, j, b)
                φp = w * _phi_bin(m, g, ĝ)
                ob = (b - 1) * NC
                for ℓ in m:lmax
                    coeffs[ob + op + ℓ + 1] += P[ℓ - m + 1] * φp
                end
                m == 0 && continue
                φn = w * _phi_bin(-m, g, ĝ)
                for ℓ in m:lmax
                    coeffs[ob + on + ℓ + 1] += P[ℓ - m + 1] * φn
                end
            end
        end
    end
    return coeffs
end

# =============================================================================
# Spherical setup / run split. `sph_setup` reads the grid (`materialize`, the latitude quadrature, the
# ring table, the longitude DFT matrix, the Legendre tables) and `sph_run!` transforms a field against
# what it returns. The one-shot composes the two and `DirectSumSphericalPlan` holds the setup. Buffers
# sized by the batch live in the setup; the one-shot reads the batch from `coeffs`.
# =============================================================================

"""
    sph_setup(layout, grid, T, lmax, B; sampling, weights) -> NamedTuple

Everything a spherical direct sum reads from `grid`: its materialized nodes, or its ring table, or its
longitude DFT matrix and latitude quadrature — whichever its layout uses — plus the Legendre recurrence
tables and the longitude buffer. `T` is the coefficient element type; a complex one adds the `exp(+imλ)`
partner transform.

Run it against a field with [`sph_run!`](@ref). `DirectSumSphericalPlan` holds it across executions.
"""
function sph_setup(::FlowGeometries.Grids.TensorRings, g, ::Type{T}, lmax::Int, B::Int;
        sampling = nothing, weights = nothing) where {T}
    FT = real(float(T))
    λ, θ, wθ, nlon, nlat, _ = _sph_structured_setup(g, FT; sampling = sampling, weights = weights,
        lmax = lmax)
    Wφ = _sph_longitude_matrix(FT, λ, lmax, nlon)
    tables = SphericalKernels.legendre_tables(FT, lmax)
    ys, ks = _order_starts(θ, tables, lmax)
    return (; Wφ, Wφc = _partner_matrix(T, Wφ), θ, wθ, nlon, nlat,
        fhat = Array{Complex{FT}, 3}(undef, lmax + 1, nlat, B),
        fhat2 = _partner_buffer(T, FT, lmax + 1, nlat, B),
        tables, ys, ks, Plm = Matrix{FT}(undef, lmax + 1, lmax + 1))
end

function sph_setup(::FlowGeometries.Grids.IsoLatitudeRings, g, ::Type{T}, lmax::Int, B::Int;
        sampling = nothing, weights = nothing) where {T}
    FT = real(float(T))
    rt = Grids._ring_table(g, FT; lmax = lmax, sampling = sampling, weights = weights)
    nr = length(rt.ranges)
    tables = SphericalKernels.legendre_tables(FT, lmax)
    ys, ks = _order_starts(rt.θ, tables, lmax)
    return (; rt, nr, N = length(rt.λ),
        fhat = zeros(Complex{FT}, lmax + 1, nr, B),
        fhat2 = _partner_buffer(T, FT, lmax + 1, nr, B),
        tables, ys, ks, Plm = Matrix{FT}(undef, lmax + 1, lmax + 1))
end

function sph_setup(::FlowGeometries.Grids.NoRings, g, ::Type{T}, lmax::Int, B::Int;
        sampling = nothing, weights = nothing) where {T}
    FT = real(float(T))
    θpt, φpt, wpt = _sph_point_data(g, FT; sampling = sampling, weights = weights)
    return (; θpt, φpt, wpt, N = length(θpt),
        tables = SphericalKernels.legendre_tables(FT, lmax),
        Plm = Matrix{FT}(undef, lmax + 1, lmax + 1), trig = _lon_scratch(FT, lmax))
end

"""
    sph_run!(coeffs, layout, setup, field, lmax, B) -> ks

Fill `coeffs` `(lmax+1, 2lmax+1, B)` with the spherical spectrum of `field` against a
[`sph_setup`](@ref) result, and return `(0:lmax, -lmax:lmax)`.

A tensor layout runs one longitude DFT for every latitude, a ring layout one per ring, and a point cloud
projects per node; all three then contract over θ against the Legendre tables.
"""
function sph_run!(coeffs::AbstractArray{C}, ::FlowGeometries.Grids.TensorRings, s, field::AbstractArray,
        lmax::Int, B::Int) where {C <: Number}
    _sph_longitude_dft!(s.fhat, s.Wφ, field, s.nlon, s.nlat, B)
    _sph_longitude_dft!(s.fhat2, s.Wφc, field, s.nlon, s.nlat, B)
    fill!(coeffs, zero(C))
    _sph_order_accumulate!(coeffs, s.fhat, s.fhat2, s.θ, s.wθ, s.ys, s.ks, s.tables, 0:lmax, lmax, B,
        view(s.Plm, :, 1))
    return (0:lmax, -lmax:lmax)
end

# Iso-latitude rings (HEALPix in RING order, ring and reduced-Gaussian grids, a masked or spheroid tensor
# grid): colatitude is constant along a ring, so the longitude sum runs once per `(ring, m)` and the
# θ-Legendre contraction over rings, `O(N·lmax + nrings·lmax²)`.
function sph_run!(coeffs::AbstractArray{C}, ::FlowGeometries.Grids.IsoLatitudeRings, s, field::AbstractArray,
        lmax::Int, B::Int) where {C <: Number}
    fill!(coeffs, zero(C))
    _sph_ring_longitude!(s.fhat, s.fhat2, field, s.N, s.rt, 1:s.nr, lmax, B, real(float(C)))
    _sph_order_accumulate!(coeffs, s.fhat, s.fhat2, s.rt.θ, nothing, s.ys, s.ks, s.tables, 0:lmax, lmax, B,
        view(s.Plm, :, 1))
    return (0:lmax, -lmax:lmax)
end

# A grid without iso-latitude rings — scattered nodes, the cubed sphere, the icosahedral geodesic,
# Yin-Yang, HEALPix in NESTED order — takes the per-point projection.
function sph_run!(coeffs::AbstractArray{CT}, ::FlowGeometries.Grids.NoRings, s, field::AbstractArray,
        lmax::Int, B::Int) where {CT <: Number}
    Nθc = lmax + 1
    Nφc = 2 * lmax + 1
    N = s.N
    F = reshape(field, N, B)
    C = reshape(coeffs, Nθc, Nφc, B)
    fill!(coeffs, zero(CT))
    @inbounds for p in 1:N
        _sph_project_point!(C, F, s.Plm, s.trig, s.tables, s.θpt[p], s.φpt[p], s.wpt[p], p, lmax, B)
    end
    return (0:lmax, -lmax:lmax)
end

# Node `p`'s weighted contribution to every coefficient. A node of zero weight contributes nothing, and
# skipping it keeps a masked cell's value out of the sum entirely: masked data is commonly NaN, and
# `0 * NaN` is NaN.
@inline function _sph_project_point!(C, F, Plm::AbstractMatrix{FT}, trig, tables, θp, φp, wp, p::Int,
        lmax::Int, B::Int) where {FT}
    iszero(wp) && return C
    SphericalKernels.fill_legendre!(Plm, tables, cos(θp), sin(θp), lmax)
    cs, sn = trig
    _lon_trig!(cs, sn, FT(φp), lmax)
    @inbounds for l in 0:lmax
        for m in -l:l
            gw = _real_sph_tab(Plm, l, m, cs, sn) * wp      # real SH (FSH convention)
            idx = sph_mode_index(l, m)
            for b in 1:B
                C[idx, b] += F[p, b] * gw
            end
        end
    end
    return C
end

# One-shot: set up against this grid, then run. A repeated caller holds the setup in a plan.
function _sph_direct!(layout, coeffs::AbstractArray{C}, g, field::AbstractArray, lmax::Int;
        sampling = nothing, weights = nothing) where {C <: Number}
    B = length(coeffs) ÷ ((lmax + 1) * (2 * lmax + 1))
    s = sph_setup(layout, g, C, lmax, B; sampling = sampling, weights = weights)
    return sph_run!(coeffs, layout, s, field, lmax, B)
end

"""
    sph_synth_setup(grid, ::Type{FT}, lmax) -> NamedTuple

Everything the spherical inverse reads from `grid`: its nodes, the Legendre recurrence tables and the
per-node `Plm` scratch. Synthesis carries no quadrature, so this is the node list only.

Run it against coefficients with [`sph_synth_run!`](@ref); `DirectSumSynthesisPlan` holds it.
"""
function sph_synth_setup(g, ::Type{FT}, lmax::Int) where {FT}
    θraw, φraw = Grids._sph_points(g)
    return (; θpt = FT.(θraw), φpt = FT.(φraw), N = length(θraw),
        tables = SphericalKernels.legendre_tables(FT, lmax),
        Plm = Matrix{FT}(undef, lmax + 1, lmax + 1), trig = _lon_scratch(FT, lmax))
end

"""
    sph_synth_run!(out, setup, coeffs, lmax) -> out

Evaluate the spherical harmonic sum at the setup's nodes, writing `out` `(N, batch…)`.
"""
function sph_synth_run!(out::AbstractArray{OT}, s, coeffs::AbstractArray, lmax::Int) where {OT <: Number}
    N = s.N
    B = length(out) ÷ N
    O = reshape(out, N, B)
    C = reshape(coeffs, lmax + 1, 2 * lmax + 1, B)
    fill!(O, zero(OT))
    @inbounds for p in 1:N
        _sph_synth_point!(O, C, s.Plm, s.trig, s.tables, s.θpt[p], s.φpt[p], p, lmax, B)
    end
    return out
end

# The harmonic sum at node `p`.
@inline function _sph_synth_point!(O, C, Plm::AbstractMatrix{FT}, trig, tables, θp, φp, p::Int, lmax::Int,
        B::Int) where {FT}
    SphericalKernels.fill_legendre!(Plm, tables, cos(θp), sin(θp), lmax)
    cs, sn = trig
    _lon_trig!(cs, sn, FT(φp), lmax)
    @inbounds for l in 0:lmax
        for m in -l:l
            Ylm = _real_sph_tab(Plm, l, m, cs, sn)          # real SH (FSH convention)
            idx = sph_mode_index(l, m)
            for b in 1:B
                O[p, b] += C[idx, b] * Ylm
            end
        end
    end
    return O
end

# One-shot: set up against this grid, then run.
_synthesize_spherical_direct!(
    out::AbstractArray{OT},
    g::FlowGeometries.Grids.AbstractGrid{<:Grids.SphericalHarmonicGeometry},
    coeffs::AbstractArray, lmax::Int,
) where {OT <: Number} =
    sph_synth_run!(out, sph_synth_setup(g, real(float(OT)), lmax), coeffs, lmax)

end # module DirectSum
