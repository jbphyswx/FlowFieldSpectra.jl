module Grids

using FlowGeometries: FlowGeometries
using FlowTransformBindings: FlowTransformBindings as FTB

export physical_wavenumbers

"""
    PointwiseCartesian

Cartesian grids whose coordinates are stored per point: a node cloud, and a curvilinear grid, whose
coordinate arrays hold one value per cell. Both index linearly in the field's own order. A tensor grid
has its own, more specific, factorized methods.
"""
const PointwiseCartesian = Union{
    FlowGeometries.Grids.AbstractUnstructuredGrid{<:FlowGeometries.Geometry.AbstractCartesianGeometry},
    FlowGeometries.Grids.AbstractCurvilinearGrid{<:FlowGeometries.Geometry.AbstractCartesianGeometry},
}

"""
    SphericalHarmonicGeometry

Geometries whose points are addressed by a direction, so a spherical-harmonic transform applies to a
field sampled on them: a sphere, and a spheroid.

A spheroid node's direction is its geocentric one, which [`_colatitude`](@ref) gives, and the transform
returns the field's surface expansion over those directions.
"""
const SphericalHarmonicGeometry = Union{
    FlowGeometries.Geometry.AbstractSphericalGeometry,
    FlowGeometries.Geometry.AbstractEllipsoidalGeometry,
}

# =============================================================================
# What the transforms read from a FlowGeometries grid: wavenumbers, quadrature, point lists, and the
# bridge from FlowGeometries' `(λ, φ_lat)` (longitude, geographic latitude) to the transform's `(θ, φ)`
# (colatitude, longitude).
# =============================================================================

# -----------------------------------------------------------------------------
# Physical wavenumbers.
# -----------------------------------------------------------------------------

# The layout depends on the field's realness, so the form without a `Val` raises.
@inline physical_wavenumbers(g::FlowGeometries.Grids.AbstractGrid, ms::NTuple{D, Int}) where {D} =
    throw(ArgumentError(
        "physical_wavenumbers needs the field's realness, which fixes the layout: pass `Val(true)` for " *
        "a real field (axis 1 halved to `ms[1]÷2+1` nonnegative modes) or `Val(false)` for a complex " *
        "one (every mode, native order). `calculate_spectrum` returns the matching axes as its second " *
        "value, so a caller reducing its coefficients uses those."))

"""
    physical_wavenumbers(Ls, box, ::Val{real}) -> NTuple
    physical_wavenumbers(grid, ms, ::Val{real}) -> NTuple

Packed native-order wavenumber axes matching the packed coefficient layout: for a real transform axis 1
is a nonnegative rfft axis (`FlowTransformBindings.RFFTAxis`, length `box[1]÷2+1`) and axes `2:D` are
full fft axes (`FlowTransformBindings.FFTAxis`); for a complex transform every axis is a full fft axis.
`box[d]` is the coefficient box's mode count along `d`, which on `grid` is the [`mode_box`](@ref) of an
`ms`-mode transform. A Fourier length of zero takes the spacing of unit length.
"""
@inline physical_wavenumbers(Ls::NTuple{D, <:Real}, box::NTuple{D, Int}, r::Val) where {D} =
    FTB.wavenumber_axes(map(L -> iszero(L) ? one(L) : L, Ls), box, r)

@inline function physical_wavenumbers(
        g::FlowGeometries.Grids.AbstractGrid{<:FlowGeometries.Geometry.AbstractCartesianGeometry},
        ms::NTuple{D, Int}, r::Val) where {D}
    FT = eltype(g)
    Ls = ntuple(d -> axis_range(FT, g, d), Val(D))
    return physical_wavenumbers(Ls, mode_box(g, ms, r), r)
end

# -----------------------------------------------------------------------------
# Cartesian quadrature. A transform estimates the domain average
# `C(k) = Σ_j w_j f_j e^{-i k·(x_j - x₀)} / Σ_j w_j` with `w` the grid's per-node measure and `x₀` each
# direction's smallest coordinate, the first sample of an ascending axis, so an FFT of the samples
# returns it with no phase applied. Every backend normalizes by the node count `N`, and
#
#     Σ_j w_j f_j e^{-ikx} / Σw  ==  (1/N) Σ_j (N w_j / Σw) f_j e^{-ikx},
#
# so scaling the field by `α_j = N w_j / Σw` once, before any backend sees it, turns each backend's
# `1/N` into the measure-weighted quadrature. `α ≡ 1` where the measure is constant, which covers every
# uniform grid and every scattered grid whose nodes carry equal weight.
# -----------------------------------------------------------------------------

"""
    quadrature_scale(grid, ::Type{FT}, N) -> Union{Nothing, AbstractVector}

Per-node factor `α_j = N·w_j/Σw` taking a node-count-normalized transform to the measure-weighted
quadrature over `grid`, or `nothing` where the measure is constant and `α_j` is exactly one. `N` is the
number of spatial samples the field carries; the result is flat, of length `N`, in the field's
column-major point order.

Read from `FlowGeometries.Grids.measure`, which keeps a rectilinear grid's `SeparableMeasure` and a ring
layout's `RingwiseVector` compact, so the result is lazy on such a grid.
"""
function quadrature_scale(g::FlowGeometries.Grids.AbstractGrid{<:FlowGeometries.Geometry.AbstractCartesianGeometry},
        ::Type{FT}, N::Integer) where {FT}
    m = FlowGeometries.Grids.measure(g)
    length(m) == N || throw(DimensionMismatch(
        "grid measure has $(length(m)) entries for $N samples"))
    lo, hi = extrema(m)
    lo == hi && return nothing                    # constant measure ⇒ α ≡ 1
    s = sum(m)
    (isfinite(s) && s > 0) || throw(ArgumentError(
        "grid measure sums to $s, so it cannot set quadrature weights"))
    return vec((FT(N) / FT(s)) .* m)
end

# -----------------------------------------------------------------------------
# Field shape and point lists, read from the grid.
# -----------------------------------------------------------------------------

"""
    field_batch_shape(grid, field) -> Tuple

Trailing batch sizes of `field` on `grid`: the dims after the leading `ndims(grid)` spatial ones, per the
`(spatial…, batch…)` contract `Problem` states. `ndims(grid)` is `1` for a node cloud and `N` for a
grid that indexes as an `N`-D array.
"""
field_batch_shape(g::FlowGeometries.Grids.AbstractGrid, field) =
    ntuple(i -> size(field, ndims(g) + i), ndims(field) - ndims(g))

"""
    axis_geometry(::Type{FT}, grid, D) -> (offsets, ranges)

Per-axis origin and Fourier length: `offsets[d]` is the smallest coordinate along direction `d`, the `x₀`
every phase is measured from, and `ranges[d]` the [`axis_range`](@ref) of direction `d`.
"""
function axis_geometry(::Type{FT}, g::FlowGeometries.Grids.AbstractGrid, D::Int) where {FT}
    offsets = ntuple(d -> FT(FlowGeometries.Grids.bounds(g, d)[1]), D)
    ranges = ntuple(d -> axis_range(FT, g, d), D)
    return offsets, ranges
end

# A transform of part of a grid's points takes the whole grid's origins and Fourier lengths as `box`, so
# the parts' coefficients sit on one wavenumber lattice and sum to the whole's.
axis_geometry(::Type{FT}, g::FlowGeometries.Grids.AbstractGrid, D::Int, ::Nothing) where {FT} =
    axis_geometry(FT, g, D)
axis_geometry(::Type{FT}, ::FlowGeometries.Grids.AbstractGrid, D::Int, box::Tuple) where {FT} =
    (map(FT, box[1]), map(FT, box[2]))

"""
    relative_coordinates(grid, offsets) -> NTuple

Each direction's coordinates less its origin, `x - x₀`: the positions a phase `e^{-ik·(x - x₀)}` reads.
"""
relative_coordinates(g::FlowGeometries.Grids.AbstractGrid, offsets::Tuple) =
    map((x, o) -> x .- o, FlowGeometries.Grids.coordinates(g), offsets)

"""
    axis_range(FT, grid, d) -> FT

Direction `d`'s Fourier length: the length `FlowGeometries.Grids.domain_length` gives the cells along
`d`, which is the period where `d` wraps. A transform weights each sample by its measure, and the
measure integrates over those cells, so the Fourier modes are periodic over the same length.
"""
@inline axis_range(::Type{FT}, g::FlowGeometries.Grids.AbstractGrid, d::Integer) where {FT} =
    FT(FlowGeometries.Grids.domain_length(g, d))

"""
    point_coordinates(::Type{FT}, grid, D) -> (coords::NTuple{D}, spatial::Tuple)

The `D` per-point coordinate vectors of `grid` in the field's own column-major order, and the spatial
shape a field on it carries. Read from `FlowGeometries.Grids.materialize`, which allocates `D·n` numbers
on a tensor grid; a repeated caller holds the result in a plan.
"""
function point_coordinates(::Type{FT}, g::FlowGeometries.Grids.AbstractGrid, D::Int) where {FT}
    pts = FlowGeometries.Grids.materialize(g)
    return ntuple(d -> _as_points(FT, pts[d]), D), size(g)
end

# Flat and of element type `FT`, converting only when one of the two differs (a node cloud's own vectors
# then pass through untouched).
_as_points(::Type{FT}, v::AbstractVector{FT}) where {FT} = v
_as_points(::Type{FT}, v::AbstractArray) where {FT} = FT.(vec(v))

# -----------------------------------------------------------------------------
# Spherical θ/φ/weights bridge. FlowGeometries stores `(λ, φ_lat)` = (longitude, geographic latitude);
# the transform wants `(θ, φ)` = (colatitude, longitude). Colatitude θ = π/2 − φ_lat.
# -----------------------------------------------------------------------------

"""
    _sph_points(grid) -> (θ, φ)

Per-point colatitude/longitude lists for any spherical `grid`, in the transform's
`(θ = colatitude, φ = longitude)` convention and the field's column-major order (longitude fastest on a
structured `(nlon, nlat)` grid). The nodes come from `FlowGeometries.Grids.materialize`, so a
pixelization yields its pixel centres.
"""
function _sph_points(g::FlowGeometries.Grids.AbstractGrid{<:FlowGeometries.Geometry.AbstractSphericalGeometry})
    λ, φlat = FlowGeometries.Grids.materialize(g)   # (longitude, geographic latitude) per node
    FT = eltype(g)
    half = FT(π) / 2
    return (half .- FT.(φlat)), FT.(λ)
end

# A spheroid stores the geodetic latitude, so each node's colatitude is read through `_colatitude`. A
# surface grid `(λ, φ)` is the domain of a spherical-harmonic expansion; a grid carrying height as a third
# direction is not.
function _sph_points(g::FlowGeometries.Grids.AbstractGrid{<:FlowGeometries.Geometry.AbstractEllipsoidalGeometry})
    pts = FlowGeometries.Grids.materialize(g)
    length(pts) == 2 || throw(ArgumentError(
        "a spherical-harmonic transform on a spheroid reads a SURFACE grid `(λ, φ)`; this grid carries " *
        "$(length(pts)) coordinate directions. A field over `(λ, φ, h)` expands one height level at a " *
        "time — build the surface grid and pass the levels as trailing batch dims."))
    geo = FlowGeometries.Grids.grid_geometry(g)
    FT = eltype(g)
    λ = FT.(pts[1])
    φ = pts[2]
    θ = similar(λ)
    @inbounds for i in eachindex(θ)
        θ[i] = _colatitude(geo, FT(φ[i]))
    end
    return θ, λ
end

"""
    _colatitude(geometry, φ) -> FT

Colatitude of the direction the surface point at latitude `φ` lies in: `π/2` less its
`Geometry.geocentric_latitude`, which is `φ` on a sphere and the direction of the geodetic `φ`'s point on
a spheroid. Longitude does not enter, so a latitude row is one iso-latitude ring.
"""
@inline _colatitude(geo::FlowGeometries.Geometry.AbstractGeometry, φ::FT) where {FT} =
    FT(π) / 2 - FT(FlowGeometries.Geometry.geocentric_latitude(geo, φ))

_colatitude(g::FlowGeometries.Grids.AbstractGrid, φ::FT) where {FT} =
    _colatitude(FlowGeometries.Grids.grid_geometry(g), φ)

# -----------------------------------------------------------------------------
# Spherical layout routing, from how FlowGeometries reports a grid's cells lying on iso-latitude rings.
# -----------------------------------------------------------------------------

"""
    _sph_layout(grid) -> FlowGeometries.Grids.AbstractRingStructure

Which spherical transform `grid` takes, named by `FlowGeometries.Grids.ring_structure`: `TensorRings()`
is the tensor path, one longitude matrix shared by complete latitude rows under a sphere's latitude
quadrature; `IsoLatitudeRings()` the ring path, a longitude sum per ring; `NoRings()` the per-point
projection. A tensor grid with a mask, whose rows are incomplete, or on a spheroid, whose latitudes are
geodetic, takes the ring path.
"""
_sph_layout(g::FlowGeometries.Grids.AbstractGrid) = _sph_layout(FlowGeometries.Grids.ring_structure(g), g)
_sph_layout(s::FlowGeometries.Grids.AbstractRingStructure, _) = s
_sph_layout(::FlowGeometries.Grids.TensorRings, g) =
    _tensor_layout(FlowGeometries.Grids.mask(g), FlowGeometries.Grids.grid_geometry(g))

_tensor_layout(::FlowGeometries.Grids.AllActive, ::FlowGeometries.Geometry.AbstractSphericalGeometry) =
    FlowGeometries.Grids.TensorRings()
_tensor_layout(_, _) = FlowGeometries.Grids.IsoLatitudeRings()

# The sampling whose band limit the ring path checks a degree against. A sampling states its rule for a
# sphere's colatitudes, and a spheroid's directions are the geocentric ones, so a spheroid grid states none.
_quadrature_sampling(g::FlowGeometries.Grids.AbstractGrid) = FlowGeometries.Grids.sampling(g)
_quadrature_sampling(::FlowGeometries.Grids.AbstractGrid{<:FlowGeometries.Geometry.AbstractEllipsoidalGeometry}) =
    nothing

# -----------------------------------------------------------------------------
# Masked fields. A grid's mask says which cells participate; the extent a transform integrates over is
# `FlowGeometries.Grids.covered_area`.
# -----------------------------------------------------------------------------

"""
    _zeroed_inactive(field, grid) -> AbstractArray

`field` with its inactive cells set to zero, or `field` itself where every cell is active.

A transform consuming complete longitude rows (a library SHT on its own grid, a longitude `rfft`) is
linear in the field, so a zeroed datum drops out of its sum exactly, and a `NaN` under the mask is
cleared. The weights stay the sampling's own.
"""
_zeroed_inactive(field, g) = _zeroed_inactive(field, FlowGeometries.Grids.mask(g))

_zeroed_inactive(field, ::FlowGeometries.Grids.AllActive) = field

function _zeroed_inactive(field, mask::AbstractArray{Bool})
    out = similar(field)
    copyto!(out, field)
    N = length(mask)
    B = length(out) ÷ N
    z = zero(eltype(out))
    @inbounds for b in 1:B, i in 1:N
        mask[i] || (out[i + (b - 1) * N] = z)
    end
    return out
end

"""
    AbstractInactiveCells

How a masked spherical grid's inactive cells enter a transform, passed as `inactive`: [`ZeroFill`](@ref)
or [`Exclude`](@ref).
"""
abstract type AbstractInactiveCells end

"""
    ZeroFill <: AbstractInactiveCells

Inactive cells count as zero: the transform of the field extended by zero over the whole grid. The
quadrature reads the grid's own weights, the inactive cells' contributing nothing, and the least-squares
fit (`solve = true`) runs over every node, an inactive one reading zero.
"""
struct ZeroFill <: AbstractInactiveCells end

"""
    Exclude <: AbstractInactiveCells

Inactive cells are left out: the quadrature or the fit over the active cells as their own point set. The
quadrature reads the active cells' weights rescaled to the whole grid's total ([`_inactive_scale`](@ref)),
as a grid of the active cells alone weighs them, and the least-squares fit (`solve = true`) runs over the
active nodes.
"""
struct Exclude <: AbstractInactiveCells end

"""
    _inactive_scale(inactive, grid, ::Type{FT}; sampling = nothing, weights = nothing) -> FT

The factor on a spherical quadrature over `grid` under `inactive`: one under [`ZeroFill`](@ref) or with
every cell active, and under [`Exclude`](@ref) the total of the grid's weights unmasked over the total of
its active cells' ([`_sph_point_weights`](@ref)).
"""
_inactive_scale(::ZeroFill, g, ::Type{FT}; sampling = nothing, weights = nothing) where {FT} = one(FT)
_inactive_scale(::Exclude, g, ::Type{FT}; sampling = nothing, weights = nothing) where {FT} =
    _inactive_scale(FlowGeometries.Grids.mask(g), g, FT, sampling, weights)
_inactive_scale(::FlowGeometries.Grids.AllActive, g, ::Type{FT}, sampling, weights) where {FT} = one(FT)
_inactive_scale(::AbstractArray{Bool}, g, ::Type{FT}, sampling, weights) where {FT} =
    _unmasked_total(g, FT, sampling, weights) / FT(sum(_sph_point_weights(g, FT; sampling, weights)))

# The total of a spherical grid's weights with every cell active: the latitude rule times `2π` on a
# structured sphere, `4π` for the grid's own node weights, the sum of explicit ones.
_unmasked_total(g::FlowGeometries.Grids.AbstractStructuredGrid{<:FlowGeometries.Geometry.AbstractSphericalGeometry},
    ::Type{FT}, sampling, weights) where {FT} =
    FT(2π) * FT(sum(_sht_weights(g, length(FlowGeometries.Grids.coordinates(g, 2)); sampling, weights)))
_unmasked_total(::FlowGeometries.Grids.AbstractGrid, ::Type{FT}, sampling, weights) where {FT} =
    weights === nothing ? FT(4π) : FT(sum(weights))

"""
    _inactive_weights(inactive, grid, ::Type{FT}; sampling = nothing, weights = nothing) -> weights

The `weights` a spherical quadrature over `grid` reads under `inactive`: `weights` itself where
[`_inactive_scale`](@ref) is one, and otherwise the grid's weights times it, per latitude on a structured
sphere and per node elsewhere, as `weights` is read there.
"""
function _inactive_weights(s::AbstractInactiveCells, g, ::Type{FT}; sampling = nothing,
        weights = nothing) where {FT}
    c = _inactive_scale(s, g, FT; sampling, weights)
    isone(c) && return weights
    return c .* _base_weights(g, FT, sampling, weights)
end

_base_weights(g::FlowGeometries.Grids.AbstractStructuredGrid{<:FlowGeometries.Geometry.AbstractSphericalGeometry},
    ::Type{FT}, sampling, weights) where {FT} =
    FT.(_sht_weights(g, length(FlowGeometries.Grids.coordinates(g, 2)); sampling, weights))
_base_weights(g::FlowGeometries.Grids.AbstractGrid, ::Type{FT}, sampling, weights) where {FT} =
    collect(FT, _sph_point_weights(g, FT; sampling, weights))

"""
    _transform_nodes(inactive, grid, solve) -> (act, live)

The nodes a spherical transform over `grid` holds under `inactive`, `act` their indices (`nothing` for
every node), and which of them read the field, `live` a flag per held node (`nothing` where every one
does). A quadrature gives an inactive node no weight, so it holds the active nodes. A least-squares fit
(`solve`) holds every node under [`ZeroFill`](@ref), an inactive one reading zero, and the active nodes
under [`Exclude`](@ref).
"""
_transform_nodes(s::AbstractInactiveCells, g::FlowGeometries.Grids.AbstractGrid, solve::Bool) =
    _transform_nodes(s, FlowGeometries.Grids.mask(g), solve)
_transform_nodes(::AbstractInactiveCells, ::FlowGeometries.Grids.AllActive, ::Bool) = (nothing, nothing)
_transform_nodes(s::AbstractInactiveCells, mask::AbstractArray{Bool}, solve::Bool) =
    (solve && s isa ZeroFill) ? (nothing, vec(mask)) : (findall(vec(mask)), nothing)

# `v` at the points `act`, or `v` itself where every point is active.
_at_points(v::AbstractVector, ::Nothing) = v
_at_points(v::AbstractVector, act::AbstractVector{Int}) = v[act]

# -----------------------------------------------------------------------------
# The coefficient box of a Cartesian transform.
# -----------------------------------------------------------------------------

"""
    mode_box(grid, ms, ::Val{real}) -> NTuple

The extents of the coefficient box an `ms`-mode Cartesian transform of a real field on `grid` fills: an
even `ms[d]` on an axis whose samples tell `+ms[d]/2` from `-ms[d]/2` apart widens to `ms[d] + 1`
(`FlowTransformBindings.box_size`), so that the packed half reaches every conjugate partner. A uniform axis
of a structured grid transformed at its own length samples the two as one mode and keeps `ms[d]`. A
complex field's box is `ms`.
"""
mode_box(g, ms::NTuple{D, Int}, ::Val{true}) where {D} =
    FTB.box_size(ms, ntuple(d -> !_nyquist_shared(g, d, ms[d]), Val(D)))
mode_box(g, ms::NTuple{D, Int}, ::Val{false}) where {D} = ms

"""
    series_modes(grid, ms, ::Val{real}) -> NTuple

The mode counts, in FFT order, of the complex series an `ms`-mode synthesis on `grid` evaluates: for a
real field `mode_box(grid, ms, Val(true))[1]` on axis 1 and `ms` on the others, whose real part, with the
coefficients `DirectSum.native_coefficients!` writes, is the series of the packed half over the whole
coefficient box; `ms` for a complex field.
"""
series_modes(g, ms::NTuple{D, Int}, ::Val{true}) where {D} = (mode_box(g, ms, Val(true))[1], Base.tail(ms)...)
series_modes(g, ms::NTuple{D, Int}, ::Val{false}) where {D} = ms

_nyquist_shared(g::FlowGeometries.Grids.AbstractStructuredGrid, d::Int, m::Int) =
    FlowGeometries.Grids.isuniform(g, d) && m == size(g, d)
_nyquist_shared(::FlowGeometries.Grids.AbstractGrid, ::Int, ::Int) = false

"""
    _warn_bandlimit(sampling, nlat, lmax)

Warn, once per logger, when degree `lmax` exceeds what `nlat` latitude samples of `sampling` resolve; the
coefficients are still returned. The relation is the sampling's own `SphericalSampling.bandlimit`:
`nlat÷2 − 1` for Driscoll–Healy, `nlat − 1` for Gauss–Legendre and Clenshaw–Curtis.
"""
function _warn_bandlimit(s, nlat::Integer, lmax::Integer)
    lim = _bandlimit_or_nothing(s, nlat)
    (lim === nothing || lmax <= lim) && return nothing
    @warn("requested spherical degree exceeds what this grid's sampling resolves; coefficients above " *
          "its band limit are not supported by its quadrature",
        sampling = nameof(typeof(s)), nlat, lmax, bandlimit = lim,
        exact_quadrature = FlowGeometries.SphericalSampling.admits_exact_bandlimited_quadrature(s),
        maxlog = 1)
    return nothing
end

"""
    _warn_longitude_bandlimit(nlon, lmax)

Warn when `nlon` longitudes cannot separate the orders `|m| ≤ lmax`. The trigonometric polynomials of
degree `lmax` span `2lmax + 1` dimensions, so on fewer longitudes one of them vanishes at every node and
the orders alias; on equispaced longitudes `cos(mλ)` and `cos((nlon − m)λ)` take the same samples. Warned,
as [`_warn_bandlimit`](@ref) warns for the latitudes.
"""
function _warn_longitude_bandlimit(nlon::Integer, lmax::Integer)
    2 * lmax + 1 <= nlon && return nothing
    @warn("requested spherical order exceeds what this grid's longitudes resolve; orders above " *
          "(nlon - 1) ÷ 2 alias onto lower ones", nlon, lmax, resolved = (nlon - 1) ÷ 2, maxlog = 1)
    return nothing
end

# A sampling states a band-limit relation only where it has one: an equal-area pixelization defines no
# `bandlimit` method, a lat–lon grid's is user-chosen and raises, and Driscoll–Healy's requires an even
# latitude count. Absent a relation there is nothing to check against.
function _bandlimit_or_nothing(s, nlat::Integer)
    hasmethod(FlowGeometries.SphericalSampling.bandlimit, Tuple{typeof(s), Int}) || return nothing
    try
        return FlowGeometries.SphericalSampling.bandlimit(s, nlat)
    catch e
        e isa ArgumentError && return nothing
        rethrow()
    end
end

"""
    _ring_table(grid, ::Type{FT}; lmax=nothing, sampling=nothing, weights=nothing) -> (; ranges, θ, λ, wpt)

Per iso-latitude ring: its slice of the flattened point vector and its colatitude, read at the ring's first
point; plus every point's longitude and quadrature weight. The weights are
[`_sph_point_weights`](@ref)`(grid, FT; sampling, weights)`, the ones the per-point projection reads, so
the ring transform is that projection reassociated for every quadrature a caller can state.
"""
function _ring_table(g, ::Type{FT}; lmax = nothing, sampling = nothing, weights = nothing) where {FT}
    nr = FlowGeometries.Grids.nrings(g)
    if lmax !== nothing
        s = sampling === nothing ? _quadrature_sampling(g) : sampling
        s === nothing || _warn_bandlimit(s, nr, lmax)
    end
    ranges = Vector{UnitRange{Int}}(undef, nr)
    θ = Vector{FT}(undef, nr)
    @inbounds for r in 1:nr
        ranges[r] = FlowGeometries.Grids.ring_range(g, r)
        θ[r] = _colatitude(g, FT(FlowGeometries.Grids.ring_latitude(g, r)))
    end
    return (; ranges, θ, λ = FT.(first(FlowGeometries.Grids.materialize(g))),
        wpt = _sph_point_weights(g, FT; sampling = sampling, weights = weights))
end

"""
    _sht_weights(grid, nlat; sampling=nothing, weights=nothing, lmax=nothing) -> Vector

Per-colatitude quadrature weights for a structured spherical `grid`, from its own node-set recipe
`FlowGeometries.Grids.sampling(grid)` via `SphericalSampling.latitude_weights` (`Σw = 2`, carrying the
`sinθ` Jacobian). `weights` or `sampling` override it; `lmax` is checked against the band limit.

Coordinates alone do not fix the quadrature, so a grid built from raw axes, which records no recipe,
raises unless the caller states one.
"""
function _sht_weights(g::FlowGeometries.Grids.AbstractGrid{<:SphericalHarmonicGeometry},
        nlat::Integer;
        sampling::Union{Nothing, FlowGeometries.SphericalSampling.AbstractSphericalSampling} = nothing,
        weights = nothing, lmax = nothing)
    weights !== nothing && return weights
    s = sampling === nothing ? FlowGeometries.Grids.sampling(g) : sampling
    s isa FlowGeometries.SphericalSampling.AbstractSphericalSampling || throw(ArgumentError(
        "this spherical grid records no sampling recipe, so its latitude quadrature is undetermined " *
        "(coordinates alone do not fix it). Build it from a sampling " *
        "(`structured_grid(GaussLegendreSampling(), nlat)`), or pass `sampling=` / `weights=`."))
    lmax === nothing || _warn_bandlimit(s, nlat, lmax)
    return FlowGeometries.SphericalSampling.latitude_weights(eltype(g), s, nlat)
end

"""
    _sph_point_weights(grid, ::Type{FT}; sampling=nothing, weights=nothing) -> AbstractVector{FT}

Per-point quadrature weights of a spherical `grid` in the field's column-major order, an inactive point at
zero: on a structured sphere the latitude rule of [`_sht_weights`](@ref) times `2π/nlon`, and on every
other grid the per-node weights of [`_sph_node_weights`](@ref). Both total `4π` over an unmasked grid.
`sampling` and `weights` override the grid's own rule as those two state, so `weights` holds one value per
latitude on a structured sphere and one per node elsewhere.
"""
function _sph_point_weights(
        g::FlowGeometries.Grids.AbstractStructuredGrid{<:FlowGeometries.Geometry.AbstractSphericalGeometry},
        ::Type{FT}; sampling = nothing, weights = nothing) where {FT}
    nlon = length(FlowGeometries.Grids.coordinates(g, 1))
    nlat = length(FlowGeometries.Grids.coordinates(g, 2))
    wlat = _sht_weights(g, nlat; sampling = sampling, weights = weights)
    length(wlat) == nlat || throw(DimensionMismatch("$(length(wlat)) latitude weights for $nlat latitudes"))
    dλ = FT(2π) / nlon
    mask = FlowGeometries.Grids.mask(g)
    wpt = Vector{FT}(undef, nlon * nlat)
    # Point `p` is `iλ + (jφ-1)·nlon`, the mask's own linear order.
    @inbounds for p in eachindex(wpt)
        wpt[p] = mask[p] ? FT(wlat[(p - 1) ÷ nlon + 1]) * dλ : zero(FT)
    end
    return wpt
end

_sph_point_weights(g::FlowGeometries.Grids.AbstractGrid{<:SphericalHarmonicGeometry}, ::Type{FT};
    sampling = nothing, weights = nothing) where {FT} = _sph_node_weights(g, FT, length(g), weights)

"""
    _sph_node_weights(grid, ::Type{FT}, N, weights) -> Vector{FT}

Per-node quadrature weights for a scattered spherical `grid`, from its own per-node measure rescaled to
sum to `4π` — the transform's convention, the same total the structured path's
`latitude_weights ⊗ 2π/nlon` carries. `weights` overrides it, one per node, an inactive node still at
zero. Only the relative measures matter, so a grid of true dual-cell areas integrates exactly and a grid
of equal-area nodes gives `4π/N`.
"""
function _sph_node_weights(g::FlowGeometries.Grids.AbstractGrid{<:SphericalHarmonicGeometry},
        ::Type{FT}, N::Integer, weights) where {FT}
    if weights !== nothing
        length(weights) == N || throw(DimensionMismatch("$(length(weights)) weights for $N nodes"))
        return _masked_weights(FlowGeometries.Grids.mask(g), FT.(weights), one(FT), FT)
    end
    m = FlowGeometries.Grids.measure(g)
    length(m) == N || throw(DimensionMismatch(
        "grid measure has $(length(m)) entries for $N nodes"))
    s = sum(m)
    (isfinite(s) && s > 0) || throw(ArgumentError(
        "grid measure sums to $s, so it cannot set quadrature weights; pass `weights=`."))
    # Divided by the whole grid's measure, each cell's measure is its solid angle on the unit sphere, and
    # the active ones total the covered solid angle: `4π` unmasked, `4π·f_sky` under a mask.
    return _masked_weights(FlowGeometries.Grids.mask(g), m, FT(4π) / FT(s), FT)
end

# No mask: every cell participates, so the scaled measure stands as it is (lazy where separable).
_masked_weights(::FlowGeometries.Grids.AllActive, m, c, ::Type{FT}) where {FT} = vec(c .* m)

# An inactive node gets weight zero, which drops it from every quadrature sum reading these weights. The
# zero pattern is arbitrary, so this materializes.
function _masked_weights(mask::AbstractArray{Bool}, m, c, ::Type{FT}) where {FT}
    w = Vector{FT}(undef, length(m))
    @inbounds for i in eachindex(w)
        w[i] = mask[i] ? FT(m[i]) * c : zero(FT)
    end
    return w
end

end # module Grids
