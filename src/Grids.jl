module Grids

using FlowGeometries: FlowGeometries
using ..Packing: Packing

export physical_wavenumbers

"""
    PointwiseCartesian

Cartesian grids whose coordinates are stored per point: a node cloud, and a curvilinear grid, whose
coordinate arrays hold one value per cell.

Both index linearly in the field's own order, so a transform reads them identically — the direct sum, its
Nyquist twin, both inverses, and a NUFFT provider's point list. A tensor grid stores axes instead and has
its own factorized methods, which are more specific than this.
"""
const PointwiseCartesian = Union{
    FlowGeometries.Grids.AbstractUnstructuredGrid{<:FlowGeometries.Geometry.AbstractCartesianGeometry},
    FlowGeometries.Grids.AbstractCurvilinearGrid{<:FlowGeometries.Geometry.AbstractCartesianGeometry},
}

"""
    SphericalHarmonicGeometry

Geometries whose points are addressed by a DIRECTION, so a spherical-harmonic transform applies to a
field sampled on them: a sphere, and a spheroid.

A spherical harmonic is orthonormal over directions, and a spheroid node's direction is its geocentric
one, which [`_colatitude`](@ref) reads through the geometry's own embedding. What such a transform
returns is the SURFACE expansion of the field: the spheroid's radius varies with latitude, and that
radial dependence belongs to the solid-harmonic expansion, which is a different transform.
"""
const SphericalHarmonicGeometry = Union{
    FlowGeometries.Geometry.AbstractSphericalGeometry,
    FlowGeometries.Geometry.AbstractEllipsoidalGeometry,
}

# =============================================================================
# FlowGeometries adapter. FFS builds directly on FlowGeometries' grid/geometry types — a Cartesian grid
# is a `StructuredGrid`/`UnstructuredGrid` over a `CartesianGeometry`, a spherical grid the same over a
# `SphericalGeometry`. Callers construct grids with FlowGeometries; FFS reads them with FlowGeometries'
# own accessors (`size`, `ndims`, `length`, `coordinates`, `isuniform`, `period`, …). There is no FFS
# grid-construction or interface wrapper layer. This module holds only the two pieces FlowGeometries has
# no equivalent for:
#   • `physical_wavenumbers` — the spectral wavenumber grid (reads FlowGeometries' `period` for `L`);
#   • the θ/φ/quadrature bridge from FlowGeometries' `(λ, φ_lat)` = (longitude, geographic latitude) to
#     the transform's `(θ, φ)` = (colatitude, longitude).
# =============================================================================

# -----------------------------------------------------------------------------
# Physical wavenumbers (spectral; period-based). No FlowGeometries equivalent.
# -----------------------------------------------------------------------------

# A caller has to say whether the field is real, since that decides the layout: a real transform halves
# axis 1 and a complex one keeps every mode. The two-argument form states that requirement.
@inline physical_wavenumbers(g::FlowGeometries.Grids.AbstractGrid, ms::NTuple{D, Int}) where {D} =
    throw(ArgumentError(
        "physical_wavenumbers needs the field's realness, which fixes the layout: pass `Val(true)` for " *
        "a real field (axis 1 halved to `ms[1]÷2+1` nonnegative modes) or `Val(false)` for a complex " *
        "one (every mode, native order). `calculate_spectrum` returns the matching axes as its second " *
        "value, so a caller reducing its coefficients uses those."))

"""
    physical_wavenumbers(Ls, ms, ::Val{real}) -> NTuple
    physical_wavenumbers(grid, ms, ::Val{real}) -> NTuple

Packed native-order wavenumber axes matching the packed coefficient layout: for a real transform axis 1
is a nonnegative rfft axis (`Packing.RFFTAxis`, length `ms[1]÷2+1`) and axes `2:D` are full fft axes
(`Packing.FFTAxis`); for a complex transform every axis is a full fft axis. `ms[d]` is the full
transform length along `d`.
"""
@inline physical_wavenumbers(Ls::NTuple{D, <:Real}, ms::NTuple{D, Int}, ::Val{R}) where {D, R} =
    ntuple(d -> (d == 1 && R) ? Packing.RFFTAxis(Packing._axis_scale(Ls[d]), ms[d]) :
                                Packing.FFTAxis(Packing._axis_scale(Ls[d]), ms[d]), Val(D))

@inline function physical_wavenumbers(
        g::FlowGeometries.Grids.AbstractGrid{<:FlowGeometries.Geometry.AbstractCartesianGeometry},
        ms::NTuple{D, Int}, r::Val) where {D}
    FT = eltype(g)
    Ls = ntuple(d -> axis_range(FT, g, d), Val(D))
    return physical_wavenumbers(Ls, ms, r)
end

# -----------------------------------------------------------------------------
# Cartesian quadrature. A transform estimates the domain average
# `C(k) = Σ_j w_j f_j e^{-i k·x_j} / Σ_j w_j` with `w` the grid's per-node measure. Every backend here
# normalizes by the plain node count instead, and
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
number of spatial samples the field carries. Flat and of length `N`, indexed by the same column-major
point order the field and the coordinates use, so it reads against a strengths vector and stages to a
device as one.

Read from `Grids.measure`, never `measure_array`: a rectilinear grid's measure is a `SeparableMeasure`
and a ring layout's a `RingwiseVector`, whose `sum` and `extrema` cost `O(Σ Nᵈ)` and `O(nrings)` against
a dense `O(∏ Nᵈ)`, and whose scaling by a constant stays in that representation. The result is therefore
lazy on such a grid, carrying `O(Σ Nᵈ)` numbers for `N` entries.
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
grid that indexes as an `N`-D array, so a path serving both reads its batch from here.
"""
field_batch_shape(g::FlowGeometries.Grids.AbstractGrid, field) =
    ntuple(i -> size(field, ndims(g) + i), ndims(field) - ndims(g))

"""
    axis_geometry(::Type{FT}, grid, D) -> (offsets, ranges)

Per-axis origin and Fourier length: `offsets[d]` is the smallest coordinate along direction `d`, and
`ranges[d]` its wrap length where the direction wraps and `1` where it does not.

`offsets` comes from `FlowGeometries.Grids.bounds`, which is `O(1)` on a structured grid (an axis is
monotone, so its extremes are its endpoints) against an `O(N)` scan of the coordinates.

`ranges` reads `isperiodic` before `period`, whose own contract states it is meaningful only where the
direction wraps: a grid may declare a direction non-periodic while carrying a nonzero `period` entry, so
the flag decides. A non-periodic direction gets `1`, making its wavenumbers raw per-sample.
"""
function axis_geometry(::Type{FT}, g::FlowGeometries.Grids.AbstractGrid, D::Int) where {FT}
    offsets = ntuple(d -> FT(FlowGeometries.Grids.bounds(g, d)[1]), D)
    ranges = ntuple(d -> axis_range(FT, g, d), D)
    return offsets, ranges
end

"""`axis_range(FT, grid, d)` — direction `d`'s Fourier length: its wrap period, or `1` where it does not wrap."""
@inline axis_range(::Type{FT}, g::FlowGeometries.Grids.AbstractGrid, d::Integer) where {FT} =
    FlowGeometries.Grids.isperiodic(g, d) ? FT(FlowGeometries.Grids.period(g, d)) : one(FT)

"""
    point_coordinates(::Type{FT}, grid, D) -> (coords::NTuple{D}, spatial::Tuple)

The `D` per-point coordinate vectors of `grid` in the field's own column-major order, and the spatial
shape a field on it carries.

`FlowGeometries.Grids.materialize` answers this for every architecture: a node cloud returns its nodes,
a tensor grid the expansion of its axes, a curvilinear grid its per-cell values, and a pixelization its
pixel centres. It allocates `D·n` numbers on a tensor grid (its docstring says so), so a repeated caller
holds the result in a plan.
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
`(θ = colatitude, φ = longitude)` convention.

The nodes come from `FlowGeometries.Grids.materialize`, which every grid architecture answers: on an
unstructured grid it returns the nodes themselves, on a structured `(nlon, nlat)` grid the expansion in
column-major order (longitude fastest, matching `size` and the field layout), and on a pixelization
(HEALPix, cubed-sphere, icosahedral, ring, Yin-Yang) the pixel centres. Those pixelizations answer
neither `coordinates(grid, d)` nor `sampling(grid)`, so `materialize` is the accessor that spans them.
"""
function _sph_points(g::FlowGeometries.Grids.AbstractGrid{<:FlowGeometries.Geometry.AbstractSphericalGeometry})
    λ, φlat = FlowGeometries.Grids.materialize(g)   # (longitude, geographic latitude) per node
    FT = eltype(g)
    half = FT(π) / 2
    return (half .- FT.(φlat)), FT.(λ)
end

# A spheroid stores the GEODETIC latitude, which is not the colatitude of the direction its node lies in,
# so each node's colatitude is read through `_colatitude`. A surface grid `(λ, φ)` is the domain of a
# spherical-harmonic expansion; a grid carrying height as a third direction is not.
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
        θ[i] = _colatitude(geo, λ[i], FT(φ[i]))
    end
    return θ, λ
end

"""
    _colatitude(geometry, λ, φ) -> FT

Colatitude of the direction the node `(λ, φ)` lies in.

On a sphere that is `π/2 − φ`. On a spheroid the stored `φ` is the geodetic latitude, whose difference
from the geocentric one reaches ~0.19° at mid-latitudes on Earth's ellipsoid, so the node is embedded
through the geometry's own `geodetic_to_cartesian` and the direction read back from it. The result is
independent of `λ` on both (a spheroid is a surface of revolution), so a latitude row remains one
iso-latitude ring and the ring-factorized transform still applies.
"""
@inline _colatitude(::FlowGeometries.Geometry.AbstractSphericalGeometry, λ::FT, φ::FT) where {FT} =
    FT(π) / 2 - φ

@inline function _colatitude(geo::FlowGeometries.Geometry.AbstractEllipsoidalGeometry,
        λ::FT, φ::FT) where {FT}
    p = FlowGeometries.Geometry.geodetic_to_cartesian(geo, (λ, φ))
    r = hypot(p.x, p.y, p.z)
    return acos(clamp(FT(p.z) / FT(r), -one(FT), one(FT)))
end

# The colatitude of a whole latitude row, which both factorized paths read once per ring.
_colatitude(g::FlowGeometries.Grids.AbstractGrid, φ::FT) where {FT} =
    _colatitude(FlowGeometries.Grids.grid_geometry(g), zero(FT), φ)

# -----------------------------------------------------------------------------
# Spherical layout routing. A sampling declares its own structure
# (`SphericalSampling.is_tensor_product` / `is_iso_latitude`), and the transform picks its algorithm from
# that. HEALPix and the reduced-Gaussian rings are iso-latitude without being tensor products, so a split
# on the grid's Julia type alone cannot express them and sends them to the per-point projection where a
# per-ring factorization applies.
# -----------------------------------------------------------------------------

"""Tensor-product `(nlon, nlat)` sphere: one longitude axis shared by every latitude."""
struct TensorSphere end

"""Iso-latitude rings whose longitude count varies by ring (HEALPix, reduced Gaussian)."""
struct RingSphere end

"""No iso-latitude structure: a point cloud (Fibonacci, cubed-sphere, icosahedral, Yin-Yang)."""
struct ScatteredSphere end

"""
    _sph_sampling(grid) -> AbstractSphericalSampling or nothing

The grid's own node-set recipe. A structured grid records one and carries it in its TYPE, so the traits
read from it fold at compile time. A `HEALPixGrid` answers from its resolution parameter, `nside` fixing
its recipe entirely. Anything else records none.
"""
_sph_sampling(g::FlowGeometries.Grids.AbstractGrid) = FlowGeometries.Grids.sampling(g)
_sph_sampling(g::FlowGeometries.Grids.HEALPixGrid) =
    FlowGeometries.SphericalSampling.HEALPixSampling(FlowGeometries.Grids.nside(g))

"""
    _sph_layout(grid) -> TensorSphere | RingSphere | ScatteredSphere

Which spherical algorithm `grid` admits, taken from its sampling's declared traits.
"""
function _sph_layout(g::FlowGeometries.Grids.AbstractGrid{<:FlowGeometries.Geometry.AbstractSphericalGeometry})
    s = _sph_sampling(g)
    s === nothing && return ScatteredSphere()
    FlowGeometries.SphericalSampling.is_tensor_product(s) && return TensorSphere()
    FlowGeometries.SphericalSampling.is_iso_latitude(s) && return RingSphere()
    return ScatteredSphere()
end

# A structured spherical grid stores `(λ, φ)` axes, so it is a tensor product by construction, whether or
# not it records the sampling that generated them. A grid built from raw axes therefore stays on the
# tensor path and reaches `_sht_weights`, which states that its latitude quadrature is undetermined.
_sph_layout(g::FlowGeometries.Grids.AbstractStructuredGrid{<:FlowGeometries.Geometry.AbstractSphericalGeometry}) =
    _structured_layout(FlowGeometries.Grids.mask(g))

_structured_layout(::FlowGeometries.Grids.AllActive) = TensorSphere()

# The tensor path shares one longitude matrix across every latitude, which describes a complete longitude
# row. A mask leaves partial rows, and the latitudes are iso-latitude rings, so a masked structured grid
# takes the ring path, whose longitude sum runs per ring and skips inactive cells.
_structured_layout(::AbstractArray{Bool}) = RingSphere()

# A structured spherical grid's rings are its latitude rows, in the longitude-fastest order
# `_sph_points` and the field layout both use.
_nrings(g::FlowGeometries.Grids.AbstractStructuredGrid{<:SphericalHarmonicGeometry}) =
    length(FlowGeometries.Grids.coordinates(g, 2))
function _ring_range(g::FlowGeometries.Grids.AbstractStructuredGrid{<:SphericalHarmonicGeometry},
        r::Int)
    nlon = length(FlowGeometries.Grids.coordinates(g, 1))
    return ((r - 1) * nlon + 1):(r * nlon)
end

# A `RingGrid` holds arbitrary per-ring latitudes and counts, so it records no sampling whose traits could
# be read; it answers the ring accessors itself.
_sph_layout(::FlowGeometries.Grids.RingGrid) = RingSphere()

# A spheroid's latitude rows are iso-latitude — `_colatitude` reads the same value along a row — so a
# structured spheroid grid takes the ring path, whose weight is each ring's own measure. A point set with
# no such rows takes the per-point projection.
_sph_layout(::FlowGeometries.Grids.AbstractGrid{<:FlowGeometries.Geometry.AbstractEllipsoidalGeometry}) =
    ScatteredSphere()
_sph_layout(::FlowGeometries.Grids.AbstractStructuredGrid{<:FlowGeometries.Geometry.AbstractEllipsoidalGeometry}) =
    RingSphere()

# The sampling whose latitude quadrature a factorized path may use. A sampling states its rule for a
# SPHERE's colatitudes, and a spheroid's directions are the geocentric ones, so a spheroid grid states
# none and its rings take their own measure as the rule.
_quadrature_sampling(g::FlowGeometries.Grids.AbstractGrid) = _sph_sampling(g)
_quadrature_sampling(::FlowGeometries.Grids.AbstractGrid{<:FlowGeometries.Geometry.AbstractEllipsoidalGeometry}) =
    nothing

# Ring count and a ring's slice of the flattened point vector. `RingGrid` answers both directly; a HEALPix
# grid answers through its sampling, the grid-level accessors having methods for `RingGrid` alone
# (FlowGeometries.jl#18 proposes forwarding them).
_nrings(g::FlowGeometries.Grids.RingGrid) = FlowGeometries.Grids.nrings(g)
_ring_range(g::FlowGeometries.Grids.RingGrid, r::Int) = FlowGeometries.Grids.ring_range(g, r)
_nrings(g::FlowGeometries.Grids.HEALPixGrid) =
    FlowGeometries.SphericalSampling.nrings(_sph_sampling(g))
_ring_range(g::FlowGeometries.Grids.HEALPixGrid, r::Int) =
    FlowGeometries.SphericalSampling.ring_range(_sph_sampling(g), r)

# -----------------------------------------------------------------------------
# Covered extent. A grid's mask says which cells participate, and `size`/`length` report the FULL shape
# (`Base.size(grid) = size(mask(grid))`), so the extent a transform actually integrates over is the
# active cells' measure. Whether a grid carries a mask is part of its type (`AllActive{N}` against a
# `BitVector`), so the unmasked form resolves at compile time onto the measure's own specialized `sum`.
# -----------------------------------------------------------------------------

"""
    _zeroed_inactive(field, grid) -> AbstractArray

`field` with its inactive cells set to zero, or `field` itself where every cell is active.

A transform that consumes complete longitude rows — a library SHT on its own grid, or a longitude `rfft`
— is linear in the field, so zeroing an inactive cell's datum removes it from the sum exactly, matching
what the per-point paths get by skipping it. The weights stay the sampling's own, so the active cells
still carry their own solid angles and total the covered one. Writing zero also clears a `NaN`, which
masked data commonly holds.
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
    covered_area(grid) -> T

Total measure of the cells that participate: the extent a transform over `grid` integrates over.

Equal to `sum(measure(grid))` on an unmasked grid — `4πR²` for a layout tiling the whole sphere, which a
`GridMeasure` answers without visiting a cell — and the active cells' share of it under a mask.
"""
covered_area(g::FlowGeometries.Grids.AbstractGrid) = _covered_area(FlowGeometries.Grids.mask(g), g)

# No mask: the measure's own `sum`, which is `O(Σ Nᵈ)` separable, `O(nrings)` ring-wise, `O(1)` formulaic.
_covered_area(::FlowGeometries.Grids.AllActive, g) = sum(FlowGeometries.Grids.measure(g))

function _covered_area(mask::AbstractArray{Bool}, g)
    m = FlowGeometries.Grids.measure(g)
    T = eltype(g)
    s = zero(T)
    @inbounds for i in eachindex(mask)
        mask[i] && (s += T(m[i]))
    end
    return s
end

"""
    sky_fraction(grid) -> T

Fraction of `grid`'s cells' total measure that participates: `covered_area(grid) / sum(measure(grid))`.
One on an unmasked grid. On a partial-sky spherical grid this is `f_sky`, so a caller wanting the
full-sphere-normalized convention scales coefficients by `inv(sky_fraction(grid))`.
"""
function sky_fraction(g::FlowGeometries.Grids.AbstractGrid)
    total = sum(FlowGeometries.Grids.measure(g))
    return covered_area(g) / total
end

# Per-ring latitude weights where the sampling defines them, else `nothing`. A sampling may define no
# `latitude_weights` method at all (an equal-area pixelization), or define one that declines for its own
# reasons (McEwen–Wiaux states that the sine-series rule is not exact on its nodes).
# Which rule a ring's weight comes from, by DISPATCH: each method returns one concrete type, so this is
# type-stable per specialization and costs no method-table query.
#
# `AbstractSpectralQuadratureSampling` is exactly the set with a latitude rule (Gauss–Legendre,
# Driscoll–Healy, Clenshaw–Curtis, McEwen–Wiaux). Everything else — an equal-area pixelization, a lat–lon
# grid, a `RingGrid` recording no sampling — takes its own cell measure, which for a `RingGrid` is built
# from the Gaussian weights already. McEwen–Wiaux states the method and declines the rule, so it raises
# from here, as the tensor path does for that grid unmasked.
_ring_latitude_weights(::Type{FT}, ::Nothing, nr::Int) where {FT} = nothing
_ring_latitude_weights(::Type{FT}, ::FlowGeometries.SphericalSampling.AbstractSphericalSampling,
    nr::Int) where {FT} = nothing
_ring_latitude_weights(::Type{FT},
    s::FlowGeometries.SphericalSampling.AbstractSpectralQuadratureSampling, nr::Int) where {FT} =
    FlowGeometries.SphericalSampling.latitude_weights(FT, s, nr)

# The two rules as separate methods, so the choice above is resolved by one dispatch and each ring loop
# sees a concrete type.
function _fill_ring_weights!(w, ::Nothing, ranges, m, sc, ::Type{FT}) where {FT}
    @inbounds for r in eachindex(w)
        w[r] = FT(m[first(ranges[r])]) * sc
    end
    return w
end

# `latitude_weights` carries `Σw = 2`, so `wⱼ·(2π/nlonⱼ)` totals `4π` over the rings with no further
# normalization — the rule `latitude_weights`' own docstring states for ring grids.
function _fill_ring_weights!(w, wlat::AbstractVector, ranges, m, sc, ::Type{FT}) where {FT}
    @inbounds for r in eachindex(w)
        w[r] = FT(wlat[r]) * (FT(2π) / length(ranges[r]))
    end
    return w
end

"""
    _warn_bandlimit(sampling, nlat, lmax)

Warn when degree `lmax` exceeds what `nlat` latitude samples of `sampling` support, so the coefficients
above it are not resolved by that grid's quadrature.

Warned, never raised: an over-resolved request still returns coefficients, and a caller may want them.
The relation is the sampling's own `SphericalSampling.bandlimit` — notably `nlat÷2 − 1` for
Driscoll–Healy against `nlat − 1` for Gauss–Legendre and Clenshaw–Curtis. `maxlog = 1` caps this per
logger instance, so a fresh logger still observes it.
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
    _ring_table(grid, ::Type{FT}; lmax=nothing) -> (; ranges, θ, w, λ)

Per iso-latitude ring: its slice of the flattened point vector, its colatitude, and the per-point
quadrature weight inside it; plus every point's longitude.

Colatitude and the weight are read at the ring's first point, both being constant along a ring. The
weight comes from `Grids.measure`, rescaled so the weights total `4π` — the transform's convention, the
same total [`_sph_node_weights`](@ref) carries. A ring layout's measure is a `RingwiseVector` and an
equal-area one's an `Axes.ConstantVector`, whose `sum` is `O(nrings)` and `O(1)`, so no dense
`∏N`-sized measure is built.
"""
function _ring_table(g, ::Type{FT}; lmax = nothing) where {FT}
    nr = _nrings(g)
    if lmax !== nothing
        s = _quadrature_sampling(g)
        s === nothing || _warn_bandlimit(s, nr, lmax)
    end
    λraw, φlat = FlowGeometries.Grids.materialize(g)
    m = FlowGeometries.Grids.measure(g)
    total = sum(m)
    (isfinite(total) && total > 0) || throw(ArgumentError(
        "grid measure sums to $total, so it cannot set quadrature weights; pass `weights=`."))
    sc = FT(4π) / FT(total)
    # A sampling that defines latitude weights states the QUADRATURE for its rings, which is a different
    # thing from the geometric cell area `measure` reports: Gaussian weights are chosen for exactness, so
    # on a Gauss–Legendre grid the two differ and only the former integrates a band-limited field
    # exactly. Where a sampling defines none — HEALPix, whose equal-area pixel is its own weight, and a
    # `RingGrid`, whose `ring_area` is built from the Gaussian weights already — the measure is the rule.
    ranges = Vector{UnitRange{Int}}(undef, nr)
    θ = Vector{FT}(undef, nr)
    w = Vector{FT}(undef, nr)
    @inbounds for r in 1:nr
        rr = _ring_range(g, r)
        ranges[r] = rr
        θ[r] = _colatitude(g, FT(φlat[first(rr)]))
    end
    _fill_ring_weights!(w, _ring_latitude_weights(FT, _quadrature_sampling(g), nr), ranges, m, sc, FT)
    # A ring's weight is one number, so a partially masked ring cannot be expressed by zeroing it; the
    # mask travels with the table and the longitude sum skips inactive points instead.
    return (; ranges, θ, w, λ = FT.(λraw), mask = FlowGeometries.Grids.mask(g))
end

"""
    _sht_weights(grid, nlat; sampling=nothing, weights=nothing) -> Vector

Per-colatitude quadrature weights for a structured spherical `grid`, from its own node-set recipe
`FlowGeometries.Grids.sampling(grid)` via `SphericalSampling.latitude_weights` (`Σw = 2`, carrying the
`sinθ` Jacobian — the transform's convention). `weights` or `sampling` override it.

The recipe is the only thing that determines the quadrature: Gauss–Legendre and an arbitrary lat–lon
grid are the same coordinates to round-off, and only the recipe says which rule is exact on them. A grid
built from raw axes records none, so it raises here and the caller states the quadrature.
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
    _sph_node_weights(grid, ::Type{FT}, N, weights) -> Vector{FT}

Per-node quadrature weights for a scattered spherical `grid`, from its own per-node measure rescaled to
sum to `4π` — the transform's convention, the same total the structured path's
`latitude_weights ⊗ 2π/nlon` carries. `weights` overrides it. Only the relative measures matter, so a
grid of true dual-cell areas integrates exactly and a grid of equal-area nodes gives `4π/N`.
"""
function _sph_node_weights(g::FlowGeometries.Grids.AbstractGrid{<:SphericalHarmonicGeometry},
        ::Type{FT}, N::Integer, weights) where {FT}
    weights === nothing || return FT.(weights)
    m = FlowGeometries.Grids.measure(g)
    length(m) == N || throw(DimensionMismatch(
        "grid measure has $(length(m)) entries for $N nodes"))
    s = sum(m)
    (isfinite(s) && s > 0) || throw(ArgumentError(
        "grid measure sums to $s, so it cannot set quadrature weights; pass `weights=`."))
    # Dividing by the FULL measure total turns each cell's measure into its solid angle on the unit
    # sphere, so summing the active ones gives the covered solid angle — `4π` on an unmasked grid and
    # `4π·f_sky` under a mask, which is the convention `covered_area` states.
    return _masked_weights(FlowGeometries.Grids.mask(g), m, FT(4π) / FT(s), FT)
end

# No mask: every cell participates, so the scaled measure stands as it is (lazy where separable).
_masked_weights(::FlowGeometries.Grids.AllActive, m, c, ::Type{FT}) where {FT} = vec(c .* m)

# An inactive node gets weight zero, which drops it from every quadrature sum without any loop or kernel
# knowing about masks — the per-point paths, the threaded accumulators and the device kernel all read
# these weights. The zero pattern is arbitrary, so this one materializes.
function _masked_weights(mask::AbstractArray{Bool}, m, c, ::Type{FT}) where {FT}
    w = Vector{FT}(undef, length(m))
    @inbounds for i in eachindex(w)
        w[i] = mask[i] ? FT(m[i]) * c : zero(FT)
    end
    return w
end

end # module Grids
