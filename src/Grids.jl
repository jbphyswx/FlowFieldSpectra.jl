module Grids

export AbstractGrid,
    AbstractCartesianGrid,
    AbstractSphericalGrid,
    UniformCartesianGrid,
    NonuniformCartesianGrid,
    ScatteredCartesianGrid,
    StructuredSphericalGrid,
    ScatteredSphericalGrid,
    AbstractQuadrature,
    ClenshawCurtis,
    GaussLegendre,
    Equiangular,
    physical_wavenumbers,
    spatial_dims,
    ndims_spatial,
    spatial_size,
    npoints

# =============================================================================
# Grid taxonomy.
#
# The fundamental distinction is TOPOLOGY, encoded in the type:
#   • tensor-product / rectilinear — points are the Cartesian product of `D` 1-D axes; a field is a
#     `D`-dimensional tensor `(N_1, …, N_D, batch…)`. Uniform spacing (`AbstractRange` axes) is
#     FFT-eligible; nonuniform spacing (`AbstractVector` axes, e.g. Gaussian latitudes / stretched z)
#     needs NUFFT/DirectSum. Nonuniform-in-space-but-gridded (lat/lon) lives here.
#   • scattered — a genuine point cloud of `N` points; a field is `(N, batch…)`. Coordinates are
#     `D` per-axis point vectors of length `N` (the layout FINUFFT/cuFINUFFT consume directly).
#
# `D` = number of SPATIAL (transformed) dims. The batch axes are every trailing array dim beyond the
# ones the grid consumes; the split is taken from the grid via `ndims_spatial`/`spatial_size`, never
# guessed (see `Problem.jl`).
# =============================================================================

"""
    AbstractGrid{FT, D}

Abstract supertype for all coordinate grids. `FT` is the coordinate element type and `D` the number
of physical (spatial, transformed) dimensions. The grid type *is* the coordinate system, so backends
dispatch on it rather than guessing from coordinate magnitudes.
"""
abstract type AbstractGrid{FT, D} end

"""
    AbstractCartesianGrid{FT, D} <: AbstractGrid{FT, D}

Cartesian grids in `D` dimensions (uniform / nonuniform tensor-product, or scattered).
"""
abstract type AbstractCartesianGrid{FT, D} <: AbstractGrid{FT, D} end

"""
    AbstractSphericalGrid{FT} <: AbstractGrid{FT, 2}

Spherical ``(\\theta, \\phi)`` grids (structured-quadrature or scattered).
"""
abstract type AbstractSphericalGrid{FT} <: AbstractGrid{FT, 2} end

"""
    AbstractQuadrature

Quadrature scheme for a structured spherical grid. Dispatched on type, not a symbol.
"""
abstract type AbstractQuadrature end

"""`ClenshawCurtis()` — Clenshaw–Curtis latitude nodes (default)."""
struct ClenshawCurtis <: AbstractQuadrature end

"""`GaussLegendre()` — Gauss–Legendre latitude nodes."""
struct GaussLegendre <: AbstractQuadrature end

"""`Equiangular()` — equiangular latitude nodes."""
struct Equiangular <: AbstractQuadrature end

# -----------------------------------------------------------------------------
# Cartesian grids
# -----------------------------------------------------------------------------

"""
    UniformCartesianGrid(axes::NTuple{D,AbstractRange}; domain_size=nothing)
    UniformCartesianGrid(axis::AbstractRange; domain_size=nothing)
    UniformCartesianGrid(; domain, n)

Tensor-product Cartesian grid with **uniform** spacing along every axis — the FFT-eligible grid. Each
`axes[d]` is a 1-D `AbstractRange` of length `N_d` (one value per grid line, *not* `prod(N)` per-point
coordinates). A field on this grid is a `D`-dimensional tensor `(N_1, …, N_D, batch…)`.

`domain_size[d]` is the physical period along axis `d`; when omitted it is `step(axes[d]) * N_d`
(the periodic-domain convention, so `range(0, L, N+1)[1:N]` recovers `L`). The `domain=…, n=…`
keyword form builds `range(0, L_d, n_d + 1)[1:n_d]` for each axis.
"""
struct UniformCartesianGrid{FT, D, A<:Tuple} <: AbstractCartesianGrid{FT, D}
    axes::A                       # D-tuple of AbstractRange, each length N_d
    domain_size::NTuple{D, FT}
end

"""
    NonuniformCartesianGrid(axes::NTuple{D,AbstractVector}; domain_size=nothing)
    NonuniformCartesianGrid(axis::AbstractVector; domain_size=nothing)

Tensor-product Cartesian grid with **nonuniform** spacing along one or more axes (e.g. Gaussian
latitudes, a stretched vertical grid). Still fully gridded: each `axes[d]` is a 1-D coordinate vector
of length `N_d`, and a field is a tensor `(N_1, …, N_D, batch…)`. FFTW is not valid; use
`NUFFTBackend` or `DirectSumBackend`. `domain_size` defaults to the per-axis coordinate span.
"""
struct NonuniformCartesianGrid{FT, D, A<:Tuple} <: AbstractCartesianGrid{FT, D}
    axes::A                       # D-tuple of AbstractVector, each length N_d
    domain_size::NTuple{D, FT}
end

"""
    ScatteredCartesianGrid(coords::NTuple{D,AbstractVector}; domain_size=nothing)

Arbitrary scattered points in `D`-dimensional Cartesian space (a genuine point cloud — no product
structure). `coords[d]` is the length-`N` vector of the `d`-th coordinate of every point; a field is
`(N, batch…)`. This per-axis-vector layout is what NUFFT/cuFINUFFT consume directly. Suitable for
`NUFFTBackend` (``D \\le 3``) and `DirectSumBackend` (any `D`). `domain_size` defaults to the
coordinate bounding box.
"""
struct ScatteredCartesianGrid{FT, D, C<:Tuple} <: AbstractCartesianGrid{FT, D}
    coords::C                     # D-tuple of AbstractVector, each length N (points)
    domain_size::NTuple{D, FT}
end

# -----------------------------------------------------------------------------
# Spherical grids
# -----------------------------------------------------------------------------

"""
    StructuredSphericalGrid(θ, φ; weights=nothing, quad=ClenshawCurtis())

Structured spherical quadrature grid given by a colatitude axis `θ` (length `Nθ`) and a longitude
axis `φ` (length `Nφ`) — a tensor-product `(Nθ, Nφ)` grid. A field is `(Nθ, Nφ, batch…)`. `weights`
are the per-colatitude quadrature weights (length `Nθ`) or `nothing`. Suitable for `SHTBackend`.
"""
struct StructuredSphericalGrid{FT, Aθ<:AbstractVector, Aφ<:AbstractVector, W, Q<:AbstractQuadrature} <: AbstractSphericalGrid{FT}
    θ::Aθ                         # colatitude axis, length Nθ
    φ::Aφ                         # longitude axis, length Nφ
    weights::W                    # per-θ quadrature weights (length Nθ) or nothing → uniform
    quad::Q
end

"""
    ScatteredSphericalGrid(θ, φ; weights=nothing)

Arbitrary scattered points ``(\\theta, \\phi)`` on the sphere (length-`N` per-point vectors). A field
is `(N, batch…)`. Suitable for `NUFSHTBackend` and `DirectSumBackend`.
"""
struct ScatteredSphericalGrid{FT, C<:Tuple, W} <: AbstractSphericalGrid{FT}
    coords::C                     # (θ, φ), each length N
    weights::W
end

# -----------------------------------------------------------------------------
# Interface — the spatial/batch split is taken from HERE, never guessed.
# -----------------------------------------------------------------------------

"""
    spatial_dims(grid) -> Int

Number of physical/spatial (transformed) dimensions `D`. For a scattered grid this is the *ambient*
dimension (number of coordinate axes), which can exceed `ndims_spatial`.
"""
spatial_dims(::AbstractGrid{FT, D}) where {FT, D} = D

"""
    ndims_spatial(grid) -> Int

Number of *leading array dimensions* a field on this grid consumes: `D` for a tensor-product Cartesian
grid or a structured spherical grid, and `1` for any scattered/point-cloud grid (the single point
axis). Every array dimension after these is a batch dimension.
"""
ndims_spatial(::UniformCartesianGrid{FT, D}) where {FT, D} = D
ndims_spatial(::NonuniformCartesianGrid{FT, D}) where {FT, D} = D
ndims_spatial(::ScatteredCartesianGrid) = 1
ndims_spatial(::StructuredSphericalGrid) = 2
ndims_spatial(::ScatteredSphericalGrid) = 1

"""
    spatial_size(grid) -> NTuple

Sizes of the leading spatial array dimensions: `(N_1, …, N_D)` for a tensor-product Cartesian grid,
`(Nθ, Nφ)` for a structured spherical grid, and `(N,)` for a scattered grid.
"""
spatial_size(g::UniformCartesianGrid) = map(length, g.axes)
spatial_size(g::NonuniformCartesianGrid) = map(length, g.axes)
spatial_size(g::ScatteredCartesianGrid) = (length(g.coords[1]),)
spatial_size(g::StructuredSphericalGrid) = (length(g.θ), length(g.φ))
spatial_size(g::ScatteredSphericalGrid) = (length(g.coords[1]),)

"""
    npoints(grid) -> Int

Total number of spatial sample points: `prod(spatial_size(grid))` (`∏ N_d` for a tensor grid, `N` for
a scattered grid).
"""
npoints(g::AbstractGrid) = prod(spatial_size(g))

# -----------------------------------------------------------------------------
# Construction helpers
# -----------------------------------------------------------------------------

# Periodic-domain size for a uniform axis: step·N (so range(0, L, N+1)[1:N] ⇒ L).
@inline _uniform_domain(ax::AbstractRange, ::Type{FT}) where {FT} = FT(step(ax)) * length(ax)
# Bounding-box span for an arbitrary axis / coordinate vector.
@inline function _span(v, ::Type{FT}) where {FT}
    lo, hi = extrema(v)
    return FT(hi - lo)
end

_float_eltype(v) = float(eltype(v))

# UniformCartesianGrid: axes are ranges; domain defaults to step·N per axis.
function UniformCartesianGrid(axes::Tuple; domain_size = nothing)
    D = length(axes)
    all(ax -> ax isa AbstractRange, axes) ||
        throw(ArgumentError("UniformCartesianGrid axes must be AbstractRanges (uniform spacing); use NonuniformCartesianGrid for vector axes"))
    FT = _float_eltype(axes[1])
    ax = ntuple(d -> axes[d], D)
    ds = domain_size === nothing ?
        ntuple(d -> _uniform_domain(ax[d], FT), D) :
        ntuple(d -> FT(domain_size[d]), D)
    return UniformCartesianGrid{FT, D, typeof(ax)}(ax, ds)
end
UniformCartesianGrid(axis::AbstractRange; kwargs...) = UniformCartesianGrid((axis,); kwargs...)

# Convenience: build periodic uniform axes from physical extent + point count.
function UniformCartesianGrid(; domain::Tuple, n::Tuple)
    length(domain) == length(n) || throw(ArgumentError("domain and n must have equal length"))
    FT = float(eltype(domain))
    axes = ntuple(length(n)) do d
        L = FT(domain[d])
        range(zero(FT), L; length = n[d] + 1)[1:n[d]]
    end
    return UniformCartesianGrid(axes; domain_size = ntuple(d -> FT(domain[d]), length(n)))
end

# NonuniformCartesianGrid / ScatteredCartesianGrid: 1-D vectors; domain defaults to the span.
for G in (:NonuniformCartesianGrid, :ScatteredCartesianGrid)
    @eval function $G(vs::Tuple; domain_size = nothing)
        D = length(vs)
        FT = _float_eltype(vs[1])
        cc = ntuple(d -> vs[d], D)
        ds = domain_size === nothing ?
            ntuple(d -> _span(cc[d], FT), D) :
            ntuple(d -> FT(domain_size[d]), D)
        return $G{FT, D, typeof(cc)}(cc, ds)
    end
    @eval $G(v::AbstractVector; kwargs...) = $G((v,); kwargs...)
end

function StructuredSphericalGrid(θ, φ; weights = nothing, quad::AbstractQuadrature = ClenshawCurtis())
    FT = _float_eltype(θ)
    return StructuredSphericalGrid{FT, typeof(θ), typeof(φ), typeof(weights), typeof(quad)}(θ, φ, weights, quad)
end

function ScatteredSphericalGrid(θ, φ; weights = nothing)
    FT = _float_eltype(θ)
    cc = (θ, φ)
    return ScatteredSphericalGrid{FT, typeof(cc), typeof(weights)}(cc, weights)
end

# -----------------------------------------------------------------------------
# Physical wavenumbers (single definition; axis/domain-based, not coord-based).
# -----------------------------------------------------------------------------

"""
    physical_wavenumbers(domain_size::NTuple{D}, ms::NTuple{D}, ::Type{FT}) -> NTuple{D,<:AbstractRange}

Centered physical wavenumber ranges matching the FFTW/FINUFFT `fftshift`ed mode ordering
`[-m÷2, (m-1)÷2]`, scaled by `2π / L` along each axis. A zero domain size is treated as length `1`.
"""
@inline function physical_wavenumbers(domain_size::NTuple{D, FT}, ms::NTuple{D, Int}, ::Type{FT}) where {D, FT}
    return ntuple(Val(D)) do d
        L = domain_size[d]
        scale = FT(2π) / (L == 0 ? one(FT) : L)
        range(FT(-(ms[d] ÷ 2)), stop = FT((ms[d] - 1) ÷ 2), length = ms[d]) .* scale
    end
end

"""
    physical_wavenumbers(grid::AbstractCartesianGrid, ms) -> NTuple{D,<:AbstractRange}

Physical wavenumber ranges for a Cartesian grid at spectral resolution `ms`.
"""
@inline function physical_wavenumbers(g::AbstractCartesianGrid{FT, D}, ms::NTuple{D, Int}) where {FT, D}
    return physical_wavenumbers(g.domain_size, ms, FT)
end

end # module Grids
