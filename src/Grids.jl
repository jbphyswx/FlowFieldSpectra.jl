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
    infer_grid,
    physical_wavenumbers,
    spatial_dims,
    npoints

"""
    AbstractGrid{FT, D}

Abstract supertype for all coordinate grids. `FT` is the coordinate element type and
`D` is the number of physical (spatial) dimensions. Grids make the coordinate system
*explicit* so backends dispatch on the grid type rather than guessing from coordinate
magnitudes.
"""
abstract type AbstractGrid{FT, D} end

"""
    AbstractCartesianGrid{FT, D} <: AbstractGrid{FT, D}

Cartesian grids in `D` dimensions (uniform, nonuniform tensor-product, or scattered).
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
    UniformCartesianGrid(coords; domain_size=nothing)

Scattered/listed coordinates that lie on a *uniform* rectilinear lattice (one value per
grid point, e.g. produced by `vec` over a `range` mesh). Suitable for the `FFTBackend`.

`coords` is a `D`-tuple of equal-length coordinate vectors (`coords[d][j]` is the `d`-th
coordinate of point `j`). `domain_size` is the physical extent (period) along each axis;
when `nothing` it is inferred from the coordinate bounding box.
"""
struct UniformCartesianGrid{FT, D, C<:Tuple} <: AbstractCartesianGrid{FT, D}
    coords::C
    domain_size::NTuple{D, FT}
end

"""
    NonuniformCartesianGrid(coords; domain_size=nothing)

Tensor-product Cartesian grid with nonuniform spacing along one or more axes. FFTW is not
valid here; use `NUFFTBackend` or `DirectSumBackend`.
"""
struct NonuniformCartesianGrid{FT, D, C<:Tuple} <: AbstractCartesianGrid{FT, D}
    coords::C
    domain_size::NTuple{D, FT}
end

"""
    ScatteredCartesianGrid(coords; domain_size=nothing)

Arbitrary scattered points in `D`-dimensional Cartesian space. Suitable for `NUFFTBackend`
(``D \\le 3``) and `DirectSumBackend` (any `D`).
"""
struct ScatteredCartesianGrid{FT, D, C<:Tuple} <: AbstractCartesianGrid{FT, D}
    coords::C
    domain_size::NTuple{D, FT}
end

# -----------------------------------------------------------------------------
# Spherical grids
# -----------------------------------------------------------------------------

"""
    StructuredSphericalGrid(θ, φ; weights=nothing, quad=:clenshaw_curtis)

Structured spherical quadrature grid given as flattened colatitude/longitude node lists
`(θ, φ)` (each of length `N = Nθ·Nφ`). Suitable for `SHTBackend`.
"""
struct StructuredSphericalGrid{FT, C<:Tuple, W, Q<:AbstractQuadrature} <: AbstractSphericalGrid{FT}
    coords::C            # (θ, φ), each length N
    weights::W           # quadrature weights (length N) or nothing → uniform 4π/N
    quad::Q
end

"""
    ScatteredSphericalGrid(θ, φ; weights=nothing)

Arbitrary scattered points ``(\\theta, \\phi)`` on the sphere. Suitable for
`NUFSHTBackend` and `DirectSumBackend`.
"""
struct ScatteredSphericalGrid{FT, C<:Tuple, W} <: AbstractSphericalGrid{FT}
    coords::C            # (θ, φ), each length N
    weights::W
end

# -----------------------------------------------------------------------------
# Accessors
# -----------------------------------------------------------------------------

"""
    spatial_dims(grid) -> Int

Number of physical/spatial dimensions of the grid.
"""
spatial_dims(::AbstractGrid{FT, D}) where {FT, D} = D

"""
    npoints(grid) -> Int

Number of spatial sample points in the grid.
"""
npoints(g::AbstractGrid) = length(g.coords[1])

# -----------------------------------------------------------------------------
# Construction helpers (fold domain_size inference into one place)
# -----------------------------------------------------------------------------

@inline function _infer_domain_size(coords::NTuple{D, Any}, ::Type{FT}) where {D, FT}
    return ntuple(Val(D)) do d
        lo, hi = extrema(coords[d])
        FT(hi - lo)
    end
end

@inline function _resolve_domain_size(coords::NTuple{D, Any}, domain_size, ::Type{FT}) where {D, FT}
    domain_size === nothing && return _infer_domain_size(coords, FT)
    return ntuple(d -> FT(domain_size[d]), Val(D))
end

for G in (:UniformCartesianGrid, :NonuniformCartesianGrid, :ScatteredCartesianGrid)
    @eval function $G(coords::Tuple; domain_size = nothing)
        D = length(coords)
        FT = float(eltype(coords[1]))
        cc = ntuple(d -> coords[d], D)
        ds = _resolve_domain_size(cc, domain_size, FT)
        return $G{FT, D, typeof(cc)}(cc, ds)
    end
end

function StructuredSphericalGrid(θ, φ; weights = nothing, quad::AbstractQuadrature = ClenshawCurtis())
    FT = float(eltype(θ))
    cc = (θ, φ)
    return StructuredSphericalGrid{FT, typeof(cc), typeof(weights), typeof(quad)}(cc, weights, quad)
end

function ScatteredSphericalGrid(θ, φ; weights = nothing)
    FT = float(eltype(θ))
    cc = (θ, φ)
    return ScatteredSphericalGrid{FT, typeof(cc), typeof(weights)}(cc, weights)
end

# -----------------------------------------------------------------------------
# Physical wavenumbers (single definition; was copy-pasted across backends)
# -----------------------------------------------------------------------------

"""
    physical_wavenumbers(domain_size::NTuple{D}, ms::NTuple{D}, ::Type{FT}) -> NTuple{D,<:AbstractRange}

Centered physical wavenumber ranges matching the FFTW/FINUFFT `fftshift`ed mode ordering
`[-m÷2, (m-1)÷2]`, scaled by `2π / L` along each axis. A zero domain size is treated as
length `1` to avoid division by zero.
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

# -----------------------------------------------------------------------------
# Heuristic classification — the ONLY surviving copy, opt-in via infer_grid
# -----------------------------------------------------------------------------

"""
    _looks_spherical(coords, ms) -> Bool

Legacy heuristic: a 2-tuple of coordinates is treated as spherical when `coords[1] ∈
[0, π]`, `coords[2] ∈ [0, 2π]`, and `ms[2] == 2*ms[1] - 1`. Fragile (a small Cartesian
domain in that range matches) — used only by [`infer_grid`](@ref) as a backwards-compatible
convenience, and warns once when it fires implicitly.
"""
@inline function _looks_spherical(coords::Tuple, ms::Tuple)
    length(coords) == 2 || return false
    length(ms) == 2 || return false
    e1 = extrema(coords[1])
    e2 = extrema(coords[2])
    return e1[1] >= -1e-5 && e1[2] <= (π + 1e-3) &&
           e2[1] >= -1e-5 && e2[2] <= (2π + 1e-3) &&
           ms[2] == 2 * ms[1] - 1
end

const _SPHERICAL_WARNED = Ref(false)

"""
    infer_grid(coords::Tuple, ms::Tuple; spherical=nothing, domain_size=nothing, weights=nothing,
               uniform=false) -> AbstractGrid

Construct an explicit grid from the legacy `(coords, ms)` arguments — a backwards-compatible
adapter for the old positional API. Pass `spherical=true`/`false` to force the coordinate
system; when `nothing`, the (fragile) range heuristic is used and warns once on implicit
spherical detection. `uniform=true` selects `UniformCartesianGrid` (for the FFT backend);
otherwise a Cartesian grid defaults to `ScatteredCartesianGrid`. This adapter does runtime
classification by design; all hot-path dispatch happens on the returned grid's concrete type.
"""
function infer_grid(
    coords::Tuple,
    ms::Tuple;
    spherical::Union{Nothing, Bool} = nothing,
    domain_size = nothing,
    weights = nothing,
    uniform::Bool = false,
)
    is_sph = spherical === true || (spherical === nothing && _looks_spherical(coords, ms))
    if is_sph
        if spherical === nothing && !_SPHERICAL_WARNED[]
            _SPHERICAL_WARNED[] = true
            @warn "FlowFieldSpectra: coordinates were classified as spherical by the legacy range " *
                  "heuristic. Construct a grid explicitly (e.g. `ScatteredSphericalGrid`) or pass " *
                  "`spherical=true`/`spherical=false` to silence this." maxlog = 1
        end
        return weights === nothing ? ScatteredSphericalGrid(coords[1], coords[2]) :
               ScatteredSphericalGrid(coords[1], coords[2]; weights = weights)
    else
        return uniform ? UniformCartesianGrid(coords; domain_size = domain_size) :
               ScatteredCartesianGrid(coords; domain_size = domain_size)
    end
end

end # module Grids
