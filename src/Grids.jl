module Grids

using FlowGeometries: FlowGeometries

export physical_wavenumbers

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

"""
    physical_wavenumbers(domain_size::NTuple{D}, ms::NTuple{D}, ::Type{FT}) -> NTuple{D,<:AbstractRange}

Centered physical wavenumber ranges matching the FFTW/FINUFFT `fftshift`ed mode ordering
`[-m÷2, (m-1)÷2]`, scaled by `2π / L` along each axis. A zero domain size is treated as length `1`.
"""
# `map` over the two tuples with a top-level (non-capturing) function — NOT `ntuple(d -> …)`, whose
# closure boxes the captured `domain_size`/`ms` when `@inbounds` is disabled under `--check-bounds=yes`.
@inline physical_wavenumbers(domain_size::NTuple{D, FT}, ms::NTuple{D, Int}, ::Type{FT}) where {D, FT} =
    map(_wavenumber_axis, domain_size, ms)

# One axis of centered physical wavenumbers, built directly as a `start:step:…` range (no `.*`
# broadcast) so it stays allocation-free even with `@inbounds` off.
@inline function _wavenumber_axis(L::FT, m::Int) where {FT}
    scale = FT(2π) / (L == 0 ? one(FT) : L)
    return range(FT(-(m ÷ 2)) * scale; step = scale, length = m)
end

"""
    physical_wavenumbers(grid, ms::NTuple{D,Int}) -> NTuple{D,<:AbstractRange}

Physical wavenumber ranges for a Cartesian `grid` at spectral resolution `ms`. The Fourier period `L`
per axis is FlowGeometries' `period(grid, d)` (`2π/L` scaling); a non-periodic direction reports period
`0` and is treated as `L = 1` (raw per-sample wavenumbers).
"""
@inline function physical_wavenumbers(
        g::FlowGeometries.Grids.AbstractGrid{<:FlowGeometries.Geometry.AbstractCartesianGeometry},
        ms::NTuple{D, Int}) where {D}
    FT = eltype(g)
    Ls = ntuple(d -> FT(FlowGeometries.Grids.period(g, d)), Val(D))
    return physical_wavenumbers(Ls, ms, FT)
end

# -----------------------------------------------------------------------------
# Spherical θ/φ/weights bridge. FlowGeometries stores `(λ, φ_lat)` = (longitude, geographic latitude);
# the transform wants `(θ, φ)` = (colatitude, longitude). Colatitude θ = π/2 − φ_lat.
# -----------------------------------------------------------------------------

"""
    _sph_points(grid) -> (θ, φ)

Per-point colatitude/longitude lists for a spherical `grid`. A structured `(nlon, nlat)` grid is
expanded in FlowGeometries' column-major order (longitude fastest, matching `size`); a scattered grid
returns its nodes. Both in the transform's `(θ = colatitude, φ = longitude)` convention.
"""
function _sph_points(g::FlowGeometries.Grids.AbstractStructuredGrid{<:FlowGeometries.Geometry.AbstractSphericalGeometry})
    λ = FlowGeometries.Grids.coordinates(g, 1)      # longitude, length nlon
    φlat = FlowGeometries.Grids.coordinates(g, 2)   # geographic latitude, length nlat
    FT = eltype(g)
    nlon = length(λ); nlat = length(φlat)
    θ = Vector{FT}(undef, nlon * nlat)
    φ = Vector{FT}(undef, nlon * nlat)
    half = FT(π) / 2
    @inbounds for jφ in 1:nlat, iλ in 1:nlon
        p = iλ + (jφ - 1) * nlon                    # column-major: longitude fastest
        θ[p] = half - FT(φlat[jφ])
        φ[p] = FT(λ[iλ])
    end
    return θ, φ
end

function _sph_points(g::FlowGeometries.Grids.AbstractUnstructuredGrid{<:FlowGeometries.Geometry.AbstractSphericalGeometry})
    λ = FlowGeometries.Grids.coordinates(g, 1)
    φlat = FlowGeometries.Grids.coordinates(g, 2)
    FT = eltype(g)
    half = FT(π) / 2
    return (half .- FT.(φlat)), FT.(λ)
end

"""
    _sht_weights(grid, nlat; sampling=nothing, weights=nothing) -> Union{Nothing,Vector}

Per-colatitude quadrature weights for a spherical `grid`: an explicit `weights` vector if given;
otherwise those of the FlowGeometries `sampling` via `SphericalSampling.latitude_weights` (`Σw = 2`,
carrying the `sinθ` Jacobian — exactly the transform's convention); otherwise `nothing` (uniform).
"""
function _sht_weights(g::FlowGeometries.Grids.AbstractGrid{<:FlowGeometries.Geometry.AbstractSphericalGeometry},
        nlat::Integer;
        sampling::Union{Nothing, FlowGeometries.SphericalSampling.AbstractSphericalSampling} = nothing,
        weights = nothing)
    weights !== nothing && return weights
    sampling !== nothing &&
        return FlowGeometries.SphericalSampling.latitude_weights(eltype(g), sampling, nlat)
    return nothing
end

end # module Grids
