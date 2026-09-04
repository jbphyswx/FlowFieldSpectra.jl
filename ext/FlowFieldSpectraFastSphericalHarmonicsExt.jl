module FlowFieldSpectraFastSphericalHarmonicsExt

using FastSphericalHarmonics: FastSphericalHarmonics as FSH
using FlowFieldSpectra: FlowFieldSpectra as FFS
using ComputationalBackends: ComputationalBackends
using SpectralBackends: SpectralBackends
using FlowGeometries: FlowGeometries

# =============================================================================
# Structured Spherical Harmonic Transform via FastSphericalHarmonics (FastTransforms). `sph_transform!`
# is an exact analysis on FastTransforms' OWN grid, addressed purely by array POSITION: rows are the
# ascending colatitudes `θ_i = (i-½)π/Nθ`, columns the ascending longitudes `φ_j = 2π j/Nφ`
# (`Nφ = 2Nθ-1`) — `FastSphericalHarmonics.sph_points`. It reads no coordinates and carries no explicit
# quadrature weights, so it is NOT a `Σ f·Y·w` sum and does not equal the DirectSum reference on this
# (non-Gauss) grid; its guarantee is the exact round trip with `sph_evaluate`.
#
# FFS fields live on an FlowGeometries `(nlon, nlat)` grid in the grid's own coordinate order, which
# need not be FastTransforms' order. So we map each batch slab onto FastTransforms' grid using the
# grid's own coordinates (permute nlat→ascending-colatitude, nlon→ascending-longitude), and VALIDATE
# that the nodes are exactly FastTransforms' Clenshaw–Curtis grid — a wrong grid errors rather than
# returning a silently wrong transform. CPU-only; the execution axis is a documented no-op.
# =============================================================================

function FFS._calculate_spectrum_sht(g::FlowGeometries.Grids.AbstractStructuredGrid{<:FlowGeometries.Geometry.AbstractSphericalGeometry},
        field::AbstractArray, ms::Tuple; kwargs...)
    FT = real(float(eltype(field)))
    lmax = ms[1] - 1
    Nθ = lmax + 1                     # nlat
    Nφ = 2 * lmax + 1                 # nlon
    ss = size(g)
    ss == (Nφ, Nθ) || throw(ArgumentError(
        "FSHTSpectralBackend requires a structured (nlon, nlat) = ($Nφ, $Nθ) grid matching ms; got size = $ss"))

    rowperm, colperm = _ft_perms(FT, g, Nθ, Nφ)
    B = length(field) ÷ (Nθ * Nφ)
    # The analysis is linear in the field and consumes the whole grid, so an inactive cell is left out by
    # zeroing its datum.
    Fr = reshape(FFS.Grids._zeroed_inactive(field, g), Nφ, Nθ, B)     # (nlon, nlat, batch)
    # `sph_transform!` is a real analysis, so a real field's coefficients are real outright, and a complex
    # field goes through it one real component at a time — the analysis is linear, so the components
    # recombine as `C = C(Re f) + i·C(Im f)`.
    coeffs = zeros(FFS.sph_coeff_type(eltype(field), FT), Nθ, Nφ, B)
    slab = Matrix{FT}(undef, Nθ, Nφ)
    @inbounds for b in 1:B
        _ft_stage!(slab, Fr, rowperm, colperm, b, FT, real)
        FSH.sph_transform!(slab)                               # exact analysis, in place
        _ft_gather!(coeffs, slab, lmax, b, one(FT))
        if !(eltype(field) <: Real)
            _ft_stage!(slab, Fr, rowperm, colperm, b, FT, imag)
            FSH.sph_transform!(slab)
            _ft_gather!(coeffs, slab, lmax, b, im)
        end
    end
    batch = ntuple(i -> size(field, 2 + i), ndims(field) - 2)
    return reshape(coeffs, Nθ, Nφ, batch...), (0:lmax, -lmax:lmax)
end

# One field component, in the grid's `(nlon, nlat)` order, written into FastTransforms' `(θ, φ)` slab.
function _ft_stage!(slab, Fr, rowperm, colperm, b::Int, ::Type{FT}, part) where {FT}
    @inbounds for (ic, jc) in enumerate(colperm), (ir, jr) in enumerate(rowperm)
        slab[ir, ic] = FT(part(Fr[jc, jr, b]))
    end
    return slab
end

# The analyzed slab, added into `coeffs` in FFS's own mode layout at weight `scale`.
function _ft_gather!(coeffs, slab, lmax::Int, b::Int, scale)
    @inbounds for l in 0:lmax, m in -l:l
        coeffs[FFS.sph_mode_index(l, m), b] += scale * slab[FSH.sph_mode(l, m)]
    end
    return coeffs
end

# The grid's own axis order → FastTransforms' ascending-(colatitude, longitude) order, and whether the
# nodes ARE that Clenshaw–Curtis grid. Shared by the forward, the inverse and the applicability query, so
# one grid check serves them all.
function _ft_sort(::Type{FT}, g, Nθ::Int) where {FT}
    Θ, Φ = FSH.sph_points(Nθ)
    θ = FT(π) / 2 .- collect(FT, FlowGeometries.Grids.coordinates(g, 2))
    λ = collect(FT, FlowGeometries.Grids.coordinates(g, 1))
    rowperm = sortperm(θ)
    colperm = sortperm(λ)
    on_grid = isapprox(θ[rowperm], Θ; atol = sqrt(eps(FT))) &&
        isapprox(λ[colperm], Φ; atol = sqrt(eps(FT)))
    return rowperm, colperm, on_grid
end

function _ft_perms(::Type{FT}, g, Nθ::Int, Nφ::Int) where {FT}
    rowperm, colperm, on_grid = _ft_sort(FT, g, Nθ)
    on_grid || throw(ArgumentError(
        "FSHTSpectralBackend requires the FastSphericalHarmonics Clenshaw–Curtis grid " *
        "(colatitude θ=(i-½)π/Nθ, longitude φ=2πj/Nφ); this grid's nodes differ. Sample on that grid " *
        "(FastSphericalHarmonics.sph_points), or use DirectSumSpectralBackend / NUFSHTSpectralBackend."))
    return rowperm, colperm
end

# `AutoSpectralBackend` asks before selecting this backend, so the same node check answers without
# raising. The size test runs first, since the node lists are read at `(Nφ, Nθ)`.
function FFS._sht_applicable(g::FlowGeometries.Grids.AbstractStructuredGrid{<:FlowGeometries.Geometry.AbstractSphericalGeometry},
        ms::Tuple)
    length(ms) == 2 && ms[1] >= 1 || return false
    lmax = ms[1] - 1
    size(g) == (2 * lmax + 1, lmax + 1) || return false
    return _ft_sort(real(float(eltype(g))), g, lmax + 1)[3]
end

# =============================================================================
# Inverse (synthesis) via `sph_evaluate!`, the exact inverse of the `sph_transform!` the forward runs.
# Coefficients arrive in `sph_mode_index` layout and are copied into FastTransforms' `sph_mode` slab,
# evaluated on its own grid, then written back through the same permutations the forward used. A complex
# coefficient array is evaluated one real component at a time, which the transform's linearity permits.
# =============================================================================

function FFS._synthesize(::SpectralBackends.AbstractFSHTSpectralBackend,
        ::ComputationalBackends.AbstractExecutionBackend,
        g::FlowGeometries.Grids.AbstractStructuredGrid{<:FlowGeometries.Geometry.AbstractSphericalGeometry},
        coeffs::AbstractArray, ms::NTuple{2, Int}; real_output::Bool = true, iflag::Int = 1,
        ks = nothing, kwargs...)
    FT = real(float(eltype(coeffs)))
    lmax = ms[1] - 1
    Nθ = lmax + 1
    Nφ = 2 * lmax + 1
    ss = size(g)
    ss == (Nφ, Nθ) || throw(ArgumentError(
        "FSHTSpectralBackend requires a structured (nlon, nlat) = ($Nφ, $Nθ) grid matching ms; got size = $ss"))
    size(coeffs)[1:2] == (Nθ, Nφ) || throw(DimensionMismatch(
        "spherical coefficients must be (Nθ, Nφ) = ($Nθ, $Nφ) on the spectral dims; got $(size(coeffs)[1:2])."))
    rowperm, colperm = _ft_perms(FT, g, Nθ, Nφ)
    B = length(coeffs) ÷ (Nθ * Nφ)
    C = reshape(coeffs, Nθ, Nφ, B)
    batch = ntuple(i -> size(coeffs, 2 + i), ndims(coeffs) - 2)
    ET = real_output ? FT : Complex{FT}
    out = zeros(ET, Nφ, Nθ, batch...)
    O = reshape(out, Nφ, Nθ, B)
    slab = Matrix{FT}(undef, Nθ, Nφ)
    ncomp = real_output ? 1 : 2
    @inbounds for b in 1:B, comp in 1:ncomp
        fill!(slab, zero(FT))
        for l in 0:lmax, m in -l:l
            z = C[FFS.sph_mode_index(l, m), b]
            slab[FSH.sph_mode(l, m)] = comp == 1 ? real(z) : imag(z)
        end
        FSH.sph_evaluate!(slab)                                # exact synthesis, in place
        for (ic, jc) in enumerate(colperm), (ir, jr) in enumerate(rowperm)
            v = slab[ir, ic]                                   # FastTransforms (θ ir, φ ic) → (nlon jc, nlat jr)
            O[jc, jr, b] += comp == 1 ? ET(v) : ET(im * v)
        end
    end
    return out
end

end # module FlowFieldSpectraFastSphericalHarmonicsExt
