module FlowFieldSpectraFastSphericalHarmonicsExt

using FastSphericalHarmonics: FastSphericalHarmonics as FSH
using FlowFieldSpectra: FlowFieldSpectra as FFS
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

    # FastTransforms' grid + the permutations that put this grid's axes into its order.
    Θ, Φ = FSH.sph_points(Nθ)                                  # ascending colatitude / longitude
    θ = FT(π) / 2 .- collect(FT, FlowGeometries.Grids.coordinates(g, 2))   # colatitude per nlat index
    λ = collect(FT, FlowGeometries.Grids.coordinates(g, 1))               # longitude per nlon index
    rowperm = sortperm(θ)                                     # → ascending colatitude
    colperm = sortperm(λ)                                     # → ascending longitude
    (isapprox(θ[rowperm], Θ; atol = sqrt(eps(FT))) && isapprox(λ[colperm], Φ; atol = sqrt(eps(FT)))) || throw(ArgumentError(
        "FSHTSpectralBackend requires the FastSphericalHarmonics Clenshaw–Curtis grid " *
        "(colatitude θ=(i-½)π/Nθ, longitude φ=2πj/Nφ); this grid's nodes differ. Sample on that grid " *
        "(FastSphericalHarmonics.sph_points), or use DirectSumSpectralBackend / NUFSHTSpectralBackend."))

    B = length(field) ÷ (Nθ * Nφ)
    Fr = reshape(field, Nφ, Nθ, B)                            # (nlon, nlat, batch)
    coeffs = zeros(Complex{FT}, Nθ, Nφ, B)                    # spectral coefficient array
    slab = Matrix{FT}(undef, Nθ, Nφ)
    @inbounds for b in 1:B
        for (ic, jc) in enumerate(colperm), (ir, jr) in enumerate(rowperm)
            slab[ir, ic] = FT(Fr[jc, jr, b])                  # (nlon jc, nlat jr) → FastTransforms (θ ir, φ ic)
        end
        FSH.sph_transform!(slab)                               # exact analysis, in place
        for l in 0:lmax, m in -l:l
            coeffs[FFS.sph_mode_index(l, m), b] = slab[FSH.sph_mode(l, m)]
        end
    end
    batch = ntuple(i -> size(field, 2 + i), ndims(field) - 2)
    return reshape(coeffs, Nθ, Nφ, batch...), (0:lmax, -lmax:lmax)
end

end # module FlowFieldSpectraFastSphericalHarmonicsExt
