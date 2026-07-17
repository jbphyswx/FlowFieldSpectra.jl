module FlowFieldSpectraFastSphericalHarmonicsExt

using FastSphericalHarmonics: FastSphericalHarmonics as FSH
using FlowFieldSpectra: FlowFieldSpectra as FFS

# =============================================================================
# Structured Spherical Harmonic Transform via FastSphericalHarmonics, tensor-native. The field arrives
# as `(Nθ, Nφ, batch…)` — we slice each batch's `(Nθ, Nφ)` plane straight into `sph_transform!`, with
# NO `vec`→`reshape` round-trip. FastSphericalHarmonics/FastTransforms is CPU-only (see the plan's
# ceilings); the execution axis is a documented no-op here (Serial and Threaded route in identically).
# =============================================================================

function FFS._calculate_spectrum_sht(g::FFS.StructuredSphericalGrid, field::AbstractArray, ms::Tuple; kwargs...)
    FT = real(float(eltype(field)))
    lmax = ms[1] - 1
    Nθ = lmax + 1
    Nφ = 2 * lmax + 1
    ss = FFS.spatial_size(g)
    ss == (Nθ, Nφ) || throw(ArgumentError(
        "SHTBackend requires a structured (Nθ, Nφ) = ($Nθ, $Nφ) grid matching ms; got spatial_size = $ss"))
    B = length(field) ÷ (Nθ * Nφ)
    Fr = reshape(field, Nθ, Nφ, B)
    coeffs = zeros(Complex{FT}, Nθ, Nφ, B)
    @inbounds for b in 1:B
        slab = Matrix{FT}(@view Fr[:, :, b])          # dense copy; sph_transform! mutates in place
        FSH.sph_transform!(slab)                       # real/imag packed in the FSH layout
        for l in 0:lmax
            for m in -l:l
                coeffs[FFS.sph_mode_index(l, m), b] = slab[FSH.sph_mode(l, m)]
            end
        end
    end
    batch = ntuple(i -> size(field, 2 + i), ndims(field) - 2)
    return reshape(coeffs, Nθ, Nφ, batch...), (0:lmax, -lmax:lmax)
end

end # module FlowFieldSpectraFastSphericalHarmonicsExt
