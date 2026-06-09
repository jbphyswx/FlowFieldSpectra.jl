module FlowFieldSpectraFastSphericalHarmonicsExt

using FastSphericalHarmonics: FastSphericalHarmonics
using FlowFieldSpectra: FlowFieldSpectra as FFS, SHTBackend, sph_mode_index

"""
    calculate_spectrum(::SHTBackend, coords_vecs, fields_vecs, ms; ...)

Compute structured Spherical Harmonic Transform (SHT) using FastSphericalHarmonics.
"""
function FFS._calculate_spectrum_sht(
    coords_vecs::Tuple,
    fields_vecs::Tuple,
    ms::Tuple;
    kwargs...,
)
    FT = eltype(coords_vecs[1])
    NU = length(fields_vecs)
    lmax = ms[1] - 1
    Nθ = lmax + 1
    Nφ = 2 * lmax + 1

    # Validate inputs
    for d in 1:2
        length(coords_vecs[d]) == Nθ * Nφ || throw(DimensionMismatch("SHT requires coordinates length to match grid size Nθ*Nφ = $(Nθ*Nφ)"))
    end
    for k in 1:NU
        length(fields_vecs[k]) == Nθ * Nφ || throw(DimensionMismatch("SHT requires field components length to match grid size Nθ*Nφ = $(Nθ*Nφ)"))
    end

    # Preallocate complex coefficients
    # Although FastSphericalHarmonics returns a real matrix where real/imag parts of C_l^m
    # are stored in different columns, we map them to a standard Complex array.
    coeffs = zeros(Complex{FT}, Nθ, Nφ, NU)

    for k in 1:NU
        # Copy input field and reshape to grid
        grid_data = copy(reshape(fields_vecs[k], Nθ, Nφ))
        
        # In-place transform to spherical harmonic coefficients (stores real/imag parts)
        FastSphericalHarmonics.sph_transform!(grid_data)

        # Map to complex coefficients coeffs
        for l in 0:lmax
            for m in -l:l
                idx = sph_mode_index(l, m)
                # FastSphericalHarmonics.sph_mode(l, m) returns the CartesianIndex of the real coefficient
                fsh_idx = FastSphericalHarmonics.sph_mode(l, m)
                coeffs[idx, k] = grid_data[fsh_idx]
            end
        end
    end

    return coeffs, (0:lmax, -lmax:lmax)
end

end # module FlowFieldSpectraFastSphericalHarmonicsExt
