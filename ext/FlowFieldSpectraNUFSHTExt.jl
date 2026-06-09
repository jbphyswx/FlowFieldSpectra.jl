module FlowFieldSpectraNUFSHTExt

using NUFSHT: NUFSHT
using FlowFieldSpectra: FlowFieldSpectra as FFS, NUFSHTBackend, sph_mode_index

"""
    calculate_spectrum(::NUFSHTBackend, coords_vecs, fields_vecs, ms; tol=1e-8, solve=false, maxiter=500, rtol=1e-6, ...)

Compute unstructured Spherical Harmonic Transform (SHT) using NUFSHT.
"""
function FFS._calculate_spectrum_nufsht(
    coords_vecs::Tuple,
    fields_vecs::Tuple,
    ms::Tuple;
    tol::Real = 1e-8,
    solve::Bool = false,
    maxiter::Int = 500,
    rtol::Real = 1e-6,
    kwargs...,
)
    FT = eltype(coords_vecs[1])
    NU = length(fields_vecs)
    lmax = ms[1] - 1
    Nθ = lmax + 1
    Nφ = 2 * lmax + 1

    θ, φ = coords_vecs

    # Create NUFSHT plan
    plan = NUFSHT.make_plan(θ, φ, lmax; tol = tol, T = FT)

    # Preallocate complex coefficients
    coeffs = zeros(Complex{FT}, Nθ, Nφ, NU)

    for k in 1:NU
        C_real = zeros(FT, Nθ, Nφ)

        if solve
            NUFSHT.nusht_solve!(C_real, fields_vecs[k], plan; maxiter = maxiter, rtol = rtol)
        else
            NUFSHT.nusht_type1!(C_real, fields_vecs[k], plan)
        end

        # Map to complex coefficients coeffs
        for l in 0:lmax
            for m in -l:l
                idx = sph_mode_index(l, m)
                fsh_idx = NUFSHT.FastSphericalHarmonics.sph_mode(l, m)
                coeffs[idx, k] = C_real[fsh_idx]
            end
        end
    end

    return coeffs, (0:lmax, -lmax:lmax)
end

end # module FlowFieldSpectraNUFSHTExt
