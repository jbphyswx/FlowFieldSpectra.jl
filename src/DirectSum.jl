module DirectSum

using ..Types: DirectSumBackend
using ..Grids: physical_wavenumbers
using ..SphericalKernels: legendre_tables, fill_legendre!

export sph_mode_index

"""
    sph_mode_index(l::Int, m::Int)

Return the `CartesianIndex` corresponding to degree `l` and order `m` in the
standard 2D coefficient array of size `(lmax+1, 2lmax+1)`.
"""
@inline function sph_mode_index(l::Int, m::Int)
    row = l - abs(m) + 1
    col = m == 0 ? 1 : (m < 0 ? 2 * abs(m) : 2 * m + 1)
    return CartesianIndex(row, col)
end

# Cartesian Direct Sum Transform - SERIAL VERSION
function _calculate_spectrum_cartesian_direct!(
    coeffs::AbstractArray{Complex{FT}},
    coords_vecs::Tuple,
    fields_vecs::Tuple,
    ms::NTuple{D, Int},
    iflag::Int,
    domain_size::Union{Nothing, Tuple} = nothing,
) where {FT, D}
    N = length(coords_vecs[1])
    NU = length(fields_vecs)

    # Validate output size
    expected_size = (ms..., NU)
    size(coeffs) == expected_size || throw(DimensionMismatch("coeffs size $(size(coeffs)) != expected $expected_size"))

    # 1. Coordinate ranges for physical wavenumbers
    ranges = ntuple(Val(D)) do d
        if domain_size !== nothing
            return FT(domain_size[d])
        else
            min_x, max_x = extrema(coords_vecs[d])
            return FT(max_x - min_x)
        end
    end

    # Generate physical wavenumbers consistent with FFTW/FINUFFT (shared definition)
    ks_phys = physical_wavenumbers(ranges, ms, FT)

    # Zero out coeffs (in case of reuse)
    fill!(coeffs, zero(Complex{FT}))

    # O(N * M) Cartesian direct Fourier sum - SERIAL
    @inbounds for I in CartesianIndices(ms)
        for j in 1:N
            # Compute phase = k ⋅ x (manual loop for zero allocation)
            phi = zero(FT)
            for d in 1:D
                phi += ks_phys[d][I[d]] * coords_vecs[d][j]
            end
            phi = -iflag * phi

            W = cis(phi)  # cis(x) = exp(im*x), more efficient

            for u_idx in 1:NU
                coeffs[I, u_idx] += fields_vecs[u_idx][j] * W
            end
        end
    end

    coeffs ./= N
    return ks_phys
end

# Spherical Direct SHT Projection - SERIAL VERSION
function _calculate_spectrum_spherical_direct!(
    coeffs::AbstractArray{Complex{FT}},
    coords_vecs::Tuple,
    fields_vecs::Tuple,
    lmax::Int,
    weights::Union{Nothing, AbstractVector},
) where {FT}
    N = length(coords_vecs[1])
    NU = length(fields_vecs)
    Nθ = lmax + 1
    Nφ = 2 * lmax + 1

    # Validate output size
    expected_size = (Nθ, Nφ, NU)
    size(coeffs) == expected_size || throw(DimensionMismatch("coeffs size $(size(coeffs)) != expected $expected_size"))

    θ = coords_vecs[1]
    φ = coords_vecs[2]

    # Use uniform weights if not provided
    w = weights === nothing ? fill(FT(4π) / N, N) : weights

    # Zero out coeffs (in case of reuse)
    fill!(coeffs, zero(Complex{FT}))

    # Precompute recurrence coefficients once; reuse a per-point Legendre table buffer.
    tables = legendre_tables(FT, lmax)
    Plm = Matrix{FT}(undef, lmax + 1, lmax + 1)

    @inbounds for j in 1:N
        θj = θ[j]
        φj = φ[j]
        wj = w[j]

        xj = cos(θj)
        sj = sin(θj)

        # Fill P_l^m(cos θj) for all (l, m≥0) once for this point.
        fill_legendre!(Plm, tables, xj, sj, lmax)

        for l in 0:lmax
            for m in -l:l
                abs_m = abs(m)
                P_l_m = Plm[l+1, abs_m+1]

                factor = (m < 0 && isodd(abs_m)) ? -one(FT) : one(FT)
                phase = cis(m * φj)
                Y_lm = factor * P_l_m * phase

                idx = sph_mode_index(l, m)
                fj_conj_Ylm_wj = conj(Y_lm) * wj
                for u_idx in 1:NU
                    coeffs[idx, u_idx] += fields_vecs[u_idx][j] * fj_conj_Ylm_wj
                end
            end
        end
    end

    return (0:lmax, -lmax:lmax)
end

end # module DirectSum
