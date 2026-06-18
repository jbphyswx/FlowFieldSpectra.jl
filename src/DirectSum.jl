module DirectSum

using ..Types: DirectSumBackend

export calculate_spectrum_direct, calculate_spectrum_direct!, sph_mode_index

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

"""
    calculate_spectrum_direct(coords_vecs, fields_vecs, ms; kwargs...)

Compute direct sum spectral coefficients for Cartesian or spherical coordinates.
Allocates output arrays. See also `calculate_spectrum_direct!` for preallocated output.
"""
function calculate_spectrum_direct(
    coords_vecs::Tuple,
    fields_vecs::Tuple,
    ms::Tuple;
    iflag::Int = 1,
    domain_size::Union{Nothing, Tuple} = nothing,
    weights::Union{Nothing, AbstractVector} = nothing,
)
    D = length(coords_vecs)
    NU = length(fields_vecs)
    N = length(coords_vecs[1])
    FT = eltype(coords_vecs[1])

    # Determine output size based on coordinate type
    if D == 2 && all(extrema(coords_vecs[1]) .<= (π + 1e-3)) && all(extrema(coords_vecs[1]) .>= -1e-5) &&
       all(extrema(coords_vecs[2]) .<= (2π + 1e-3)) && all(extrema(coords_vecs[2]) .>= -1e-5) &&
       (ms[2] == 2 * ms[1] - 1)
        # Spherical: ms = (Nθ, Nφ) = (lmax+1, 2*lmax+1)
        lmax = ms[1] - 1
        Nθ = lmax + 1
        Nφ = 2 * lmax + 1
        coeffs = zeros(Complex{FT}, Nθ, Nφ, NU)
        ks = calculate_spectrum_direct!(coeffs, coords_vecs, fields_vecs, ms; iflag, domain_size, weights)
        return (coeffs, ks)
    else
        # Cartesian: ms = (mx, my, ...)
        coeffs = zeros(Complex{FT}, ms..., NU)
        ks = calculate_spectrum_direct!(coeffs, coords_vecs, fields_vecs, ms; iflag, domain_size, weights)
        return (coeffs, ks)
    end
end

"""
    calculate_spectrum_direct!(coeffs, coords_vecs, fields_vecs, ms; kwargs...)

In-place version of `calculate_spectrum_direct`. Computes spectral coefficients
using preallocated `coeffs` array. Returns the wavenumber ranges.

# Arguments
- `coeffs`: Preallocated output array. For Cartesian: size `(ms..., NU)`. For spherical: size `(Nθ, Nφ, NU)`.
- `coords_vecs`: Tuple of coordinate vectors
- `fields_vecs`: Tuple of field vectors
- `ms`: Target spectral resolution tuple

# Returns
- `ks_phys`: Tuple of physical wavenumber ranges

# Example
```julia
# Preallocate once
coeffs = zeros(Complex{Float64}, 64, 64, 2)

# Reuse in time loop
for t in 1:nt
    calculate_spectrum_direct!(coeffs, coords, fields[t], (64, 64))
    # ... analyze coeffs ...
end
```
"""
function calculate_spectrum_direct!(
    coeffs::AbstractArray{Complex{FT}},
    coords_vecs::Tuple,
    fields_vecs::Tuple,
    ms::Tuple;
    iflag::Int = 1,
    domain_size::Union{Nothing, Tuple} = nothing,
    weights::Union{Nothing, AbstractVector} = nothing,
) where {FT}
    D = length(coords_vecs)
    NU = length(fields_vecs)
    N = length(coords_vecs[1])

    # Validate inputs
    for d in 1:D
        length(coords_vecs[d]) == N || throw(DimensionMismatch("Coordinates length mismatch"))
    end
    for u_idx in 1:NU
        length(fields_vecs[u_idx]) == N || throw(DimensionMismatch("Field length mismatch"))
    end

    if D == 2 && all(extrema(coords_vecs[1]) .<= (π + 1e-3)) && all(extrema(coords_vecs[1]) .>= -1e-5) &&
       all(extrema(coords_vecs[2]) .<= (2π + 1e-3)) && all(extrema(coords_vecs[2]) .>= -1e-5) &&
       (ms[2] == 2 * ms[1] - 1)
        # Spherical coordinate path
        lmax = ms[1] - 1
        return _calculate_spectrum_spherical_direct!(coeffs, coords_vecs, fields_vecs, lmax, weights)
    else
        # Cartesian coordinate path
        return _calculate_spectrum_cartesian_direct!(coeffs, coords_vecs, fields_vecs, ms, iflag, domain_size)
    end
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
            return domain_size[d]
        else
            min_x, max_x = extrema(coords_vecs[d])
            return max_x - min_x
        end
    end

    # Generate physical wavenumbers consistent with FFTW/FINUFFT
    ks_phys = ntuple(
        d ->
            range(FT(-ms[d] ÷ 2), stop = FT((ms[d] - 1) ÷ 2), length = ms[d]) .*
            (FT(2π) / (ranges[d] == 0 ? one(FT) : ranges[d])),
        Val(D),
    )

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

    # Compute SHT direct summation - SERIAL with on-the-fly Legendre
    @inbounds for j in 1:N
        θj = θ[j]
        φj = φ[j]
        wj = w[j]

        xj = cos(θj)
        sj = sin(θj)

        for l in 0:lmax
            for m in -l:l
                abs_m = abs(m)
                P_l_m = _normalized_legendre(l, abs_m, xj, sj)

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

"""
    _normalized_legendre(l, m, x, s)

Compute normalized associated Legendre polynomial P_l^m(x) using recurrence.
On-the-fly computation avoids storing the full matrix.
"""
@inline function _normalized_legendre(l::Int, m::Int, x::FT, s::FT)::FT where FT
    m > l && return zero(FT)

    # P_m^m via sectoral recurrence
    P_mm = one(FT) / sqrt(FT(4π))
    @inbounds for mm in 1:m
        P_mm *= -sqrt(FT(2mm + 1) / (2mm)) * s
    end

    l == m && return P_mm

    # P_{m+1}^m using first-step recurrence
    P_lm = x * sqrt(FT(2m + 3)) * P_mm
    P_lminus1_m = P_mm

    l == m + 1 && return P_lm

    # Standard recurrence for higher l
    @inbounds for ll in (m+2):l
        coeff1 = sqrt(FT(4ll^2 - 1) / (ll^2 - m^2))
        coeff2 = sqrt(FT(2ll + 1) * ((ll - 1)^2 - m^2) / ((2ll - 3) * (ll^2 - m^2)))
        P_lminus1_m, P_lm = P_lm, x * coeff1 * P_lm - coeff2 * P_lminus1_m
    end

    return P_lm
end

end # module DirectSum
