module DirectSum

using LinearAlgebra: LinearAlgebra as LA
using StaticArrays: StaticArrays as SA
using Base.Threads: Threads

using ..Types: DirectSumBackend

export calculate_spectrum_direct, sph_mode_index

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

    # Validate inputs
    for d in 1:D
        length(coords_vecs[d]) == N || throw(DimensionMismatch("Coordinates length mismatch"))
    end
    for u_idx in 1:NU
        length(fields_vecs[u_idx]) == N || throw(DimensionMismatch("Field length mismatch"))
    end

    if D == 2 && all(extrema(coords_vecs[1]) .<= (π + 1e-3)) && all(extrema(coords_vecs[1]) .>= -1e-5) &&
       all(extrema(coords_vecs[2]) .<= (2π + 1e-3)) && all(extrema(coords_vecs[2]) .>= -1e-5) &&
       (ms[2] == 2 * ms[1] - 1) # spherical signature (theta, phi) with Nphi = 2*Ntheta - 1
        # Spherical coordinate fallback: (theta, phi)
        # ms[1] is lmax + 1, ms[2] is 2*lmax + 1
        lmax = ms[1] - 1
        return _calculate_spectrum_spherical_direct(coords_vecs, fields_vecs, lmax, weights)
    else
        # Cartesian coordinate path
        return _calculate_spectrum_cartesian_direct(coords_vecs, fields_vecs, ms, iflag, domain_size)
    end
end

# Cartesian Direct Sum Transform
function _calculate_spectrum_cartesian_direct(
    coords_vecs::Tuple{T1, Vararg{T1}},
    fields_vecs::Tuple{T2, Vararg{T2}},
    ms::NTuple{D, Int},
    iflag::Int,
    domain_size::Union{Nothing, Tuple} = nothing,
) where {D, T1, T2}
    FT = eltype(coords_vecs[1])
    N = length(coords_vecs[1])
    NU = length(fields_vecs)

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

    # Preallocate coefficients
    coeffs = zeros(Complex{FT}, ms..., NU)

    # O(N * M) Cartesian direct Fourier sum
    Threads.@threads for I in CartesianIndices(ms)
        k_phys = SA.SVector{D, FT}(ntuple(d -> ks_phys[d][I[d]], Val(D)))

        for j in 1:N
            x_pos = SA.SVector{D, FT}(ntuple(d -> coords_vecs[d][j], Val(D)))

            phi = -iflag * (LA.dot(k_phys, x_pos))
            W = exp(im * phi)

            for u_idx in 1:NU
                coeffs[I, u_idx] += fields_vecs[u_idx][j] * W
            end
        end
    end

    coeffs ./= N
    return coeffs, ks_phys
end

# Spherical Direct SHT Projection
function _calculate_spectrum_spherical_direct(
    coords_vecs::Tuple,
    fields_vecs::Tuple,
    lmax::Int,
    weights::Union{Nothing, AbstractVector},
)
    FT = eltype(coords_vecs[1])
    N = length(coords_vecs[1])
    NU = length(fields_vecs)
    Nθ = lmax + 1
    Nφ = 2 * lmax + 1

    θ = coords_vecs[1]
    φ = coords_vecs[2]

    # Preallocate coefficients
    coeffs = zeros(Complex{FT}, Nθ, Nφ, NU)

    # Use uniform weights if not provided
    w = weights === nothing ? fill(FT(4π) / N, N) : weights

    # Temporary thread-local coefficient buffers to avoid data races
    n_threads = Threads.nthreads()
    thread_coeffs = [zeros(Complex{FT}, Nθ, Nφ, NU) for _ in 1:n_threads]

    # Compute SHT direct summation
    Threads.@threads for j in 1:N
        tid = Threads.threadid()
        θj = θ[j]
        φj = φ[j]
        wj = w[j]

        xj = cos(θj)
        sj = sin(θj)

        # Precompute Legendre polynomials for this point
        # P[l+1, m+1] corresponds to normalized P_l^m(xj) for m >= 0
        P = zeros(FT, Nθ, Nθ)

        # 1. Sectoral recurrence m == 0
        P[1, 1] = one(FT) / sqrt(FT(4π))
        
        # Sectoral recurrence for m > 0
        for m in 1:lmax
            P[m+1, m+1] = -sqrt(FT(2m+1) / (2m)) * sj * P[m, m]
        end

        # 2. Recurrence for l > m
        for m in 0:lmax
            if m + 1 <= lmax
                P[m+2, m+1] = xj * sqrt(FT(2m+3)) * P[m+1, m+1]
            end
            for l in (m+2):lmax
                P[l+1, m+1] = xj * sqrt(FT(4l^2 - 1) / (l^2 - m^2)) * P[l, m+1] -
                              sqrt(FT(2l+1) * ((l-1)^2 - m^2) / ((2l-3) * (l^2 - m^2))) * P[l-1, m+1]
            end
        end

        # Accumulate projection for each l, m
        for l in 0:lmax
            for m in -l:l
                # Y_l^m = P_l^|m| * exp(i*m*phi)
                # If m < 0, Y_l^m = (-1)^m * P_l^|m| * exp(i*m*phi)
                abs_m = abs(m)
                factor = (m < 0 && isodd(abs_m)) ? -one(FT) : one(FT)
                
                phase = exp(im * m * φj)
                Y_lm = factor * P[l+1, abs_m+1] * phase

                # Adjoint projection coefficient
                idx = sph_mode_index(l, m)
                for u_idx in 1:NU
                    thread_coeffs[tid][idx, u_idx] += fields_vecs[u_idx][j] * conj(Y_lm) * wj
                end
            end
        end
    end

    # Sum up thread-local buffers
    for tid in 1:n_threads
        coeffs .+= thread_coeffs[tid]
    end

    return coeffs, (0:lmax, -lmax:lmax)
end

end # module DirectSum
