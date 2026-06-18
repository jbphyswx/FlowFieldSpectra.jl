module FlowFieldSpectraOhMyThreadsExt

using OhMyThreads: OhMyThreads as OMT
using FlowFieldSpectra: FlowFieldSpectra as FFS

# =============================================================================
# Threaded DirectSum Dispatch
# =============================================================================

function FFS._calculate_spectrum_threaded(
    coords_vecs::Tuple,
    fields_vecs::Tuple,
    ms::Tuple;
    kwargs...
)
    D = length(coords_vecs)
    NU = length(fields_vecs)
    FT = eltype(coords_vecs[1])

    # Determine coordinate type (Spherical or Cartesian)
    if D == 2 && all(extrema(coords_vecs[1]) .<= (π + 1e-3)) && all(extrema(coords_vecs[1]) .>= -1e-5) &&
       all(extrema(coords_vecs[2]) .<= (2π + 1e-3)) && all(extrema(coords_vecs[2]) .>= -1e-5) &&
       (ms[2] == 2 * ms[1] - 1)
        # Spherical
        lmax = ms[1] - 1
        Nθ = lmax + 1
        Nφ = 2 * lmax + 1
        coeffs = zeros(Complex{FT}, Nθ, Nφ, NU)
        ks = FFS._calculate_spectrum_threaded!(coeffs, coords_vecs, fields_vecs, ms; kwargs...)
        return (coeffs, ks)
    else
        # Cartesian
        coeffs = zeros(Complex{FT}, ms..., NU)
        ks = FFS._calculate_spectrum_threaded!(coeffs, coords_vecs, fields_vecs, ms; kwargs...)
        return (coeffs, ks)
    end
end

function FFS._calculate_spectrum_threaded!(
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
        return _calculate_spectrum_spherical_threaded!(coeffs, coords_vecs, fields_vecs, lmax, weights)
    else
        # Cartesian coordinate path
        return _calculate_spectrum_cartesian_threaded!(coeffs, coords_vecs, fields_vecs, ms, iflag, domain_size)
    end
end

# =============================================================================
# Threaded Cartesian Direct Sum
# =============================================================================

function _calculate_spectrum_cartesian_threaded!(
    coeffs::AbstractArray{Complex{FT}},
    coords_vecs::Tuple,
    fields_vecs::Tuple,
    ms::Tuple,
    iflag::Int,
    domain_size::Union{Nothing, Tuple} = nothing,
) where {FT}
    D = length(coords_vecs)
    NU = length(fields_vecs)
    N = length(coords_vecs[1])

    # Coordinate ranges for physical wavenumbers
    ranges = ntuple(Val(D)) do d
        if domain_size !== nothing
            return domain_size[d]
        else
            min_x, max_x = extrema(coords_vecs[d])
            return max_x - min_x
        end
    end

    # Generate physical wavenumbers
    ks_phys = ntuple(
        d ->
            range(FT(-ms[d] ÷ 2), stop = FT((ms[d] - 1) ÷ 2), length = ms[d]) .*
            (FT(2π) / (ranges[d] == 0 ? one(FT) : ranges[d])),
        Val(D),
    )

    # Zero out coeffs
    fill!(coeffs, zero(Complex{FT}))

    # Threaded accumulation using OMT.@tasks (race-condition safe)
    OMT.@tasks for I in CartesianIndices(ms)
        @inbounds for j in 1:N
            phi = zero(FT)
            for d in 1:D
                phi += ks_phys[d][I[d]] * coords_vecs[d][j]
            end
            phi = -iflag * phi

            W = cis(phi)

            for u_idx in 1:NU
                coeffs[I, u_idx] += fields_vecs[u_idx][j] * W
            end
        end
    end

    coeffs ./= N
    return ks_phys
end

# =============================================================================
# Threaded Spherical Direct Sum
# =============================================================================

function _calculate_spectrum_spherical_threaded!(
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

    expected_size = (Nθ, Nφ, NU)
    size(coeffs) == expected_size || throw(DimensionMismatch("coeffs size $(size(coeffs)) != expected $expected_size"))

    θ = coords_vecs[1]
    φ = coords_vecs[2]
    w = weights === nothing ? fill(FT(4π) / N, N) : weights

    # Zero out coeffs
    fill!(coeffs, zero(Complex{FT}))

    # Use tmapreduce for race-condition-free accumulation
    partial_coeffs = OMT.tmapreduce(
        j -> _compute_point_contribution_threaded(j, θ, φ, fields_vecs, w, lmax, Nθ, Nφ, NU, FT),
        (a, b) -> begin
            a[1] .+= b[1]
            a
        end,
        1:N,
        init = (zeros(Complex{FT}, Nθ, Nφ, NU),)
    )

    copyto!(coeffs, partial_coeffs[1])

    return (0:lmax, -lmax:lmax)
end

@inline function _compute_point_contribution_threaded(
    j, θ, φ, fields_vecs, w, lmax, Nθ, Nφ, NU, FT
)
    θj = θ[j]
    φj = φ[j]
    wj = w[j]

    xj = cos(θj)
    sj = sin(θj)

    local_coeffs = zeros(Complex{FT}, Nθ, Nφ, NU)

    @inbounds for l in 0:lmax
        for m in -l:l
            abs_m = abs(m)
            P_l_m = _normalized_legendre_threaded(l, abs_m, xj, sj)

            factor = (m < 0 && isodd(abs_m)) ? -one(FT) : one(FT)
            phase = cis(m * φj)
            Y_lm = factor * P_l_m * phase

            idx = FFS.sph_mode_index(l, m)
            fj_conj_Ylm_wj = conj(Y_lm) * wj
            for u_idx in 1:NU
                local_coeffs[idx, u_idx] += fields_vecs[u_idx][j] * fj_conj_Ylm_wj
            end
        end
    end

    return (local_coeffs,)
end

@inline function _normalized_legendre_threaded(l::Int, m::Int, x::FT, s::FT)::FT where FT
    m > l && return zero(FT)

    # Sectoral recurrence
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

end # module FlowFieldSpectraOhMyThreadsExt
