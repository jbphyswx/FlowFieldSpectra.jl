module FlowFieldSpectraFINUFFTExt

using FINUFFT: FINUFFT
using FlowFieldSpectra: FlowFieldSpectra as FFS, NUFFTBackend

"""
    calculate_spectrum(::NUFFTBackend, coords_vecs, fields_vecs, ms; eps=1e-9, iflag=1, domain_size=nothing, ...)

Compute non-uniform fast Fourier transform using FINUFFT.
"""
function FFS._calculate_spectrum_nufft(
    coords_vecs::Tuple,
    fields_vecs::Tuple,
    ms::Tuple;
    eps::Real = 1e-9,
    iflag::Int = 1,
    domain_size::Union{Nothing, Tuple} = nothing,
    kwargs...,
)
    D = length(coords_vecs)
    NU = length(fields_vecs)
    N = length(coords_vecs[1])
    FT = eltype(coords_vecs[1])

    # Validate dimensions
    D <= 3 || throw(ArgumentError("FINUFFT only supports up to 3 dimensions"))
    for d in 1:D
        length(coords_vecs[d]) == N || throw(DimensionMismatch("Coordinates length mismatch"))
    end
    for k in 1:NU
        length(fields_vecs[k]) == N || throw(DimensionMismatch("Field length mismatch"))
    end

    # 1. Filter out NaNs/Infs
    valid_mask = trues(N)
    for d in 1:D
        valid_mask .&= isfinite.(coords_vecs[d])
    end
    for k in 1:NU
        valid_mask .&= isfinite.(fields_vecs[k])
    end

    N_valid = count(valid_mask)
    if N_valid == 0
        return zeros(Complex{FT}, ms..., NU),
               ntuple(i -> range(zero(FT), stop = zero(FT), length = ms[i]), Val(D))
    end

    # 2. Rescale coordinates to [0, 2π] range as required by FINUFFT Type 1
    scaled_x = ntuple(Val(D)) do d
        x = coords_vecs[d][valid_mask]
        min_x, max_x = extrema(x)
        range_calc = max_x - min_x
        range_x = domain_size === nothing ? range_calc : domain_size[d]

        if range_x ≈ 0
            return (zeros(FT, N_valid), min_x, one(FT))
        end
        return (FT(2π) .* (x .- min_x) ./ range_x, min_x, range_x)
    end

    xs = ntuple(i -> scaled_x[i][1], Val(D))
    offsets = ntuple(i -> scaled_x[i][2], Val(D))
    ranges_calc = ntuple(i -> scaled_x[i][3], Val(D))
    ranges = domain_size === nothing ? ranges_calc : domain_size

    # 3. Call FINUFFT Type 1 (non-uniform points to uniform modes)
    coeffs = zeros(Complex{FT}, ms..., NU)

    for k in 1:NU
        uk = Complex{FT}.(fields_vecs[k][valid_mask])
        coeffs_k = zeros(Complex{FT}, ms...)

        if D == 1
            FINUFFT.nufft1d1!(xs[1], uk, -iflag, eps, coeffs_k)
        elseif D == 2
            FINUFFT.nufft2d1!(xs[1], xs[2], uk, -iflag, eps, coeffs_k)
        elseif D == 3
            FINUFFT.nufft3d1!(xs[1], xs[2], xs[3], uk, -iflag, eps, coeffs_k)
        end
        Base.selectdim(coeffs, D + 1, k) .= coeffs_k
    end

    # 4. Phase shift to correct for min_x coordinate offset (Translation Property)
    # If we mapped x -> (x - min_x) * 2π/L, we must multiply the modes by exp(ik * min_x * 2π/L)
    ks = ntuple(
        i -> range(FT(-ms[i] ÷ 2), stop = FT((ms[i] - 1) ÷ 2), length = ms[i]),
        Val(D),
    )

    for d in 1:D
        k_vec = ks[d]
        L = ranges[d]
        # Match direction of transform
        phase = exp.(im .* (-iflag) .* k_vec .* (offsets[d] * FT(2π) / L))

        # Apply phase correction along dimension d
        if D == 1
            for k in 1:NU
                coeffs[:, k] .*= phase
            end
        elseif D == 2
            if d == 1
                for k in 1:NU, j in 1:ms[2]
                    coeffs[:, j, k] .*= phase
                end
            else
                for k in 1:NU, i in 1:ms[1]
                    coeffs[i, :, k] .*= phase
                end
            end
        elseif D == 3
            if d == 1
                for k in 1:NU, j in 1:ms[2], l in 1:ms[3]
                    coeffs[:, j, l, k] .*= phase
                end
            elseif d == 2
                for k in 1:NU, i in 1:ms[1], l in 1:ms[3]
                    coeffs[i, :, l, k] .*= phase
                end
            else
                for k in 1:NU, i in 1:ms[1], j in 1:ms[2]
                    coeffs[i, j, :, k] .*= phase
                end
            end
        end
    end

    # 5. Scale by 1/N
    coeffs ./= N_valid

    # 6. Physical wavenumbers
    ks_phys = ntuple(
        d -> ks[d] .* (FT(2π) / (ranges[d] == 0 ? one(FT) : ranges[d])),
        Val(D),
    )

    return coeffs, ks_phys
end

end # module FlowFieldSpectraFINUFFTExt
