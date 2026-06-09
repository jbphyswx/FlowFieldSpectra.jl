module FlowFieldSpectraFFTWExt

using FFTW: FFTW
using FlowFieldSpectra: FlowFieldSpectra as FFS, FFTBackend

"""
    calculate_spectrum(::FFTBackend, coords_vecs, fields_vecs, ms; ...)

Compute fast Fourier transform for uniform Cartesian grids using FFTW.
"""
function FFS._calculate_spectrum_fft(
    coords_vecs::Tuple,
    fields_vecs::Tuple,
    ms::Tuple;
    iflag::Int = 1,
    domain_size::Union{Nothing, Tuple} = nothing,
    kwargs...,
)
    D = length(coords_vecs)
    FT = eltype(coords_vecs[1])
    NU = length(fields_vecs)

    # Validate inputs
    for d in 1:D
        length(coords_vecs[d]) == prod(ms) || throw(DimensionMismatch("FFT requires coordinates length to match grid size prod(ms) = $(prod(ms))"))
    end
    for k in 1:NU
        length(fields_vecs[k]) == prod(ms) || throw(DimensionMismatch("FFT requires field components length to match grid size prod(ms) = $(prod(ms))"))
    end

    # Preallocate coefficients
    coeffs = zeros(Complex{FT}, ms..., NU)

    # Perform FFT for each component
    for k in 1:NU
        uk = reshape(fields_vecs[k], ms...)

        if iflag == 1
            coeffs_k = FFTW.fft(uk)
        else
            coeffs_k = FFTW.ifft(uk) .* prod(ms)
        end

        # Shift modes to match frequency ordering [-M/2, M/2-1]
        selectdim(coeffs, D + 1, k) .= FFTW.fftshift(coeffs_k)
    end

    # Scale by 1/N
    coeffs ./= prod(ms)

    # Physical wavenumbers
    ranges = ntuple(Val(D)) do d
        if domain_size !== nothing
            return domain_size[d]
        else
            min_x, max_x = extrema(coords_vecs[d])
            return max_x - min_x
        end
    end

    ks_phys = ntuple(
        d ->
            range(FT(-ms[d] ÷ 2), stop = FT((ms[d] - 1) ÷ 2), length = ms[d]) .*
            (FT(2π) / (ranges[d] == 0 ? one(FT) : ranges[d])),
        Val(D),
    )

    return coeffs, ks_phys
end

end # module FlowFieldSpectraFFTWExt
