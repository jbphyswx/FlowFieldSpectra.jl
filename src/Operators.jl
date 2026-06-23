module Operators

export spectral_divergence, spectral_vorticity, compensate, band_energy

# Physical wavenumber along axis `d` at spectral CartesianIndex `I`.
@inline _kd(ks_phys, d, I) = @inbounds ks_phys[d][I[d]]

"""
    spectral_divergence(ks_phys::Tuple, coeffs::AbstractArray{Complex{T}}) -> AbstractArray

Spectral divergence ``\\widehat{\\nabla\\cdot u} = i\\,\\sum_d k_d\\,\\hat u_d`` of a `D`-component
vector field whose Fourier coefficients are `coeffs` of shape `(ms..., D)`. Returns a
`(ms..., 1)` coefficient array; take its [`isotropic_spectrum`](@ref) for the divergence
(compressive) spectrum. Defined for `D = 1, 2, 3`.
"""
function spectral_divergence(ks_phys::Tuple, coeffs::AbstractArray{Complex{T}, N}) where {T, N}
    D = length(ks_phys)
    N == D + 1 || throw(ArgumentError("coeffs must have shape (ms..., NU)"))
    ms = size(coeffs)[1:D]
    NU = size(coeffs, N)
    NU == D || throw(ArgumentError("divergence needs NU = D = $D components, got $NU"))
    out = zeros(Complex{T}, ms..., 1)
    @inbounds for I in CartesianIndices(ms)
        acc = zero(Complex{T})
        for d in 1:D
            acc += im * T(_kd(ks_phys, d, I)) * coeffs[I, d]
        end
        out[I, 1] = acc
    end
    return out
end

"""
    spectral_vorticity(ks_phys::Tuple, coeffs::AbstractArray{Complex{T}}) -> AbstractArray

Spectral vorticity ``\\hat\\omega = i\\,k \\times \\hat u`` of a vector field with coefficients
`coeffs` of shape `(ms..., D)`:
- `D = 2`: scalar out-of-plane vorticity → `(ms..., 1)`.
- `D = 3`: 3-component vorticity vector → `(ms..., 3)`.

Take the [`isotropic_spectrum`](@ref) of the result for the enstrophy spectrum
``Z(k) = \\tfrac12 |\\hat\\omega|^2``. (`D = 1` has no curl.)
"""
function spectral_vorticity(ks_phys::Tuple, coeffs::AbstractArray{Complex{T}, N}) where {T, N}
    D = length(ks_phys)
    N == D + 1 || throw(ArgumentError("coeffs must have shape (ms..., NU)"))
    ms = size(coeffs)[1:D]
    NU = size(coeffs, N)
    if D == 2
        NU == 2 || throw(ArgumentError("2D vorticity needs 2 velocity components, got $NU"))
        out = zeros(Complex{T}, ms..., 1)
        @inbounds for I in CartesianIndices(ms)
            kx = T(_kd(ks_phys, 1, I))
            ky = T(_kd(ks_phys, 2, I))
            out[I, 1] = im * (kx * coeffs[I, 2] - ky * coeffs[I, 1])
        end
        return out
    elseif D == 3
        NU == 3 || throw(ArgumentError("3D vorticity needs 3 velocity components, got $NU"))
        out = zeros(Complex{T}, ms..., 3)
        @inbounds for I in CartesianIndices(ms)
            kx = T(_kd(ks_phys, 1, I))
            ky = T(_kd(ks_phys, 2, I))
            kz = T(_kd(ks_phys, 3, I))
            cx = coeffs[I, 1]
            cy = coeffs[I, 2]
            cz = coeffs[I, 3]
            out[I, 1] = im * (ky * cz - kz * cy)
            out[I, 2] = im * (kz * cx - kx * cz)
            out[I, 3] = im * (kx * cy - ky * cx)
        end
        return out
    else
        throw(ArgumentError("vorticity is undefined for D = $D (need 2 or 3)"))
    end
end

"""
    compensate(k_bins, E_k, p) -> Vector

Compensated spectrum ``k^p E(k)`` (e.g. `p = 5/3` for a Kolmogorov inertial-range plateau,
`p = 1` for a premultiplied / variance-preserving log-`k` plot, `p = 2` for the
2D-incompressible enstrophy identity ``Z(k) = k^2 E(k)``).
"""
compensate(k_bins::AbstractVector, E_k::AbstractVector, p::Real) = @. k_bins^p * E_k

"""
    band_energy(k_bins, E_k, k1, k2) -> Real

Energy integrated over the wavenumber band ``[k_1, k_2]``: ``\\int_{k_1}^{k_2} E(k)\\,dk`` via the
trapezoidal rule over the bins whose centers fall in the band.
"""
function band_energy(k_bins::AbstractVector{T}, E_k::AbstractVector, k1::Real, k2::Real) where {T}
    lo, hi = promote(T(k1), T(k2))
    total = zero(eltype(E_k))
    @inbounds for i in 1:(length(k_bins)-1)
        ka, kb = k_bins[i], k_bins[i+1]
        (kb < lo || ka > hi) && continue
        total += (kb - ka) * (E_k[i] + E_k[i+1]) / 2
    end
    return total
end

end # module Operators
