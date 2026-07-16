module Operators

export spectral_divergence, spectral_vorticity, compensate, band_energy

# =============================================================================
# Spectral-differentiation operators. The velocity components live on the FIRST batch dim (size `D`),
# so coefficients are `(ms…, D, extra_batch…)`; the extra trailing batch dims (levels/time/…) are
# carried through. Output keeps `(ms…, ncomp_out, extra_batch…)`.
# =============================================================================

# Reshape `(ms…, D, extra…)` ↔ `(M, D, E)` (M = ∏ms spectral modes, E = ∏extra batch slices).
@inline _mDE(a, M, D, E) = reshape(a, M, D, E)

"""
    spectral_divergence(ks_phys::Tuple, coeffs) -> AbstractArray

Spectral divergence ``\\widehat{\\nabla\\cdot u} = i\\sum_d k_d \\hat u_d`` of a `D`-component vector
field with coefficients `(ms…, D, extra…)`. Returns `(ms…, 1, extra…)`. Defined for `D = 1, 2, 3`.
"""
function spectral_divergence(ks_phys::Tuple, coeffs::AbstractArray{Complex{T}, N}) where {T, N}
    D = length(ks_phys)
    N >= D + 1 || throw(ArgumentError("coeffs must have shape (ms…, D, extra…)"))
    ms = ntuple(d -> size(coeffs, d), D)
    ncomp = size(coeffs, D + 1)
    ncomp == D || throw(ArgumentError("divergence needs D = $D components on dim $(D+1), got $ncomp"))
    extra = ntuple(i -> size(coeffs, D + 1 + i), N - D - 1)
    M = prod(ms)
    E = prod(extra; init = 1)
    out = zeros(Complex{T}, ms..., 1, extra...)
    C = _mDE(coeffs, M, D, E)
    O = _mDE(out, M, 1, E)
    @inbounds for (mi, I) in enumerate(CartesianIndices(ms))
        for e in 1:E
            acc = zero(Complex{T})
            for d in 1:D
                acc += im * T(ks_phys[d][I[d]]) * C[mi, d, e]
            end
            O[mi, 1, e] = acc
        end
    end
    return out
end

"""
    spectral_vorticity(ks_phys::Tuple, coeffs) -> AbstractArray

Spectral vorticity ``\\hat\\omega = i\\,k \\times \\hat u`` of a vector field with coefficients
`(ms…, D, extra…)`: `D = 2` → scalar out-of-plane vorticity `(ms…, 1, extra…)`; `D = 3` → 3-component
vorticity `(ms…, 3, extra…)`.
"""
function spectral_vorticity(ks_phys::Tuple, coeffs::AbstractArray{Complex{T}, N}) where {T, N}
    D = length(ks_phys)
    N >= D + 1 || throw(ArgumentError("coeffs must have shape (ms…, D, extra…)"))
    ms = ntuple(d -> size(coeffs, d), D)
    ncomp = size(coeffs, D + 1)
    extra = ntuple(i -> size(coeffs, D + 1 + i), N - D - 1)
    M = prod(ms)
    E = prod(extra; init = 1)
    if D == 2
        ncomp == 2 || throw(ArgumentError("2D vorticity needs 2 components, got $ncomp"))
        out = zeros(Complex{T}, ms..., 1, extra...)
        C = _mDE(coeffs, M, 2, E)
        O = _mDE(out, M, 1, E)
        @inbounds for (mi, I) in enumerate(CartesianIndices(ms))
            kx = T(ks_phys[1][I[1]])
            ky = T(ks_phys[2][I[2]])
            for e in 1:E
                O[mi, 1, e] = im * (kx * C[mi, 2, e] - ky * C[mi, 1, e])
            end
        end
        return out
    elseif D == 3
        ncomp == 3 || throw(ArgumentError("3D vorticity needs 3 components, got $ncomp"))
        out = zeros(Complex{T}, ms..., 3, extra...)
        C = _mDE(coeffs, M, 3, E)
        O = _mDE(out, M, 3, E)
        @inbounds for (mi, I) in enumerate(CartesianIndices(ms))
            kx = T(ks_phys[1][I[1]])
            ky = T(ks_phys[2][I[2]])
            kz = T(ks_phys[3][I[3]])
            for e in 1:E
                cx = C[mi, 1, e]
                cy = C[mi, 2, e]
                cz = C[mi, 3, e]
                O[mi, 1, e] = im * (ky * cz - kz * cy)
                O[mi, 2, e] = im * (kz * cx - kx * cz)
                O[mi, 3, e] = im * (kx * cy - ky * cx)
            end
        end
        return out
    else
        throw(ArgumentError("vorticity is undefined for D = $D (need 2 or 3)"))
    end
end

"""
    compensate(k_bins, E_k, p) -> AbstractArray

Compensated spectrum ``k^p E(k)`` (e.g. `p = 5/3` Kolmogorov plateau, `p = 2` for `Z(k)=k²E(k)`).
`E_k` may be `(num_bins,)` or `(num_bins, batch…)`; `k_bins` broadcasts along the wavenumber axis.
"""
compensate(k_bins::AbstractVector, E_k::AbstractArray, p::Real) = (k_bins .^ p) .* E_k

"""
    band_energy(k_bins, E_k, k1, k2) -> Real

Energy integrated over the wavenumber band ``[k_1, k_2]`` via the trapezoidal rule over the bins whose
centers fall in the band. `E_k` is a 1D spectrum `(num_bins,)`.
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
