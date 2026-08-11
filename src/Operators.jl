module Operators

export spectral_divergence, spectral_divergence!, spectral_vorticity, spectral_vorticity!,
    compensate, band_energy

# =============================================================================
# Spectral-differentiation operators. The velocity components live on the FIRST batch dim (size `D`),
# so coefficients are `(ms…, D, extra_batch…)`; the extra trailing batch dims (levels/time/…) are
# carried through. Output keeps `(ms…, ncomp_out, extra_batch…)`.
# =============================================================================

# Column-major linear indexing into `(ms…, ncomp, extra…)`: element `(mi, c, e)` (mode `mi ∈ 1:M`,
# component `c`, extra-batch `e`) lives at `mi + (c-1)·M + (e-1)·M·ncomp`. Used instead of `reshape` so
# the `!` variants allocate nothing (a `reshape` header is a per-call allocation under bounds checking).
@inline _lin(mi, M, c, e, ncomp) = mi + (c - 1) * M + (e - 1) * M * ncomp

"""
    spectral_divergence!(out, ks_phys::Tuple, coeffs) -> out

In-place [`spectral_divergence`](@ref): writes the divergence into preallocated `out` of shape
`(ms…, 1, extra…)`. Allocation-free — reuse `out` across a batch/time loop.
"""
function spectral_divergence!(out::AbstractArray{Complex{T}}, ks_phys::Tuple,
        coeffs::AbstractArray{Complex{T}, N}) where {T, N}
    D = length(ks_phys)
    N >= D + 1 || throw(ArgumentError("coeffs must have shape (ms…, D, extra…)"))
    ms = ntuple(d -> size(coeffs, d), D)
    ncomp = size(coeffs, D + 1)
    ncomp == D || throw(ArgumentError("divergence needs D = $D components on dim $(D+1), got $ncomp"))
    extra = ntuple(i -> size(coeffs, D + 1 + i), N - D - 1)
    size(out) == (ms..., 1, extra...) ||
        throw(DimensionMismatch("out must have shape (ms…, 1, extra…) = $((ms..., 1, extra...)), got $(size(out))"))
    M = prod(ms)
    E = prod(extra; init = 1)
    @inbounds for (mi, I) in enumerate(CartesianIndices(ms))
        for e in 1:E
            acc = zero(Complex{T})
            for d in 1:D
                acc += im * T(ks_phys[d][I[d]]) * coeffs[_lin(mi, M, d, e, D)]
            end
            out[_lin(mi, M, 1, e, 1)] = acc
        end
    end
    return out
end

"""
    spectral_divergence(ks_phys::Tuple, coeffs) -> AbstractArray

Spectral divergence ``\\widehat{\\nabla\\cdot u} = i\\sum_d k_d \\hat u_d`` of a `D`-component vector
field with coefficients `(ms…, D, extra…)`. Returns `(ms…, 1, extra…)`. Defined for `D = 1, 2, 3`.
"""
function spectral_divergence(ks_phys::Tuple, coeffs::AbstractArray{Complex{T}, N}) where {T, N}
    D = length(ks_phys)
    N >= D + 1 || throw(ArgumentError("coeffs must have shape (ms…, D, extra…)"))
    ms = ntuple(d -> size(coeffs, d), D)
    extra = ntuple(i -> size(coeffs, D + 1 + i), N - D - 1)
    out = Array{Complex{T}}(undef, ms..., 1, extra...)
    return spectral_divergence!(out, ks_phys, coeffs)
end

"""
    spectral_vorticity!(out, ks_phys::Tuple, coeffs) -> out

In-place [`spectral_vorticity`](@ref): writes the vorticity into preallocated `out` — shape
`(ms…, 1, extra…)` for `D = 2`, `(ms…, 3, extra…)` for `D = 3`. Allocation-free.
"""
function spectral_vorticity!(out::AbstractArray{Complex{T}}, ks_phys::Tuple,
        coeffs::AbstractArray{Complex{T}, N}) where {T, N}
    D = length(ks_phys)
    N >= D + 1 || throw(ArgumentError("coeffs must have shape (ms…, D, extra…)"))
    ms = ntuple(d -> size(coeffs, d), D)
    ncomp = size(coeffs, D + 1)
    extra = ntuple(i -> size(coeffs, D + 1 + i), N - D - 1)
    M = prod(ms)
    E = prod(extra; init = 1)
    if D == 2
        ncomp == 2 || throw(ArgumentError("2D vorticity needs 2 components, got $ncomp"))
        size(out) == (ms..., 1, extra...) ||
            throw(DimensionMismatch("out must have shape (ms…, 1, extra…) = $((ms..., 1, extra...)), got $(size(out))"))
        @inbounds for (mi, I) in enumerate(CartesianIndices(ms))
            kx = T(ks_phys[1][I[1]])
            ky = T(ks_phys[2][I[2]])
            for e in 1:E
                out[_lin(mi, M, 1, e, 1)] =
                    im * (kx * coeffs[_lin(mi, M, 2, e, 2)] - ky * coeffs[_lin(mi, M, 1, e, 2)])
            end
        end
        return out
    elseif D == 3
        ncomp == 3 || throw(ArgumentError("3D vorticity needs 3 components, got $ncomp"))
        size(out) == (ms..., 3, extra...) ||
            throw(DimensionMismatch("out must have shape (ms…, 3, extra…) = $((ms..., 3, extra...)), got $(size(out))"))
        @inbounds for (mi, I) in enumerate(CartesianIndices(ms))
            kx = T(ks_phys[1][I[1]])
            ky = T(ks_phys[2][I[2]])
            kz = T(ks_phys[3][I[3]])
            for e in 1:E
                cx = coeffs[_lin(mi, M, 1, e, 3)]
                cy = coeffs[_lin(mi, M, 2, e, 3)]
                cz = coeffs[_lin(mi, M, 3, e, 3)]
                out[_lin(mi, M, 1, e, 3)] = im * (ky * cz - kz * cy)
                out[_lin(mi, M, 2, e, 3)] = im * (kz * cx - kx * cz)
                out[_lin(mi, M, 3, e, 3)] = im * (kx * cy - ky * cx)
            end
        end
        return out
    else
        throw(ArgumentError("vorticity is undefined for D = $D (need 2 or 3)"))
    end
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
    extra = ntuple(i -> size(coeffs, D + 1 + i), N - D - 1)
    ncomp_out = D == 2 ? 1 : D == 3 ? 3 : throw(ArgumentError("vorticity is undefined for D = $D (need 2 or 3)"))
    out = Array{Complex{T}}(undef, ms..., ncomp_out, extra...)
    return spectral_vorticity!(out, ks_phys, coeffs)
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
