module Averaging

export welch_power_spectrum, coherence_spectrum

# =============================================================================
# Variance-reduced (Welch / ensemble) estimators. The trailing batch dims of the coefficient array are
# the independent segments / realizations: their periodograms are averaged before (Welch) or together
# with (coherence) the radial binning. `coeffs` is `(ms…, realization_batch…)`.
# =============================================================================

@inline function _radial_setup(ks_phys::NTuple{D, Any}, ms::NTuple{D, Int}, num_bins::Int, ::Type{T}) where {D, T}
    k_max = minimum(ntuple(d -> maximum(abs, ks_phys[d]), D))
    num_bins <= 0 && (num_bins = minimum(ms) ÷ 2)
    dk = k_max / num_bins
    k_bins = [T(0.5) * ((i - 1) * dk + i * dk) for i in 1:num_bins]
    kd2 = ntuple(d -> [T(v)^2 for v in ks_phys[d]], D)
    return num_bins, dk, k_max, k_bins, kd2
end

"""
    welch_power_spectrum(ks_phys::Tuple, coeffs; num_bins=0)

Variance-reduced (Welch / ensemble-averaged) isotropic power spectrum. The trailing batch dims of
`coeffs` `(ms…, realization…)` index independent segments/realizations whose periodograms are averaged
before radial binning. Returns `(k_bins, E_k)`.
"""
function welch_power_spectrum(ks_phys::Tuple, coeffs::AbstractArray{Complex{T}, N};
        num_bins::Int = 0) where {T, N}
    D = length(ks_phys)
    N >= D || throw(ArgumentError("coeffs must have ≥ $D spectral dims"))
    ms = ntuple(d -> size(coeffs, d), D)
    nreal = prod(ntuple(i -> size(coeffs, D + i), N - D); init = 1)
    num_bins, dk, k_max, k_bins, kd2 = _radial_setup(ks_phys, ms, num_bins, T)
    C = reshape(coeffs, prod(ms), nreal)
    E_k = zeros(T, num_bins)
    @inbounds for (mi, I) in enumerate(CartesianIndices(ms))
        kmag = zero(T)
        for d in 1:D
            kmag += kd2[d][I[d]]
        end
        kmag = sqrt(kmag)
        kmag > k_max && continue
        p = zero(T)
        for e in 1:nreal
            p += abs2(C[mi, e])
        end
        p /= nreal
        bin = clamp(floor(Int, kmag / dk) + 1, 1, num_bins)
        E_k[bin] += T(0.5) * p
    end
    E_k ./= dk
    return k_bins, E_k
end

"""
    coherence_spectrum(ks_phys::Tuple, cf, cg; num_bins=0) -> (k_bins, coherence², phase)

Magnitude-squared coherence ``\\gamma^2(k) = |S_{fg}|^2 / (S_{ff} S_{gg})`` and phase between two
fields whose coefficients `cf`, `cg` share `(ms…, realization…)`. Cross/auto spectra are averaged over
the realization batch **and** over the modes in each radial bin before the ratio is formed.
"""
function coherence_spectrum(ks_phys::Tuple, cf::AbstractArray{Complex{T}, N},
        cg::AbstractArray{Complex{T}, N}; num_bins::Int = 0) where {T, N}
    D = length(ks_phys)
    size(cf) == size(cg) || throw(DimensionMismatch("cf and cg must match"))
    N >= D || throw(ArgumentError("coeffs must have ≥ $D spectral dims"))
    ms = ntuple(d -> size(cf, d), D)
    nreal = prod(ntuple(i -> size(cf, D + i), N - D); init = 1)
    num_bins, dk, k_max, k_bins, kd2 = _radial_setup(ks_phys, ms, num_bins, T)
    F = reshape(cf, prod(ms), nreal)
    G = reshape(cg, prod(ms), nreal)
    Sff = zeros(T, num_bins)
    Sgg = zeros(T, num_bins)
    Sfg = zeros(Complex{T}, num_bins)
    @inbounds for (mi, I) in enumerate(CartesianIndices(ms))
        kmag = zero(T)
        for d in 1:D
            kmag += kd2[d][I[d]]
        end
        kmag = sqrt(kmag)
        kmag > k_max && continue
        bin = clamp(floor(Int, kmag / dk) + 1, 1, num_bins)
        sff = zero(T)
        sgg = zero(T)
        sfg = zero(Complex{T})
        for e in 1:nreal
            a = F[mi, e]
            b = G[mi, e]
            sff += abs2(a)
            sgg += abs2(b)
            sfg += a * conj(b)
        end
        Sff[bin] += sff
        Sgg[bin] += sgg
        Sfg[bin] += sfg
    end
    coherence² = zeros(T, num_bins)
    phase = zeros(T, num_bins)
    @inbounds for i in 1:num_bins
        denom = Sff[i] * Sgg[i]
        coherence²[i] = denom > 0 ? clamp(abs2(Sfg[i]) / denom, zero(T), one(T)) : zero(T)
        phase[i] = angle(Sfg[i])
    end
    return k_bins, coherence², phase
end

end # module Averaging
