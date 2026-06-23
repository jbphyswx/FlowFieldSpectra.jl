module Averaging

export welch_power_spectrum, coherence_spectrum

# Shared radial-bin setup: returns (num_bins, dk, k_max, k_bins, kd2).
@inline function _radial_bins(ks_phys::NTuple{D, Any}, ms::NTuple{D, Int}, num_bins::Int,
        ::Type{T}) where {D, T}
    k_max = minimum(ntuple(d -> maximum(abs.(ks_phys[d])), D))
    num_bins <= 0 && (num_bins = minimum(ms) ÷ 2)
    bin_edges = range(zero(T), stop = k_max, length = num_bins + 1)
    dk = k_max / num_bins
    k_bins = [T(0.5) * (bin_edges[i] + bin_edges[i+1]) for i in 1:num_bins]
    kd2 = [T.(ks_phys[d]) .^ 2 for d in 1:D]
    return num_bins, dk, k_max, k_bins, kd2
end

"""
    welch_power_spectrum(ks_phys::Tuple, coeffs::AbstractArray{Complex{T}}; num_bins=0)

Variance-reduced (Welch / ensemble-averaged) isotropic power spectrum. `coeffs` has shape
`(ms..., n_realizations)` — the trailing axis indexes independent segments/realizations whose
periodograms are averaged before radial binning. A single periodogram has ~100% variance
regardless of resolution; averaging `M` of them cuts it by ~`1/M`. Returns `(k_bins, E_k)`.
"""
function welch_power_spectrum(ks_phys::Tuple, coeffs::AbstractArray{Complex{T}, N_dim};
        num_bins::Int = 0) where {T<:AbstractFloat, N_dim}
    D = length(ks_phys)
    @assert N_dim == D + 1 "coeffs must have shape (ms..., n_realizations)"
    ms = size(coeffs)[1:D]
    nreal = size(coeffs, N_dim)
    num_bins, dk, k_max, k_bins, kd2 = _radial_bins(ks_phys, ms, num_bins, T)
    E_k = zeros(T, num_bins)

    @inbounds for I in CartesianIndices(ms)
        k_mag = zero(T)
        for d in 1:D
            k_mag += kd2[d][I[d]]
        end
        k_mag = sqrt(k_mag)
        k_mag > k_max && continue
        p = zero(T)
        for e in 1:nreal
            p += abs2(coeffs[I, e])
        end
        p /= nreal                              # mean periodogram across realizations
        bin = clamp(floor(Int, k_mag / dk) + 1, 1, num_bins)
        E_k[bin] += T(0.5) * p
    end
    E_k ./= dk
    return k_bins, E_k
end

"""
    coherence_spectrum(ks_phys::Tuple, cf, cg; num_bins=0) -> (k_bins, coherence², phase)

Magnitude-squared coherence ``\\gamma^2(k) = |S_{fg}|^2 / (S_{ff} S_{gg})`` and phase
``\\phi(k) = \\angle S_{fg}`` between two fields whose coefficients `cf`, `cg` share shape
`(ms..., n_realizations)`. The cross- and auto-spectra are averaged over the realization axis
**and** over the modes in each radial bin before forming the ratio — without that averaging
coherence is identically 1 and meaningless. `γ² ∈ [0,1]`: 1 = perfectly linearly related at that
scale, 0 = uncorrelated.
"""
function coherence_spectrum(ks_phys::Tuple, cf::AbstractArray{Complex{T}, N_dim},
        cg::AbstractArray{Complex{T}, N_dim}; num_bins::Int = 0) where {T<:AbstractFloat, N_dim}
    D = length(ks_phys)
    @assert N_dim == D + 1 "coeffs must have shape (ms..., n_realizations)"
    size(cf) == size(cg) || throw(DimensionMismatch("cf and cg must match"))
    ms = size(cf)[1:D]
    nreal = size(cf, N_dim)
    num_bins, dk, k_max, k_bins, kd2 = _radial_bins(ks_phys, ms, num_bins, T)

    Sff = zeros(T, num_bins)
    Sgg = zeros(T, num_bins)
    Sfg = zeros(Complex{T}, num_bins)
    @inbounds for I in CartesianIndices(ms)
        k_mag = zero(T)
        for d in 1:D
            k_mag += kd2[d][I[d]]
        end
        k_mag = sqrt(k_mag)
        k_mag > k_max && continue
        bin = clamp(floor(Int, k_mag / dk) + 1, 1, num_bins)
        sff = zero(T)
        sgg = zero(T)
        sfg = zero(Complex{T})
        for e in 1:nreal
            a = cf[I, e]
            b = cg[I, e]
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
