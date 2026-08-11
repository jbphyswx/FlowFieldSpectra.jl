module Averaging

export welch_power_spectrum, welch_power_spectrum!, coherence_spectrum, coherence_spectrum!

# =============================================================================
# Variance-reduced (Welch / ensemble) estimators. The trailing batch dims of the coefficient array are
# the independent segments / realizations: their periodograms are averaged before (Welch) or together
# with (coherence) the radial binning. `coeffs` is `(ms…, realization_batch…)`.
# =============================================================================

# Fill `k_bins` (length nb) with radial bin centers; return `(dk, k_max)`. No allocation.
@inline function _fill_kbins!(k_bins::AbstractVector{T}, ks_phys::NTuple{D, Any}) where {T, D}
    k_max = minimum(ntuple(d -> maximum(abs, ks_phys[d]), D))
    nb = length(k_bins)
    dk = k_max / nb
    @inbounds for i in 1:nb
        k_bins[i] = T(0.5) * ((i - 1) * dk + i * dk)
    end
    return dk, k_max
end

@inline _resolve_nb(num_bins::Int, ms::NTuple) = num_bins > 0 ? num_bins : minimum(ms) ÷ 2

"""
    welch_power_spectrum!(E_k, k_bins, ks_phys, coeffs; num_bins=0) -> nothing

In-place, allocation-free [`welch_power_spectrum`](@ref): fills preallocated `E_k` and `k_bins` (both
length `num_bins`). Reusable across a loop with zero steady-state heap traffic.
"""
function welch_power_spectrum!(E_k::AbstractVector{T}, k_bins::AbstractVector{T}, ks_phys::Tuple,
        coeffs::AbstractArray{Complex{T}, N}; num_bins::Int = 0) where {T, N}
    D = length(ks_phys)
    N >= D || throw(ArgumentError("coeffs must have ≥ $D spectral dims"))
    ms = ntuple(d -> size(coeffs, d), D)
    nb = length(k_bins)
    length(E_k) == nb || throw(DimensionMismatch("E_k and k_bins must have equal length"))
    num_bins > 0 && num_bins != nb && throw(ArgumentError("num_bins=$num_bins ≠ length(k_bins)=$nb"))
    M = prod(ms)
    nreal = length(coeffs) ÷ M
    dk, k_max = _fill_kbins!(k_bins, ks_phys)
    fill!(E_k, zero(T))
    # Linear indexing `coeffs[mi + (e-1)·M]` (mode mi, realization e) avoids a reshape-header alloc.
    @inbounds for (mi, I) in enumerate(CartesianIndices(ms))
        kmag = zero(T)
        for d in 1:D
            kv = T(ks_phys[d][I[d]])
            kmag += kv * kv
        end
        kmag = sqrt(kmag)
        kmag > k_max && continue
        p = zero(T)
        for e in 1:nreal
            p += abs2(coeffs[mi + (e - 1) * M])
        end
        p /= nreal
        bin = clamp(floor(Int, kmag / dk) + 1, 1, nb)
        E_k[bin] += T(0.5) * p
    end
    E_k ./= dk
    return nothing
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
    nb = _resolve_nb(num_bins, ms)
    k_bins = Vector{T}(undef, nb)
    E_k = Vector{T}(undef, nb)
    welch_power_spectrum!(E_k, k_bins, ks_phys, coeffs)
    return k_bins, E_k
end

"""
    coherence_spectrum!(coherence², phase, k_bins, ks_phys, cf, cg; num_bins=0) -> nothing

In-place [`coherence_spectrum`](@ref): fills preallocated `coherence²`, `phase`, `k_bins` (each length
`num_bins`). Uses `O(num_bins)` internal scratch for the complex cross-spectrum accumulator.
"""
function coherence_spectrum!(coherence²::AbstractVector{T}, phase::AbstractVector{T},
        k_bins::AbstractVector{T}, ks_phys::Tuple, cf::AbstractArray{Complex{T}, N},
        cg::AbstractArray{Complex{T}, N}; num_bins::Int = 0) where {T, N}
    D = length(ks_phys)
    size(cf) == size(cg) || throw(DimensionMismatch("cf and cg must match"))
    N >= D || throw(ArgumentError("coeffs must have ≥ $D spectral dims"))
    ms = ntuple(d -> size(cf, d), D)
    nb = length(k_bins)
    (length(coherence²) == nb && length(phase) == nb) ||
        throw(DimensionMismatch("coherence², phase, k_bins must have equal length"))
    num_bins > 0 && num_bins != nb && throw(ArgumentError("num_bins=$num_bins ≠ length(k_bins)=$nb"))
    M = prod(ms)
    nreal = length(cf) ÷ M
    dk, k_max = _fill_kbins!(k_bins, ks_phys)
    # `coherence²`/`phase` double as the Sff/Sgg real accumulators; the complex cross-spectrum Sfg needs
    # its own accumulator (the only allocation — O(num_bins)).
    fill!(coherence², zero(T))
    fill!(phase, zero(T))
    Sfg = zeros(Complex{T}, nb)
    @inbounds for (mi, I) in enumerate(CartesianIndices(ms))
        kmag = zero(T)
        for d in 1:D
            kv = T(ks_phys[d][I[d]])
            kmag += kv * kv
        end
        kmag = sqrt(kmag)
        kmag > k_max && continue
        bin = clamp(floor(Int, kmag / dk) + 1, 1, nb)
        sff = zero(T)
        sgg = zero(T)
        sfg = zero(Complex{T})
        for e in 1:nreal
            a = cf[mi + (e - 1) * M]
            b = cg[mi + (e - 1) * M]
            sff += abs2(a)
            sgg += abs2(b)
            sfg += a * conj(b)
        end
        coherence²[bin] += sff
        phase[bin] += sgg
        Sfg[bin] += sfg
    end
    @inbounds for i in 1:nb
        denom = coherence²[i] * phase[i]                       # Sff · Sgg
        coherence²[i] = denom > 0 ? clamp(abs2(Sfg[i]) / denom, zero(T), one(T)) : zero(T)
        phase[i] = angle(Sfg[i])
    end
    return nothing
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
    nb = _resolve_nb(num_bins, ms)
    k_bins = Vector{T}(undef, nb)
    coherence² = Vector{T}(undef, nb)
    phase = Vector{T}(undef, nb)
    coherence_spectrum!(coherence², phase, k_bins, ks_phys, cf, cg)
    return k_bins, coherence², phase
end

end # module Averaging
