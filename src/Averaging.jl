module Averaging

using ..Packing: Packing
using ..Normalization: Normalization
using ..Reductions: Reductions

export welch_power_spectrum, welch_power_spectrum!, coherence_spectrum, coherence_spectrum!

# =============================================================================
# Variance-reduced (Welch / ensemble) estimators. The trailing batch dims of the coefficient array are
# the independent segments / realizations: their periodograms are averaged before (Welch) or together
# with (coherence) the radial binning. `coeffs` is `(ms…, realization_batch…)`.
# =============================================================================

@inline _resolve_nb(num_bins::Int, ks_phys::Tuple, ::Type{T}, cut::Val) where {T} =
    first(Reductions._radial_extent(ks_phys, num_bins, T, cut))

"""
    welch_power_spectrum!(E_k, k_bins, ks_phys, coeffs; num_bins=0, cutoff=true,
                          convention=SpectralConvention()) -> nothing

In-place, allocation-free [`welch_power_spectrum`](@ref): fills preallocated `E_k` and `k_bins` (both
length `num_bins`). Reusable across a loop with zero steady-state heap traffic.
"""
function welch_power_spectrum!(E_k::AbstractVector{T}, k_bins::AbstractVector{T}, ks_phys::Tuple,
        coeffs::AbstractArray{Complex{T}, N}; num_bins::Int = 0, cutoff::Bool = true,
        convention::Normalization.SpectralConvention = Normalization.SpectralConvention()) where {T, N}
    D = length(ks_phys)
    N >= D || throw(ArgumentError("coeffs must have ≥ $D spectral dims"))
    Reductions._radial_sided(convention.sided)
    length(E_k) == length(k_bins) || throw(DimensionMismatch("E_k and k_bins must have equal length"))
    req = Reductions._requested_bins(k_bins, num_bins, ks_phys, T, cutoff)
    cutoff ? _fill_welch!(E_k, k_bins, ks_phys, coeffs, convention, req, Val(true)) :
             _fill_welch!(E_k, k_bins, ks_phys, coeffs, convention, req, Val(false))
    return nothing
end

# `k_bins` holds the bins' mode counts until the bin centres are written into it.
function _fill_welch!(E_k::AbstractVector{T}, k_bins::AbstractVector{T}, ks_phys::Tuple,
        coeffs::AbstractArray{Complex{T}, N}, convention::Normalization.SpectralConvention, req::Int,
        cut::Val{C}) where {T, N, C}
    D = length(ks_phys)
    nb, dk, k_top = Reductions._radial_extent(ks_phys, req, T, cut)
    fill!(E_k, zero(T))
    fill!(k_bins, zero(T))
    twin = Packing.axis_twin(first(ks_phys))
    real_idx = CartesianIndices(ntuple(i -> size(coeffs, D + i), Val(N - D)))
    nreal = length(real_idx)
    @inbounds for I in CartesianIndices(ntuple(d -> size(coeffs, d), Val(D)))
        kmag = sqrt(Packing.ksq(T, ks_phys, I))
        C && kmag > k_top && continue
        n = T(Packing.mode_fold(ks_phys, I))
        p = zero(T)
        for Ir in real_idx
            J = CartesianIndex(I, Ir)
            p += C ? n * abs2(coeffs[J]) : Reductions._native_energy(ks_phys, coeffs, J, twin)
        end
        bin = clamp(floor(Int, kmag / dk) + 1, 1, nb)
        k_bins[bin] += n
        E_k[bin] += T(0.5) * p / nreal
    end
    Reductions._shell_values!(E_k, k_bins, dk, ks_phys, convention.estimator)
    Reductions._scale_bins!(E_k, dk, convention.scaling)
    @inbounds for i in 1:nb
        k_bins[i] = T(0.5) * ((i - 1) * dk + i * dk)
    end
    return E_k
end

"""
    welch_power_spectrum(ks_phys::Tuple, coeffs; num_bins=0, cutoff=true, convention=SpectralConvention())

Variance-reduced (Welch / ensemble-averaged) isotropic power spectrum. The trailing batch dims of
`coeffs` `(ms…, realization…)` index independent segments/realizations whose periodograms are averaged
before radial binning. `cutoff` and `convention` are read as in [`isotropic_spectrum`](@ref). Returns
`(k_bins, E_k)`.
"""
function welch_power_spectrum(ks_phys::Tuple, coeffs::AbstractArray{Complex{T}, N};
        num_bins::Int = 0, cutoff::Bool = true,
        convention::Normalization.SpectralConvention = Normalization.SpectralConvention()) where {T, N}
    D = length(ks_phys)
    N >= D || throw(ArgumentError("coeffs must have ≥ $D spectral dims"))
    nb = cutoff ? _resolve_nb(num_bins, ks_phys, T, Val(true)) : _resolve_nb(num_bins, ks_phys, T, Val(false))
    k_bins = Vector{T}(undef, nb)
    E_k = Vector{T}(undef, nb)
    welch_power_spectrum!(E_k, k_bins, ks_phys, coeffs; num_bins = cutoff ? 0 : num_bins, cutoff, convention)
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
    _, dk, k_max = Reductions._radial_extent(ks_phys, nb, T, Val(true))
    @inbounds for i in 1:nb
        k_bins[i] = T(0.5) * ((i - 1) * dk + i * dk)
    end
    # `coherence²`/`phase` double as the Sff/Sgg real accumulators; the complex cross-spectrum Sfg needs
    # its own accumulator (the only allocation — O(num_bins)).
    fill!(coherence², zero(T))
    fill!(phase, zero(T))
    Sfg = zeros(Complex{T}, nb)
    @inbounds for (mi, I) in enumerate(CartesianIndices(ms))
        kmag = sqrt(Packing.ksq(T, ks_phys, I))
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
        # Fold the missing negative half: the auto-spectra double on interior halved modes, the cross
        # spectrum takes its conjugate partner (`X+conj(X)`). Inert on a full layout (`w==1`).
        w = Packing.mode_fold(ks_phys, I)
        coherence²[bin] += w * sff
        phase[bin] += w * sgg
        Sfg[bin] += sfg + (w - one(w)) * conj(sfg)
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
the realization batch **and** over the modes in each radial bin before the ratio is formed; a bin's
normalization cancels in the ratio, so the estimator of [`isotropic_spectrum`](@ref) does not enter.
"""
function coherence_spectrum(ks_phys::Tuple, cf::AbstractArray{Complex{T}, N},
        cg::AbstractArray{Complex{T}, N}; num_bins::Int = 0) where {T, N}
    D = length(ks_phys)
    size(cf) == size(cg) || throw(DimensionMismatch("cf and cg must match"))
    N >= D || throw(ArgumentError("coeffs must have ≥ $D spectral dims"))
    nb = _resolve_nb(num_bins, ks_phys, T, Val(true))
    k_bins = Vector{T}(undef, nb)
    coherence² = Vector{T}(undef, nb)
    phase = Vector{T}(undef, nb)
    coherence_spectrum!(coherence², phase, k_bins, ks_phys, cf, cg)
    return k_bins, coherence², phase
end

end # module Averaging
