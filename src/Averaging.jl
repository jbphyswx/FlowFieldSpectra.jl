module Averaging

using ComputationalBackends: ComputationalBackends
using FlowTransformBindings: FlowTransformBindings as FTB
using ..Normalization: Normalization
using ..Reductions: Reductions

export welch_power_spectrum, welch_power_spectrum!, coherence_spectrum, coherence_spectrum!

# =============================================================================
# Variance-reduced (Welch / ensemble) estimators. The trailing batch dims of the coefficient array are
# the independent segments / realizations: their periodograms are averaged before (Welch) or together
# with (coherence) the radial binning. `coeffs` is `(ms…, realization_batch…)`.
# =============================================================================

@inline _resolve_nb(num_bins::Int, ks_phys::Tuple, ::Type{T}, cut::Bool) where {T} =
    first(Reductions._radial_extent(ks_phys, num_bins, T, cut))

"""
    welch_power_spectrum!(E_k, k_bins, ks_phys, coeffs; num_bins=0, cutoff=true,
                          convention=SpectralConvention(), execution=AutoBackend()) -> nothing

In-place [`welch_power_spectrum`](@ref), allocating nothing on `SerialBackend`: fills preallocated `E_k` and
`k_bins` (both length `num_bins`).
"""
function welch_power_spectrum!(E_k::AbstractVector{T}, k_bins::AbstractVector{T}, ks_phys::Tuple,
        coeffs::AbstractArray{Complex{T}, N}; num_bins::Int = 0, cutoff::Bool = true,
        convention::Normalization.SpectralConvention = Normalization.SpectralConvention(),
        execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend()) where {T, N}
    D = length(ks_phys)
    N >= D || throw(ArgumentError("coeffs must have ≥ $D spectral dims"))
    Reductions._radial_sided(convention.sided)
    length(E_k) == length(k_bins) || throw(DimensionMismatch("E_k and k_bins must have equal length"))
    req = Reductions._requested_bins(k_bins, num_bins, ks_phys, T, cutoff)
    _fill_welch!(Reductions._reduction_execution(execution), E_k, k_bins, ks_phys, coeffs, convention, req, cutoff)
    return nothing
end

# `k_bins` holds the bins' mode counts until the bin centres are written into it.
function _fill_welch!(exec::ComputationalBackends.AbstractExecutionBackend, E_k::AbstractVector{T},
        k_bins::AbstractVector{T}, ks_phys::Tuple, coeffs::AbstractArray{Complex{T}, N},
        convention::Normalization.SpectralConvention, req::Int, cut::Bool) where {T, N}
    nb, dk, k_top = Reductions._radial_extent(ks_phys, req, T, cut)
    fill!(E_k, zero(T))
    fill!(k_bins, zero(T))
    _welch_accumulate!(exec, E_k, k_bins, coeffs, ks_phys, dk, k_top, cut)
    Reductions._shell_values!(E_k, k_bins, dk, ks_phys, convention.estimator)
    Reductions._scale_bins!(E_k, dk, convention.scaling)
    @inbounds for i in 1:nb
        k_bins[i] = T(0.5) * ((i - 1) * dk + i * dk)
    end
    return E_k
end

"""
    _welch_accumulate!(exec, E_k, nmodes, coeffs, ks, dk, k_top, cut) -> E_k

Add to `E_k[bin]` the mean over the realizations (the batch of `coeffs`) of `½ n |C|²` over the modes in
radial bin `bin`, `n` a mode's fold weight, and `n` to `nmodes[bin]` once per mode; `cut` leaves out
`|k| > k_top`. The rows are split on `exec` (`Reductions._split_sum!`). The GPUArraysCore extension runs a
device `coeffs` on its device.
"""
function _welch_accumulate!(exec::ComputationalBackends.AbstractExecutionBackend, E_k::AbstractVector{T},
        nmodes::AbstractVector{T}, coeffs::AbstractArray{Complex{T}, N}, ks_phys::Tuple, dk::T, k_top::T,
        cut::Bool) where {T, N}
    D = length(ks_phys)
    nb = length(E_k)
    k1 = first(ks_phys)
    sz = ntuple(d -> size(coeffs, d), Val(D))
    NM = prod(sz)
    NM == 0 && return E_k
    R = length(coeffs) ÷ NM
    s = T(0.5) / R
    Reductions._split_sum!(exec, (E_k, nmodes), Reductions._nrows(sz)) do (Ep, np), rows
        Reductions._foreach_radial(ks_phys, sz, dk, k_top, cut, nb, rows) do bin, i1, off, _
            w = T(FTB.fold_weight(k1, i1))
            acc = zero(T)
            @inbounds for r in 1:R
                acc += abs2(coeffs[off + i1 + (r - 1) * NM])
            end
            @inbounds np[bin] += w
            @inbounds Ep[bin] += s * w * acc
        end
    end
    return E_k
end

"""
    welch_power_spectrum(ks_phys::Tuple, coeffs; num_bins=0, cutoff=true, convention=SpectralConvention(),
                         execution=AutoBackend())

Variance-reduced (Welch / ensemble-averaged) isotropic power spectrum. The trailing batch dims of
`coeffs` `(ms…, realization…)` index independent segments/realizations whose periodograms are averaged
before radial binning. `cutoff`, `convention` and `execution` are read as in
[`isotropic_spectrum`](@ref Reductions.isotropic_spectrum).
Returns `(k_bins, E_k)`.
"""
function welch_power_spectrum(ks_phys::Tuple, coeffs::AbstractArray{Complex{T}, N};
        num_bins::Int = 0, cutoff::Bool = true,
        convention::Normalization.SpectralConvention = Normalization.SpectralConvention(),
        execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend()) where {T, N}
    D = length(ks_phys)
    N >= D || throw(ArgumentError("coeffs must have ≥ $D spectral dims"))
    nb = _resolve_nb(num_bins, ks_phys, T, cutoff)
    k_bins = Vector{T}(undef, nb)
    E_k = Vector{T}(undef, nb)
    welch_power_spectrum!(E_k, k_bins, ks_phys, coeffs; num_bins = cutoff ? 0 : num_bins, cutoff, convention,
        execution)
    return k_bins, E_k
end

"""
    coherence_spectrum!(coherence², phase, k_bins, ks_phys, cf, cg; num_bins=0, execution=AutoBackend()) -> nothing

In-place [`coherence_spectrum`](@ref): fills preallocated `coherence²`, `phase`, `k_bins` (each length
`num_bins`). Uses `O(num_bins)` internal scratch for the complex cross-spectrum accumulator. `execution` is
read as in [`isotropic_spectrum`](@ref Reductions.isotropic_spectrum).
"""
function coherence_spectrum!(coherence²::AbstractVector{T}, phase::AbstractVector{T},
        k_bins::AbstractVector{T}, ks_phys::Tuple, cf::AbstractArray{Complex{T}, N},
        cg::AbstractArray{Complex{T}, N}; num_bins::Int = 0,
        execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend()) where {T, N}
    D = length(ks_phys)
    size(cf) == size(cg) || throw(DimensionMismatch("cf and cg must match"))
    N >= D || throw(ArgumentError("coeffs must have ≥ $D spectral dims"))
    nb = length(k_bins)
    (length(coherence²) == nb && length(phase) == nb) ||
        throw(DimensionMismatch("coherence², phase, k_bins must have equal length"))
    num_bins > 0 && num_bins != nb && throw(ArgumentError("num_bins=$num_bins ≠ length(k_bins)=$nb"))
    _, dk, k_max = Reductions._radial_extent(ks_phys, nb, T, true)
    @inbounds for i in 1:nb
        k_bins[i] = T(0.5) * ((i - 1) * dk + i * dk)
    end
    # `coherence²` and `phase` accumulate Sff and Sgg; the complex Sfg has its own buffer.
    fill!(coherence², zero(T))
    fill!(phase, zero(T))
    Sfg = zeros(Complex{T}, nb)
    _coherence_accumulate!(Reductions._reduction_execution(execution), coherence², phase, Sfg, cf, cg, ks_phys, dk,
        k_max)
    @inbounds for i in 1:nb
        denom = coherence²[i] * phase[i]                       # Sff · Sgg
        coherence²[i] = denom > 0 ? clamp(abs2(Sfg[i]) / denom, zero(T), one(T)) : zero(T)
        phase[i] = angle(Sfg[i])
    end
    return nothing
end

"""
    _coherence_accumulate!(exec, Sff, Sgg, Sfg, cf, cg, ks, dk, k_max) -> Sfg

Add to bin `bin` of `Sff`, `Sgg` and `Sfg` the sums over the realizations and the modes up to `k_max` in
that radial bin of `n|a|²`, `n|b|²` and `x + (n − 1) conj(x)`, `a`, `b` the two coefficients, `x = a conj(b)`
and `n` the mode's fold weight: the missing negative half of a halved axis folded in, inert on a full
layout. The rows are split on `exec` (`Reductions._split_sum!`). The GPUArraysCore extension runs device
coefficients on their device, with every sum halved, which the coherence and phase do not read.
"""
function _coherence_accumulate!(exec::ComputationalBackends.AbstractExecutionBackend, Sff::AbstractVector{T},
        Sgg::AbstractVector{T}, Sfg::AbstractVector, cf::AbstractArray{Complex{T}, N},
        cg::AbstractArray{Complex{T}, N}, ks_phys::Tuple, dk::T, k_max::T) where {T, N}
    D = length(ks_phys)
    nb = length(Sfg)
    sz = ntuple(d -> size(cf, d), Val(D))
    NM = prod(sz)
    NM == 0 && return Sfg
    R = length(cf) ÷ NM
    k1 = first(ks_phys)
    Reductions._split_sum!(exec, (Sff, Sgg, Sfg), Reductions._nrows(sz)) do (pff, pgg, pfg), rows
        Reductions._foreach_radial(ks_phys, sz, dk, k_max, true, nb, rows) do bin, i1, off, _
            w = T(FTB.fold_weight(k1, i1))
            @inbounds for r in 1:R
                p = off + i1 + (r - 1) * NM
                a = cf[p]
                b = cg[p]
                x = a * conj(b)
                pff[bin] += w * abs2(a)
                pgg[bin] += w * abs2(b)
                pfg[bin] += x + (w - one(w)) * conj(x)
            end
        end
    end
    return Sfg
end

"""
    coherence_spectrum(ks_phys::Tuple, cf, cg; num_bins=0, execution=AutoBackend()) -> (k_bins, coherence², phase)

Magnitude-squared coherence ``\\gamma^2(k) = |S_{fg}|^2 / (S_{ff} S_{gg})`` and phase between two
fields whose coefficients `cf`, `cg` share `(ms…, realization…)`. Cross/auto spectra are averaged over
the realization batch **and** over the modes in each radial bin before the ratio is formed; a bin's
normalization cancels in the ratio, so the estimator of
[`isotropic_spectrum`](@ref Reductions.isotropic_spectrum) does not enter. `execution` is read as in
[`isotropic_spectrum`](@ref Reductions.isotropic_spectrum).
"""
function coherence_spectrum(ks_phys::Tuple, cf::AbstractArray{Complex{T}, N},
        cg::AbstractArray{Complex{T}, N}; num_bins::Int = 0,
        execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend()) where {T, N}
    D = length(ks_phys)
    size(cf) == size(cg) || throw(DimensionMismatch("cf and cg must match"))
    N >= D || throw(ArgumentError("coeffs must have ≥ $D spectral dims"))
    nb = _resolve_nb(num_bins, ks_phys, T, true)
    k_bins = Vector{T}(undef, nb)
    coherence² = Vector{T}(undef, nb)
    phase = Vector{T}(undef, nb)
    coherence_spectrum!(coherence², phase, k_bins, ks_phys, cf, cg; execution)
    return k_bins, coherence², phase
end

end # module Averaging
