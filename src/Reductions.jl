module Reductions

using FlowTransformBindings: FlowTransformBindings as FTB
using ..DirectSum: DirectSum, sph_mode_index
using ..Normalization: Normalization

export isotropic_spectrum, isotropic_spectrum!, transect_spectrum, transect_spectrum!,
    spherical_energy_spectrum, spherical_energy_spectrum!,
    cross_spectrum, cospectrum, quadspectrum, anisotropic_spectrum

# =============================================================================
# Reductions preserve the batch. Coefficients are `(spectral…, batch…)`, the first `D = length(ks)` dims
# spectral; a reduction bins or integrates the spectral dims and keeps every batch dim, returning
# `E(kbin, batch…)`. `dims=` folds designated batch axes into the energy (`dims=D+1` sums the
# vector-component axis into a kinetic-energy spectrum).
# =============================================================================

# Wavenumber sums over the heterogeneous axis tuple, folded by recursion so `ks[d]` with a runtime `d`
# never boxes: `ksq` is `Σ_d k_d(I_d)²`, `kmax` the largest shell the mode box holds whole
# (`min_d max|k_d|`), `kcorner` the box's corner `√(Σ_d max|k_d|²)`, and `dk_product` the product of the
# spacings of the axes in `dims`.
@inline ksq(::Type{T}, ks::Tuple, I::CartesianIndex) where {T} = ksq(T, ks, I, 1)
@inline function ksq(::Type{T}, ks::Tuple, I::CartesianIndex, d::Int) where {T}
    kv = T(first(ks)[I[d]])
    return kv * kv + ksq(T, Base.tail(ks), I, d + 1)
end
@inline ksq(::Type{T}, ::Tuple{}, ::CartesianIndex, ::Int) where {T} = zero(T)

@inline kmax(::Type{T}, ks::Tuple) where {T} = min(T(FTB.max_abs(first(ks))), kmax(T, Base.tail(ks)))
@inline kmax(::Type{T}, ::Tuple{}) where {T} = T(Inf)

@inline kcorner(::Type{T}, ks::Tuple) where {T} = sqrt(_kcorner_sq(T, ks))
@inline _kcorner_sq(::Type{T}, ks::Tuple) where {T} = T(FTB.max_abs(first(ks)))^2 + _kcorner_sq(T, Base.tail(ks))
@inline _kcorner_sq(::Type{T}, ::Tuple{}) where {T} = zero(T)

@inline dk_product(::Type{T}, ks::Tuple, dims::Tuple) where {T} = _dkp(T, ks, dims, 1)
@inline _dkp(::Type{T}, ks::Tuple, dims::Tuple, d::Int) where {T} =
    (d in dims ? T(first(ks)[2] - first(ks)[1]) : one(T)) * _dkp(T, Base.tail(ks), dims, d + 1)
@inline _dkp(::Type{T}, ::Tuple{}, ::Tuple, ::Int) where {T} = one(T)

# Default radial bin count: half the shortest axis's full mode count, the same for a real field's packed
# half as for a complex field's full spectrum.
@inline default_bins(ks::Tuple) = minimum(map(FTB.full_length, ks)) ÷ 2

# Absolute batch dims to fold, validated ⊆ (D+1):N. Accepts an Int or a tuple; `()` folds nothing.
@inline _fold_dims(dims::Integer, D, N) = _fold_dims((Int(dims),), D, N)
@inline function _fold_dims(dims, D, N)
    fold = Tuple(Int(d) for d in dims)
    all(d -> D < d <= N, fold) ||
        throw(ArgumentError("dims=$dims must reference batch dimensions $(D+1):$N (spectral dims 1:$D are always reduced)"))
    return fold
end

# Radial-bin setup shared by the isotropic-style reductions: `(total, dk, k_top)`, `total` bins of width
# `dk = k_max/num_bins` spanning `[0, k_top]`, with `k_max = min_d max|k_d|` the largest shell the mode
# box holds whole (`num_bins = 0` takes the default count). `Val(true)` stops at `k_max`; `Val(false)`
# appends bins of the same width until they reach the box's corner, so the bins below `k_max` are the
# same either way.
@inline function _radial_extent(ks_phys::Tuple, num_bins::Int, ::Type{T}, ::Val{true}) where {T}
    k_max = kmax(T, ks_phys)
    num_bins <= 0 && (num_bins = default_bins(ks_phys))
    return num_bins, k_max / num_bins, k_max
end
@inline function _radial_extent(ks_phys::Tuple, num_bins::Int, ::Type{T}, ::Val{false}) where {T}
    nb, dk, k_max = _radial_extent(ks_phys, num_bins, T, Val(true))
    total = nb + ceil(Int, (kcorner(T, ks_phys) - k_max) / dk)
    return total, dk, total * dk
end

@inline _kbin_centers(num_bins::Int, dk::T) where {T} = [T(0.5) * ((i - 1) * dk + i * dk) for i in 1:num_bins]

# Allocating setup (returns the k-bin centers too) for the allocating reductions.
@inline function _radial_setup(ks_phys::Tuple, num_bins::Int, ::Type{T}, cut::Val = Val(true)) where {T}
    num_bins, dk, k_top = _radial_extent(ks_phys, num_bins, T, cut)
    return num_bins, dk, k_top, _kbin_centers(num_bins, dk)
end

# Volume of the unit ball in `D` dimensions: `V₀ = 1`, `V₁ = 2`, `V_D = (2π/D) V_{D−2}`.
@inline _unit_ball(::Type{T}, ::Val{0}) where {T} = one(T)
@inline _unit_ball(::Type{T}, ::Val{1}) where {T} = T(2)
@inline _unit_ball(::Type{T}, ::Val{D}) where {T, D} = T(2π) / D * _unit_ball(T, Val(D - 2))

"""
    _shell_values!(E, nmodes, dk, ks, estimator) -> E

The bins' values from the energy `E` (bins first, batch after) they accumulated over `nmodes` native
modes each. [`Normalization.ShellSum`](@ref) keeps the sum; [`Normalization.ModeAverage`](@ref) scales
bin `b` by `V_D ((b·dk)ᴰ − ((b−1)·dk)ᴰ) / ∏Δk_d / nmodes[b]`.
"""
_shell_values!(E, nmodes, dk, ks::Tuple, ::Normalization.ShellSum) = E
function _shell_values!(E::AbstractArray, nmodes::AbstractVector{T}, dk::T, ks::Tuple,
        ::Normalization.ModeAverage) where {T}
    D = length(ks)
    cell = dk_product(T, ks, ntuple(identity, Val(D)))
    vD = _unit_ball(T, Val(D))
    nb = length(nmodes)
    @inbounds for b in 1:nb
        whole = vD * ((b * dk)^D - ((b - 1) * dk)^D) / cell
        f = nmodes[b] > 0 ? whole / nmodes[b] : T(NaN)
        for j in b:nb:length(E)
            E[j] *= f
        end
    end
    return E
end

# =============================================================================
# Isotropic (radial) energy spectrum — E(k, batch…)
# =============================================================================

"""
    isotropic_spectrum(ks_phys::Tuple, coeffs; num_bins=0, dims=(), cutoff=true,
                       convention=SpectralConvention())

1D radially-integrated (isotropic) energy spectrum of `coeffs` `(ms…, batch…)`, binning over the `D =
length(ks_phys)` spectral dims and **preserving all batch dims** → `(k_bins, E)` with `E` of shape
`(num_bins, batch…)`. `dims` (absolute batch-dim index/indices `> D`) folds those axes into the energy
(e.g. `dims=D+1` sums a vector-component axis into a single kinetic-energy spectrum). A bin collects
`½|C|²` over its modes of the coefficient box, a packed half standing for both signs of `k₁`.

`cutoff = true` bins the modes with `|k| ≤ k_max = min_d max|k_d|`, the shells the mode box holds whole.
`cutoff = false` bins every mode, the bins continuing at the same width to the corner of the mode box;
under `ShellSum` the bins then hold `½ Σ |C|²` over the whole coefficient box, `½⟨|f|²⟩` for a transform
on a uniform grid at its own length. `num_bins = 0` takes half the shortest axis's full mode count up to
`k_max`.

`convention.estimator` takes the bin's sum (`ShellSum()`, the default) or its mode average times a whole
shell's mode count (`ModeAverage()`); `convention.scaling` divides by the bin width (`DensityScaling()`,
the default) or not (`PowerScaling()`). A radial bin covers every direction, so its sum already includes
each mode's `−k` partner and `convention.sided` is fixed at `OneSided()` here; see
[`Normalization.SpectralConvention`](@ref).
"""
function isotropic_spectrum(ks_phys::Tuple, coeffs::AbstractArray{Complex{T}, N};
        num_bins::Int = 0, dims = (), cutoff::Bool = true,
        convention::Normalization.SpectralConvention = Normalization.SpectralConvention()) where {T, N}
    D = length(ks_phys)
    N >= D || throw(ArgumentError("coeffs must have ≥ $D spectral dims (got $N)"))
    _radial_sided(convention.sided)
    fold = _fold_dims(dims, D, N)
    P = isempty(fold) ? abs2.(coeffs) : dropdims(sum(abs2, coeffs; dims = fold); dims = fold)
    return cutoff ? _bin_isotropic(ks_phys, P, num_bins, convention, Val(true)) :
                    _bin_isotropic(ks_phys, P, num_bins, convention, Val(false))
end

# A radial bin sums over every direction, so each mode's `−k` partner is already in it and there is no
# signed radial axis to keep. `TwoSided` names a spectrum this reduction does not produce.
_radial_sided(::Normalization.OneSided) = nothing
_radial_sided(s::Normalization.AbstractSidedness) = throw(ArgumentError(
    "a radial spectrum has no signed wavenumber axis: each bin sums over all directions, so both signs " *
    "of every mode are already in it. Pass `sided = OneSided()` (the default) here, and use " *
    "`transect_spectrum` for a spectrum against a signed axis wavenumber (got $(nameof(typeof(s)))))."))

# `dk` division for a density; identity for per-bin power.
_scale_bins!(E, dk, ::Normalization.DensityScaling) = (E ./= dk)
_scale_bins!(E, dk, ::Normalization.PowerScaling) = E

# Barrier: `P` (real power `|C|²`, shape (ms…, kept_batch…)) has a concrete rank here → type-stable. Each
# mode carries its fold weight: the halved axis's mirror holds the partners of the stored modes, at the
# same `|k|`, and nothing else in the coefficient box. `Val(true)` stops at `k_top = k_max`.
function _bin_isotropic(ks_phys::Tuple, P::AbstractArray{T, NP}, num_bins::Int,
        convention::Normalization.SpectralConvention, cut::Val{C}) where {T, NP, C}
    D = length(ks_phys)
    num_bins, dk, k_top, k_bins = _radial_setup(ks_phys, num_bins, T, cut)
    bat = CartesianIndices(ntuple(i -> size(P, D + i), Val(NP - D)))
    E = zeros(T, num_bins, size(bat)...)
    nmodes = zeros(T, num_bins)
    @inbounds for I in CartesianIndices(ntuple(d -> size(P, d), Val(D)))
        kmag = sqrt(ksq(T, ks_phys, I))
        C && kmag > k_top && continue
        bin = clamp(floor(Int, kmag / dk) + 1, 1, num_bins)
        n = T(FTB.mode_fold(ks_phys, I))
        nmodes[bin] += n
        for Ib in bat
            E[bin, Ib] += T(0.5) * n * P[I, Ib]
        end
    end
    _shell_values!(E, nmodes, dk, ks_phys, convention.estimator)
    _scale_bins!(E, dk, convention.scaling)
    return k_bins, E
end

"""
    isotropic_spectrum!(E, k_bins, ks_phys, coeffs; num_bins=0, cutoff=true, convention=SpectralConvention())

In-place, allocation-free isotropic spectrum that **preserves all batch dims** (no folding): fills
preallocated `E` (shape `(nb, batch…)`) and `k_bins` (length `nb`). With the cutoff `nb` bins span
`[0, k_max]`; with `cutoff = false`, `num_bins` of them span `[0, k_max]` (`0`: the default count) and
`nb` is the total that reaches the corner of the mode box, the length `isotropic_spectrum` returns.
`cutoff` and `convention` are read as in [`isotropic_spectrum`](@ref).
"""
function isotropic_spectrum!(E::AbstractArray{T}, k_bins::AbstractVector{T}, ks_phys::Tuple,
        coeffs::AbstractArray{Complex{T}, N}; num_bins::Int = 0, cutoff::Bool = true,
        convention::Normalization.SpectralConvention = Normalization.SpectralConvention()) where {T, N}
    D = length(ks_phys)
    N >= D || throw(ArgumentError("coeffs must have ≥ $D spectral dims (got $N)"))
    _radial_sided(convention.sided)
    req = _requested_bins(k_bins, num_bins, ks_phys, T, cutoff)
    cutoff ? _fill_isotropic!(E, k_bins, ks_phys, coeffs, convention, req, Val(true)) :
             _fill_isotropic!(E, k_bins, ks_phys, coeffs, convention, req, Val(false))
    return nothing
end

"""
    _requested_bins(k_bins, num_bins, ks, T, cutoff) -> Int

The bin count up to `k_max` of an in-place radial reduction. Under `cutoff` it is `length(k_bins)`. With
`cutoff = false` it is `num_bins` (or the default), and those bins plus the ones appended up to the mode
box's corner must number `length(k_bins)`.
"""
function _requested_bins(k_bins::AbstractVector, num_bins::Int, ks_phys::Tuple, ::Type{T}, cutoff::Bool) where {T}
    nb = length(k_bins)
    if cutoff
        num_bins > 0 && num_bins != nb && throw(ArgumentError("num_bins=$num_bins ≠ length(k_bins)=$nb"))
        return nb
    end
    total = first(_radial_extent(ks_phys, num_bins, T, Val(false)))
    total == nb || throw(DimensionMismatch(
        "cutoff = false with num_bins = $num_bins fills $total bins up to the corner of the mode box; " *
        "k_bins has $nb"))
    return num_bins
end

# `k_bins` holds the bins' mode counts until the bin centres are written into it.
function _fill_isotropic!(E::AbstractArray{T}, k_bins::AbstractVector{T}, ks_phys::Tuple,
        coeffs::AbstractArray{Complex{T}, N}, convention::Normalization.SpectralConvention, req::Int,
        cut::Val{C}) where {T, N, C}
    D = length(ks_phys)
    nb, dk, k_top = _radial_extent(ks_phys, req, T, cut)
    fill!(E, zero(T))
    fill!(k_bins, zero(T))
    bat = CartesianIndices(ntuple(i -> size(coeffs, D + i), Val(N - D)))
    @inbounds for I in CartesianIndices(ntuple(d -> size(coeffs, d), Val(D)))
        kmag = sqrt(ksq(T, ks_phys, I))
        C && kmag > k_top && continue
        bin = clamp(floor(Int, kmag / dk) + 1, 1, nb)
        n = T(FTB.mode_fold(ks_phys, I))
        k_bins[bin] += n
        for Ib in bat
            E[bin, Ib] += T(0.5) * n * abs2(coeffs[CartesianIndex(I, Ib)])
        end
    end
    _shell_values!(E, k_bins, dk, ks_phys, convention.estimator)
    _scale_bins!(E, dk, convention.scaling)
    @inbounds for i in 1:nb
        k_bins[i] = T(0.5) * ((i - 1) * dk + i * dk)
    end
    return E
end

# =============================================================================
# Anisotropy-resolved 2D spectrum E(k, θ, batch…)
# =============================================================================

"""
    anisotropic_spectrum(ks_phys::Tuple, coeffs; num_k_bins=0, num_θ_bins=16, dims=(),
                         convention=SpectralConvention())

Anisotropy-resolved 2D energy spectrum `E(k, θ, batch…)` for a 2D field: bin `½|C|²` by wavenumber
magnitude up to `k_max` and polar angle, preserving batch. The `k = 0` mode has no angle and is spread
evenly over the `θ` bins of the first `k` bin, so integrating over `θ` recovers the isotropic spectrum
in every bin. Under `ModeAverage()` a bin's mean is scaled by the mode count of its whole annular sector,
`(dθ/2)(k₊² − k₋²)/(Δk₁Δk₂)`; `DensityScaling()` divides by `dk·dθ`.
"""
function anisotropic_spectrum(ks_phys::Tuple, coeffs::AbstractArray{Complex{T}, N};
        num_k_bins::Int = 0, num_θ_bins::Int = 16, dims = (),
        convention::Normalization.SpectralConvention = Normalization.SpectralConvention()) where {T, N}
    length(ks_phys) == 2 || throw(ArgumentError("anisotropic_spectrum is defined for 2D fields"))
    N >= 2 || throw(ArgumentError("coeffs must have ≥ 2 spectral dims"))
    _radial_sided(convention.sided)
    fold = _fold_dims(dims, 2, N)
    P = isempty(fold) ? abs2.(coeffs) : dropdims(sum(abs2, coeffs; dims = fold); dims = fold)
    return _bin_anisotropic(ks_phys, P, num_k_bins, num_θ_bins, convention)
end

function _bin_anisotropic(ks_phys::Tuple, P::AbstractArray{T, NP}, num_k_bins::Int, num_θ_bins::Int,
        convention::Normalization.SpectralConvention) where {T, NP}
    k_max = kmax(T, ks_phys)
    num_k_bins <= 0 && (num_k_bins = default_bins(ks_phys))
    dk = k_max / num_k_bins
    dθ = T(2π) / num_θ_bins
    k_bins = [T(0.5) * dk + (i - 1) * dk for i in 1:num_k_bins]
    θ_bins = [-T(π) + (j - T(0.5)) * dθ for j in 1:num_θ_bins]
    bat = CartesianIndices(ntuple(i -> size(P, 2 + i), Val(NP - 2)))
    E = zeros(T, num_k_bins, num_θ_bins, size(bat)...)
    nmodes = zeros(T, num_k_bins, num_θ_bins)
    @inbounds for I in CartesianIndices((size(P, 1), size(P, 2)))
        kx = T(ks_phys[1][I[1]])
        ky = T(ks_phys[2][I[2]])
        kmag = sqrt(kx^2 + ky^2)
        kmag > k_max && continue
        if iszero(kmag)
            # `k = 0` has no direction: its energy spreads evenly over the angles, so the first `k` bin
            # summed over `θ` holds it whole, as the isotropic spectrum's does.
            for iθ in 1:num_θ_bins
                nmodes[1, iθ] += one(T) / num_θ_bins
                for Ib in bat
                    E[1, iθ, Ib] += T(0.5) * P[I, Ib] / num_θ_bins
                end
            end
            continue
        end
        ik = clamp(floor(Int, kmag / dk) + 1, 1, num_k_bins)
        # The halved axis stores its even Nyquist slot as +k₁max, which a full axis labels −k₁max, so
        # that sign gives the angle of the two-sided spectrum.
        kxa = (FTB.is_halved(ks_phys[1]) && FTB.is_nyquist(ks_phys[1], I[1])) ? -kx : kx
        θ = atan(ky, kxa)
        iθ = clamp(floor(Int, (θ + T(π)) / dθ) + 1, 1, num_θ_bins)
        nmodes[ik, iθ] += one(T)
        for Ib in bat
            E[ik, iθ, Ib] += T(0.5) * P[I, Ib]
        end
        # A halved-axis interior mode's conjugate partner at −k carries the same power at angle θ±π; on a
        # full layout `mode_fold` is 1 and the −k mode is binned on its own pass, so this is inert there.
        if FTB.mode_fold(ks_phys, I) != one(T)
            θm = θ > 0 ? θ - T(π) : θ + T(π)
            iθm = clamp(floor(Int, (θm + T(π)) / dθ) + 1, 1, num_θ_bins)
            nmodes[ik, iθm] += one(T)
            for Ib in bat
                E[ik, iθm, Ib] += T(0.5) * P[I, Ib]
            end
        end
    end
    _sector_values!(E, nmodes, dk, dθ, ks_phys, convention.estimator)
    _scale_bins!(E, dk * dθ, convention.scaling)
    return k_bins, θ_bins, E
end

_sector_values!(E, nmodes, dk, dθ, ks::Tuple, ::Normalization.ShellSum) = E
function _sector_values!(E::AbstractArray, nmodes::AbstractMatrix{T}, dk::T, dθ::T, ks::Tuple,
        ::Normalization.ModeAverage) where {T}
    cell = dk_product(T, ks, (1, 2))
    nk, nθ = size(nmodes)
    @inbounds for iθ in 1:nθ, ik in 1:nk
        whole = dθ / 2 * ((ik * dk)^2 - ((ik - 1) * dk)^2) / cell
        f = nmodes[ik, iθ] > 0 ? whole / nmodes[ik, iθ] : T(NaN)
        for j in (ik + (iθ - 1) * nk):(nk * nθ):length(E)
            E[j] *= f
        end
    end
    return E
end

# =============================================================================
# Cross-spectrum  S_fg(k, batch…) = ½ f̂ · conj(ĝ)
# =============================================================================

"""
    cross_spectrum(ks_phys::Tuple, coeffs_f, coeffs_g; num_bins=0, dims=(), convention=SpectralConvention())

Radially-binned cross-spectrum `S_fg(k, batch…)`, a bin collecting `½ f̂ conj(ĝ)` over its modes up to
`k_max`; `coeffs_f`, `coeffs_g` share shape `(ms…, batch…)`. `convention` is read as in
[`isotropic_spectrum`](@ref). Real part → co-spectrum, negative imag part → quad spectrum.
"""
function cross_spectrum(ks_phys::Tuple, coeffs_f::AbstractArray{Complex{T}, N},
        coeffs_g::AbstractArray{Complex{T}, N}; num_bins::Int = 0, dims = (),
        convention::Normalization.SpectralConvention = Normalization.SpectralConvention()) where {T, N}
    D = length(ks_phys)
    size(coeffs_f) == size(coeffs_g) || throw(DimensionMismatch("coeffs_f and coeffs_g must match"))
    N >= D || throw(ArgumentError("coeffs must have ≥ $D spectral dims"))
    _radial_sided(convention.sided)
    fold = _fold_dims(dims, D, N)
    X = coeffs_f .* conj.(coeffs_g)
    P = isempty(fold) ? X : dropdims(sum(X; dims = fold); dims = fold)
    return _bin_cross(ks_phys, P, num_bins, convention)
end

function _bin_cross(ks_phys::Tuple, P::AbstractArray{Complex{T}, NP}, num_bins::Int,
        convention::Normalization.SpectralConvention) where {T, NP}
    D = length(ks_phys)
    num_bins, dk, k_max, k_bins = _radial_setup(ks_phys, num_bins, T)
    bat = CartesianIndices(ntuple(i -> size(P, D + i), Val(NP - D)))
    S = zeros(Complex{T}, num_bins, size(bat)...)
    nmodes = zeros(T, num_bins)
    @inbounds for I in CartesianIndices(ntuple(d -> size(P, d), Val(D)))
        kmag = sqrt(ksq(T, ks_phys, I))
        kmag > k_max && continue
        bin = clamp(floor(Int, kmag / dk) + 1, 1, num_bins)
        # A halved-axis interior mode's conjugate partner at −k is stored once; the two-sided cross bin
        # is X + conj(X). `mode_fold` is 2 there and 1 at dc/Nyquist and on a full layout (where the −k
        # mode is stored separately), so `(w−1)` selects the partner term without a branch.
        w = T(FTB.mode_fold(ks_phys, I))
        nmodes[bin] += w
        for Ib in bat
            x = P[I, Ib]
            S[bin, Ib] += T(0.5) * (x + (w - one(w)) * conj(x))
        end
    end
    _shell_values!(S, nmodes, dk, ks_phys, convention.estimator)
    _scale_bins!(S, dk, convention.scaling)
    return k_bins, S
end

"""`cospectrum(ks, cf, cg; …)` — `Re S_fg(k, batch…)` (in-phase, flux-carrying part)."""
function cospectrum(ks_phys::Tuple, coeffs_f, coeffs_g; kwargs...)
    k, S = cross_spectrum(ks_phys, coeffs_f, coeffs_g; kwargs...)
    return k, real.(S)
end

"""`quadspectrum(ks, cf, cg; …)` — `-Im S_fg(k, batch…)` (90°-out-of-phase part)."""
function quadspectrum(ks_phys::Tuple, coeffs_f, coeffs_g; kwargs...)
    k, S = cross_spectrum(ks_phys, coeffs_f, coeffs_g; kwargs...)
    return k, -imag.(S)
end

# =============================================================================
# Transect spectrum: integrate out chosen spectral dims, keep the rest and the batch
# =============================================================================

"""
    transect_spectrum(ks_phys::Tuple, coeffs, dims::Tuple)

Integrate the spectral energy density `½|C|²` along the spectral dimensions `dims` (1-indexed, ⊆
`1:D`), scaling by their wavenumber spacing. Returns `(ks_reduced, E_reduced)` where `E_reduced` has
the *kept* spectral dims followed by the batch dims.

`Σ E_reduced / ∏dk` over the kept axes is the field's folded Parseval total over the coefficient box,
whichever axes are kept: a kept full axis carries both signs of its wavenumber outright, and a kept halved
axis reports `|k₁|` with the `−k₁` energy folded in. So a kept full axis's entries at `±k` are equal for a
real field's spectrum and differ for a complex field's, where the sign carries the propagation direction.
"""
function transect_spectrum(ks_phys::Tuple, coeffs::AbstractArray{Complex{T}, N}, dims::Tuple) where {T, N}
    D = length(ks_phys)
    all(d -> 1 <= d <= D, dims) || throw(ArgumentError("transect dims must be spectral (1:$D)"))
    kept_shape = Tuple(size(coeffs, d) for d in 1:N if !(d <= D && d in dims))
    E_reduced = zeros(T, kept_shape...)
    transect_spectrum!(E_reduced, ks_phys, coeffs, dims)
    ks_reduced = Tuple(ks_phys[d] for d in 1:D if !(d in dims))
    return ks_reduced, E_reduced
end

"""
    transect_spectrum!(E_reduced, ks_phys, coeffs, dims) -> nothing

In-place, allocation-free [`transect_spectrum`](@ref): fills preallocated `E_reduced` (kept spectral
dims + batch dims) with the `dims`-integrated `½|C|²` density.
"""
function transect_spectrum!(E_reduced::AbstractArray{T}, ks_phys::Tuple,
        coeffs::AbstractArray{Complex{T}, N}, dims::Tuple) where {T, N}
    D = length(ks_phys)
    all(d -> 1 <= d <= D, dims) || throw(ArgumentError("transect dims must be spectral (1:$D)"))
    dk_prod = dk_product(T, ks_phys, dims)
    fill!(E_reduced, zero(T))
    # Column-major linear index into E_reduced (kept spectral dims + all batch dims), built inline so
    # the reduction over the summed spectral dims is allocation-free.
    halved_out = FTB.is_halved(ks_phys[1]) && (1 in dims)
    # A kept halved axis reports `|k₁|`, and the field's `−k₁` energy is stored nowhere else, so
    # each interior stored mode carries its `fold_weight` of 2 (1 at dc and, for even `n₁`, at Nyquist).
    halved_kept = FTB.is_halved(ks_phys[1]) && !(1 in dims)
    @inbounds for I in CartesianIndices(coeffs)
        lin = 1
        stride = 1
        for d in 1:N
            if !(d <= D && d in dims)
                lin += (I[d] - 1) * stride
                stride *= size(coeffs, d)
            end
        end
        if halved_out
            # Integrating out the halved axis: each stored row stands for itself and, where its fold
            # weight is 2, for the `k₁ < 0` row at the same kept `k_·`, whose magnitude is that of the
            # stored `(k₁, −k_·)`.
            E_reduced[lin] += T(0.5) * abs2(coeffs[I])
            if FTB.fold_weight(ks_phys[1], I[1]) == 2
                J = CartesianIndex(ntuple(d -> d == 1 ? I[1] :
                        (d <= D ? FTB.neg_index(ks_phys[d], I[d]) : I[d]), Val(N)))
                E_reduced[lin] += T(0.5) * abs2(coeffs[J])
            end
        else
            w = halved_kept ? T(FTB.fold_weight(ks_phys[1], I[1])) : one(T)
            E_reduced[lin] += w * T(0.5) * abs2(coeffs[I])
        end
    end
    E_reduced .*= dk_prod
    return nothing
end

# =============================================================================
# Spherical degree energy spectrum — E(ℓ, batch…)
# =============================================================================

"""
    spherical_energy_spectrum(coeffs; lmax=size(coeffs,1)-1)

Degree energy spectrum `E(ℓ, batch…) = ½ Σ_{m=-ℓ}^{ℓ} |C_ℓ^m|²` of spherical-harmonic coefficients
`(Nθ, Nφ, batch…)`. Returns `(0:lmax, E_l)`, `E_l` of shape `(lmax+1, batch…)` and real element type,
from real or complex coefficients.
"""
function spherical_energy_spectrum(coeffs::AbstractArray{<:Number, N};
        lmax::Int = size(coeffs, 1) - 1) where {N}
    N >= 2 || throw(ArgumentError("coeffs must have ≥ 2 spectral dims (Nθ, Nφ)"))
    T = real(float(eltype(coeffs)))
    batch = ntuple(i -> size(coeffs, 2 + i), N - 2)
    E = zeros(T, lmax + 1, batch...)
    _accumulate_degree!(E, coeffs, lmax)
    return 0:lmax, E
end

"""
    spherical_energy_spectrum!(E_l, coeffs; lmax=size(coeffs,1)-1) -> nothing

In-place [`spherical_energy_spectrum`](@ref): fills preallocated `E_l` (shape `(lmax+1, batch…)`).
"""
function spherical_energy_spectrum!(E_l::AbstractArray{T}, coeffs::AbstractArray{<:Number, N};
        lmax::Int = size(coeffs, 1) - 1) where {T, N}
    fill!(E_l, zero(T))
    _accumulate_degree!(E_l, coeffs, lmax)
    return nothing
end

function _accumulate_degree!(E::AbstractArray{T}, coeffs::AbstractArray{<:Number, N}, lmax::Int) where {T, N}
    Nθ = size(coeffs, 1)
    Nφ = size(coeffs, 2)
    NθNφ = Nθ * Nφ
    B = length(coeffs) ÷ NθNφ
    Lp1 = lmax + 1
    # Degree ℓ / order m / batch b lives at `coeffs[row + (col-1)·Nθ + (b-1)·Nθ·Nφ]`. Indexed linearly,
    # since a `reshape` allocates under `--check-bounds=yes`.
    @inbounds for b in 1:B
        cbase = (b - 1) * NθNφ
        ebase = (b - 1) * Lp1
        for l in 0:lmax
            acc = zero(T)
            for m in -l:l
                idx = sph_mode_index(l, m)
                acc += abs2(coeffs[idx[1] + (idx[2] - 1) * Nθ + cbase])
            end
            E[l + 1 + ebase] += T(0.5) * acc
        end
    end
    return E
end

end # module Reductions
