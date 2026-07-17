module Reductions

using ..DirectSum: DirectSum, sph_mode_index

export isotropic_spectrum, isotropic_spectrum!, transect_spectrum, transect_spectrum!,
    spherical_energy_spectrum, spherical_energy_spectrum!,
    cross_spectrum, cospectrum, quadspectrum, anisotropic_spectrum

# =============================================================================
# Reductions are BATCH-PRESERVING. Coefficients are `(spectral…, batch…)` (the first `D = length(ks)`
# dims are spectral; the rest are batch). A reduction bins/integrates over the spectral dims and keeps
# every batch dim, returning `E(kbin, batch…)` — so a whole `(kx, ky, comp, z, t)` stack reduces in one
# vectorized pass with no per-slice reshape loop. `dims=` opts into folding designated batch axes into
# the energy (e.g. `dims=D+1` sums the vector-component axis → a kinetic-energy spectrum).
# =============================================================================

# Absolute batch dims to fold, validated ⊆ (D+1):N. Accepts an Int or a tuple; `()` folds nothing.
@inline _fold_dims(dims::Integer, D, N) = _fold_dims((Int(dims),), D, N)
@inline function _fold_dims(dims, D, N)
    fold = Tuple(Int(d) for d in dims)
    all(d -> D < d <= N, fold) ||
        throw(ArgumentError("dims=$dims must reference batch dimensions $(D+1):$N (spectral dims 1:$D are always reduced)"))
    return fold
end

# Radial-bin setup shared by the isotropic-style reductions.
# Non-allocating radial extent (num_bins, dk, k_max): the max resolved isotropic wavenumber is the
# min over axes of each axis's max |k|, formed without per-axis temporaries.
@inline function _radial_extent(ks_phys::NTuple{D, Any}, ms::NTuple{D, Int}, num_bins::Int, ::Type{T}) where {D, T}
    k_max = T(Inf)
    @inbounds for d in 1:D
        axmax = zero(T)
        for v in ks_phys[d]
            a = abs(T(v))
            axmax = ifelse(a > axmax, a, axmax)
        end
        k_max = min(k_max, axmax)
    end
    num_bins <= 0 && (num_bins = minimum(ms) ÷ 2)
    return num_bins, k_max / num_bins, k_max
end

@inline _kbin_centers(num_bins::Int, dk::T) where {T} = [T(0.5) * ((i - 1) * dk + i * dk) for i in 1:num_bins]

# Allocating setup (returns the k-bin centers too) for the allocating reductions.
@inline function _radial_setup(ks_phys::NTuple{D, Any}, ms::NTuple{D, Int}, num_bins::Int, ::Type{T}) where {D, T}
    num_bins, dk, k_max = _radial_extent(ks_phys, ms, num_bins, T)
    return num_bins, dk, k_max, _kbin_centers(num_bins, dk)
end

# =============================================================================
# Isotropic (radial) energy spectrum — E(k, batch…)
# =============================================================================

"""
    isotropic_spectrum(ks_phys::Tuple, coeffs; num_bins=0, dims=())

1D radially-integrated (isotropic) energy spectrum of `coeffs` `(ms…, batch…)`, binning over the `D =
length(ks_phys)` spectral dims and **preserving all batch dims** → `(k_bins, E)` with `E` of shape
`(num_bins, batch…)`. `dims` (absolute batch-dim index/indices `> D`) folds those axes into the energy
(e.g. `dims=D+1` sums a vector-component axis into a single kinetic-energy spectrum).
`E(k) = (1/2dk) Σ_{|k|∈bin} Σ_{folded} |C|²`.
"""
function isotropic_spectrum(ks_phys::Tuple, coeffs::AbstractArray{Complex{T}, N};
        num_bins::Int = 0, dims = ()) where {T, N}
    D = length(ks_phys)
    N >= D || throw(ArgumentError("coeffs must have ≥ $D spectral dims (got $N)"))
    fold = _fold_dims(dims, D, N)
    P = isempty(fold) ? abs2.(coeffs) : dropdims(sum(abs2, coeffs; dims = fold); dims = fold)
    return _bin_isotropic(ks_phys, P, num_bins)
end

# Barrier: `P` (real power, shape (ms…, kept_batch…)) has a concrete rank here → type-stable.
function _bin_isotropic(ks_phys::Tuple, P::AbstractArray{T, NP}, num_bins::Int) where {T, NP}
    D = length(ks_phys)
    ms = ntuple(d -> size(P, d), D)
    num_bins, dk, k_max, k_bins = _radial_setup(ks_phys, ms, num_bins, T)
    E = zeros(T, num_bins, ntuple(i -> size(P, D + i), NP - D)...)
    kd2 = ntuple(d -> [T(v)^2 for v in ks_phys[d]], D)
    @inbounds for I in CartesianIndices(P)
        kmag = zero(T)
        for d in 1:D
            kmag += kd2[d][I[d]]
        end
        kmag = sqrt(kmag)
        kmag > k_max && continue
        bin = clamp(floor(Int, kmag / dk) + 1, 1, num_bins)
        keep = ntuple(i -> I[D + i], NP - D)
        E[bin, keep...] += T(0.5) * P[I]
    end
    E ./= dk
    return k_bins, E
end

"""
    isotropic_spectrum!(E, k_bins, ks_phys, coeffs; num_bins=0)

In-place, allocation-free isotropic spectrum that **preserves all batch dims** (no folding): fills
preallocated `E` (shape `(num_bins, batch…)`) and `k_bins` (length `num_bins`). Reusable across a time
loop with zero steady-state heap traffic.
"""
function isotropic_spectrum!(E::AbstractArray{T}, k_bins::AbstractVector{T}, ks_phys::Tuple,
        coeffs::AbstractArray{Complex{T}, N}; num_bins::Int = 0) where {T, N}
    D = length(ks_phys)
    N >= D || throw(ArgumentError("coeffs must have ≥ $D spectral dims (got $N)"))
    ms = ntuple(d -> size(coeffs, d), D)
    nb = length(k_bins)
    num_bins > 0 && num_bins != nb &&
        throw(ArgumentError("num_bins=$num_bins ≠ length(k_bins)=$nb"))
    _, dk, k_max = _radial_extent(ks_phys, ms, nb, T)      # no k_bins allocation (fill the arg below)
    @inbounds for i in 1:nb
        k_bins[i] = T(0.5) * ((i - 1) * dk + i * dk)
    end
    fill!(E, zero(T))
    @inbounds for I in CartesianIndices(coeffs)
        kmag = zero(T)
        for d in 1:D
            kv = T(ks_phys[d][I[d]])
            kmag += kv * kv
        end
        kmag = sqrt(kmag)
        kmag > k_max && continue
        bin = clamp(floor(Int, kmag / dk) + 1, 1, nb)
        keep = ntuple(i -> I[D + i], N - D)
        E[bin, keep...] += T(0.5) * abs2(coeffs[I])
    end
    E ./= dk
    return nothing
end

# =============================================================================
# Anisotropy-resolved 2D spectrum E(k, θ, batch…)
# =============================================================================

"""
    anisotropic_spectrum(ks_phys::Tuple, coeffs; num_k_bins=0, num_θ_bins=16, dims=())

Anisotropy-resolved 2D energy spectrum `E(k, θ, batch…)` for a 2D field: bin `½|C|²` by wavenumber
magnitude and polar angle, preserving batch. Integrating over `θ` recovers the isotropic spectrum.
"""
function anisotropic_spectrum(ks_phys::Tuple, coeffs::AbstractArray{Complex{T}, N};
        num_k_bins::Int = 0, num_θ_bins::Int = 16, dims = ()) where {T, N}
    length(ks_phys) == 2 || throw(ArgumentError("anisotropic_spectrum is defined for 2D fields"))
    N >= 2 || throw(ArgumentError("coeffs must have ≥ 2 spectral dims"))
    fold = _fold_dims(dims, 2, N)
    P = isempty(fold) ? abs2.(coeffs) : dropdims(sum(abs2, coeffs; dims = fold); dims = fold)
    return _bin_anisotropic(ks_phys, P, num_k_bins, num_θ_bins)
end

function _bin_anisotropic(ks_phys::Tuple, P::AbstractArray{T, NP}, num_k_bins::Int, num_θ_bins::Int) where {T, NP}
    ms = (size(P, 1), size(P, 2))
    k_max = min(maximum(abs, ks_phys[1]), maximum(abs, ks_phys[2]))
    num_k_bins <= 0 && (num_k_bins = minimum(ms) ÷ 2)
    dk = k_max / num_k_bins
    dθ = T(2π) / num_θ_bins
    k_bins = [T(0.5) * dk + (i - 1) * dk for i in 1:num_k_bins]
    θ_bins = [-T(π) + (j - T(0.5)) * dθ for j in 1:num_θ_bins]
    E = zeros(T, num_k_bins, num_θ_bins, ntuple(i -> size(P, 2 + i), NP - 2)...)
    @inbounds for I in CartesianIndices(P)
        kx = T(ks_phys[1][I[1]])
        ky = T(ks_phys[2][I[2]])
        kmag = sqrt(kx^2 + ky^2)
        (kmag > k_max || kmag == 0) && continue
        ik = clamp(floor(Int, kmag / dk) + 1, 1, num_k_bins)
        iθ = clamp(floor(Int, (atan(ky, kx) + T(π)) / dθ) + 1, 1, num_θ_bins)
        keep = ntuple(i -> I[2 + i], NP - 2)
        E[ik, iθ, keep...] += T(0.5) * P[I]
    end
    E ./= (dk * dθ)
    return k_bins, θ_bins, E
end

# =============================================================================
# Cross-spectrum  S_fg(k, batch…) = ½ f̂ · conj(ĝ)
# =============================================================================

"""
    cross_spectrum(ks_phys::Tuple, coeffs_f, coeffs_g; num_bins=0, dims=())

Radially-binned cross-spectrum `S_fg(k, batch…) = ½ Σ_{|k|∈bin} Σ_{folded} f̂ conj(ĝ)`; `coeffs_f`,
`coeffs_g` share shape `(ms…, batch…)`. Real part → co-spectrum, negative imag part → quad spectrum.
"""
function cross_spectrum(ks_phys::Tuple, coeffs_f::AbstractArray{Complex{T}, N},
        coeffs_g::AbstractArray{Complex{T}, N}; num_bins::Int = 0, dims = ()) where {T, N}
    D = length(ks_phys)
    size(coeffs_f) == size(coeffs_g) || throw(DimensionMismatch("coeffs_f and coeffs_g must match"))
    N >= D || throw(ArgumentError("coeffs must have ≥ $D spectral dims"))
    fold = _fold_dims(dims, D, N)
    X = coeffs_f .* conj.(coeffs_g)
    P = isempty(fold) ? X : dropdims(sum(X; dims = fold); dims = fold)
    return _bin_cross(ks_phys, P, num_bins)
end

function _bin_cross(ks_phys::Tuple, P::AbstractArray{Complex{T}, NP}, num_bins::Int) where {T, NP}
    D = length(ks_phys)
    ms = ntuple(d -> size(P, d), D)
    num_bins, dk, k_max, k_bins = _radial_setup(ks_phys, ms, num_bins, T)
    S = zeros(Complex{T}, num_bins, ntuple(i -> size(P, D + i), NP - D)...)
    kd2 = ntuple(d -> [T(v)^2 for v in ks_phys[d]], D)
    @inbounds for I in CartesianIndices(P)
        kmag = zero(T)
        for d in 1:D
            kmag += kd2[d][I[d]]
        end
        kmag = sqrt(kmag)
        kmag > k_max && continue
        bin = clamp(floor(Int, kmag / dk) + 1, 1, num_bins)
        keep = ntuple(i -> I[D + i], NP - D)
        S[bin, keep...] += T(0.5) * P[I]
    end
    S ./= dk
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
# Transect spectrum — integrate out specific SPECTRAL dims, preserve the rest + batch
# =============================================================================

"""
    transect_spectrum(ks_phys::Tuple, coeffs, dims::Tuple)

Integrate the spectral energy density `½|C|²` along the spectral dimensions `dims` (1-indexed, ⊆
`1:D`), scaling by their wavenumber spacing. Returns `(ks_reduced, E_reduced)` where `E_reduced` has
the *kept* spectral dims followed by the batch dims.
"""
function transect_spectrum(ks_phys::Tuple, coeffs::AbstractArray{Complex{T}, N}, dims::Tuple) where {T, N}
    D = length(ks_phys)
    all(d -> 1 <= d <= D, dims) || throw(ArgumentError("transect dims must be spectral (1:$D)"))
    dk_prod = one(T)
    for d in dims
        dk_prod *= T(ks_phys[d][2] - ks_phys[d][1])
    end
    P = dropdims(sum(x -> T(0.5) * abs2(x), coeffs; dims = dims); dims = dims)
    P .*= dk_prod
    ks_reduced = Tuple(ks_phys[d] for d in 1:D if !(d in dims))
    return ks_reduced, P
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
    dk_prod = one(T)
    for d in dims
        dk_prod *= T(ks_phys[d][2] - ks_phys[d][1])
    end
    fill!(E_reduced, zero(T))
    # Column-major linear index into E_reduced (kept spectral dims + all batch dims), built inline so
    # the reduction over the summed spectral dims is allocation-free.
    @inbounds for I in CartesianIndices(coeffs)
        lin = 1
        stride = 1
        for d in 1:N
            if !(d <= D && d in dims)
                lin += (I[d] - 1) * stride
                stride *= size(coeffs, d)
            end
        end
        E_reduced[lin] += T(0.5) * abs2(coeffs[I])
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
`(Nθ, Nφ, batch…)`. Returns `(0:lmax, E_l)`, `E_l` of shape `(lmax+1, batch…)`.
"""
function spherical_energy_spectrum(coeffs::AbstractArray{Complex{T}, N};
        lmax::Int = size(coeffs, 1) - 1) where {T, N}
    N >= 2 || throw(ArgumentError("coeffs must have ≥ 2 spectral dims (Nθ, Nφ)"))
    batch = ntuple(i -> size(coeffs, 2 + i), N - 2)
    E = zeros(T, lmax + 1, batch...)
    _accumulate_degree!(E, coeffs, lmax)
    return 0:lmax, E
end

"""
    spherical_energy_spectrum!(E_l, coeffs; lmax=size(coeffs,1)-1) -> nothing

In-place [`spherical_energy_spectrum`](@ref): fills preallocated `E_l` (shape `(lmax+1, batch…)`).
"""
function spherical_energy_spectrum!(E_l::AbstractArray{T}, coeffs::AbstractArray{Complex{T}, N};
        lmax::Int = size(coeffs, 1) - 1) where {T, N}
    fill!(E_l, zero(T))
    _accumulate_degree!(E_l, coeffs, lmax)
    return nothing
end

function _accumulate_degree!(E::AbstractArray{T}, coeffs::AbstractArray{Complex{T}, N}, lmax::Int) where {T, N}
    Nθ = size(coeffs, 1)
    Nφ = size(coeffs, 2)
    NθNφ = Nθ * Nφ
    B = length(coeffs) ÷ NθNφ
    Lp1 = lmax + 1
    # Linear indexing (no `reshape`): degree ℓ / order m / batch b lives at
    # `coeffs[row + (col-1)·Nθ + (b-1)·Nθ·Nφ]`. Avoids the reshape-header alloc under --check-bounds=yes.
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
