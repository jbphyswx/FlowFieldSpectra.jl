module Reductions

using Statistics: mean
using ..DirectSum: sph_mode_index

export isotropic_spectrum, isotropic_spectrum!, transect_spectrum, transect_spectrum!,
    spherical_energy_spectrum, spherical_energy_spectrum!,
    cross_spectrum, cospectrum, quadspectrum, anisotropic_spectrum

"""
    isotropic_spectrum(ks_phys::Tuple, coeffs::AbstractArray; num_bins::Int=minimum(size(coeffs)[1:end-1]) ÷ 2)

Compute the 1D radially integrated (isotropic) energy spectrum from ND Cartesian Fourier coefficients.

# Arguments
- `ks_phys::Tuple`: A tuple of physical wavenumber ranges along each dimension (e.g., returned by `calculate_spectrum`).
- `coeffs::AbstractArray`: Complex coefficients of shape `(m1, m2, ..., mD, NU)` where `NU` is the number of fields.

# Returns
- `k_bins::Vector`: The centers of the radial wavenumber bins.
- `E_k::Vector`: The spectral density ``E(k)`` such that the integral ``\\int E(k) dk`` matches the total spatial energy of the fields.

# Details
The 1D isotropic energy spectrum is computed by binning the multi-dimensional spectral coefficients according to their radial wavenumber magnitude ``k_{mag} = \\sqrt{k_x^2 + k_y^2 + \\dots}``.
For each bin, we sum the spectral energy:
``E(k) = \\frac{1}{2 \\, dk} \\sum_{k_{mag} \\in [k, k+dk)} \\sum_{c=1}^{NU} |C(I, c)|^2``
where ``dk`` is the bin width. This conversion turns the discrete sum into a spectral density.

# Example
```julia
k_bins, E_k = isotropic_spectrum(ks_phys, coeffs; num_bins=32)
```
"""
function isotropic_spectrum(
    ks_phys::Tuple,
    coeffs::AbstractArray{Complex{T}, N_dim};
    num_bins::Int = 0,
) where {T<:AbstractFloat, N_dim}
    D = length(ks_phys)
    @assert N_dim == D + 1 "coeffs must have shape (m1, ..., mD, NU)"

    ms = size(coeffs)[1:D]
    NU = size(coeffs, N_dim)

    # Calculate wavenumber magnitude for each grid point
    # Find max wavenumber along each dimension to establish radial Nyquist limit
    max_ks = [maximum(abs.(ks_phys[d])) for d in 1:D]
    k_max = minimum(max_ks) # Nyquist radius for complete spherical shells

    if num_bins <= 0
        num_bins = minimum(ms) ÷ 2
    end

    bin_edges = range(zero(T), stop = k_max, length = num_bins + 1)
    dk = k_max / num_bins

    k_bins = [T(0.5) * (bin_edges[i] + bin_edges[i+1]) for i in 1:num_bins]
    E_k = zeros(T, num_bins)

    # Precompute squared wavenumbers per axis (avoids re-squaring shared values).
    kd2 = [T.(ks_phys[d]) .^ 2 for d in 1:D]

    # Accumulate energy in bins: E(k) = 1/2 * sum |C|^2
    @inbounds for I in CartesianIndices(ms)
        k_mag = zero(T)
        for d in 1:D
            k_mag += kd2[d][I[d]]
        end
        k_mag = sqrt(k_mag)

        if k_mag <= k_max
            bin_idx = clamp(floor(Int, k_mag / dk) + 1, 1, num_bins)
            energy = zero(T)
            for c in 1:NU
                energy += abs2(coeffs[I, c])
            end
            E_k[bin_idx] += T(0.5) * energy
        end
    end

    # Normalize by bin width dk to convert sum to spectral density
    E_k ./= dk

    return k_bins, E_k
end

"""
    anisotropic_spectrum(ks_phys::Tuple, coeffs; num_k_bins=0, num_θ_bins=16)

Anisotropy-resolved 2D energy spectrum ``E(k, \\theta)`` for a 2D Cartesian field: bin the
energy ``\\tfrac12\\sum_c|C|^2`` by wavenumber magnitude ``k=|\\mathbf k|`` and polar angle
``\\theta=\\mathrm{atan}(k_y, k_x)\\in(-\\pi,\\pi]``. Returns `(k_bins, θ_bins, E)` where `E` is
`(num_k_bins, num_θ_bins)`, normalized as a density (per `dk·dθ`). Integrating over `θ` recovers
the isotropic spectrum.
"""
function anisotropic_spectrum(
    ks_phys::Tuple,
    coeffs::AbstractArray{Complex{T}, N_dim};
    num_k_bins::Int = 0,
    num_θ_bins::Int = 16,
) where {T<:AbstractFloat, N_dim}
    D = length(ks_phys)
    D == 2 || throw(ArgumentError("anisotropic_spectrum is defined for 2D fields (D=2), got D=$D"))
    @assert N_dim == 3 "coeffs must have shape (mx, my, NU)"
    ms = size(coeffs)[1:2]
    NU = size(coeffs, 3)

    k_max = minimum((maximum(abs.(ks_phys[1])), maximum(abs.(ks_phys[2]))))
    num_k_bins <= 0 && (num_k_bins = minimum(ms) ÷ 2)

    dk = k_max / num_k_bins
    dθ = T(2π) / num_θ_bins
    k_bins = [T(0.5) * dk + (i - 1) * dk for i in 1:num_k_bins]
    θ_bins = [-T(π) + (j - T(0.5)) * dθ for j in 1:num_θ_bins]
    E = zeros(T, num_k_bins, num_θ_bins)

    @inbounds for I in CartesianIndices(ms)
        kx = T(ks_phys[1][I[1]])
        ky = T(ks_phys[2][I[2]])
        k_mag = sqrt(kx^2 + ky^2)
        (k_mag > k_max || k_mag == 0) && continue
        ik = clamp(floor(Int, k_mag / dk) + 1, 1, num_k_bins)
        θ = atan(ky, kx)                       # (-π, π]
        iθ = clamp(floor(Int, (θ + T(π)) / dθ) + 1, 1, num_θ_bins)
        energy = zero(T)
        for c in 1:NU
            energy += abs2(coeffs[I, c])
        end
        E[ik, iθ] += T(0.5) * energy
    end
    E ./= (dk * dθ)
    return k_bins, θ_bins, E
end

"""
    cross_spectrum(ks_phys::Tuple, coeffs_f, coeffs_g; num_bins=0)

Radially-binned **cross-spectrum** ``S_{fg}(k) = \\tfrac{1}{2}\\sum_{c}\\hat f_c\\,\\overline{\\hat g_c}``
of two fields whose coefficients `coeffs_f`, `coeffs_g` share shape `(ms..., NU)`. Returns
`(k_bins, S)` with `S` complex.

Its real part is the **co-spectrum** ([`cospectrum`](@ref)) — the scale-by-scale covariance whose
integral recovers ``\\langle f\\,g\\rangle`` (e.g. the momentum-flux co-spectrum ``\\langle u'w'\\rangle``);
its negative imaginary part is the **quadrature spectrum** ([`quadspectrum`](@ref)). Coherence and
phase additionally require segment/ensemble averaging to be meaningful.
"""
function cross_spectrum(
    ks_phys::Tuple,
    coeffs_f::AbstractArray{Complex{T}, N_dim},
    coeffs_g::AbstractArray{Complex{T}, N_dim};
    num_bins::Int = 0,
) where {T<:AbstractFloat, N_dim}
    D = length(ks_phys)
    @assert N_dim == D + 1 "coeffs must have shape (m1, ..., mD, NU)"
    size(coeffs_f) == size(coeffs_g) || throw(DimensionMismatch("coeffs_f and coeffs_g must match"))

    ms = size(coeffs_f)[1:D]
    NU = size(coeffs_f, N_dim)

    max_ks = [maximum(abs.(ks_phys[d])) for d in 1:D]
    k_max = minimum(max_ks)
    num_bins <= 0 && (num_bins = minimum(ms) ÷ 2)

    bin_edges = range(zero(T), stop = k_max, length = num_bins + 1)
    dk = k_max / num_bins
    k_bins = [T(0.5) * (bin_edges[i] + bin_edges[i+1]) for i in 1:num_bins]
    S = zeros(Complex{T}, num_bins)

    kd2 = [T.(ks_phys[d]) .^ 2 for d in 1:D]
    @inbounds for I in CartesianIndices(ms)
        k_mag = zero(T)
        for d in 1:D
            k_mag += kd2[d][I[d]]
        end
        k_mag = sqrt(k_mag)
        if k_mag <= k_max
            bin_idx = clamp(floor(Int, k_mag / dk) + 1, 1, num_bins)
            acc = zero(Complex{T})
            for c in 1:NU
                acc += coeffs_f[I, c] * conj(coeffs_g[I, c])
            end
            S[bin_idx] += T(0.5) * acc
        end
    end
    S ./= dk
    return k_bins, S
end

"""
    cospectrum(ks_phys, coeffs_f, coeffs_g; num_bins=0) -> (k_bins, Co)

Co-spectrum ``\\mathrm{Co}_{fg}(k) = \\mathrm{Re}\\,S_{fg}(k)`` — the in-phase, flux-carrying part.
"""
function cospectrum(ks_phys::Tuple, coeffs_f, coeffs_g; num_bins::Int = 0)
    k_bins, S = cross_spectrum(ks_phys, coeffs_f, coeffs_g; num_bins = num_bins)
    return k_bins, real.(S)
end

"""
    quadspectrum(ks_phys, coeffs_f, coeffs_g; num_bins=0) -> (k_bins, Q)

Quadrature spectrum ``Q_{fg}(k) = -\\mathrm{Im}\\,S_{fg}(k)`` — the 90°-out-of-phase part.
"""
function quadspectrum(ks_phys::Tuple, coeffs_f, coeffs_g; num_bins::Int = 0)
    k_bins, S = cross_spectrum(ks_phys, coeffs_f, coeffs_g; num_bins = num_bins)
    return k_bins, -imag.(S)
end

"""
    transect_spectrum(ks_phys::Tuple, coeffs::AbstractArray, dims::Tuple)

Integrate the spectral energy density along specific dimensions of an ND Cartesian field, reducing its dimensionality.

# Arguments
- `ks_phys::Tuple`: Tuple of physical wavenumber ranges along each dimension.
- `coeffs::AbstractArray`: Complex coefficients of shape `(m1, m2, ..., mD, NU)`.
- `dims::Tuple`: Dimensions to sum over / integrate out (1-indexed).

# Returns
- `ks_reduced::Tuple`: Physical wavenumber ranges for the remaining (non-reduced) dimensions.
- `E_reduced::Array`: The reduced spectral energy density.

# Details
This is useful for computing transect or slice spectra. For instance, in a 2D field, summing over the second dimension (`dims=(2,)`) computes the 1D energy spectrum along the first dimension (e.g., zonal energy spectrum). The sum is scaled by the wavenumber spacing ``dk_d`` of the integrated dimensions to preserve energy density units.

# Example
```julia
# Reduce a 2D spectrum to a 1D transect spectrum along the first axis (integrate out y-axis, dim 2)
ks_1d, E_1d = transect_spectrum(ks_phys, coeffs, (2,))
```
"""
function transect_spectrum(
    ks_phys::Tuple,
    coeffs::AbstractArray{Complex{T}, N_dim},
    dims::Tuple,
) where {T<:AbstractFloat, N_dim}
    D = length(ks_phys)
    @assert N_dim == D + 1 "coeffs must have shape (m1, ..., mD, NU)"

    ms = size(coeffs)[1:D]
    NU = size(coeffs, N_dim)

    keep_dims = Tuple(d for d in 1:D if !(d in dims))
    sum_dims = Tuple(d for d in 1:D if d in dims)

    # Allocate only the reduced output (no full ms-sized intermediate grid).
    out_shape = Tuple(ms[d] for d in keep_dims)
    E_reduced = zeros(T, out_shape)

    dk_prod = one(T)
    for d in sum_dims
        dk_prod *= T(ks_phys[d][2] - ks_phys[d][1])
    end

    _accumulate_transect!(E_reduced, coeffs, ms, NU, keep_dims, out_shape)
    E_reduced .*= dk_prod

    ks_reduced = Tuple(ks_phys[d] for d in keep_dims)
    return ks_reduced, E_reduced
end

# Accumulate 1/2 Σ|C|^2 from an (ms..., NU) coeff array into the reduced output indexed by the
# kept dimensions. Uses precomputed strides + linear indexing — allocation-free, no per-mode
# tuple construction (so it scales to large 3D grids).
function _accumulate_transect!(E_reduced::AbstractArray{T}, coeffs, ms::NTuple{D, Int}, NU::Int,
        keep_dims::Tuple, out_shape::Tuple) where {T, D}
    nkeep = length(keep_dims)
    out_strides = ones(Int, nkeep)
    @inbounds for kk in 2:nkeep
        out_strides[kk] = out_strides[kk-1] * out_shape[kk-1]
    end
    flatE = vec(E_reduced)
    @inbounds for I in CartesianIndices(ms)
        e = zero(T)
        for c in 1:NU
            e += abs2(coeffs[I, c])
        end
        lin = 1
        kk = 0
        for d in keep_dims
            kk += 1
            lin += (I[d] - 1) * out_strides[kk]
        end
        flatE[lin] += T(0.5) * e
    end
    return E_reduced
end

"""
    spherical_energy_spectrum(coeffs::AbstractArray{Complex{T}, 3}; lmax::Int=size(coeffs, 1)-1) where {T}

Compute the spherical harmonic power spectrum per degree ``l`` (also known as the degree energy spectrum):
``E(l) = \\frac{1}{2} \\sum_{m=-l}^{l} \\sum_{c=1}^{NU} |C_l^m(c)|^2``

# Arguments
- `coeffs::AbstractArray{Complex{T}, 3}`: Spherical harmonic coefficients of size `(Ntheta, Nphi, NU)` or `(lmax+1, 2lmax+1, NU)`.
- `lmax::Int`: Maximum spherical degree to compute (defaults to `Ntheta - 1`).

# Returns
- `degrees::AbstractRange`: Range of degrees `0:lmax`.
- `E_l::Vector`: Spherical energy spectrum array of length `lmax + 1`.

# Details
The degree energy spectrum represents the distribution of flow energy across different spherical spatial scales (with degree ``l`` roughly corresponding to wavelength ``2\\pi R / l``).

# Example
```julia
degrees, E_l = spherical_energy_spectrum(coeffs)
```
"""
function spherical_energy_spectrum(
    coeffs::AbstractArray{Complex{T}, 3};
    lmax::Int = size(coeffs, 1) - 1,
) where {T<:AbstractFloat}
    Nθ, Nφ, NU = size(coeffs)
    @assert Nθ >= lmax + 1 "Coefficients size mismatch"

    E_l = zeros(T, lmax + 1)

    for l in 0:lmax
        energy = zero(T)
        for m in -l:l
            idx = sph_mode_index(l, m)
            for c in 1:NU
                energy += abs2(coeffs[idx, c])
            end
        end
        E_l[l+1] = T(0.5) * energy
    end

    return 0:lmax, E_l
end

"""
    isotropic_spectrum!(E_k::Vector, k_bins::Vector, ks_phys::Tuple, coeffs::AbstractArray; num_bins::Int=0)

In-place version of `isotropic_spectrum`. Computes the 1D isotropic energy spectrum into preallocated arrays.

# Arguments
- `E_k::Vector{T}`: Preallocated output array for energy spectrum values, length `num_bins`.
- `k_bins::Vector{T}`: Preallocated output array for wavenumber bin centers, length `num_bins`.
- `ks_phys::Tuple`: Physical wavenumber ranges.
- `coeffs::AbstractArray`: Complex spectral coefficients.

# Returns
Nothing. Results are written to `E_k` and `k_bins`.

# Example
```julia
num_bins = 32
E_k = zeros(Float64, num_bins)
k_bins = zeros(Float64, num_bins)
isotropic_spectrum!(E_k, k_bins, ks, coeffs; num_bins=num_bins)
```
"""
function isotropic_spectrum!(
    E_k::Vector{T},
    k_bins::Vector{T},
    ks_phys::Tuple,
    coeffs::AbstractArray{Complex{T}, N_dim};
    num_bins::Int = 0,
) where {T<:AbstractFloat, N_dim}
    D = length(ks_phys)
    @assert N_dim == D + 1 "coeffs must have shape (m1, ..., mD, NU)"

    ms = size(coeffs)[1:D]
    NU = size(coeffs, N_dim)

    # Calculate wavenumber magnitude for each grid point
    max_ks = [maximum(abs.(ks_phys[d])) for d in 1:D]
    k_max = minimum(max_ks)

    if num_bins <= 0
        num_bins = minimum(ms) ÷ 2
    end

    @assert length(E_k) == num_bins "E_k length must equal num_bins"
    @assert length(k_bins) == num_bins "k_bins length must equal num_bins"

    bin_edges = range(zero(T), stop = k_max, length = num_bins + 1)
    dk = k_max / num_bins

    # Fill k_bins
    for i in 1:num_bins
        k_bins[i] = T(0.5) * (bin_edges[i] + bin_edges[i+1])
    end

    # Zero out E_k (in case of reuse)
    fill!(E_k, zero(T))

    # Precompute squared wavenumbers per axis.
    kd2 = [T.(ks_phys[d]) .^ 2 for d in 1:D]

    # Accumulate energy in bins
    @inbounds for I in CartesianIndices(ms)
        k_mag = zero(T)
        for d in 1:D
            k_mag += kd2[d][I[d]]
        end
        k_mag = sqrt(k_mag)

        if k_mag <= k_max
            bin_idx = clamp(floor(Int, k_mag / dk) + 1, 1, num_bins)

            energy = zero(T)
            for c in 1:NU
                energy += abs2(coeffs[I, c])
            end
            E_k[bin_idx] += T(0.5) * energy
        end
    end

    # Normalize by bin width
    E_k ./= dk

    return nothing
end

"""
    transect_spectrum!(E_reduced, ks_reduced, ks_phys, coeffs, dims)

In-place version of `transect_spectrum`. Note: this function allocates internally for intermediate grid reductions.
"""
function transect_spectrum!(
    E_reduced::AbstractArray{T},
    ks_reduced::Vector,
    ks_phys::Tuple,
    coeffs::AbstractArray{Complex{T}, N_dim},
    dims::Tuple,
) where {T<:AbstractFloat, N_dim}
    D = length(ks_phys)
    @assert N_dim == D + 1 "coeffs must have shape (m1, ..., mD, NU)"

    ms = size(coeffs)[1:D]
    NU = size(coeffs, N_dim)

    keep_dims = Tuple(d for d in 1:D if !(d in dims))
    sum_dims = Tuple(d for d in 1:D if d in dims)

    out_shape = Tuple(ms[d] for d in keep_dims)
    @assert size(E_reduced) == out_shape "E_reduced size mismatch"

    dk_prod = one(T)
    for d in sum_dims
        dk_prod *= T(ks_phys[d][2] - ks_phys[d][1])
    end

    fill!(E_reduced, zero(T))
    _accumulate_transect!(E_reduced, coeffs, ms, NU, keep_dims, out_shape)
    E_reduced .*= dk_prod

    empty!(ks_reduced)
    for d in keep_dims
        push!(ks_reduced, ks_phys[d])
    end

    return nothing
end

"""
    spherical_energy_spectrum!(E_l::Vector, coeffs::AbstractArray{Complex{T}, 3}; lmax::Int=size(coeffs, 1)-1)

In-place version of `spherical_energy_spectrum`. Computes spherical energy spectrum into preallocated `E_l`.

# Arguments
- `E_l::Vector{T}`: Preallocated output array of length `lmax + 1`.
- `coeffs::AbstractArray{Complex{T}, 3}`: Spherical harmonic coefficients.

# Returns
Nothing. Results written to `E_l`.

# Example
```julia
lmax = 32
E_l = zeros(Float64, lmax + 1)
spherical_energy_spectrum!(E_l, coeffs; lmax=lmax)
```
"""
function spherical_energy_spectrum!(
    E_l::Vector{T},
    coeffs::AbstractArray{Complex{T}, 3};
    lmax::Int = size(coeffs, 1) - 1,
) where {T<:AbstractFloat}
    Nθ, Nφ, NU = size(coeffs)
    @assert Nθ >= lmax + 1 "Coefficients size mismatch"
    @assert length(E_l) == lmax + 1 "E_l length must be lmax + 1"

    # Zero out (in case of reuse)
    fill!(E_l, zero(T))

    @inbounds for l in 0:lmax
        energy = zero(T)
        for m in -l:l
            idx = sph_mode_index(l, m)
            for c in 1:NU
                energy += abs2(coeffs[idx, c])
            end
        end
        E_l[l+1] = T(0.5) * energy
    end

    return nothing
end

end # module Reductions
