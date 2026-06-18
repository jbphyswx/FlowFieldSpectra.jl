module Reductions

using Statistics: mean
using ..DirectSum: sph_mode_index

export isotropic_spectrum, isotropic_spectrum!, transect_spectrum, transect_spectrum!, spherical_energy_spectrum, spherical_energy_spectrum!

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

    # Accumulate energy in bins
    # E(k) = 1/2 * sum |C|^2
    for I in CartesianIndices(ms)
        # Compute physical wavenumber magnitude
        k_mag = zero(T)
        for d in 1:D
            k_mag += ks_phys[d][I[d]]^2
        end
        k_mag = sqrt(k_mag)

        # Skip points outside the Nyquist circle/sphere
        if k_mag <= k_max
            # Find bin index
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

    # Compute energy density at each mode
    # E = 1/2 * sum |C|^2
    energy_grid = zeros(T, ms...)
    for I in CartesianIndices(ms)
        val = zero(T)
        for c in 1:NU
            val += abs2(coeffs[I, c])
        end
        energy_grid[I] = T(0.5) * val
    end

    # Dimensions to sum over
    sum_dims = filter(d -> d in dims, 1:D)
    keep_dims = filter(d -> !(d in dims), 1:D)

    # Sum along dims
    reduced_energy = sum(energy_grid, dims=Tuple(sum_dims))

    # Wavenumber spacing for summed dimensions to convert sum to integration density
    for d in sum_dims
        dk_d = ks_phys[d][2] - ks_phys[d][1]
        reduced_energy .*= dk_d
    end

    # Reshape and extract
    out_shape = Tuple(ms[d] for d in keep_dims)
    E_reduced = reshape(reduced_energy, out_shape...)
    ks_reduced = Tuple(ks_phys[d] for d in keep_dims)

    return ks_reduced, E_reduced
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

    # Accumulate energy in bins
    @inbounds for I in CartesianIndices(ms)
        k_mag = zero(T)
        for d in 1:D
            k_mag += ks_phys[d][I[d]]^2
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

    # Compute energy density
    energy_grid = zeros(T, ms...)
    for I in CartesianIndices(ms)
        val = zero(T)
        for c in 1:NU
            val += abs2(coeffs[I, c])
        end
        energy_grid[I] = T(0.5) * val
    end

    sum_dims = filter(d -> d in dims, 1:D)
    keep_dims = filter(d -> !(d in dims), 1:D)

    reduced_energy = sum(energy_grid, dims=Tuple(sum_dims))

    for d in sum_dims
        dk_d = ks_phys[d][2] - ks_phys[d][1]
        reduced_energy .*= dk_d
    end

    out_shape = Tuple(ms[d] for d in keep_dims)
    @assert size(E_reduced) == out_shape "E_reduced size mismatch"

    # Copy to output
    copyto!(E_reduced, reshape(reduced_energy, out_shape))

    # Fill ks_reduced
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
