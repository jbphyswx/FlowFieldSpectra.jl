module Operators

using FlowTransformBindings: FlowTransformBindings as FTB

export spectral_divergence, spectral_divergence!, spectral_vorticity, spectral_vorticity!,
    compensate, band_energy

# =============================================================================
# Spectral-differentiation operators. The velocity components are the first batch dim (size `D`), so
# coefficients are `(ms…, D, extra_batch…)`; the extra trailing batch dims (levels, time, …) are carried
# through. Output keeps `(ms…, ncomp_out, extra_batch…)`. Each operator is one fused broadcast against the
# wavenumber axes laid out along their own dims in the coefficients' memory, so device coefficients run on
# their device.
# =============================================================================

# Axis `d`'s wavenumbers as an `N`-dim array of size `n_d` along dim `d` and one elsewhere: a reshape of a
# FlowTransformBindings axis, which computes each wavenumber from its index wherever the broadcast runs,
# and otherwise a copy in the memory of `coeffs`.
_k_along(::AbstractArray{<:Any, N}, ks::FTB.AbstractWavenumberAxis, d::Int) where {N} =
    reshape(ks, ntuple(i -> i == d ? length(ks) : 1, Val(N)))
function _k_along(coeffs::AbstractArray{<:Any, N}, ks::AbstractVector, d::Int) where {N}
    T = real(eltype(coeffs))
    k = T[ks[i] for i in eachindex(ks)]
    a = similar(coeffs, T, ntuple(i -> i == d ? length(k) : 1, Val(N)))
    return copyto!(a, k)
end

# Every spectral axis's wavenumbers so laid out.
_k_arrays(coeffs::AbstractArray, ks_phys::Tuple) = _k_arrays(coeffs, ks_phys, 1)
_k_arrays(::AbstractArray, ::Tuple{}, ::Int) = ()
_k_arrays(coeffs::AbstractArray, ks::Tuple, d::Int) =
    (_k_along(coeffs, first(ks), d), _k_arrays(coeffs, Base.tail(ks), d + 1)...)

# Component `c` of the vector field on dim `D + 1`, keeping that dim.
@inline _component(coeffs::AbstractArray{<:Any, N}, ::Val{D}, c::Int) where {N, D} =
    view(coeffs, ntuple(_ -> Colon(), Val(D))..., c:c, ntuple(_ -> Colon(), Val(N - D - 1))...)

# The spectral shape and the extra batch shape of `coeffs` `(ms…, ncomp, extra…)`, checked against
# `ks_phys`.
function _shapes(ks_phys::Tuple, coeffs::AbstractArray{<:Any, N}) where {N}
    D = length(ks_phys)
    N >= D + 1 || throw(ArgumentError("coeffs must have shape (ms…, D, extra…)"))
    ms = ntuple(d -> size(coeffs, d), Val(length(ks_phys)))
    map(length, ks_phys) == ms ||
        throw(DimensionMismatch("ks_phys lengths $(map(length, ks_phys)) must match coeff spectral sizes $ms"))
    extra = ntuple(i -> size(coeffs, D + 1 + i), Val(N - length(ks_phys) - 1))
    return ms, extra
end

"""
    spectral_divergence!(out, ks_phys::Tuple, coeffs) -> out

In-place [`spectral_divergence`](@ref): writes the divergence into preallocated `out` of shape
`(ms…, 1, extra…)`, on the coefficients' device.
"""
function spectral_divergence!(out::AbstractArray{Complex{T}}, ks_phys::Tuple,
        coeffs::AbstractArray{Complex{T}, N}) where {T, N}
    D = length(ks_phys)
    ms, extra = _shapes(ks_phys, coeffs)
    ncomp = size(coeffs, D + 1)
    ncomp == D || throw(ArgumentError("divergence needs D = $D components on dim $(D+1), got $ncomp"))
    size(out) == (ms..., 1, extra...) ||
        throw(DimensionMismatch("out must have shape (ms…, 1, extra…) = $((ms..., 1, extra...)), got $(size(out))"))
    vD = Val(length(ks_phys))
    terms = map((k, d) -> Broadcast.broadcasted(*, k, _component(coeffs, vD, d)), _k_arrays(coeffs, ks_phys),
        ntuple(identity, vD))
    Broadcast.materialize!(out, Broadcast.broadcasted(*, im, Broadcast.broadcasted(+, terms...)))
    return out
end

"""
    spectral_divergence(ks_phys::Tuple, coeffs) -> AbstractArray

Spectral divergence ``\\widehat{\\nabla\\cdot u} = i\\sum_d k_d \\hat u_d`` of a `D`-component vector
field with coefficients `(ms…, D, extra…)`. Returns `(ms…, 1, extra…)` where the coefficients live.
"""
function spectral_divergence(ks_phys::Tuple, coeffs::AbstractArray{Complex{T}, N}) where {T, N}
    ms, extra = _shapes(ks_phys, coeffs)
    return spectral_divergence!(similar(coeffs, ms..., 1, extra...), ks_phys, coeffs)
end

"""
    spectral_vorticity!(out, ks_phys::Tuple, coeffs) -> out

In-place [`spectral_vorticity`](@ref): writes the vorticity into preallocated `out` — shape
`(ms…, 1, extra…)` for `D = 2`, `(ms…, 3, extra…)` for `D = 3` — on the coefficients' device.
"""
function spectral_vorticity!(out::AbstractArray{Complex{T}}, ks_phys::Tuple,
        coeffs::AbstractArray{Complex{T}, N}) where {T, N}
    ms, extra = _shapes(ks_phys, coeffs)
    return _vorticity!(out, _k_arrays(coeffs, ks_phys), coeffs, ms, extra)
end

function _vorticity!(out, (kx, ky)::NTuple{2, AbstractArray}, coeffs, ms, extra)
    size(coeffs, 3) == 2 || throw(ArgumentError("2D vorticity needs 2 components, got $(size(coeffs, 3))"))
    size(out) == (ms..., 1, extra...) ||
        throw(DimensionMismatch("out must have shape (ms…, 1, extra…) = $((ms..., 1, extra...)), got $(size(out))"))
    u, v = _component(coeffs, Val(2), 1), _component(coeffs, Val(2), 2)
    out .= im .* (kx .* v .- ky .* u)
    return out
end

function _vorticity!(out, (kx, ky, kz)::NTuple{3, AbstractArray}, coeffs, ms, extra)
    size(coeffs, 4) == 3 || throw(ArgumentError("3D vorticity needs 3 components, got $(size(coeffs, 4))"))
    size(out) == (ms..., 3, extra...) ||
        throw(DimensionMismatch("out must have shape (ms…, 3, extra…) = $((ms..., 3, extra...)), got $(size(out))"))
    u, v, w = _component(coeffs, Val(3), 1), _component(coeffs, Val(3), 2), _component(coeffs, Val(3), 3)
    _component(out, Val(3), 1) .= im .* (ky .* w .- kz .* v)
    _component(out, Val(3), 2) .= im .* (kz .* u .- kx .* w)
    _component(out, Val(3), 3) .= im .* (kx .* v .- ky .* u)
    return out
end

_vorticity!(out, ka::Tuple, coeffs, ms, extra) =
    throw(ArgumentError("vorticity is undefined for D = $(length(ka)) (need 2 or 3)"))

"""
    spectral_vorticity(ks_phys::Tuple, coeffs) -> AbstractArray

Spectral vorticity ``\\hat\\omega = i\\,k \\times \\hat u`` of a vector field with coefficients
`(ms…, D, extra…)`: `D = 2` → scalar out-of-plane vorticity `(ms…, 1, extra…)`; `D = 3` → 3-component
vorticity `(ms…, 3, extra…)`, where the coefficients live.
"""
function spectral_vorticity(ks_phys::Tuple, coeffs::AbstractArray{Complex{T}, N}) where {T, N}
    ms, extra = _shapes(ks_phys, coeffs)
    out = similar(coeffs, ms..., _vorticity_components(Val(length(ks_phys))), extra...)
    return spectral_vorticity!(out, ks_phys, coeffs)
end

_vorticity_components(::Val{2}) = 1
_vorticity_components(::Val{3}) = 3
_vorticity_components(::Val{D}) where {D} =
    throw(ArgumentError("vorticity is undefined for D = $D (need 2 or 3)"))

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
