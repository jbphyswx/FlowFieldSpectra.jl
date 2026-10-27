module Reductions

using ComputationalBackends: ComputationalBackends
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

# =============================================================================
# Execution. A reduction runs on the calling process: `execution` resolves as the transforms resolve it,
# a process backend to its local one (`_reduction_execution`, defined by the package once
# `_resolve_execution` is). Coefficients on a device are reduced on their device.
# =============================================================================

function _reduction_execution end

"""
    _split_sum!(f!, exec, accs, n) -> accs

`f!(parts, r)` adds the work of the indices `r` into the arrays `parts`, over ranges `r` partitioning `1:n`.
On `SerialBackend` that is `f!(accs, 1:n)`. On `ThreadedBackend` (the OhMyThreads extension) each task adds
its contiguous range into zeroed copies of `accs`, which are then added into `accs`; with `accs = ()` the
tasks write disjoint results themselves.
"""
_split_sum!(f!::F, ::ComputationalBackends.AbstractSerialBackend, accs::Tuple, n::Int) where {F} =
    (f!(accs, 1:n); accs)
_split_sum!(f!, ::ComputationalBackends.AbstractThreadedBackend, args...) =
    throw(ArgumentError("ThreadedBackend is not loaded. Run `using OhMyThreads`."))
_split_sum!(f!, exec::ComputationalBackends.AbstractExecutionBackend, args...) = throw(ArgumentError(
    "a reduction of host arrays runs on SerialBackend or ThreadedBackend; got $(nameof(typeof(exec)))"))

# Absolute batch dims to fold, validated ⊆ (D+1):N and distinct. Accepts an Int or a tuple; `()` folds
# nothing.
@inline _fold_dims(dims::Integer, D, N) = _fold_dims((Int(dims),), D, N)
@inline function _fold_dims(dims::Tuple, D, N)
    fold = map(Int, dims)
    (all(d -> D < d <= N, fold) && _distinct(fold)) ||
        throw(ArgumentError("dims=$dims must reference distinct batch dimensions $(D+1):$N (spectral dims 1:$D are always reduced)"))
    return fold
end

# Radial-bin setup shared by the isotropic-style reductions: `(total, dk, k_top)`, `total` bins of width
# `dk = k_max/num_bins` spanning `[0, k_top]`, with `k_max = min_d max|k_d|` the largest shell the mode
# box holds whole (`num_bins = 0` takes the default count). With `cut` the bins stop at `k_max`; with
# `cut = false` bins of the same width continue to the box's corner, so the bins below `k_max` are the
# same either way.
@inline function _radial_extent(ks_phys::Tuple, num_bins::Int, ::Type{T}, cut::Bool) where {T}
    k_max = kmax(T, ks_phys)
    nb = num_bins <= 0 ? default_bins(ks_phys) : num_bins
    dk = k_max / nb
    cut && return nb, dk, k_max
    total = nb + ceil(Int, (kcorner(T, ks_phys) - k_max) / dk)
    return total, dk, total * dk
end

@inline _kbin_centers(num_bins::Int, dk::T) where {T} = [T(0.5) * ((i - 1) * dk + i * dk) for i in 1:num_bins]

# A mode's contribution to its bin from its fold weight `n` and its entry: `½ n P` from a power,
# `½ n |C|²` from a coefficient, and `½ (X + (n − 1) X̄)` from a cross product, the `n − 1` adding the
# stored mode's conjugate partner on a halved axis.
struct _HalfPower end
struct _HalfAbs2 end
struct _HalfCross end
@inline (::_HalfPower)(n, p) = n * p / 2
@inline (::_HalfAbs2)(n, c) = n * abs2(c) / 2
@inline (::_HalfCross)(n, x) = (x + (n - one(n)) * conj(x)) / 2

# The radial bin of mode `i` of axis 1 in a row whose other axes give `Σ k_d² = krest`: `⌊|k|/dk⌋ + 1`
# clamped to `nb`, or `0` where `cut` leaves `|k| > k_top` out. `unsafe_trunc` is `floor` on the
# non-negative finite quotient.
@inline function _mode_bin(k1, i::Int, krest::T, dk::T, k_top::T, cut::Bool, nb::Int) where {T}
    kmag = sqrt(T(@inbounds k1[i])^2 + krest)
    return ifelse(cut & (kmag > k_top), 0, min(unsafe_trunc(Int, kmag / dk) + 1, nb))
end

"""
    _foreach_radial(f, ks, sz, dk, k_top, cut, nb, rows)

`f(bin, i1, off, Irest)` for every mode of the rows `rows` (linear indices over axes `2:D`) of a spectral
box of size `sz` that has a radial bin (`_mode_bin`): `i1` its index on axis 1, `off` the linear
index before its row and `Irest` the row's indices on the other axes.
"""
function _foreach_radial(f::F, ks::Tuple, sz::NTuple{D, Int}, dk::T, k_top::T, cut::Bool, nb::Int,
        rows::UnitRange{Int}) where {F, D, T}
    k1 = first(ks)
    n1 = sz[1]
    R = CartesianIndices(Base.tail(sz))
    @inbounds for r in rows
        Irest = R[r]
        krest = ksq(T, Base.tail(ks), Irest, 1)
        off = (r - 1) * n1
        for i1 in 1:n1
            bin = _mode_bin(k1, i1, krest, dk, k_top, cut, nb)
            bin == 0 || f(bin, i1, off, Irest)
        end
    end
    return nothing
end

# The rows of a spectral box of size `sz`: the product of its sizes past axis 1.
@inline _nrows(sz::Tuple) = prod(Base.tail(sz); init = 1)

"""
    _radial_accumulate!(exec, E, nmodes, val, A, ks, dk, k_top, cut) -> E

Add `val(n, A[I, b])` to `E[bin, b]` and `n` to `nmodes[bin]` for every mode `I` of `A`
`(ms…, batch…)`, `n` its fold weight and `bin` the radial bin of width `dk` holding `|k(I)|`; `cut`
leaves out `|k| > k_top`. The rows are split on `exec` ([`_split_sum!`](@ref)). The GPUArraysCore
extension runs a device `A` on its device, `E` and `nmodes` staying on the host.
"""
function _radial_accumulate!(exec::ComputationalBackends.AbstractExecutionBackend, E::AbstractArray,
        nmodes::AbstractVector{T}, val::V, A::AbstractArray, ks_phys::Tuple, dk::T, k_top::T,
        cut::Bool) where {T, V}
    D = length(ks_phys)
    nb = length(nmodes)
    k1 = first(ks_phys)
    sz = ntuple(d -> size(A, d), Val(D))
    NM = prod(sz)
    NM == 0 && return E
    nbat = length(A) ÷ NM
    _split_sum!(exec, (E, nmodes), _nrows(sz)) do (Ep, np), rows
        _foreach_radial(ks_phys, sz, dk, k_top, cut, nb, rows) do bin, i1, off, _
            n = T(FTB.fold_weight(k1, i1))
            @inbounds np[bin] += n
            @inbounds for b in 1:nbat
                Ep[bin + (b - 1) * nb] += val(n, A[off + i1 + (b - 1) * NM])
            end
        end
    end
    return E
end

# Allocating setup (returns the k-bin centers too) for the allocating reductions.
@inline function _radial_setup(ks_phys::Tuple, num_bins::Int, ::Type{T}, cut::Bool = true) where {T}
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
                       convention=SpectralConvention(), execution=AutoBackend())

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

`execution` splits the binning over threads on `ThreadedBackend` (with `using OhMyThreads`) and runs it on one
on `SerialBackend`; `AutoBackend` resolves as the transforms resolve it, and a process backend to the calling
process's own. Coefficients on a device are reduced on their device.
"""
function isotropic_spectrum(ks_phys::Tuple, coeffs::AbstractArray{Complex{T}, N};
        num_bins::Int = 0, dims = (), cutoff::Bool = true,
        convention::Normalization.SpectralConvention = Normalization.SpectralConvention(),
        execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend()) where {T, N}
    D = length(ks_phys)
    N >= D || throw(ArgumentError("coeffs must have ≥ $D spectral dims (got $N)"))
    _radial_sided(convention.sided)
    fold = _fold_dims(dims, D, N)
    P = isempty(fold) ? abs2.(coeffs) : dropdims(sum(abs2, coeffs; dims = fold); dims = fold)
    return _bin_isotropic(_reduction_execution(execution), ks_phys, P, num_bins, convention, cutoff)
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
# same `|k|`, and nothing else in the coefficient box. `cut` stops at `k_top = k_max`.
function _bin_isotropic(exec::ComputationalBackends.AbstractExecutionBackend, ks_phys::Tuple,
        P::AbstractArray{T, NP}, num_bins::Int, convention::Normalization.SpectralConvention, cut::Bool) where {T, NP}
    D = length(ks_phys)
    num_bins, dk, k_top, k_bins = _radial_setup(ks_phys, num_bins, T, cut)
    E = zeros(T, num_bins, ntuple(i -> size(P, D + i), Val(NP - D))...)
    nmodes = zeros(T, num_bins)
    _radial_accumulate!(exec, E, nmodes, _HalfPower(), P, ks_phys, dk, k_top, cut)
    _shell_values!(E, nmodes, dk, ks_phys, convention.estimator)
    _scale_bins!(E, dk, convention.scaling)
    return k_bins, E
end

"""
    isotropic_spectrum!(E, k_bins, ks_phys, coeffs; num_bins=0, cutoff=true, convention=SpectralConvention(),
                        execution=AutoBackend())

In-place isotropic spectrum that **preserves all batch dims** (no folding): fills preallocated `E` (shape
`(nb, batch…)`) and `k_bins` (length `nb`), allocating nothing on `SerialBackend`. With the cutoff `nb` bins
span `[0, k_max]`; with `cutoff = false`, `num_bins` of them span `[0, k_max]` (`0`: the default count) and
`nb` is the total that reaches the corner of the mode box, the length `isotropic_spectrum` returns.
`cutoff`, `convention` and `execution` are read as in [`isotropic_spectrum`](@ref).
"""
function isotropic_spectrum!(E::AbstractArray{T}, k_bins::AbstractVector{T}, ks_phys::Tuple,
        coeffs::AbstractArray{Complex{T}, N}; num_bins::Int = 0, cutoff::Bool = true,
        convention::Normalization.SpectralConvention = Normalization.SpectralConvention(),
        execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend()) where {T, N}
    D = length(ks_phys)
    N >= D || throw(ArgumentError("coeffs must have ≥ $D spectral dims (got $N)"))
    _radial_sided(convention.sided)
    req = _requested_bins(k_bins, num_bins, ks_phys, T, cutoff)
    _fill_isotropic!(_reduction_execution(execution), E, k_bins, ks_phys, coeffs, convention, req, cutoff)
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
    total = first(_radial_extent(ks_phys, num_bins, T, false))
    total == nb || throw(DimensionMismatch(
        "cutoff = false with num_bins = $num_bins fills $total bins up to the corner of the mode box; " *
        "k_bins has $nb"))
    return num_bins
end

# `k_bins` holds the bins' mode counts until the bin centres are written into it.
function _fill_isotropic!(exec::ComputationalBackends.AbstractExecutionBackend, E::AbstractArray{T},
        k_bins::AbstractVector{T}, ks_phys::Tuple, coeffs::AbstractArray{Complex{T}, N},
        convention::Normalization.SpectralConvention, req::Int, cut::Bool) where {T, N}
    nb, dk, k_top = _radial_extent(ks_phys, req, T, cut)
    fill!(E, zero(T))
    fill!(k_bins, zero(T))
    _radial_accumulate!(exec, E, k_bins, _HalfAbs2(), coeffs, ks_phys, dk, k_top, cut)
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
                         convention=SpectralConvention(), execution=AutoBackend())

Anisotropy-resolved 2D energy spectrum `E(k, θ, batch…)` for a 2D field: bin `½|C|²` by wavenumber
magnitude up to `k_max` and polar angle, preserving batch. The `k = 0` mode has no angle and is spread
evenly over the `θ` bins of the first `k` bin, so integrating over `θ` recovers the isotropic spectrum
in every bin. Under `ModeAverage()` a bin's mean is scaled by the mode count of its whole annular sector,
`(dθ/2)(k₊² − k₋²)/(Δk₁Δk₂)`; `DensityScaling()` divides by `dk·dθ`. `execution` is read as in
[`isotropic_spectrum`](@ref).
"""
function anisotropic_spectrum(ks_phys::Tuple, coeffs::AbstractArray{Complex{T}, N};
        num_k_bins::Int = 0, num_θ_bins::Int = 16, dims = (),
        convention::Normalization.SpectralConvention = Normalization.SpectralConvention(),
        execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend()) where {T, N}
    length(ks_phys) == 2 || throw(ArgumentError("anisotropic_spectrum is defined for 2D fields"))
    N >= 2 || throw(ArgumentError("coeffs must have ≥ 2 spectral dims"))
    _radial_sided(convention.sided)
    fold = _fold_dims(dims, 2, N)
    P = isempty(fold) ? abs2.(coeffs) : dropdims(sum(abs2, coeffs; dims = fold); dims = fold)
    return _bin_anisotropic(_reduction_execution(execution), ks_phys, P, num_k_bins, num_θ_bins, convention)
end

function _bin_anisotropic(exec::ComputationalBackends.AbstractExecutionBackend, ks_phys::Tuple,
        P::AbstractArray{T, NP}, num_k_bins::Int, num_θ_bins::Int,
        convention::Normalization.SpectralConvention) where {T, NP}
    k_max = kmax(T, ks_phys)
    num_k_bins <= 0 && (num_k_bins = default_bins(ks_phys))
    dk = k_max / num_k_bins
    dθ = T(2π) / num_θ_bins
    k_bins = [T(0.5) * dk + (i - 1) * dk for i in 1:num_k_bins]
    θ_bins = [-T(π) + (j - T(0.5)) * dθ for j in 1:num_θ_bins]
    E = zeros(T, num_k_bins, num_θ_bins, ntuple(i -> size(P, 2 + i), Val(NP - 2))...)
    nmodes = zeros(T, num_k_bins, num_θ_bins)
    _sector_accumulate!(exec, E, nmodes, P, ks_phys, dk, dθ, k_max)
    _sector_values!(E, nmodes, dk, dθ, ks_phys, convention.estimator)
    _scale_bins!(E, dk * dθ, convention.scaling)
    return k_bins, θ_bins, E
end

"""
    _sector_accumulate!(exec, E, nmodes, P, ks, dk, dθ, k_max) -> E

Add `½ P[I, b]` to `E[ik, iθ, b]` and one to `nmodes[ik, iθ]` for every mode `I` of the 2D power `P`
up to `k_max`, `(ik, iθ)` the bin of its magnitude and angle; a halved-axis mode's partner at `−k` lands
in the opposite angle, and `k = 0` is spread evenly over the angles of the first bin. The rows are split on
`exec` ([`_split_sum!`](@ref)). The GPUArraysCore extension runs a device `P` on its device.
"""
function _sector_accumulate!(exec::ComputationalBackends.AbstractExecutionBackend, E::AbstractArray{T},
        nmodes::AbstractMatrix{T}, P::AbstractArray, ks_phys::Tuple, dk::T, dθ::T, k_max::T) where {T}
    nk, nθ = size(nmodes)
    k1, k2 = ks_phys
    sz = (size(P, 1), size(P, 2))
    NM = prod(sz)
    NM == 0 && return E
    nbat = length(P) ÷ NM
    S = nk * nθ
    _split_sum!(exec, (E, nmodes), _nrows(sz)) do (Ep, np), rows
        _foreach_radial(ks_phys, sz, dk, k_max, true, nk, rows) do ik, i1, off, Irest
            kx = T(k1[i1])
            ky = T(k2[Irest[1]])
            p = off + i1
            if iszero(kx) && iszero(ky)
                # `k = 0` has no direction: its energy spreads evenly over the angles, so the first `k` bin
                # summed over `θ` holds it whole, as the isotropic spectrum's does.
                @inbounds for iθ in 1:nθ
                    np[1 + (iθ - 1) * nk] += one(T) / nθ
                    for b in 1:nbat
                        Ep[1 + (iθ - 1) * nk + (b - 1) * S] += T(0.5) * P[p + (b - 1) * NM] / nθ
                    end
                end
                return nothing
            end
            # The halved axis stores its even Nyquist slot as +k₁max, which a full axis labels −k₁max, so
            # that sign gives the angle of the two-sided spectrum.
            kxa = (FTB.is_halved(k1) && FTB.is_nyquist(k1, i1)) ? -kx : kx
            θ = atan(ky, kxa)
            _sector_add!(Ep, np, P, ik, _angle_bin(θ, dθ, nθ), nk, S, p, NM, nbat)
            # A halved-axis interior mode's conjugate partner at −k carries the same power at angle θ±π; on
            # a full layout the fold weight is 1 and the −k mode is binned on its own.
            if FTB.fold_weight(k1, i1) != 1
                _sector_add!(Ep, np, P, ik, _angle_bin(θ > 0 ? θ - T(π) : θ + T(π), dθ, nθ), nk, S, p, NM, nbat)
            end
            return nothing
        end
    end
    return E
end

@inline _angle_bin(θ::T, dθ::T, nθ::Int) where {T} = clamp(floor(Int, (θ + T(π)) / dθ) + 1, 1, nθ)

# Mode `p`'s power into sector `(ik, iθ)` of every field, and one into the sector's mode count.
@inline function _sector_add!(E, nmodes, P, ik::Int, iθ::Int, nk::Int, S::Int, p::Int, NM::Int, nbat::Int)
    j = ik + (iθ - 1) * nk
    @inbounds nmodes[j] += one(eltype(nmodes))
    @inbounds for b in 1:nbat
        E[j + (b - 1) * S] += eltype(E)(0.5) * P[p + (b - 1) * NM]
    end
    return nothing
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
    cross_spectrum(ks_phys::Tuple, coeffs_f, coeffs_g; num_bins=0, dims=(), convention=SpectralConvention(),
                   execution=AutoBackend())

Radially-binned cross-spectrum `S_fg(k, batch…)`, a bin collecting `½ f̂ conj(ĝ)` over its modes up to
`k_max`; `coeffs_f`, `coeffs_g` share shape `(ms…, batch…)`. `convention` and `execution` are read as in
[`isotropic_spectrum`](@ref). Real part → co-spectrum, negative imag part → quad spectrum.
"""
function cross_spectrum(ks_phys::Tuple, coeffs_f::AbstractArray{Complex{T}, N},
        coeffs_g::AbstractArray{Complex{T}, N}; num_bins::Int = 0, dims = (),
        convention::Normalization.SpectralConvention = Normalization.SpectralConvention(),
        execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend()) where {T, N}
    D = length(ks_phys)
    size(coeffs_f) == size(coeffs_g) || throw(DimensionMismatch("coeffs_f and coeffs_g must match"))
    N >= D || throw(ArgumentError("coeffs must have ≥ $D spectral dims"))
    _radial_sided(convention.sided)
    fold = _fold_dims(dims, D, N)
    X = coeffs_f .* conj.(coeffs_g)
    P = isempty(fold) ? X : dropdims(sum(X; dims = fold); dims = fold)
    return _bin_cross(_reduction_execution(execution), ks_phys, P, num_bins, convention)
end

function _bin_cross(exec::ComputationalBackends.AbstractExecutionBackend, ks_phys::Tuple,
        P::AbstractArray{Complex{T}, NP}, num_bins::Int, convention::Normalization.SpectralConvention) where {T, NP}
    D = length(ks_phys)
    num_bins, dk, k_max, k_bins = _radial_setup(ks_phys, num_bins, T)
    S = zeros(Complex{T}, num_bins, ntuple(i -> size(P, D + i), Val(NP - D))...)
    nmodes = zeros(T, num_bins)
    _radial_accumulate!(exec, S, nmodes, _HalfCross(), P, ks_phys, dk, k_max, true)
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
    transect_spectrum(ks_phys::Tuple, coeffs, dims::Tuple; execution=AutoBackend())

Integrate the spectral energy density `½|C|²` along the spectral dimensions `dims` (1-indexed, ⊆
`1:D`), scaling by their wavenumber spacing. Returns `(ks_reduced, E_reduced)` where `E_reduced` has
the *kept* spectral dims followed by the batch dims. `execution` is read as in [`isotropic_spectrum`](@ref).

`Σ E_reduced / ∏dk` over the kept axes is the field's folded Parseval total over the coefficient box,
whichever axes are kept: a kept full axis carries both signs of its wavenumber outright, and a kept halved
axis reports `|k₁|` with the `−k₁` energy folded in. So a kept full axis's entries at `±k` are equal for a
real field's spectrum and differ for a complex field's, where the sign carries the propagation direction.
"""
function transect_spectrum(ks_phys::Tuple, coeffs::AbstractArray{Complex{T}, N}, dims::Tuple;
        execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend()) where {T, N}
    D = length(ks_phys)
    _check_transect(dims, D)
    kept_shape = ntuple(i -> size(coeffs, _kept_dim(i, dims, D)), Val(N - length(dims)))
    E_reduced = zeros(T, kept_shape...)
    transect_spectrum!(E_reduced, ks_phys, coeffs, dims; execution)
    return _kept_axes(ks_phys, dims, 1), E_reduced
end

_check_transect(dims::Tuple, D::Int) = (all(d -> 1 <= d <= D, dims) && _distinct(dims)) ||
    throw(ArgumentError("transect dims must be distinct spectral dims (1:$D); got $dims"))

# Whether no entry of `t` repeats.
_distinct(::Tuple{}) = true
_distinct(t::Tuple) = !(first(t) in Base.tail(t)) && _distinct(Base.tail(t))

# The `i`-th dim not among the spectral `dims`.
function _kept_dim(i::Int, dims::Tuple, D::Int)
    d = 0
    while i > 0
        d += 1
        (d <= D && d in dims) || (i -= 1)
    end
    return d
end

# The wavenumber axes not among `dims`, in order.
_kept_axes(::Tuple{}, ::Tuple, ::Int) = ()
_kept_axes(ks::Tuple, dims::Tuple, d::Int) =
    d in dims ? _kept_axes(Base.tail(ks), dims, d + 1) : (first(ks), _kept_axes(Base.tail(ks), dims, d + 1)...)

"""
    transect_spectrum!(E_reduced, ks_phys, coeffs, dims; execution=AutoBackend()) -> nothing

In-place [`transect_spectrum`](@ref), allocating nothing on `SerialBackend`: fills preallocated
`E_reduced` (kept spectral dims + batch dims) with the `dims`-integrated `½|C|²` density.
"""
function transect_spectrum!(E_reduced::AbstractArray{T}, ks_phys::Tuple,
        coeffs::AbstractArray{Complex{T}, N}, dims::Tuple;
        execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend()) where {T, N}
    _check_transect(dims, length(ks_phys))
    dk_prod = dk_product(T, ks_phys, dims)
    fill!(E_reduced, zero(T))
    _transect_accumulate!(_reduction_execution(execution), E_reduced, ks_phys, coeffs, dims)
    E_reduced .*= dk_prod
    return nothing
end

"""
    _transect_accumulate!(exec, E_reduced, ks, coeffs, dims) -> E_reduced

Add `½|C|²` of every entry of `coeffs` to the entry of `E_reduced` its kept indices name, with the fold
weights and partners of a halved axis. The columns of `coeffs` along axis 1, one per index of dims `2…N`,
are split on `exec` ([`_split_sum!`](@ref)). The GPUArraysCore extension runs a device `coeffs` on its
device.
"""
function _transect_accumulate!(exec::ComputationalBackends.AbstractExecutionBackend, E_reduced::AbstractArray{T},
        ks_phys::Tuple, coeffs::AbstractArray{Complex{T}, N}, dims::Tuple) where {T, N}
    length(coeffs) == 0 && return E_reduced
    D = length(ks_phys)
    ncols = length(coeffs) ÷ size(coeffs, 1)
    if all(==(1), dims) && !(FTB.is_halved(ks_phys[1]) && 1 in dims && D >= 2)
        # Dims `2…N` are kept and no halved-axis mirror negates them, so the tasks write disjoint entries of
        # `E_reduced`.
        _split_sum!(exec, (), ncols) do _, r
            _transect_block!(E_reduced, ks_phys, coeffs, dims, r)
        end
    else
        _split_sum!(exec, (E_reduced,), ncols) do (Ep,), r
            _transect_block!(Ep, ks_phys, coeffs, dims, r)
        end
    end
    return E_reduced
end

# `_transect_accumulate!` over columns `r` of `coeffs`, column `t` the `t`-th index of dims `2…N`.
function _transect_block!(E_reduced::AbstractArray{T}, ks_phys::Tuple, coeffs::AbstractArray{Complex{T}, N},
        dims::Tuple, r::UnitRange{Int}) where {T, N}
    D = length(ks_phys)
    k1 = first(ks_phys)
    n1 = size(coeffs, 1)
    cols = CartesianIndices(ntuple(i -> size(coeffs, i + 1), Val(N - 1)))
    kept = ntuple(d -> !(d <= D && d in dims), Val(N))
    # Each dim's stride in E_reduced (kept spectral dims + all batch dims), zero for a summed dim.
    ostride = _kept_strides(kept, size(coeffs), 1)
    halved_out = FTB.is_halved(k1) && (1 in dims)
    # The kept spectral dims past axis 1, which the mirror of a halved-axis mode negates.
    mirrored = Base.tail(ntuple(d -> halved_out && 2 <= d <= D && kept[d], Val(N)))
    axes_n = Base.tail(ntuple(d -> d <= D ? ks_phys[d] : nothing, Val(N)))
    h = T(0.5)
    @inbounds for t in r
        J = cols[t]
        base = 1 + sum(map((i, s) -> (i - 1) * s, Tuple(J), Base.tail(ostride)))
        if first(kept)
            # A kept halved axis reports `|k₁|`, and the field's `−k₁` energy is stored nowhere else, so each
            # row carries its `fold_weight`: 2, and 1 at dc and, for even `n₁`, at Nyquist.
            s1 = first(ostride)
            for i1 in 1:n1
                E_reduced[base + (i1 - 1) * s1] += T(FTB.fold_weight(k1, i1)) * h * abs2(coeffs[i1, J])
            end
        elseif halved_out
            # Integrating out the halved axis: a stored row stands for itself and, where its fold weight is 2,
            # for its `k₁ < 0` mirror, the conjugate of the stored mode, whose kept `k_·` is negated.
            s, s2 = zero(T), zero(T)
            @simd for i1 in 1:n1
                e = abs2(coeffs[i1, J])
                s += e
                s2 += ifelse(FTB.fold_weight(k1, i1) == 2, e, zero(T))
            end
            E_reduced[base] += h * s
            E_reduced[1 + sum(map((i, s, m, k) -> (_mirror_index(m, k, i) - 1) * s, Tuple(J), Base.tail(ostride),
                mirrored, axes_n))] += h * s2
        else
            s = zero(T)
            @simd for i1 in 1:n1
                s += abs2(coeffs[i1, J])
            end
            E_reduced[base] += h * s
        end
    end
    return E_reduced
end

# Index `i` of an axis, or its negated frequency's where `m`.
@inline _mirror_index(m::Bool, k, i::Int) = m ? FTB.neg_index(k, i) : i
@inline _mirror_index(::Bool, ::Nothing, i::Int) = i

# The column-major strides of the kept dims, `s` the stride of the first kept one, zero for the others.
_kept_strides(::Tuple{}, ::Tuple{}, s::Int) = ()
_kept_strides(kept::Tuple, sz::Tuple, s::Int) = first(kept) ?
    (s, _kept_strides(Base.tail(kept), Base.tail(sz), s * first(sz))...) :
    (0, _kept_strides(Base.tail(kept), Base.tail(sz), s)...)

# =============================================================================
# Spherical degree energy spectrum — E(ℓ, batch…)
# =============================================================================

"""
    spherical_energy_spectrum(coeffs; lmax=size(coeffs,1)-1, execution=AutoBackend())

Degree energy spectrum `E(ℓ, batch…) = ½ Σ_{m=-ℓ}^{ℓ} |C_ℓ^m|²` of spherical-harmonic coefficients
`(Nθ, Nφ, batch…)`. Returns `(0:lmax, E_l)`, `E_l` of shape `(lmax+1, batch…)` and real element type,
from real or complex coefficients. `execution` is read as in [`isotropic_spectrum`](@ref).
"""
function spherical_energy_spectrum(coeffs::AbstractArray{<:Number, N}; lmax::Int = size(coeffs, 1) - 1,
        execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend()) where {N}
    N >= 2 || throw(ArgumentError("coeffs must have ≥ 2 spectral dims (Nθ, Nφ)"))
    T = real(float(eltype(coeffs)))
    batch = ntuple(i -> size(coeffs, 2 + i), Val(N - 2))
    E = zeros(T, lmax + 1, batch...)
    _accumulate_degree!(_reduction_execution(execution), E, coeffs, lmax)
    return 0:lmax, E
end

"""
    spherical_energy_spectrum!(E_l, coeffs; lmax=size(coeffs,1)-1, execution=AutoBackend()) -> nothing

In-place [`spherical_energy_spectrum`](@ref): fills preallocated `E_l` (shape `(lmax+1, batch…)`).
"""
function spherical_energy_spectrum!(E_l::AbstractArray{T}, coeffs::AbstractArray{<:Number, N};
        lmax::Int = size(coeffs, 1) - 1,
        execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend()) where {T, N}
    fill!(E_l, zero(T))
    _accumulate_degree!(_reduction_execution(execution), E_l, coeffs, lmax)
    return nothing
end

# The degrees are split on `exec` ([`_split_sum!`](@ref)), each task writing its own degrees' entries.
function _accumulate_degree!(exec::ComputationalBackends.AbstractExecutionBackend, E::AbstractArray{T},
        coeffs::AbstractArray{<:Number, N}, lmax::Int) where {T, N}
    _split_sum!(exec, (), lmax + 1) do _, ls
        _degree_block!(E, coeffs, lmax, ls)
    end
    return E
end

function _degree_block!(E::AbstractArray{T}, coeffs::AbstractArray{<:Number, N}, lmax::Int,
        ls::UnitRange{Int}) where {T, N}
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
        for l1 in ls
            l = l1 - 1
            acc = zero(T)
            for m in -l:l
                idx = sph_mode_index(l, m)
                acc += abs2(coeffs[idx[1] + (idx[2] - 1) * Nθ + cbase])
            end
            E[l1 + ebase] += T(0.5) * acc
        end
    end
    return E
end

end # module Reductions
