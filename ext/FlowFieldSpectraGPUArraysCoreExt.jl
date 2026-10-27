module FlowFieldSpectraGPUArraysCoreExt

# Spectral reductions of coefficients held on a device, whatever execution backend the call names. Each
# mode adds its contribution to its bin on the device, and only the bins, a few hundred numbers per field,
# come back to the host, which finishes them as it does a host reduction. The copy of the bins to the host
# is ordered after the kernel.

using KernelAbstractions: KernelAbstractions as KA, @index, @Const
using GPUArraysCore: GPUArraysCore
using ComputationalBackends: ComputationalBackends as CB
using FlowFieldSpectra: FlowFieldSpectra as FFS
using FlowTransformBindings: FlowTransformBindings as FTB

_upload(backend, v::AbstractVector) = copyto!(KA.allocate(backend, eltype(v), length(v)), v)

function FFS.Reductions._radial_accumulate!(::CB.AbstractExecutionBackend, E::AbstractArray,
        nmodes::AbstractVector{T}, val::V, A::GPUArraysCore.AbstractGPUArray, ks_phys::Tuple, dk::T, k_top::T,
        cut::Bool) where {T, V}
    D = length(ks_phys)
    backend = KA.get_backend(A)
    ms = ntuple(d -> size(A, d), Val(D))
    NM = prod(ms)
    B = length(A) ÷ max(NM, 1)
    nb = length(nmodes)
    NM == 0 && return E
    ksd = map(k -> _upload(backend, T[k[i] for i in eachindex(k)]), ks_phys)
    w1 = _upload(backend, vec(FTB.fold_weights(T, ks_phys)))
    cplx = eltype(E) <: Complex
    Er = KA.zeros(backend, T, nb, B)
    Ei = KA.zeros(backend, T, cplx ? nb : 1, cplx ? B : 1)
    nm = KA.zeros(backend, T, nb)
    _radial_kernel!(backend)(Er, Ei, nm, val, reshape(A, NM, B), ksd, w1, CartesianIndices(ms), dk, k_top,
        cut, cplx, nb, B; ndrange = NM)
    Eh = Array(Er)
    E .+= reshape(cplx ? complex.(Eh, Array(Ei)) : Eh, size(E))
    nmodes .+= Array(nm)
    return E
end

KA.@kernel function _radial_kernel!(Er, Ei, nm, val, @Const(A), ksd, @Const(w1), cart, dk, k_top, cut::Bool,
        cplx::Bool, nb::Int, B::Int)
    t = @index(Global)
    @inbounds begin
        I = cart[t]
        kmag = sqrt(_ksq(typeof(dk), ksd, I, 1))
        if !(cut && kmag > k_top)
            bin = clamp(unsafe_trunc(Int, floor(kmag / dk)) + 1, 1, nb)
            n = w1[ifelse(length(w1) == 1, 1, I[1])]
            KA.@atomic nm[bin] += n
            for b in 1:B
                v = val(n, A[t, b])
                KA.@atomic Er[bin, b] += real(v)
                cplx && (KA.@atomic Ei[bin, b] += imag(v))
            end
        end
    end
end

function FFS.Reductions._sector_accumulate!(::CB.AbstractExecutionBackend, E::AbstractArray{T},
        nmodes::AbstractMatrix{T}, P::GPUArraysCore.AbstractGPUArray, ks_phys::Tuple, dk::T, dθ::T, k_max::T) where {T}
    backend = KA.get_backend(P)
    n1, n2 = size(P, 1), size(P, 2)
    n1 * n2 == 0 && return E
    B = length(P) ÷ (n1 * n2)
    nk, nθ = size(nmodes)
    a1 = ks_phys[1]
    kx = _upload(backend, T[a1[i] for i in 1:n1])
    ky = _upload(backend, T[ks_phys[2][i] for i in 1:n2])
    flip = _upload(backend, Bool[FTB.is_halved(a1) && FTB.is_nyquist(a1, i) for i in 1:n1])
    pair = _upload(backend, Bool[FTB.fold_weight(a1, i) != 1 for i in 1:n1])
    Ed = KA.zeros(backend, T, nk, nθ, B)
    nm = KA.zeros(backend, T, nk, nθ)
    _sector_kernel!(backend)(Ed, nm, reshape(P, n1 * n2, B), kx, ky, flip, pair, n1, dk, dθ, k_max, B;
        ndrange = n1 * n2)
    E .+= reshape(Array(Ed), size(E))
    nmodes .+= Array(nm)
    return E
end

# The host loop of `Reductions._sector_accumulate!`, one thread per mode.
KA.@kernel function _sector_kernel!(Ed, nm, @Const(P), @Const(kx), @Const(ky), @Const(flip), @Const(pair),
        n1::Int, dk, dθ, k_max, B::Int)
    t = @index(Global)
    T = eltype(Ed)
    nk, nθ = size(nm)
    @inbounds begin
        i1 = (t - 1) % n1 + 1
        i2 = (t - 1) ÷ n1 + 1
        x = kx[i1]; y = ky[i2]
        kmag = sqrt(x^2 + y^2)
        if kmag <= k_max
            if iszero(kmag)
                for iθ in 1:nθ
                    KA.@atomic nm[1, iθ] += one(T) / nθ
                    for b in 1:B
                        KA.@atomic Ed[1, iθ, b] += T(0.5) * P[t, b] / nθ
                    end
                end
            else
                ik = clamp(unsafe_trunc(Int, floor(kmag / dk)) + 1, 1, nk)
                θ = atan(y, ifelse(flip[i1], -x, x))
                iθ = clamp(unsafe_trunc(Int, floor((θ + T(π)) / dθ)) + 1, 1, nθ)
                KA.@atomic nm[ik, iθ] += one(T)
                for b in 1:B
                    KA.@atomic Ed[ik, iθ, b] += T(0.5) * P[t, b]
                end
                if pair[i1]
                    θm = θ > 0 ? θ - T(π) : θ + T(π)
                    iθm = clamp(unsafe_trunc(Int, floor((θm + T(π)) / dθ)) + 1, 1, nθ)
                    KA.@atomic nm[ik, iθm] += one(T)
                    for b in 1:B
                        KA.@atomic Ed[ik, iθm, b] += T(0.5) * P[t, b]
                    end
                end
            end
        end
    end
end

function FFS.Reductions._transect_accumulate!(::CB.AbstractExecutionBackend, E_reduced::AbstractArray{T},
        ks_phys::Tuple, coeffs::GPUArraysCore.AbstractGPUArray{Complex{T}, N}, dims::Tuple) where {T, N}
    D = length(ks_phys)
    length(coeffs) == 0 && return E_reduced
    backend = KA.get_backend(coeffs)
    sz = size(coeffs)
    keep = ntuple(d -> !(d <= D && d in dims), Val(N))
    halved = FTB.is_halved(ks_phys[1])
    w1 = _upload(backend, T[halved ? FTB.fold_weight(ks_phys[1], i) : 1 for i in 1:sz[1]])
    neg = ntuple(d -> _upload(backend, d <= D ? Int[FTB.neg_index(ks_phys[d], i) for i in 1:sz[d]] : Int[0]),
        Val(N))
    Ed = KA.zeros(backend, T, length(E_reduced))
    _transect_kernel!(backend)(Ed, coeffs, CartesianIndices(sz), keep, w1, neg, D, halved && !keep[1],
        halved && keep[1]; ndrange = length(coeffs))
    E_reduced .+= reshape(Array(Ed), size(E_reduced))
    return E_reduced
end

# The host loop of `Reductions._transect_accumulate!`, one thread per coefficient.
KA.@kernel function _transect_kernel!(Ed, @Const(C), cart, keep, @Const(w1), neg, D::Int, halved_out::Bool,
        halved_kept::Bool)
    t = @index(Global)
    T = eltype(Ed)
    @inbounds begin
        I = cart[t]
        lin = 1
        stride = 1
        for d in 1:length(keep)
            if keep[d]
                lin += (I[d] - 1) * stride
                stride *= size(cart, d)
            end
        end
        if halved_out
            KA.@atomic Ed[lin] += T(0.5) * abs2(C[I])
            if w1[I[1]] == 2
                KA.@atomic Ed[lin] += T(0.5) * abs2(C[_partner(I, neg, D)])
            end
        else
            KA.@atomic Ed[lin] += ifelse(halved_kept, w1[I[1]], one(T)) * T(0.5) * abs2(C[I])
        end
    end
end

# The halved axis's stored partner of `I`: the same row, every other spectral index negated.
@inline _partner(I::CartesianIndex{N}, neg::Tuple, D::Int) where {N} =
    CartesianIndex(ntuple(d -> d == 1 || d > D ? I[d] : @inbounds(neg[d][I[d]]), Val(N)))

# `Σ_d k_d(I_d)²` over the device wavenumber axes, unrolled over the tuple.
@inline _ksq(::Type{T}, ksd::Tuple, I::CartesianIndex, d::Int) where {T} =
    (@inbounds first(ksd)[I[d]])^2 + _ksq(T, Base.tail(ksd), I, d + 1)
@inline _ksq(::Type{T}, ::Tuple{}, ::CartesianIndex, ::Int) where {T} = zero(T)

# Each realization binned on the device by `_radial_accumulate!` into its own column of `E` `(nb, R)`, the
# columns then summed on the host.
function _binned_realizations(exec, ::Type{E}, nmodes, val, A, ks_phys::Tuple, dk, k_top, cut::Bool) where {E}
    D = length(ks_phys)
    ms = ntuple(d -> size(A, d), Val(D))
    R = length(A) ÷ prod(ms)
    S = zeros(E, length(nmodes), R)
    FFS.Reductions._radial_accumulate!(exec, S, nmodes, val, reshape(A, ms..., R), ks_phys, dk, k_top, cut)
    return vec(sum(S; dims = 2)), R
end

function FFS.Averaging._welch_accumulate!(exec::CB.AbstractExecutionBackend, E_k::AbstractVector{T},
        nmodes::AbstractVector{T}, coeffs::GPUArraysCore.AbstractGPUArray{Complex{T}}, ks_phys::Tuple, dk::T,
        k_top::T, cut::Bool) where {T}
    s, R = _binned_realizations(exec, T, nmodes, FFS.Reductions._HalfAbs2(), coeffs, ks_phys, dk, k_top, cut)
    E_k .+= s ./ R
    return E_k
end

function FFS.Averaging._coherence_accumulate!(exec::CB.AbstractExecutionBackend, Sff::AbstractVector{T},
        Sgg::AbstractVector{T}, Sfg::AbstractVector, cf::GPUArraysCore.AbstractGPUArray{Complex{T}},
        cg::GPUArraysCore.AbstractGPUArray{Complex{T}}, ks_phys::Tuple, dk::T, k_max::T) where {T}
    nm = zeros(T, length(Sfg))
    Sff .+= first(_binned_realizations(exec, T, nm, FFS.Reductions._HalfAbs2(), cf, ks_phys, dk, k_max, true))
    Sgg .+= first(_binned_realizations(exec, T, nm, FFS.Reductions._HalfAbs2(), cg, ks_phys, dk, k_max, true))
    Sfg .+= first(_binned_realizations(exec, Complex{T}, nm, FFS.Reductions._HalfCross(), cf .* conj.(cg), ks_phys,
        dk, k_max, true))
    return Sfg
end

# The degree spectrum: one thread per degree and field sums `|C_ℓ^m|²` over its orders.
function FFS.Reductions._accumulate_degree!(::CB.AbstractExecutionBackend, E::AbstractArray{T},
        coeffs::GPUArraysCore.AbstractGPUArray, lmax::Int) where {T}
    backend = KA.get_backend(coeffs)
    Nθ, Nφ = size(coeffs, 1), size(coeffs, 2)
    B = length(coeffs) ÷ (Nθ * Nφ)
    Ed = KA.zeros(backend, T, lmax + 1, B)
    _degree_kernel!(backend)(Ed, reshape(coeffs, Nθ, Nφ, B), lmax; ndrange = (lmax + 1, B))
    E .+= reshape(Array(Ed), size(E))
    return E
end

KA.@kernel function _degree_kernel!(Ed, @Const(C), lmax::Int)
    l1, b = @index(Global, NTuple)
    l = l1 - 1
    acc = zero(eltype(Ed))
    @inbounds for m in -l:l
        rc = FFS.DirectSum.sph_mode_index(l, m)
        acc += abs2(C[rc[1], rc[2], b])
    end
    @inbounds Ed[l1, b] = acc / 2
end

end # module FlowFieldSpectraGPUArraysCoreExt
