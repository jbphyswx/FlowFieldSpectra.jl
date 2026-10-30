# =============================================================================
# Preprocessing held by a plan: each batch slice detrended and tapered under its own valid points.
# =============================================================================

"""
    PreprocessedPlan

A spectral plan that preprocesses every batch slice of a field before its inner plan transforms it, built by
`plan_spectrum(…; preprocess)`. A slice's invalid points (`valid` false, or a non-finite value where no `valid` is
given) weigh zero in its detrend fit, in its window's normalization and in the quadrature, which normalizes by the
measure of the slice's valid points: a slice's coefficients are those of the transform over its valid points alone.
The field itself is never written. The slices are preprocessed on the plan's host execution, one task per slice
range on `ThreadedBackend`.

Holds the inner plan, the [`Preprocess`](@ref) spec, the execution, the grid's per-point quadrature weights and their
total, the detrend basis past the constant (the centred trend coordinates for `LinearDetrend`, none for `Demean`,
`nothing` for `NoDetrend`), the window at the points (`nothing` for `NoWindow`), the staging slices, each slice's
normal-equation buffers, and the padding: the padded staging an all-uniform grid embeds its slices in, the factor
`1/pad^D` a point grid scales them by, or none.
"""
struct PreprocessedPlan{P<:Plans.AbstractSpectralPlan, S<:Preprocess, X, R<:AbstractFloat, VW<:AbstractVector{R}, B,
                        TW, M<:AbstractMatrix, AG<:AbstractArray{R,3}, MC<:AbstractMatrix, PD} <:
       Plans.AbstractSpectralPlan
    inner::P
    spec::S
    exec::X
    weights::VW
    total::R
    basis::B
    taper::TW
    staging::M
    gram::AG
    rhs::MC
    pad::PD
end

# The padding a plan's slices pass through on their way to the inner plan, each holding the array that plan reads.
struct _NoPad{A<:AbstractArray}
    input::A         # the staging as (spatial…, batch…)
end
struct _PadEmbed{A<:AbstractArray, D<:AbstractArray, S<:AbstractArray}
    input::A         # (padded spatial…, batch…), zero outside the grid's own block
    block::D         # the grid's own block of `input`
    src::S           # the staging as (spatial…, slices)
end
struct _PadScale{T<:AbstractFloat, A<:AbstractArray}
    factor::T        # 1/pad^D, the share of the padded measure the points carry
    input::A         # the staging as (points, batch…)
end

Plans.coefficient_size(pp::PreprocessedPlan) = Plans.coefficient_size(pp.inner)
Plans.coefficient_type(pp::PreprocessedPlan) = Plans.coefficient_type(pp.inner)
Plans.wavenumbers(pp::PreprocessedPlan) = Plans.wavenumbers(pp.inner)
Plans.analysis_status(pp::PreprocessedPlan) = Plans.analysis_status(pp.inner)
Plans.close!(pp::PreprocessedPlan) = Plans.close!(pp.inner)

"""
    calculate_spectrum!(coeffs, plan::PreprocessedPlan, field[, valid]) -> ks_phys

Preprocess each batch slice of `field` into the plan's staging under that slice's valid points, those `valid` (shaped
like `field`) marks or else those whose value is finite, and transform the result with the inner plan. A slice with no
valid point raises.
"""
calculate_spectrum!(coeffs::AbstractArray, pp::PreprocessedPlan, field::AbstractArray) =
    _preprocessed_spectrum!(coeffs, pp, field, Base.Fix1(_finite_at, field))
function calculate_spectrum!(coeffs::AbstractArray, pp::PreprocessedPlan, field::AbstractArray, valid::AbstractArray{Bool})
    size(valid) == size(field) || throw(DimensionMismatch("valid is $(size(valid)); field is $(size(field))"))
    return _preprocessed_spectrum!(coeffs, pp, field, Base.Fix1(getindex, valid))
end

@inline _finite_at(field, k::Int) = @inbounds isfinite(field[k])

function _preprocessed_spectrum!(coeffs, pp::PreprocessedPlan, field::AbstractArray, isvalid::V) where {V}
    S = pp.staging
    N, B = size(S)
    length(field) == N * B || throw(DimensionMismatch(
        "field holds $(length(field)) values; this plan preprocesses $B slice(s) of $N points — pass the matching " *
        "`batch=` to plan_spectrum"))
    Reductions._split_sum!(pp.exec, (), B) do _, slices
        for s in slices
            _preprocess_slice!(pp, field, isvalid, s)
        end
    end
    return calculate_spectrum!(coeffs, pp.inner, _placed!(pp.pad, S))
end

# Slice `s` into the staging: a first pass stages its valid values (its invalid points at zero) and sums its weights,
# its window's energy and the normal equations `(Σ w φφᵀ) c = Σ w φ f` of its fit over its valid points; the second
# writes `(f − φ·c)·q·τ` at its valid points, `q` the window's unit mean square under the slice's weights times
# `total/Σw`, which turns the inner plan's quadrature over every point into the quadrature over the slice's valid ones.
function _preprocess_slice!(pp::PreprocessedPlan{P, S0, X, R}, field, isvalid::V, s::Int) where {P, S0, X, R, V}
    S, w, Φ, τ = pp.staging, pp.weights, pp.basis, pp.taper
    E = eltype(S)
    N = size(S, 1)
    o = (s - 1) * N
    Ws = num = zero(R)
    sums = _moment_sums(Φ, R, E)
    @inbounds for j in 1:N
        ok = isvalid(o + j)
        f = ifelse(ok, E(field[o + j]), zero(E))
        wj = ifelse(ok, w[j], zero(R))
        S[j, s] = f
        Ws += wj
        num += wj * _taper_at(τ, j, R)^2
        sums = _moment_add(sums, Φ, j, wj, f)
    end
    Ws > 0 || throw(ArgumentError("slice $s has no valid point"))
    num > 0 || throw(ArgumentError("$(nameof(typeof(pp.spec.window))) has zero energy over slice $s's valid points"))
    c = _moment_solve!(view(pp.gram, :, :, s), view(pp.rhs, :, s), sums)
    q = pp.total / Ws / sqrt(num / Ws) * _pad_factor(pp.pad, R)
    @inbounds for j in 1:N
        S[j, s] = ifelse(isvalid(o + j), (S[j, s] - _fit_at(Φ, j, c)) * (q * _taper_at(τ, j, R)), zero(E))
    end
    return nothing
end

@inline _taper_at(::Nothing, _j, ::Type{R}) where {R} = one(R)
@inline _taper_at(τ::AbstractVector, j, _) = @inbounds τ[j]

# The normal equations' sums over a slice, the basis being the constant and the `D` coordinates of `Φ`: `G = Σ w φφᵀ`
# column-major in `K² = (D+1)²` values and `r = Σ w φ f` in `K`.
_moment_sums(::Nothing, ::Type, ::Type) = nothing
_moment_sums(::NTuple{D,AbstractVector}, ::Type{R}, ::Type{E}) where {D,R,E} =
    (ntuple(_ -> zero(R), Val((D + 1)^2)), ntuple(_ -> zero(E), Val(D + 1)))

@inline _moment_add(_sums, ::Nothing, _j, _wj, _f) = nothing
@inline function _moment_add(sums, Φ::NTuple{D,AbstractVector}, j, wj, f) where {D}
    G, r = sums
    K = D + 1
    φ = (one(wj), map(x -> @inbounds(x[j]), Φ)...)
    return (ntuple(m -> G[m] + wj * φ[(m - 1) % K + 1] * φ[(m - 1) ÷ K + 1], Val(K * K)),
            ntuple(a -> r[a] + (wj * φ[a]) * f, Val(K)))
end

# The fit's coefficients from the sums, solved in the slice's buffers.
_moment_solve!(_G, _r, ::Nothing) = nothing
function _moment_solve!(Gv::AbstractMatrix, rv::AbstractVector, sums::Tuple{NTuple{M,Any},NTuple{K,Any}}) where {M,K}
    G, r = sums
    @inbounds for b in 1:K, a in 1:K
        Gv[a, b] = G[(b - 1) * K + a]
    end
    @inbounds for a in 1:K
        rv[a] = r[a]
    end
    _psd_solve!(Gv, rv, K)
    return ntuple(a -> @inbounds(rv[a]), Val(K))
end

# The fit's value at point `j`.
@inline _fit_at(::Nothing, _j, _c) = false
@inline _fit_at(::Tuple{}, _j, c) = c[1]
@inline _fit_at(Φ::NTuple{D,AbstractVector}, j, c) where {D} =
    c[1] + sum(ntuple(d -> @inbounds(Φ[d][j]) * c[d + 1], Val(D)))

# `G c = r` for the symmetric positive semidefinite `G` (its lower triangle), into `r`: an in-place Cholesky whose pivots
# at or below `K·eps` of the largest diagonal entry are dropped, their coefficients zero.
function _psd_solve!(G::AbstractMatrix{R}, r::AbstractVector, K::Int) where {R}
    dmax = zero(R)
    @inbounds for a in 1:K
        dmax = max(dmax, G[a, a])
    end
    tol = K * eps(R) * dmax
    @inbounds for a in 1:K
        d = G[a, a]
        for c in 1:(a - 1)
            d -= G[a, c]^2
        end
        if d <= tol
            for b in a:K
                G[b, a] = zero(R)
            end
            continue
        end
        L = sqrt(d)
        G[a, a] = L
        for b in (a + 1):K
            x = G[b, a]
            for c in 1:(a - 1)
                x -= G[b, c] * G[a, c]
            end
            G[b, a] = x / L
        end
    end
    @inbounds for a in 1:K
        L = G[a, a]
        if iszero(L)
            r[a] = zero(eltype(r))
            continue
        end
        x = r[a]
        for c in 1:(a - 1)
            x -= G[a, c] * r[c]
        end
        r[a] = x / L
    end
    @inbounds for a in K:-1:1
        L = G[a, a]
        if iszero(L)
            r[a] = zero(eltype(r))
            continue
        end
        x = r[a]
        for b in (a + 1):K
            x -= G[b, a] * r[b]
        end
        r[a] = x / L
    end
    return r
end

# What the inner plan reads, and the factor the slices carry for it: the staging as it stands, or embedded in the padded
# block, or as points scaled by their share of the padded measure.
_placed!(p::_NoPad, _S) = p.input
_placed!(p::_PadEmbed, _S) = (copyto!(p.block, p.src); p.input)
_placed!(p::_PadScale, _S) = p.input
_pad_factor(::Union{_NoPad,_PadEmbed}, ::Type{R}) where {R} = one(R)
_pad_factor(p::_PadScale, ::Type{R}) where {R} = R(p.factor)

# The plan behind `plan_spectrum(…; preprocess)`: the inner plan built on the grid the padding leaves, at the mode
# counts it scales, transforming staging slices of the field's element type.
function _preprocessed_plan(grid::FlowGeometries.Grids.AbstractGrid, ::Type{T}, ms::Tuple, transform, e, batch::Tuple,
                            spec::Preprocess, kwargs) where {T}
    _host_preprocessing(e)
    R = real(float(T))
    E = float(T)
    N = prod(size(grid))
    B = prod(batch; init = 1)
    w = collect(R, _fit_weights(grid, R; _quadrature_keywords(values(kwargs))...))
    staging = zeros(E, N, B)
    gp, pad, scale = _plan_padding(grid, staging, batch, spec.pad)
    msp = scale == 1 ? ms : map(m -> max(1, round(Int, scale * m)), ms)
    t = _resolve_transform(transform, gp, msp, e)
    inner = Plans.plan_spectrum(t, e, gp, T, msp; batch = batch, _analysis_kwargs(t, kwargs)...)
    Φ = _detrend_basis(spec.detrend, grid, R, w)
    K = Φ === nothing ? 0 : length(Φ) + 1
    τ = spec.window isa NoWindow ? nothing : _window(grid, R, spec.window, w)
    total = zero(R)
    for j in 1:N
        total += w[j]
    end
    return PreprocessedPlan(inner, spec, e, w, total, Φ, τ, staging, zeros(R, K, K, B), zeros(E, K, B), pad)
end

# The fit's basis per point past the constant: none for a mean, and for a linear trend the trend coordinates centred on
# their weighted means, which keeps the normal equations well conditioned.
_detrend_basis(::NoDetrend, _grid, ::Type, _w) = nothing
_detrend_basis(::Demean, _grid, ::Type, _w) = ()
function _detrend_basis(::LinearDetrend, grid, ::Type{R}, w::AbstractVector{R}) where {R}
    xs = _trend_coordinates(grid, R)
    W = sum(w)
    return map(x -> (xr = collect(R, x); xr .- sum(w .* xr) / W), xs)
end

# The grid the inner plan transforms, the padding the slices take to it, and the factor scaling the mode counts.
function _plan_padding(grid, staging::AbstractMatrix{E}, batch::Tuple, pad::Float64) where {E}
    pad == 1 && return (grid, _NoPad(reshape(staging, size(grid)..., batch...)), 1)
    gp, padded, scale = _padded(grid, zeros(E, size(grid)..., batch...), pad)
    return (gp, _pad_kind(grid, gp, padded, staging, batch, pad), scale)
end

_pad_kind(grid::FlowGeometries.Grids.AbstractStructuredGrid, ::FlowGeometries.Grids.AbstractStructuredGrid, padded,
          staging::AbstractMatrix, ::Tuple, ::Float64) =
    _PadEmbed(padded, view(padded, map(n -> 1:n, size(grid))..., :), reshape(staging, size(grid)..., size(staging, 2)))
_pad_kind(grid, _gp, _padded, staging::AbstractMatrix, batch::Tuple, pad::Float64) =
    _PadScale(inv(eltype(grid)(pad))^length(FlowGeometries.Grids.materialize(grid)), reshape(staging, size(staging, 1), batch...))

# A one-shot `calculate_spectrum` with a non-identity spec on the host runs through the plan, so a field's invalid
# points weigh as they do there; another backend preprocesses the field on its own memory (`preprocess_field`).
_one_shot_plan(::_HostExec, spec::Preprocess) = !Preprocessing.is_identity(spec)
_one_shot_plan(_, _) = false

function _preprocessed_once(grid, field::AbstractArray, ms::Tuple, transform, e, spec::Preprocess, kwargs)
    pp = _preprocessed_plan(grid, eltype(field), ms, transform, e, Grids.field_batch_shape(grid, field), spec, kwargs)
    try
        coeffs = Plans.allocate_coefficients(pp)
        return (coeffs, calculate_spectrum!(coeffs, pp, field))
    finally
        Plans.close!(pp)
    end
end

_host_preprocessing(::_HostExec) = nothing
_host_preprocessing(e::ComputationalBackends.AbstractExecutionBackend) = throw(ArgumentError(
    "a plan's preprocessing runs on the host, and this plan executes on $(nameof(typeof(e))): pass `preprocess` to " *
    "`calculate_spectrum`, or preprocess the field before `calculate_spectrum!`"))
