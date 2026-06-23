module FlowFieldSpectraFINUFFTExt

using FINUFFT: FINUFFT
using FlowFieldSpectra: FlowFieldSpectra as FFS, NUFFTBackend

# =============================================================================
# Reusable FINUFFT (guru) plan for scattered Cartesian grids.
#
# The nonuniform points are FIXED by the grid, so `finufft_makeplan` + `finufft_setpts!` run
# ONCE at plan construction and `finufft_exec!` runs per call with `ntrans = n_transf` — a batch
# of co-located fields/slices (components, vertical levels, time, ...) transformed together and
# the plan reused across calls (e.g. a time loop). This is the fast path for horizontal spectra
# of an `(x, y, z, t)` field on a fixed (possibly nonuniform) horizontal grid.
# =============================================================================

mutable struct NUFFTCartesianPlan{T, D, NM, PH, KS} <: FFS.AbstractSpectralPlan
    guru::Any                       # FINUFFT guru plan (C resource)
    cj::Matrix{Complex{T}}          # strengths buffer (M, n_transf)
    fk::Array{Complex{T}, NM}       # modes buffer (ms..., n_transf)
    ms::NTuple{D, Int}
    n_transf::Int
    M::Int                          # number of nonuniform points
    phase::PH                       # (ms...) translation-correction phase
    norm::T                         # 1/M
    ks_phys::KS
end

function _nufft_plan(::Type{T}, coords::Tuple, ms::NTuple{D, Int}, domain_size::NTuple{D},
        n_transf::Int, iflag::Int, eps::Real) where {T, D}
    M = length(coords[1])
    for d in 1:D
        length(coords[d]) == M || throw(DimensionMismatch("coordinate $d length mismatch"))
    end

    # Per-axis offset (min) and physical period; scale points to FINUFFT's radian convention.
    offsets = ntuple(d -> T(minimum(coords[d])), D)
    ranges = ntuple(d -> (r = T(domain_size[d]); r == 0 ? one(T) : r), D)
    scaled = ntuple(d -> T(2π) .* (T.(coords[d]) .- offsets[d]) ./ ranges[d], D)

    # FINUFFT type-1 plan; sign is -iflag to match the e^{-ik·x} analysis convention.
    guru = FINUFFT.finufft_makeplan(1, collect(ms), -iflag, n_transf, T(eps); dtype = T)
    if D == 1
        FINUFFT.finufft_setpts!(guru, scaled[1])
    elseif D == 2
        FINUFFT.finufft_setpts!(guru, scaled[1], scaled[2])
    elseif D == 3
        FINUFFT.finufft_setpts!(guru, scaled[1], scaled[2], scaled[3])
    else
        FINUFFT.finufft_destroy!(guru)
        throw(ArgumentError("FINUFFT supports up to 3 dimensions; got $D"))
    end

    # Centered integer modes (FINUFFT modeord=0) → translation-correction phase per axis.
    k_ints = ntuple(d -> collect(-(ms[d] ÷ 2):((ms[d] - 1) ÷ 2)), D)
    phase = Array{Complex{T}, D}(undef, ms...)
    @inbounds for I in CartesianIndices(ms)
        p = one(Complex{T})
        for d in 1:D
            p *= cis(-iflag * k_ints[d][I[d]] * (offsets[d] * T(2π) / ranges[d]))
        end
        phase[I] = p
    end

    ks_phys = FFS.Grids.physical_wavenumbers(ranges, ms, T)
    cj = Matrix{Complex{T}}(undef, M, n_transf)
    fk = Array{Complex{T}, D + 1}(undef, ms..., n_transf)

    plan = NUFFTCartesianPlan{T, D, D + 1, typeof(phase), typeof(ks_phys)}(
        guru, cj, fk, ms, n_transf, M, phase, one(T) / M, ks_phys,
    )
    finalizer(p -> FINUFFT.finufft_destroy!(p.guru), plan)
    return plan
end

function FFS.plan_spectrum(::NUFFTBackend, g::FFS.AbstractCartesianGrid, ::Type{T},
        ms::NTuple{D, Int}; n_transf::Int = 1, iflag::Int = 1, eps::Real = 1e-8) where {T, D}
    return _nufft_plan(T, g.coords, ms, g.domain_size, n_transf, iflag, eps)
end

function _load_strengths!(plan::NUFFTCartesianPlan{T}, fields_vecs::Tuple) where {T}
    length(fields_vecs) == plan.n_transf ||
        throw(DimensionMismatch("expected $(plan.n_transf) fields, got $(length(fields_vecs))"))
    @inbounds for u in 1:length(fields_vecs)
        length(fields_vecs[u]) == plan.M ||
            throw(DimensionMismatch("field $u length $(length(fields_vecs[u])) != npoints=$(plan.M)"))
        col = view(plan.cj, :, u)
        for j in 1:plan.M
            col[j] = fields_vecs[u][j]
        end
    end
    return plan
end

function _load_strengths!(plan::NUFFTCartesianPlan{T}, field::AbstractArray) where {T}
    length(field) == length(plan.cj) ||
        throw(DimensionMismatch("field has $(length(field)) entries, expected $(length(plan.cj))"))
    copyto!(plan.cj, field)
    return plan
end

"""
    calculate_spectrum!(coeffs, plan::NUFFTCartesianPlan, fields) -> ks_phys

Execute a prebuilt FINUFFT guru plan in place. `fields` is a tuple of `n_transf` field vectors
(each `npoints` long) or a packed `(npoints, batch...)` array; `coeffs` has shape
`(ms..., n_transf)`. The plan and point sorting are reused across calls.
"""
function FFS.calculate_spectrum!(coeffs::AbstractArray{Complex{T}}, plan::NUFFTCartesianPlan{T, D},
        fields) where {T, D}
    size(coeffs) == (plan.ms..., plan.n_transf) ||
        throw(DimensionMismatch("coeffs size $(size(coeffs)) != $((plan.ms..., plan.n_transf))"))
    _load_strengths!(plan, fields)
    FINUFFT.finufft_exec!(plan.guru, plan.cj, plan.fk)
    coeffs .= plan.fk .* reshape(plan.phase, plan.ms..., 1) .* plan.norm
    return plan.ks_phys
end

# One-shot allocating entry (called by the core (backend, grid) dispatch).
function FFS._calculate_spectrum_nufft(
    coords_vecs::Tuple,
    fields_vecs::Tuple,
    ms::Tuple;
    iflag::Int = 1,
    eps::Real = 1e-8,
    domain_size::Union{Nothing, Tuple} = nothing,
    kwargs...,
)
    D = length(ms)
    NU = length(fields_vecs)
    T = float(real(eltype(coords_vecs[1])))
    ds = domain_size === nothing ?
         ntuple(d -> (e = extrema(coords_vecs[d]); T(e[2] - e[1])), D) :
         ntuple(d -> T(domain_size[d]), D)
    plan = _nufft_plan(T, coords_vecs, NTuple{D, Int}(ms), ds, NU, iflag, eps)
    coeffs = zeros(Complex{T}, ms..., NU)
    ks = FFS.calculate_spectrum!(coeffs, plan, fields_vecs)
    return coeffs, ks
end

end # module FlowFieldSpectraFINUFFTExt
