module FlowFieldSpectraNUFSHTExt

using NUFSHT: NUFSHT
using FlowTransformBindings: FlowTransformBindings as FTB
using FlowFieldSpectra: FlowFieldSpectra as FFS
using ComputationalBackends: ComputationalBackends
using SpectralBackends: SpectralBackends as SB
using FlowGeometries: FlowGeometries

# Nonuniform spherical harmonic transform via NUFSHT. The field `(spatial…, batch…)` transforms `C`
# co-located slices per plan call (`batch_chunk`, the whole batch by default); NUFSHT's real coefficients
# (FastSphericalHarmonics `sph_mode` layout) map into FFS's `(Nθ, Nφ, batch…)`. The plan over the fixed
# nodes is built once. Points `(θ, φ)` (colatitude, longitude) come from `Grids._sph_points`, a
# structured grid expanded in its field's column-major order.
#
# Both paths are `NUFSHT.nusht_analyze!`. With the grid's weights it is the quadrature projection
# `Σⱼ wⱼ fⱼ Y_lm(xⱼ)` `calculate_spectrum` returns, against the direct sum's per-point weights
# (`DirectSum._sph_point_data`): a structured grid's latitude quadrature, a scattered grid's rescaled node
# measure. `solve = true` calls it without weights: the least-squares fit `A c ≈ f`, which recovers the
# coefficients a field was synthesized from. A masked grid's inactive cells enter as `inactive` says
# (`Grids.ZeroFill`, `Grids.Exclude`): the plan holds the nodes `Grids._transform_nodes` names, so a
# masked cell's `NaN` is never read.

"""
    NUSHTSphericalPlan{T}

Reusable NUFSHT plan: the fixed-node NUFSHT plan (point preset + NUFFT setup) over the nodes its
`inactive` holds, the real-coefficient and staged-field buffers, those nodes' indices and which of them
read the field, and their quadrature weights, or for `solve=true` the LSMR solve workspace, all built once
for a fixed point set and batch shape. Reuse across many fields via `calculate_spectrum!`.
"""
struct NUSHTSphericalPlan{T, CT, NB, P, CR, WS, QW, FW, AC, LV, KS} <: FFS.AbstractSpectralPlan
    plan::P                       # NUFSHT.NUSHTplan over the held nodes, ntrans = C
    C_real::CR                    # (Nθ, Nφ, C) real NUFSHT coeff buffer (FSH sph_mode layout), reused
    ws::WS                        # FTB.LSMRWorkspace for solve=true (built once); nothing otherwise
    qw::QW                        # (n,) the held nodes' quadrature weights; `nothing` for solve=true
    fw::FW                        # (n, C) one component of one chunk of the field at the held nodes
    act::AC                       # the held nodes' linear indices, `nothing` for every node
    live::LV                      # per held node, whether it reads the field; `nothing` where every one does
    N::Int                        # the grid's node count, the field's stride between batch slices
    lmax::Int
    Nθ::Int
    Nφ::Int
    batch::NTuple{NB, Int}
    B::Int                        # the declared total batch
    C::Int                        # transforms per execution
    solve::Bool
    maxiter::Int
    rtol::T
    ks::KS
end

# Transforms per NUFSHT execution, `<= 0` meaning the whole batch in one. The best chunk depends on the
# NUFFT engine NUFSHT resolves and on the core count: NonuniformFFTs threads its CPU spreading over point
# blocks and treats `ntrans` as a per-buffer component axis, so a small chunk bounds its
# `nthreads × ntrans` working set; FINUFFT threads over `ntrans`, where a small chunk idles cores. The
# default is the whole batch; `benchmark/` carries the sweep.
const DEFAULT_BATCH_CHUNK = 0

# Custom show: the wrapped NUFSHT plan holds a NUFFT library plan, whose default printing can segfault.
Base.show(io::IO, p::NUSHTSphericalPlan{T}) where {T} =
    print(io, "NUSHTSphericalPlan{", T, "}(lmax=", p.lmax, ", B=", p.B, p.solve ? ", solve" : "", ")")

FFS.Plans.coefficient_size(p::NUSHTSphericalPlan) = (p.Nθ, p.Nφ, p.batch...)
FFS.Plans.coefficient_type(::NUSHTSphericalPlan{T, CT}) where {T, CT} = CT
FFS.Plans.wavenumbers(p::NUSHTSphericalPlan) = p.ks
FFS.Plans.close!(p::NUSHTSphericalPlan) = (NUFSHT.close!(p.plan); nothing)

# `fw[j, b] = part(field[node j, lo+b-1])` for the `nvalid` slices of one chunk, node `j` the plan's
# `j`-th held node and `part` the field component NUFSHT's real transform reads, zero at a node that reads
# no field; the chunk's unused slices are zero, so a partial final chunk contributes nothing through them.
function _stage_component!(fw::AbstractMatrix{FT}, field, act, live, N::Int, part, lo::Int,
        nvalid::Int) where {FT}
    n = size(fw, 1)
    @inbounds for b in axes(fw, 2)
        if b > nvalid
            for j in 1:n
                fw[j, b] = zero(FT)
            end
            continue
        end
        off = (lo + b - 2) * N
        for j in 1:n
            fw[j, b] = _reads(live, j) ? FT(part(field[off + _node(act, j)])) : zero(FT)
        end
    end
    return fw
end

@inline _node(::Nothing, j::Int) = j
@inline _node(act::AbstractVector{Int}, j::Int) = @inbounds act[j]
@inline _reads(::Nothing, ::Int) = true
@inline _reads(live::AbstractVector{Bool}, j::Int) = @inbounds live[j]

function _nusht_plan(::Type{CT}, ::Type{FT}, g, ms::Tuple, batch::NTuple{NB, Int}, exec;
        tol::Real, solve::Bool, maxiter::Int, rtol::Real, nufft, batch_chunk::Int,
        sampling = nothing, weights = nothing, inactive::FFS.AbstractInactiveCells) where {CT, FT, NB}
    lmax = ms[1] - 1
    Nθ = lmax + 1
    Nφ = 2 * lmax + 1
    θ, φ, w = FFS.DirectSum._sph_point_data(g, FT; sampling,
        weights = FFS.Grids._inactive_weights(inactive, g, FT; sampling, weights))
    N = length(θ)
    act, live = FFS.Grids._transform_nodes(inactive, g, solve)
    θa = FFS.Grids._at_points(θ, act)
    B = prod(batch; init = 1)
    C = FFS._chunk(B, B, batch_chunk)
    plan = NUFSHT.make_plan(FT, θa, FFS.Grids._at_points(φ, act), lmax; execution = exec, tol = tol,
        ntrans = C, nufft = nufft)
    try
        C_real = zeros(FT, Nθ, Nφ, C)
        ws = solve ? FTB.LSMRWorkspace(plan) : nothing
        qw = solve ? nothing : FFS.Grids._at_points(collect(FT, w), act)
        fw = zeros(FT, length(θa), C)
        ks = (0:lmax, -lmax:lmax)
        return NUSHTSphericalPlan{FT, CT, NB, typeof(plan), typeof(C_real), typeof(ws), typeof(qw), typeof(fw),
                typeof(act), typeof(live), typeof(ks)}(plan, C_real, ws, qw, fw, act, live, N, lmax, Nθ, Nφ,
            batch, B, C, solve, maxiter, FT(rtol), ks)
    catch
        NUFSHT.close!(plan)
        rethrow()
    end
end

"""
    plan_spectrum(NUFSHTSpectralBackend(), execution, grid, T, ms; batch=(), tol, solve, maxiter, rtol,
                  nufft, batch_chunk=0, sampling=nothing, weights=nothing, inactive=ZeroFill())

Reusable [`FFS.AbstractSpectralPlan`](@ref) for NUFSHT, running on one thread under `SerialBackend()` and
on `Threads.nthreads()` under `ThreadedBackend()`; execute it with `calculate_spectrum!(coeffs, plan,
field)` and release it with `close!`. `nufft` selects NUFSHT's NUFFT library (a FlowTransformBindings tag;
`AutoSpectralBackend()` lets NUFSHT pick), `sampling` and `weights` set the quadrature as
`Grids._sph_point_weights` reads them, `inactive` how a masked grid's inactive cells enter, and
`batch_chunk` how many `batch` slices one NUFSHT execution carries, the whole batch by default.
"""
function FFS.plan_spectrum(::SB.AbstractNUFSHTSpectralBackend,
        exec::Union{ComputationalBackends.AbstractSerialBackend, ComputationalBackends.AbstractThreadedBackend},
        g::FlowGeometries.Grids.AbstractGrid{<:FFS.Grids.SphericalHarmonicGeometry},
        ::Type{T}, ms::Tuple; batch::Tuple = (), tol::Real = 1.0e-8, solve::Bool = false,
        maxiter::Int = 500, rtol::Real = 1.0e-6, nufft::SB.AbstractSpectralBackend = SB.AutoSpectralBackend(),
        batch_chunk::Int = DEFAULT_BATCH_CHUNK,
        sampling = nothing, weights = nothing, inactive::FFS.AbstractInactiveCells = FFS.ZeroFill()) where {T}
    FT = real(float(T))
    return _nusht_plan(FFS.sph_coeff_type(T, FT), FT, g, ms, NTuple{length(batch), Int}(batch), exec;
        tol, solve, maxiter, rtol, nufft, batch_chunk, sampling, weights, inactive)
end

"""
    calculate_spectrum!(coeffs, plan::NUSHTSphericalPlan, field) -> ks

Fill preallocated `coeffs` `(Nθ, Nφ, batch…)` with the scattered-spherical spectrum of `field`
`(N, batch…)`, reusing `plan`'s nodes / NUFSHT plan / LSMR solve workspace (no re-planning).
"""
function FFS.calculate_spectrum!(coeffs::AbstractArray{<:Number}, plan::NUSHTSphericalPlan{T}, field) where {T}
    FFS.Plans._check_coefficients(coeffs, plan)
    length(field) == plan.N * plan.B || throw(DimensionMismatch(
        "field holds $(length(field)) values; this plan transforms $(plan.B) field(s) of $(plan.N) points — " *
        "pass the matching `batch=` to plan_spectrum"))
    fill!(coeffs, zero(eltype(coeffs)))
    _nusht_pass!(coeffs, plan, field, real, one(T))
    eltype(field) <: Real || _nusht_pass!(coeffs, plan, field, imag, im)
    return plan.ks
end

# One real field component through NUFSHT, added into `coeffs` at weight `scale`, `plan.C` batch slices
# per execution. NUFSHT's transform is real and linear, so a real field runs this once with `real` (the
# identity on it) at weight 1, and a complex field runs it again with `imag` at weight `im`.
function _nusht_pass!(coeffs, plan::NUSHTSphericalPlan, field, part, scale)
    lmax = plan.lmax
    C = plan.C
    Cr = reshape(plan.C_real, plan.Nθ, plan.Nφ, C)
    Cc = reshape(coeffs, plan.Nθ, plan.Nφ, plan.B)
    for lo in 1:C:plan.B
        nvalid = min(C, plan.B - lo + 1)
        NUFSHT.nusht_analyze!(plan.C_real,
            _stage_component!(plan.fw, field, plan.act, plan.live, plan.N, part, lo, nvalid), plan.plan;
            weights = plan.qw, ws = plan.ws, maxiter = plan.maxiter, rtol = plan.rtol)
        # Remap NUFSHT's `sph_mode` layout to FFS's `sph_mode_index`, this chunk's valid slices only.
        @inbounds for b in 1:nvalid
            for l in 0:lmax
                for m in -l:l
                    Cc[FFS.sph_mode_index(l, m), lo + b - 1] +=
                        scale * Cr[NUFSHT.FastSphericalHarmonics.sph_mode(l, m), b]
                end
            end
        end
    end
    return coeffs
end

# One-shot: build a plan for this field's batch shape, then execute it once.
function FFS._calculate_spectrum_nufsht(exec::Union{ComputationalBackends.AbstractSerialBackend, ComputationalBackends.AbstractThreadedBackend},
        g::FlowGeometries.Grids.AbstractGrid{<:FFS.Grids.SphericalHarmonicGeometry}, field::AbstractArray, ms::Tuple;
        tol::Real = 1.0e-8, solve::Bool = false, maxiter::Int = 500, rtol::Real = 1.0e-6,
        nufft::SB.AbstractSpectralBackend = SB.AutoSpectralBackend(),
        batch_chunk::Int = DEFAULT_BATCH_CHUNK,
        sampling = nothing, weights = nothing, inactive::FFS.AbstractInactiveCells = FFS.ZeroFill())
    FT = real(float(eltype(field)))
    batch = FFS.Grids.field_batch_shape(g, field)
    plan = _nusht_plan(FFS.sph_coeff_type(eltype(field), FT), FT, g, ms, batch, exec; tol, solve, maxiter,
        rtol, nufft, batch_chunk, sampling, weights, inactive)
    try
        coeffs = FFS.allocate_coefficients(plan)
        return coeffs, FFS.calculate_spectrum!(coeffs, plan, field)
    finally
        FFS.close!(plan)
    end
end

# =============================================================================
# Synthesis via `nusht_type2!`, the series evaluated at the nodes, whose adjoint the quadrature analysis
# applies to the weighted field. Coefficients arrive in `sph_mode_index` layout and are remapped into
# NUFSHT's FastSphericalHarmonics `sph_mode` real buffer. NUFSHT's coefficients are real, so a complex
# coefficient array is evaluated one real component at a time, which the transform's linearity permits.
# =============================================================================

# Reusable synthesis: the fixed-node NUFSHT plan (points preset once) plus the real coefficient buffer
# its type-2 reads and the field buffer it writes.
struct NUSHTSynthesisPlan{FT, R, NB, P, CR, FB, SP} <: FFS.AbstractSynthesisPlan
    plan::P
    Cr::CR
    f::FB
    lmax::Int
    Nθ::Int
    Nφ::Int
    N::Int
    spatial::SP
    batch::NTuple{NB, Int}
    B::Int
end

# A default show of a struct holding a NUFFT library plan can segfault.
Base.show(io::IO, p::NUSHTSynthesisPlan{FT, R}) where {FT, R} =
    print(io, "NUSHTSynthesisPlan{", FT, "}(lmax=", p.lmax, ", ", R ? "real" : "complex", ")")

FFS.Plans.field_size(p::NUSHTSynthesisPlan) = (p.spatial..., p.batch...)
FFS.Plans.field_type(::NUSHTSynthesisPlan{FT, R}) where {FT, R} = R ? FT : Complex{FT}
FFS.Plans.close!(p::NUSHTSynthesisPlan) = (NUFSHT.close!(p.plan); nothing)

function FFS.Plans.plan_synthesis(::SB.AbstractNUFSHTSpectralBackend,
        exec::Union{ComputationalBackends.AbstractSerialBackend, ComputationalBackends.AbstractThreadedBackend},
        g::FlowGeometries.Grids.AbstractGrid{<:FFS.Grids.SphericalHarmonicGeometry},
        ::Type{T}, ms::NTuple{2, Int}; batch::Tuple = (), tol::Real = 1.0e-8,
        nufft::SB.AbstractSpectralBackend = SB.AutoSpectralBackend(), kwargs...) where {T}
    FT = real(float(T))
    lmax = ms[1] - 1
    Nθ = lmax + 1
    Nφ = 2 * lmax + 1
    θ, φ = FFS.Grids._sph_points(g)
    N = length(θ)
    bt = NTuple{length(batch), Int}(batch)
    B = prod(bt; init = 1)
    plan = NUFSHT.make_plan(FT, θ, φ, lmax; execution = exec, tol = tol, ntrans = B, nufft = nufft)
    sp = size(g)
    try
        return NUSHTSynthesisPlan{FT, T <: Real, length(bt), typeof(plan), Array{FT, 3}, Matrix{FT}, typeof(sp)}(
            plan, zeros(FT, Nθ, Nφ, B), zeros(FT, N, B), lmax, Nθ, Nφ, N, sp, bt, B)
    catch
        NUFSHT.close!(plan)
        rethrow()
    end
end

# NUFSHT's coefficients are real, so a complex array is evaluated one component at a time and
# recombined, which the transform's linearity permits.
function FFS.Plans.synthesize!(out::AbstractArray, plan::NUSHTSynthesisPlan{FT, R},
        coeffs::AbstractArray) where {FT, R}
    size(out) == FFS.Plans.field_size(plan) || throw(DimensionMismatch(
        "out is $(size(out)); this plan writes $(FFS.Plans.field_size(plan))"))
    size(coeffs)[1:2] == (plan.Nθ, plan.Nφ) || throw(DimensionMismatch(
        "spherical coefficients must be (Nθ, Nφ) = ($(plan.Nθ), $(plan.Nφ)) on the spectral dims; " *
        "got $(size(coeffs)[1:2])"))
    lmax = plan.lmax
    B = plan.B
    Cc = reshape(coeffs, plan.Nθ, plan.Nφ, B)
    O = reshape(out, plan.N, B)
    ET = R ? FT : Complex{FT}
    fill!(O, zero(ET))
    ncomp = R ? 1 : 2
    @inbounds for comp in 1:ncomp
        for b in 1:B, l in 0:lmax, m in -l:l
            z = Cc[FFS.sph_mode_index(l, m), b]
            plan.Cr[NUFSHT.FastSphericalHarmonics.sph_mode(l, m), b] = comp == 1 ? real(z) : imag(z)
        end
        NUFSHT.nusht_type2!(plan.f, plan.Cr, plan.plan)
        for b in 1:B, j in 1:plan.N
            O[j, b] += comp == 1 ? ET(plan.f[j, b]) : ET(im * plan.f[j, b])
        end
    end
    return out
end

# GPU NUFSHT comes from the NUFSHT × KernelAbstractions extension; this less specific method raises when
# KernelAbstractions is not loaded.
function FFS._calculate_spectrum_nufsht(::ComputationalBackends.AbstractGPUBackend,
        g::FlowGeometries.Grids.AbstractGrid{<:FFS.Grids.SphericalHarmonicGeometry},
        field::AbstractArray, ms::Tuple; kwargs...)
    throw(ArgumentError("GPU NUFSHT requires `using KernelAbstractions` (to place the transform on the execution backend)."))
end

end # module FlowFieldSpectraNUFSHTExt
