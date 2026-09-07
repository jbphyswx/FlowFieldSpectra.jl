module FlowFieldSpectraNUFSHTExt

using NUFSHT: NUFSHT
using FlowFieldSpectra: FlowFieldSpectra as FFS
using ComputationalBackends: ComputationalBackends
using SpectralBackends: SpectralBackends as SB
using FlowGeometries: FlowGeometries

# Non-uniform Spherical Harmonic Transform via NUFSHT, tensor-native + batched. The field
# `(spatial…, batch…)` and its `ntrans = ∏batch` co-located slices transform in ONE guru-plan call;
# NUFSHT's real coefficients (`Nθ·Nφ·B`, FastSphericalHarmonics `sph_mode` layout) map into FFS's complex
# `(Nθ, Nφ, batch…)`. The plan (fixed nodes) is built once. Points `(θ, φ)` = (colatitude, longitude) come
# from the FlowGeometries adapter's convention bridge (a structured grid is expanded to per-point form in
# the same column-major order as its field).
#
# QUADRATURE. `nusht_type1!` evaluates the adjoint `Aᴴf = Σⱼ fⱼ Y_lm(xⱼ)` and takes no weights, while
# `calculate_spectrum` returns the projection `Σⱼ wⱼ fⱼ Y_lm(xⱼ)` against the grid's own weights totalling
# `4π`. The weights therefore reach it through the field, `fw = w ⊙ f` — the same identity
# `FFS.quadrature_weighted` uses on the Cartesian side. They come from `DirectSum._sph_point_data`, the
# per-point weights the direct sum itself uses, so a structured grid contributes its sampling's latitude
# quadrature, a scattered one its rescaled node measure, and a masked cell a zero. A zero-weight node
# contributes nothing and is skipped, which keeps the `NaN` masked data commonly holds out of the sum.
#
# `solve = true` consumes the RAW field: it solves the least-squares `A c ≈ f`, recovering the
# coefficients a field was synthesized from, which is a different problem from integrating against a
# quadrature.


"""
    NUSHTSphericalPlan{T}

Reusable scattered-spherical NUFSHT plan: the fixed-node NUFSHT plan (point preset + FINUFFT setup),
the reused real-coefficient buffer, and — for `solve=true` — the LSMR solve workspace, all built once for a
fixed point set and batch shape. Reuse across many fields via `calculate_spectrum!`.
"""
struct NUSHTSphericalPlan{T, CT, NB, P, CR, WS, QW, FW, KS} <: FFS.AbstractSpectralPlan
    plan::P                       # NUFSHT.NUSHTplan (fixed nodes + FINUFFT setup), ntrans = C
    C_real::CR                    # (Nθ, Nφ, C) real NUFSHT coeff buffer (FSH sph_mode layout), reused
    ws::WS                        # LSMRWorkspace for solve=true (built once); nothing otherwise
    qw::QW                        # (N,) per-node quadrature weights, Σw = 4π
    fw::FW                        # (N, C) weighted-field buffer the type-1 reads, reused
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

# Transforms per NUFSHT execution, `<= 0` meaning the whole batch in one. The optimum depends on which
# NUFFT engine NUFSHT resolves and on the core count, and the two engines scale oppositely:
# NonuniformFFTs parallelizes its CPU spreading over POINT BLOCKS across `Threads.nthreads()`
# (`blocking/cpu.jl`'s `map_blocks_to_threads!`) and treats `ntrans` as a per-buffer component axis, so a
# small chunk keeps every thread busy and bounds the `nthreads × ntrans` working set; FINUFFT treats
# `ntrans` as a parallel axis, where a small chunk leaves cores idle. So this ships as the whole batch and
# a caller tunes it against their own engine and machine; `benchmark/` carries the sweep.
const DEFAULT_BATCH_CHUNK = 0
_chunk(B::Int, bc::Int) = bc <= 0 ? max(B, 1) : clamp(bc, 1, max(B, 1))

# Custom show: the wrapped NUFSHT plan holds FINUFFT plans, whose default printing can segfault.
Base.show(io::IO, p::NUSHTSphericalPlan{T}) where {T} =
    print(io, "NUSHTSphericalPlan{", T, "}(lmax=", p.lmax, ", B=", p.B, p.solve ? ", solve" : "", ")")

FFS.Plans.coefficient_size(p::NUSHTSphericalPlan) = (p.Nθ, p.Nφ, p.batch...)
FFS.Plans.coefficient_type(::NUSHTSphericalPlan{T, CT}) where {T, CT} = CT
FFS.Plans.wavenumbers(p::NUSHTSphericalPlan) = p.ks

# `fw[j, b] = w[j] · part(field[j, lo+b-1])` for the `nvalid` slices of one chunk, the strengths whose
# adjoint is the quadrature projection; the chunk's unused slices are zero-filled, so a partial final
# chunk contributes nothing through them. A zero-weight node contributes nothing and is written as zero,
# so a masked cell's `NaN` stays out of the sum. `part` names the field component NUFSHT's real transform
# reads.
function _weighted_field!(fw::AbstractMatrix{FT}, field, w, part, lo::Int, nvalid::Int) where {FT}
    N = size(fw, 1)
    @inbounds for b in axes(fw, 2)
        if b > nvalid
            for j in 1:N
                fw[j, b] = zero(FT)
            end
            continue
        end
        off = (lo + b - 2) * N
        for j in 1:N
            wj = w[j]
            fw[j, b] = iszero(wj) ? zero(FT) : FT(wj * part(field[off + j]))
        end
    end
    return fw
end

# The real array the solve reads: one chunk's slices, with a real field's own values copied through
# unweighted and a complex field's named component taken.
function _solve_field!(fw::AbstractMatrix{FT}, field, part, lo::Int, nvalid::Int) where {FT}
    N = size(fw, 1)
    @inbounds for b in axes(fw, 2)
        if b > nvalid
            for j in 1:N
                fw[j, b] = zero(FT)
            end
            continue
        end
        off = (lo + b - 2) * N
        for j in 1:N
            fw[j, b] = FT(part(field[off + j]))
        end
    end
    return fw
end

function _nusht_plan(::Type{CT}, ::Type{FT}, g, ms::Tuple, batch::NTuple{NB, Int}, exec;
        tol::Real, solve::Bool, maxiter::Int, rtol::Real, nufft, batch_chunk::Int,
        sampling = nothing, weights = nothing) where {CT, FT, NB}
    lmax = ms[1] - 1
    Nθ = lmax + 1
    Nφ = 2 * lmax + 1
    θ, φ, w = FFS.DirectSum._sph_point_data(g, FT; sampling = sampling, weights = weights)
    N = length(θ)
    B = prod(batch; init = 1)
    C = _chunk(B, batch_chunk)
    plan = NUFSHT.make_plan(FT, θ, φ, lmax; tol = tol, ntrans = C, nthreads = FFS._backend_nthreads(exec), nufft = nufft)
    C_real = zeros(FT, Nθ, Nφ, C)
    ws = solve ? NUFSHT.LSMRWorkspace(plan) : nothing
    fw = zeros(FT, N, C)
    ks = (0:lmax, -lmax:lmax)
    return NUSHTSphericalPlan{FT, CT, NB, typeof(plan), typeof(C_real), typeof(ws), typeof(w), typeof(fw), typeof(ks)}(
        plan, C_real, ws, w, fw, lmax, Nθ, Nφ, batch, B, C, solve, maxiter, FT(rtol), ks)
end

"""
    plan_spectrum(NUFSHTSpectralBackend(), execution, grid, T, ms; batch=(), tol, solve, maxiter, rtol, nufft)

Reusable [`FFS.AbstractSpectralPlan`](@ref) for the scattered-spherical NUFSHT on a fixed point set.
Presets the points / NUFSHT plan / CG setup once; execute across many fields with
`calculate_spectrum!(coeffs, plan, field)`. `nufft` selects NUFSHT's internal NUFFT engine (a
`SpectralBackends` marker; default `AutoSpectralBackend()`, which NUFSHT resolves to FINUFFT, then
NonuniformFFTs, then its own direct summation).

`batch_chunk` is how many `batch` slices one NUFSHT execution carries, and it sizes the plan's buffers;
the default runs the whole batch in one. A smaller chunk shrinks the inner NUFFT's per-transform working
set, which pays on an engine that parallelizes over points and costs idle cores on one that parallelizes
over transforms — measure it on your own engine and core count.
"""
function FFS.plan_spectrum(::SB.AbstractNUFSHTSpectralBackend,
        exec::Union{ComputationalBackends.AbstractSerialBackend, ComputationalBackends.AbstractThreadedBackend},
        g::FlowGeometries.Grids.AbstractGrid{<:FFS.Grids.SphericalHarmonicGeometry},
        ::Type{T}, ms::Tuple; batch::Tuple = (), tol::Real = 1.0e-8, solve::Bool = false,
        maxiter::Int = 500, rtol::Real = 1.0e-6, nufft::SB.AbstractSpectralBackend = SB.AutoSpectralBackend(),
        batch_chunk::Int = DEFAULT_BATCH_CHUNK,
        sampling = nothing, weights = nothing, kwargs...) where {T}
    FT = real(float(T))
    return _nusht_plan(FFS.sph_coeff_type(T, FT), FT, g, ms, NTuple{length(batch), Int}(batch), exec;
        tol = tol, solve = solve, maxiter = maxiter, rtol = rtol, nufft = nufft,
        batch_chunk = batch_chunk, sampling = sampling, weights = weights)
end

"""
    calculate_spectrum!(coeffs, plan::NUSHTSphericalPlan, field) -> ks

Fill preallocated `coeffs` `(Nθ, Nφ, batch…)` with the scattered-spherical spectrum of `field`
`(N, batch…)`, reusing `plan`'s nodes / NUFSHT plan / LSMR solve workspace (no re-planning).
"""
function FFS.calculate_spectrum!(coeffs::AbstractArray{<:Number}, plan::NUSHTSphericalPlan{T}, field) where {T}
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
        if plan.solve
            NUFSHT.nusht_solve!(plan.C_real, _solve_field!(plan.fw, field, part, lo, nvalid), plan.plan;
                ws = plan.ws, maxiter = plan.maxiter, rtol = plan.rtol)
        else
            NUFSHT.nusht_type1!(plan.C_real,
                _weighted_field!(plan.fw, field, plan.qw, part, lo, nvalid), plan.plan)
        end
        # Remap NUFSHT's FastSphericalHarmonics `sph_mode` layout → FFS's `sph_mode_index`, per batch
        # slice. Both index functions return a `CartesianIndex{2}` into the (Nθ, Nφ) block, so index the
        # reshaped views with `[ci, b]`. Only this chunk's valid slices are written back.
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
        sampling = nothing, weights = nothing, kwargs...)
    FT = real(float(eltype(field)))
    batch = FFS.Grids.field_batch_shape(g, field)
    plan = _nusht_plan(FFS.sph_coeff_type(eltype(field), FT), FT, g, ms, batch, exec; tol = tol,
        solve = solve, maxiter = maxiter, rtol = rtol, nufft = nufft, batch_chunk = batch_chunk,
        sampling = sampling, weights = weights)
    coeffs = FFS.allocate_coefficients(plan)
    FFS.calculate_spectrum!(coeffs, plan, field)
    return coeffs, plan.ks
end

# =============================================================================
# Inverse (synthesis) via `nusht_type2!`, the adjoint pair of the `nusht_type1!` the forward runs.
# Coefficients arrive in `sph_mode_index` layout and are remapped into NUFSHT's FastSphericalHarmonics
# `sph_mode` real buffer. NUFSHT's coefficients are real, so a complex coefficient array is evaluated
# one real component at a time, which the transform's linearity permits.
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

# A default show of a struct holding FINUFFT plans can segfault.
Base.show(io::IO, p::NUSHTSynthesisPlan{FT, R}) where {FT, R} =
    print(io, "NUSHTSynthesisPlan{", FT, "}(lmax=", p.lmax, ", ", R ? "real" : "complex", ")")

FFS.Plans.field_size(p::NUSHTSynthesisPlan) = (p.spatial..., p.batch...)
FFS.Plans.field_type(::NUSHTSynthesisPlan{FT, R}) where {FT, R} = R ? FT : Complex{FT}

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
    plan = NUFSHT.make_plan(FT, θ, φ, lmax; tol = tol, ntrans = B,
        nthreads = FFS._backend_nthreads(exec), nufft = nufft)
    sp = size(g)
    return NUSHTSynthesisPlan{FT, T <: Real, length(bt), typeof(plan), Array{FT, 3}, Matrix{FT}, typeof(sp)}(
        plan, zeros(FT, Nθ, Nφ, B), zeros(FT, N, B), lmax, Nθ, Nφ, N, sp, bt, B)
end

# NUFSHT's coefficients are real, so a complex array is evaluated one component at a time and
# recombined, which the transform's linearity permits.
function FFS.Plans.synthesize!(out::AbstractArray, plan::NUSHTSynthesisPlan{FT, R},
        coeffs::AbstractArray; ks = nothing) where {FT, R}
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

function FFS._synthesize(::SB.AbstractNUFSHTSpectralBackend,
        exec::Union{ComputationalBackends.AbstractSerialBackend, ComputationalBackends.AbstractThreadedBackend},
        g::FlowGeometries.Grids.AbstractGrid{<:FFS.Grids.SphericalHarmonicGeometry},
        coeffs::AbstractArray, ms::NTuple{2, Int}; real_output::Bool = true, iflag::Int = 1,
        ks = nothing, tol::Real = 1.0e-8,
        nufft::SB.AbstractSpectralBackend = SB.AutoSpectralBackend(), kwargs...)
    FT = real(float(eltype(coeffs)))
    lmax = ms[1] - 1
    Nθ = lmax + 1
    Nφ = 2 * lmax + 1
    size(coeffs)[1:2] == (Nθ, Nφ) || throw(DimensionMismatch(
        "spherical coefficients must be (Nθ, Nφ) = ($Nθ, $Nφ) on the spectral dims; got $(size(coeffs)[1:2])."))
    θ, φ = FFS.Grids._sph_points(g)
    N = length(θ)
    batch = ntuple(i -> size(coeffs, 2 + i), ndims(coeffs) - 2)
    B = prod(batch; init = 1)
    plan = NUFSHT.make_plan(FT, θ, φ, lmax; tol = tol, ntrans = B, nthreads = FFS._backend_nthreads(exec), nufft = nufft)
    Cr = zeros(FT, Nθ, Nφ, B)
    Cc = reshape(coeffs, Nθ, Nφ, B)
    f = zeros(FT, N, B)
    ET = real_output ? FT : Complex{FT}
    # Synthesis writes the grid's own spatial shape, so a round trip compares against the forward's input
    # directly; a node cloud's `size` is `(N,)`, so the two coincide there.
    out = zeros(ET, size(g)..., batch...)
    O = reshape(out, N, B)
    ncomp = real_output ? 1 : 2
    @inbounds for comp in 1:ncomp
        for b in 1:B, l in 0:lmax, m in -l:l
            z = Cc[FFS.sph_mode_index(l, m), b]
            Cr[NUFSHT.FastSphericalHarmonics.sph_mode(l, m), b] = comp == 1 ? real(z) : imag(z)
        end
        NUFSHT.nusht_type2!(f, Cr, plan)
        for b in 1:B, j in 1:N
            O[j, b] += comp == 1 ? ET(f[j, b]) : ET(im * f[j, b])
        end
    end
    return out
end

# GPU NUFSHT (device-resident NUFSHT plan via cuFINUFFT) is provided by the NUFSHT × KernelAbstractions
# extension, which allocates the node/field arrays on the execution backend so NUFSHT.make_plan builds a
# device plan. This less-specific stub fires only when KernelAbstractions is not loaded.
function FFS._calculate_spectrum_nufsht(::ComputationalBackends.AbstractGPUBackend,
        g::FlowGeometries.Grids.AbstractGrid{<:FFS.Grids.SphericalHarmonicGeometry},
        field::AbstractArray, ms::Tuple; kwargs...)
    throw(ArgumentError("GPU NUFSHT requires `using KernelAbstractions` (to place the transform on the execution backend)."))
end

end # module FlowFieldSpectraNUFSHTExt
