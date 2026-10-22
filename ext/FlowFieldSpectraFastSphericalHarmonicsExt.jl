module FlowFieldSpectraFastSphericalHarmonicsExt

using FastSphericalHarmonics: FastSphericalHarmonics as FSH
using FFTW: FFTW
using FlowFieldSpectra: FlowFieldSpectra as FFS
using SpectralBackends: SpectralBackends
using FlowGeometries: FlowGeometries
using FlowTransformBindings: FlowTransformBindings as FTB

# Each transform runs through `FTB.with_fasttransforms_threads`, which sets FastTransforms' OpenMP count
# on the calling OS thread to the plan's `nthreads` and restores it after, and reads its plans from `cache`.
_transform!(slab, cache, nthreads::Int) =
    FTB.with_fasttransforms_threads(() -> FSH.sph_transform!(slab; cache = cache), nthreads)
_evaluate!(slab, cache, nthreads::Int) =
    FTB.with_fasttransforms_threads(() -> FSH.sph_evaluate!(slab; cache = cache), nthreads)

# Both transforms' plans for an `(Nθ, Nφ)` slab, built by one transform and one evaluation of a zero
# slab. FastTransforms builds them on the libfftw3 FFTW.jl loads, whose planner is process-global, so
# they are built under FFTW.jl's planner lock with the planner at `nthreads`; every later slab reuses them.
function _warmed_cache(::Type{FT}, Nθ::Int, Nφ::Int, nthreads::Int) where {FT}
    cache = FSH.SphPlanCache{FT}()
    slab = zeros(FT, Nθ, Nφ)
    FFTW.set_num_threads(nthreads) do
        FTB.with_fasttransforms_threads(nthreads) do
            FSH.sph_transform!(slab; cache = cache)
            FSH.sph_evaluate!(slab; cache = cache)
        end
    end
    return cache
end

# =============================================================================
# Structured spherical harmonic transform via FastSphericalHarmonics (FastTransforms). `sph_transform!`
# is an exact analysis on FastTransforms' own grid, addressed by array position: rows are the ascending
# colatitudes `θ_i = (i-½)π/Nθ`, columns the ascending longitudes `φ_j = 2π j/Nφ` (`Nφ = 2Nθ-1`), as
# `FastSphericalHarmonics.sph_points` gives them. It carries no explicit quadrature weights, so it is not
# a `Σ f·Y·w` sum and differs from the DirectSum reference on this (non-Gauss) grid; its guarantee is the
# exact round trip with `sph_evaluate`.
#
# Each batch slab maps onto FastTransforms' grid through the grid's own coordinates (nlat sorted to
# ascending colatitude, nlon to ascending longitude), and the nodes must be exactly that Clenshaw–Curtis
# grid; any other grid raises. FastTransforms runs on the host, so these methods take SerialBackend, which
# runs it on one thread, and ThreadedBackend, which runs it on `Threads.nthreads()`; the device transform
# is the KernelAbstractions extension's.
# =============================================================================

"""
    FSHTSpectralPlan{FT,CT}

Reusable FastSphericalHarmonics analysis on its own Clenshaw–Curtis grid: the permutations from the
grid's axis order to FastTransforms' ascending order, the grid's mask with the factor its `inactive` puts
on the coefficients (`FFS.Grids._inactive_scale`), the slab the analysis runs in place on, the
FastTransforms plans and the thread count they run at, built once for a fixed `lmax` and batch shape. `CT`
is the coefficient element type, real for a real field. Execute with `calculate_spectrum!(coeffs, plan,
field)`.
"""
struct FSHTSpectralPlan{FT, CT, NB, PR, PC, MK, SL <: AbstractMatrix{FT}, C, KS} <: FFS.AbstractSpectralPlan
    rowperm::PR
    colperm::PC
    mask::MK
    scale::FT
    slab::SL
    cache::C
    nthreads::Int
    lmax::Int
    Nθ::Int
    Nφ::Int
    batch::NTuple{NB, Int}
    B::Int
    ks::KS
end

Base.show(io::IO, p::FSHTSpectralPlan{FT, CT}) where {FT, CT} =
    print(io, "FSHTSpectralPlan{", FT, "}(lmax=", p.lmax, ", ", CT <: Real ? "real" : "complex", ", B=",
        p.B, ")")

FFS.Plans.coefficient_size(p::FSHTSpectralPlan) = (p.Nθ, p.Nφ, p.batch...)
FFS.Plans.coefficient_type(::FSHTSpectralPlan{FT, CT}) where {FT, CT} = CT
FFS.Plans.wavenumbers(p::FSHTSpectralPlan) = p.ks

# `(Nθ, Nφ)` for `ms`, raising unless `g` is the `(nlon, nlat) = (Nφ, Nθ)` grid they need.
function _sht_shape(g, ms::Tuple)
    (length(ms) == 2 && ms[1] >= 1) || throw(ArgumentError(
        "FSHTSpectralBackend takes ms = (lmax + 1, 2lmax + 1); got $ms"))
    Nθ = ms[1]
    Nφ = 2 * ms[1] - 1
    size(g) == (Nφ, Nθ) || throw(ArgumentError(
        "FSHTSpectralBackend requires a structured (nlon, nlat) = ($Nφ, $Nθ) grid matching ms; got size = $(size(g))"))
    return Nθ, Nφ
end

function FFS.plan_spectrum(::SpectralBackends.AbstractFSHTSpectralBackend, exec::FFS._HostExec,
        g::FlowGeometries.Grids.AbstractStructuredGrid{<:FlowGeometries.Geometry.AbstractSphericalGeometry},
        ::Type{T}, ms::Tuple; batch::Tuple = (), sampling = nothing, weights = nothing,
        inactive::FFS.AbstractInactiveCells = FFS.ZeroFill(), kwargs...) where {T}
    FT = real(float(T))
    Nθ, Nφ = _sht_shape(g, ms)
    rowperm, colperm = _ft_perms(FT, g, Nθ, Nφ)
    bt = NTuple{length(batch), Int}(batch)
    slab = Matrix{FT}(undef, Nθ, Nφ)
    nth = FFS._backend_nthreads(exec)
    cache = _warmed_cache(FT, Nθ, Nφ, nth)
    mask = FlowGeometries.Grids.mask(g)
    scale = FFS.Grids._inactive_scale(inactive, g, FT; sampling, weights)
    lmax = Nθ - 1
    ks = (0:lmax, -lmax:lmax)
    return FSHTSpectralPlan{FT, FFS.sph_coeff_type(T, FT), length(bt), typeof(rowperm), typeof(colperm),
            typeof(mask), typeof(slab), typeof(cache), typeof(ks)}(
        rowperm, colperm, mask, scale, slab, cache, nth, lmax, Nθ, Nφ, bt, prod(bt; init = 1), ks)
end

"""
    calculate_spectrum!(coeffs, plan::FSHTSpectralPlan, field) -> ks

Fill `coeffs` `(Nθ, Nφ, batch…)` with the FastSphericalHarmonics analysis of `field`
`(nlon, nlat, batch…)`, reusing the plan's permutations, slab and FastTransforms plans.
"""
function FFS.calculate_spectrum!(coeffs::AbstractArray{<:Number}, plan::FSHTSpectralPlan{FT, CT},
        field) where {FT, CT}
    FFS.Plans._check_coefficients(coeffs, plan)
    n = plan.Nθ * plan.Nφ
    length(field) == n * plan.B || throw(DimensionMismatch(
        "field holds $(length(field)) values; this plan transforms $(plan.B) field(s) of $n points — " *
        "pass the matching `batch=` to plan_spectrum"))
    (CT <: Real && eltype(field) <: Complex) && throw(ArgumentError(
        "this plan was built for a real field, whose coefficients are real; build one for " *
        "$(eltype(field)) to transform a complex field"))
    fill!(coeffs, zero(eltype(coeffs)))
    # `sph_transform!` is a real analysis, so a complex field goes through it one real component at a
    # time; the analysis is linear, so `C = C(Re f) + i·C(Im f)`.
    for b in 1:plan.B
        _sht_component!(coeffs, plan, field, b, real, plan.scale)
        CT <: Complex && _sht_component!(coeffs, plan, field, b, imag, im * plan.scale)
    end
    return plan.ks
end

function _sht_component!(coeffs, plan::FSHTSpectralPlan{FT}, field, b::Int, part, scale) where {FT}
    _ft_stage!(plan.slab, field, plan.mask, plan.rowperm, plan.colperm, b, part)
    _transform!(plan.slab, plan.cache, plan.nthreads)          # exact analysis, in place
    _ft_gather!(coeffs, plan.slab, plan.lmax, b, scale)
    return coeffs
end

function FFS._calculate_spectrum_sht(exec::FFS._HostExec,
        g::FlowGeometries.Grids.AbstractStructuredGrid{<:FlowGeometries.Geometry.AbstractSphericalGeometry},
        field::AbstractArray, ms::Tuple; sampling = nothing, weights = nothing,
        inactive::FFS.AbstractInactiveCells = FFS.ZeroFill(), kwargs...)
    plan = FFS.plan_spectrum(SpectralBackends.FSHTSpectralBackend(), exec, g, float(eltype(field)), ms;
        batch = FFS.Grids.field_batch_shape(g, field), sampling, weights, inactive)
    coeffs = FFS.allocate_coefficients(plan)
    FFS.calculate_spectrum!(coeffs, plan, field)
    return coeffs, plan.ks
end

# Whether cell `i` participates.
@inline _active(::FlowGeometries.Grids.AllActive, i::Int) = true
@inline _active(mask::AbstractArray{Bool}, i::Int) = @inbounds mask[i]

# One component of batch slice `b`, in the grid's `(nlon, nlat)` order, written into FastTransforms'
# `(θ, φ)` slab. The analysis is linear and consumes the whole grid, so an inactive cell enters as zero.
function _ft_stage!(slab::AbstractMatrix{FT}, field, mask, rowperm, colperm, b::Int, part) where {FT}
    Nφ = length(colperm)
    off = (b - 1) * Nφ * length(rowperm)
    @inbounds for (ic, jc) in enumerate(colperm), (ir, jr) in enumerate(rowperm)
        i = jc + (jr - 1) * Nφ
        slab[ir, ic] = _active(mask, i) ? FT(part(field[off + i])) : zero(FT)
    end
    return slab
end

# The analyzed slab, added into `coeffs` in FFS's own mode layout at weight `scale`.
function _ft_gather!(coeffs, slab, lmax::Int, b::Int, scale)
    @inbounds for l in 0:lmax, m in -l:l
        coeffs[FFS.sph_mode_index(l, m), b] += scale * slab[FSH.sph_mode(l, m)]
    end
    return coeffs
end

# The grid's own axis order → FastTransforms' ascending-(colatitude, longitude) order, and whether the
# nodes are that Clenshaw–Curtis grid. The forward, the inverse and the applicability query share it.
function _ft_sort(::Type{FT}, g, Nθ::Int) where {FT}
    Θ, Φ = FSH.sph_points(Nθ)
    θ = FT(π) / 2 .- collect(FT, FlowGeometries.Grids.coordinates(g, 2))
    λ = collect(FT, FlowGeometries.Grids.coordinates(g, 1))
    rowperm = sortperm(θ)
    colperm = sortperm(λ)
    on_grid = isapprox(θ[rowperm], Θ; atol = sqrt(eps(FT))) &&
        isapprox(λ[colperm], Φ; atol = sqrt(eps(FT)))
    return rowperm, colperm, on_grid
end

function _ft_perms(::Type{FT}, g, Nθ::Int, Nφ::Int) where {FT}
    rowperm, colperm, on_grid = _ft_sort(FT, g, Nθ)
    on_grid || throw(ArgumentError(
        "FSHTSpectralBackend requires the FastSphericalHarmonics Clenshaw–Curtis grid " *
        "(colatitude θ=(i-½)π/Nθ, longitude φ=2πj/Nφ); this grid's nodes differ. Sample on that grid " *
        "(FastSphericalHarmonics.sph_points), or use DirectSumSpectralBackend / NUFSHTSpectralBackend."))
    return rowperm, colperm
end

# `AutoSpectralBackend` asks before selecting this backend, so the same node check answers without
# raising. The size test runs first, since the node lists are read at `(Nφ, Nθ)`.
function FFS._sht_applicable(g::FlowGeometries.Grids.AbstractStructuredGrid{<:FlowGeometries.Geometry.AbstractSphericalGeometry},
        ms::Tuple)
    length(ms) == 2 && ms[1] >= 1 || return false
    lmax = ms[1] - 1
    size(g) == (2 * lmax + 1, lmax + 1) || return false
    return _ft_sort(real(float(eltype(g))), g, lmax + 1)[3]
end

# =============================================================================
# Inverse (synthesis) via `sph_evaluate!`, the exact inverse of the `sph_transform!` the forward runs.
# Coefficients arrive in `sph_mode_index` layout and are copied into FastTransforms' `sph_mode` slab,
# evaluated on its own grid, then written back through the same permutations the forward used. A complex
# coefficient array is evaluated one real component at a time, which the transform's linearity permits.
# =============================================================================

"""
    FSHTSynthesisPlan{FT,R}

Reusable FastSphericalHarmonics synthesis on its own Clenshaw–Curtis grid: the permutations `_ft_perms`
validates once, the slab `sph_evaluate!` transforms in place, the FastTransforms plans it evaluates with
and the thread count they run at. `R` marks a real field. Execute with `synthesize!(out, plan, coeffs)`.
"""
struct FSHTSynthesisPlan{FT, R, NB, PR, PC, SL <: AbstractMatrix{FT}, C} <: FFS.AbstractSynthesisPlan
    rowperm::PR
    colperm::PC
    slab::SL
    cache::C
    nthreads::Int
    lmax::Int
    Nθ::Int
    Nφ::Int
    batch::NTuple{NB, Int}
    B::Int
end

Base.show(io::IO, p::FSHTSynthesisPlan{FT, R}) where {FT, R} =
    print(io, "FSHTSynthesisPlan{", FT, "}(lmax=", p.lmax, ", ", R ? "real" : "complex", ")")

FFS.Plans.field_size(p::FSHTSynthesisPlan) = (p.Nφ, p.Nθ, p.batch...)
FFS.Plans.field_type(::FSHTSynthesisPlan{FT, R}) where {FT, R} = R ? FT : Complex{FT}

function FFS.Plans.plan_synthesis(::SpectralBackends.AbstractFSHTSpectralBackend, exec::FFS._HostExec,
        g::FlowGeometries.Grids.AbstractStructuredGrid{<:FlowGeometries.Geometry.AbstractSphericalGeometry},
        ::Type{T}, ms::NTuple{2, Int}; batch::Tuple = (), kwargs...) where {T}
    FT = real(float(T))
    Nθ, Nφ = _sht_shape(g, ms)
    rowperm, colperm = _ft_perms(FT, g, Nθ, Nφ)
    bt = NTuple{length(batch), Int}(batch)
    nth = FFS._backend_nthreads(exec)
    cache = _warmed_cache(FT, Nθ, Nφ, nth)
    slab = Matrix{FT}(undef, Nθ, Nφ)
    return FSHTSynthesisPlan{FT, T <: Real, length(bt), typeof(rowperm), typeof(colperm), typeof(slab),
            typeof(cache)}(rowperm, colperm, slab, cache, nth, Nθ - 1, Nθ, Nφ, bt, prod(bt; init = 1))
end

# `sph_evaluate!` is real→real, so a complex coefficient array is evaluated one component at a time and
# recombined — the identity the forward uses in reverse.
function FFS.Plans.synthesize!(out::AbstractArray, plan::FSHTSynthesisPlan{FT, R},
        coeffs::AbstractArray) where {FT, R}
    size(out) == FFS.Plans.field_size(plan) || throw(DimensionMismatch(
        "out is $(size(out)); this plan writes $(FFS.Plans.field_size(plan))"))
    size(coeffs)[1:2] == (plan.Nθ, plan.Nφ) || throw(DimensionMismatch(
        "spherical coefficients must be (Nθ, Nφ) = ($(plan.Nθ), $(plan.Nφ)) on the spectral dims; " *
        "got $(size(coeffs)[1:2])"))
    lmax = plan.lmax
    slab = plan.slab
    C = reshape(coeffs, plan.Nθ, plan.Nφ, plan.B)
    O = reshape(out, plan.Nφ, plan.Nθ, plan.B)
    ET = R ? FT : Complex{FT}
    fill!(O, zero(ET))
    ncomp = R ? 1 : 2
    @inbounds for b in 1:plan.B, comp in 1:ncomp
        fill!(slab, zero(FT))
        for l in 0:lmax, m in -l:l
            z = C[FFS.sph_mode_index(l, m), b]
            slab[FSH.sph_mode(l, m)] = comp == 1 ? real(z) : imag(z)
        end
        _evaluate!(slab, plan.cache, plan.nthreads)
        for (ic, jc) in enumerate(plan.colperm), (ir, jr) in enumerate(plan.rowperm)
            v = slab[ir, ic]
            O[jc, jr, b] += comp == 1 ? ET(v) : ET(im * v)
        end
    end
    return out
end

end # module FlowFieldSpectraFastSphericalHarmonicsExt
