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
# colatitudes `θ_i = (i-½)π/Nθ`, Fejér's first rule, columns the longitudes `φ_j = 2π j/Nφ`
# (`Nφ = 2Nθ-1`), as `FastSphericalHarmonics.sph_points` gives them. It carries no explicit quadrature
# weights, so it is not a `Σ f·Y·w` sum and differs from the DirectSum reference on this (non-Gauss) grid;
# its guarantee is the exact round trip with `sph_evaluate`.
#
# A grid is FastTransforms' when its sampling is Fejér's first rule (`Fejer1Sampling`) at `Nθ` latitudes
# and its longitude axis a `FlowGeometries.Axes.PeriodicUniformAxis` of `Nφ` points; any other grid raises. Each
# batch slab maps onto that grid through permutations: the latitudes sorted to ascending colatitude, and
# the longitudes `λ₀ + s·φ_j` taken in the order of `φ_j`. The coefficients in `φ` rotate to the grid's
# longitudes by `mλ₀` per order. FastTransforms runs on the host, so these methods take SerialBackend,
# which runs it on one thread, and ThreadedBackend, which runs it on `Threads.nthreads()`; the device
# transform is the KernelAbstractions extension's.
# =============================================================================

FFS._available(::FFS._HostFSHT) = true

"""
    FSHTSpectralPlan{FT,CT}

Reusable FastSphericalHarmonics analysis on its own grid: the permutations from the grid's axis order to
FastTransforms', the rotation from FastTransforms' longitudes to the grid's, the grid's mask with the factor
its `inactive` puts on the coefficients (`FFS.Grids._inactive_scale`), the slab the analysis runs in place
on, the FastTransforms plans and the thread count they run at, built once for a fixed `lmax` and batch
shape. `CT` is the coefficient element type, real for a real field. Execute with
`calculate_spectrum!(coeffs, plan, field)`.
"""
struct FSHTSpectralPlan{FT, CT, NB, PR, PC, RT, MK, SL <: AbstractMatrix{FT}, C, KS} <: FFS.AbstractSpectralPlan
    rowperm::PR
    colperm::PC
    rot::RT                       # `(sin mλ₀, cos mλ₀)` for `m = 1 … lmax`
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
    rowperm, colperm, rot = _ft_perms(FT, g, Nθ, Nφ)
    bt = NTuple{length(batch), Int}(batch)
    slab = Matrix{FT}(undef, Nθ, Nφ)
    nth = FFS._backend_nthreads(exec)
    cache = _warmed_cache(FT, Nθ, Nφ, nth)
    mask = FlowGeometries.Grids.mask(g)
    scale = FFS.Grids._inactive_scale(inactive, g, FT; sampling, weights)
    lmax = Nθ - 1
    ks = (0:lmax, -lmax:lmax)
    return FSHTSpectralPlan{FT, FFS.sph_coeff_type(T, FT), length(bt), typeof(rowperm), typeof(colperm),
            typeof(rot), typeof(mask), typeof(slab), typeof(cache), typeof(ks)}(
        rowperm, colperm, rot, mask, scale, slab, cache, nth, lmax, Nθ, Nφ, bt, prod(bt; init = 1), ks)
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
    _ft_gather!(coeffs, plan.slab, plan.rot, plan.lmax, b, scale)
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

# `f(ir, ic)` over an `(Nθ, Nφ)` slab in 16 × 16 tiles. The grid's `(nlon, nlat)` order is the slab's
# transpose, so one side of each copy is strided; within a tile both sides stay in cache.
@inline function _over_tiles(f::F, Nθ::Int, Nφ::Int) where {F}
    for icb in 1:16:Nφ, irb in 1:16:Nθ, ic in icb:min(icb + 15, Nφ), ir in irb:min(irb + 15, Nθ)
        f(ir, ic)
    end
    return nothing
end

# One component of batch slice `b`, in the grid's `(nlon, nlat)` order, written into FastTransforms'
# `(θ, φ)` slab. The analysis is linear and consumes the whole grid, so an inactive cell enters as zero.
function _ft_stage!(slab::AbstractMatrix{FT}, field, mask, rowperm, colperm, b::Int, part) where {FT}
    Nφ = length(colperm)
    off = (b - 1) * Nφ * length(rowperm)
    _over_tiles(length(rowperm), Nφ) do ir, ic
        @inbounds i = colperm[ic] + (rowperm[ir] - 1) * Nφ
        @inbounds slab[ir, ic] = _active(mask, i) ? FT(part(field[off + i])) : zero(FT)
    end
    return slab
end

# The analyzed slab, coefficients in FastTransforms' longitudes `φ`, rotated to the grid's `λ = λ₀ + φ`
# and added into `coeffs` at weight `scale`: `f(λ) = g(λ − λ₀)` takes order `m`'s pair `(a_c, a_s)` of
# `cos mφ`, `sin mφ` to `(a_c cos mλ₀ − a_s sin mλ₀, a_c sin mλ₀ + a_s cos mλ₀)`. FastTransforms' layout is
# FFS's (`sph_mode_index`): an order's degrees fill one column, so the loops run order by order.
function _ft_gather!(coeffs, slab, rot, lmax::Int, b::Int, scale)
    @inbounds for l in 0:lmax
        coeffs[FFS.sph_mode_index(l, 0), b] += scale * slab[FSH.sph_mode(l, 0)]
    end
    @inbounds for m in 1:lmax
        sn, cs = rot[m]
        for l in m:lmax
            ac = slab[FSH.sph_mode(l, m)]
            an = slab[FSH.sph_mode(l, -m)]
            coeffs[FFS.sph_mode_index(l, m), b] += scale * (ac * cs - an * sn)
            coeffs[FFS.sph_mode_index(l, -m), b] += scale * (ac * sn + an * cs)
        end
    end
    return coeffs
end

# Whether `g` is FastTransforms' `(Nφ, Nθ)` grid, from its sampling's type, its longitude axis's type and
# its size.
_on_ft_grid(g, Nθ::Int, Nφ::Int) =
    FlowGeometries.Grids.sampling(g) isa FlowGeometries.SphericalSampling.AbstractFejer1Sampling &&
    FlowGeometries.Grids.coordinates(g, 1) isa FlowGeometries.Axes.PeriodicUniformAxis && size(g) == (Nφ, Nθ)

# The grid's latitude order to FastTransforms' ascending colatitudes, its longitude `λ₀ + s(i−1)Δ` at
# FastTransforms' column `j` (`φ = (j−1)Δ`), and `(sin mλ₀, cos mλ₀)` per order. The forward and the
# inverse share it.
function _ft_perms(::Type{FT}, g, Nθ::Int, Nφ::Int) where {FT}
    _on_ft_grid(g, Nθ, Nφ) || throw(ArgumentError(
        "FSHTSpectralBackend requires FastTransforms' grid: Fejér's first-rule latitudes " *
        "(`Fejer1Sampling`) at Nθ = $Nθ and a periodic longitude axis of Nφ = $Nφ points. Build it with " *
        "`structured_grid(Fejer1Sampling(), $Nθ)`, or use DirectSumSpectralBackend / NUFSHTSpectralBackend."))
    rowperm = sortperm(FT(π) / 2 .- collect(FT, FlowGeometries.Grids.coordinates(g, 2)))
    λ₀, s = FFS.Grids._longitude_axis(g, FT)
    colperm = [mod(s * (j - 1), Nφ) + 1 for j in 1:Nφ]
    rot = [sincos(m * λ₀) for m in 1:(Nθ - 1)]
    return rowperm, colperm, rot
end

# `AutoSpectralBackend` asks before selecting this backend.
function FFS._sht_applicable(g::FlowGeometries.Grids.AbstractStructuredGrid{<:FlowGeometries.Geometry.AbstractSphericalGeometry},
        ms::Tuple)
    length(ms) == 2 && ms[1] >= 1 || return false
    return _on_ft_grid(g, ms[1], 2 * ms[1] - 1)
end

# =============================================================================
# Inverse (synthesis) via `sph_evaluate!`, the exact inverse of the `sph_transform!` the forward runs.
# Coefficients are rotated back to FastTransforms' longitudes into its slab, evaluated on its own grid,
# then written back through the same permutations the forward used. A complex coefficient array is
# evaluated one real component at a time, which the transform's linearity permits.
# =============================================================================

"""
    FSHTSynthesisPlan{FT,R}

Reusable FastSphericalHarmonics synthesis on its own grid: the permutations and rotation `_ft_perms`
gives, the slab `sph_evaluate!` transforms in place, the FastTransforms plans it evaluates with and the
thread count they run at. `R` marks a real field. Execute with `synthesize!(out, plan, coeffs)`.
"""
struct FSHTSynthesisPlan{FT, R, NB, PR, PC, RT, SL <: AbstractMatrix{FT}, C} <: FFS.AbstractSynthesisPlan
    rowperm::PR
    colperm::PC
    rot::RT                       # `(sin mλ₀, cos mλ₀)` for `m = 1 … lmax`
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
    rowperm, colperm, rot = _ft_perms(FT, g, Nθ, Nφ)
    bt = NTuple{length(batch), Int}(batch)
    nth = FFS._backend_nthreads(exec)
    cache = _warmed_cache(FT, Nθ, Nφ, nth)
    slab = Matrix{FT}(undef, Nθ, Nφ)
    return FSHTSynthesisPlan{FT, T <: Real, length(bt), typeof(rowperm), typeof(colperm), typeof(rot),
            typeof(slab), typeof(cache)}(rowperm, colperm, rot, slab, cache, nth, Nθ - 1, Nθ, Nφ, bt,
        prod(bt; init = 1))
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
        part = comp == 1 ? real : imag
        fill!(slab, zero(FT))
        # The forward's rotation inverted, order by order: order `m`'s pair rotated by `−mλ₀`.
        for l in 0:lmax
            slab[FSH.sph_mode(l, 0)] = part(C[FFS.sph_mode_index(l, 0), b])
        end
        for m in 1:lmax
            sn, cs = plan.rot[m]
            for l in m:lmax
                cc = part(C[FFS.sph_mode_index(l, m), b])
                cn = part(C[FFS.sph_mode_index(l, -m), b])
                slab[FSH.sph_mode(l, m)] = cc * cs + cn * sn
                slab[FSH.sph_mode(l, -m)] = cn * cs - cc * sn
            end
        end
        _evaluate!(slab, plan.cache, plan.nthreads)
        _over_tiles(plan.Nθ, plan.Nφ) do ir, ic
            @inbounds v = slab[ir, ic]
            @inbounds O[plan.colperm[ic], plan.rowperm[ir], b] += comp == 1 ? ET(v) : ET(im * v)
        end
    end
    return out
end

end # module FlowFieldSpectraFastSphericalHarmonicsExt
