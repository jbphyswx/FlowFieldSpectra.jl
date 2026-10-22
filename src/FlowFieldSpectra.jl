module FlowFieldSpectra

# First, so its `__init__` selects the OpenMP runtime's thread-local mode before FastTransforms loads
# that runtime; see `FlowTransformBindings.with_fasttransforms_threads`.
using FlowTransformBindings: FlowTransformBindings as FTB
using PrecompileTools: PrecompileTools
using LinearAlgebra: LinearAlgebra
using ComputationalBackends: ComputationalBackends
using SpectralBackends: SpectralBackends
using FlowGeometries: FlowGeometries

include("Grids.jl")
include("Preprocessing.jl")
include("Normalization.jl")
include("Problem.jl")
include("Plans.jl")
include("SphericalKernels.jl")
include("DirectSum.jl")
include("Reductions.jl")
include("Operators.jl")
include("Averaging.jl")
include("LombScargle.jl")

using .Grids: Grids, physical_wavenumbers, AbstractInactiveCells, ZeroFill, Exclude
using .Preprocessing: AbstractWindow, NoWindow, Hann, Hamming, Blackman, Tukey, AbstractDetrend, NoDetrend, Demean, LinearDetrend, Preprocess, dpss
using .Normalization: AbstractSidedness, OneSided, TwoSided, AbstractScaling, DensityScaling, PowerScaling,
    AbstractShellEstimator, ShellSum, ModeAverage, SpectralConvention
using .Problem: Problem, TransformProblem, batch_shape, stack_fields
using .Plans: Plans, AbstractSpectralPlan, plan_spectrum,
    coefficient_size, coefficient_type, wavenumbers, allocate_coefficients,
    AbstractSynthesisPlan, plan_synthesis, synthesize!, field_size, field_type, allocate_field, close!
using .DirectSum: DirectSum, sph_mode_index
using .Reductions: isotropic_spectrum, isotropic_spectrum!, transect_spectrum, transect_spectrum!, spherical_energy_spectrum, spherical_energy_spectrum!, cross_spectrum, cospectrum, quadspectrum, anisotropic_spectrum
using .Operators: spectral_divergence, spectral_divergence!, spectral_vorticity, spectral_vorticity!,
    compensate, band_energy
using .Averaging: welch_power_spectrum, welch_power_spectrum!, coherence_spectrum, coherence_spectrum!
using .LombScargle: lomb_scargle, lomb_scargle!


# Preprocessing & Normalization (typed configuration)
export AbstractWindow, NoWindow, Hann, Hamming, Blackman, Tukey
export AbstractDetrend, NoDetrend, Demean, LinearDetrend, Preprocess, dpss
export AbstractSidedness, OneSided, TwoSided, AbstractScaling, DensityScaling, PowerScaling,
    AbstractShellEstimator, ShellSum, ModeAverage, SpectralConvention
export TransformProblem
export AbstractInactiveCells, ZeroFill, Exclude
export AbstractSpectralPlan, plan_spectrum, preprocess_field
export coefficient_size, coefficient_type, wavenumbers, allocate_coefficients
export AbstractSynthesisPlan, plan_synthesis, synthesize!, field_size, field_type, allocate_field, close!
# APIs
export calculate_spectrum, calculate_spectrum!, synthesize, isotropic_spectrum, isotropic_spectrum!, transect_spectrum, transect_spectrum!, spherical_energy_spectrum, spherical_energy_spectrum!, sph_mode_index, sph_coeff_type
export spectral_divergence, spectral_divergence!, spectral_vorticity, spectral_vorticity!,
    compensate, band_energy
export cross_spectrum, cospectrum, quadspectrum, anisotropic_spectrum
export welch_power_spectrum, welch_power_spectrum!, coherence_spectrum, coherence_spectrum!,
    lomb_scargle, lomb_scargle!
export plot_spectrum, compare_spectra, compare_spectral_analysis

"""
    sph_coeff_type(::Type{T}, ::Type{FT}) -> Type

Element type of a spherical coefficient array for a field of element type `T` on a grid of element type
`FT`: `FT` for a real field and `Complex{FT}` for a complex one. The basis is the real spherical
harmonics, so a real field's coefficients are real.
"""
sph_coeff_type(::Type{T}, ::Type{FT}) where {T <: Real, FT} = FT
sph_coeff_type(::Type{T}, ::Type{FT}) where {T, FT} = Complex{FT}

"""
    preprocess_field(grid, field, spec::Preprocess; sampling=nothing, weights=nothing)
        -> (grid′, field′, ms_scale)

`field` detrended, tapered and zero-padded per `spec`, with the grid and mode-count scale the padding
implies; `(grid, field, 1)` for an identity spec. The caller's array is never mutated.

Both detrends are least-squares fits under the transform's weights: the cell measure on a Cartesian
grid, the spherical quadrature `sampling` / `weights` on a sphere. `Demean` zeroes the `k = 0`
coefficient; `LinearDetrend` fits `a + b·x` in the point coordinates, which on a sphere are the unit
vector's components, so it zeroes every degree `l ≤ 1`. `spec.window` multiplies each point by the window
at its fractional position across each Cartesian direction ([`Preprocessing.point_window`](@ref)).
`spec.pad > 1` lengthens a Cartesian domain `pad` times with zeros: as samples on an all-uniform grid, and
otherwise as a node grid over the same points with the field scaled by `1/pad^D`. A window or padding on
a sphere raises.
"""
preprocess_field(g::FlowGeometries.Grids.AbstractGrid, field::AbstractArray, ::Nothing;
    sampling = nothing, weights = nothing) = (g, field, 1)

function preprocess_field(g::FlowGeometries.Grids.AbstractGrid, field::AbstractArray,
        spec::Preprocess; sampling = nothing, weights = nothing)
    Preprocessing.is_identity(spec) && return (g, field, 1)
    nsp = ndims(g)
    FT = real(float(eltype(field)))
    out = similar(field, float(eltype(field)))
    copyto!(out, field)
    if !(spec.detrend isa NoDetrend && spec.window isa NoWindow)
        w = _fit_weights(g, FT; sampling = sampling, weights = weights)
        _detrend!(out, g, spec.detrend, nsp, w, FT)
        spec.window isa NoWindow ||
            Preprocessing.apply_window!(out, _window(g, FT, spec.window, w), nsp)
    end
    spec.pad == 1 && return (g, out, 1)
    return _padded(g, out, spec.pad)
end

_detrend!(out, g, ::NoDetrend, nsp::Int, w, ::Type{FT}) where {FT} = out
_detrend!(out, g, d::Demean, nsp::Int, w, ::Type{FT}) where {FT} =
    Preprocessing.detrend_spatial!(out, d, nsp, w)
_detrend!(out, g, d::LinearDetrend, nsp::Int, w, ::Type{FT}) where {FT} =
    Preprocessing.detrend_spatial!(out, d, nsp, w, _trend_coordinates(g, FT))

# The weights a transform integrates a field on `g` with, up to a constant, one per point in column-major
# order: the cell measure on a Cartesian grid, and on a spherical one the per-point quadrature every
# spherical transform reads, an inactive point at zero.
function _fit_weights(g::FlowGeometries.Grids.AbstractGrid{<:FlowGeometries.Geometry.AbstractCartesianGeometry},
        ::Type{FT}; sampling = nothing, weights = nothing) where {FT}
    (sampling === nothing && weights === nothing) || throw(ArgumentError(
        "`sampling` and `weights` state a spherical quadrature; a Cartesian grid integrates under its " *
        "own cell measure, which `FlowGeometries.Grids.measure` reads."))
    return vec(FlowGeometries.Grids.measure(g))
end
_fit_weights(g::FlowGeometries.Grids.AbstractGrid{<:Grids.SphericalHarmonicGeometry}, ::Type{FT};
    sampling = nothing, weights = nothing) where {FT} =
    Grids._sph_point_weights(g, FT; sampling = sampling, weights = weights)

# The per-point coordinates a linear trend is fitted in: the position on a Cartesian grid, and on a
# sphere the unit vector `(sinθ cosλ, sinθ sinλ, cosθ)`, whose components span the degree-1 harmonics.
_trend_coordinates(g::FlowGeometries.Grids.AbstractGrid{<:FlowGeometries.Geometry.AbstractCartesianGeometry},
    ::Type{FT}) where {FT} = map(p -> Grids._as_points(FT, p), FlowGeometries.Grids.materialize(g))
function _trend_coordinates(g::FlowGeometries.Grids.AbstractGrid{<:Grids.SphericalHarmonicGeometry},
        ::Type{FT}) where {FT}
    θ, λ = Grids._sph_points(g)
    return (FT.(sin.(θ) .* cos.(λ)), FT.(sin.(θ) .* sin.(λ)), FT.(cos.(θ)))
end

_window(g::FlowGeometries.Grids.AbstractGrid{<:FlowGeometries.Geometry.AbstractCartesianGeometry},
    ::Type{FT}, win, w) where {FT} = Preprocessing.point_window(FT, win, _trend_coordinates(g, FT), w)
_window(g::FlowGeometries.Grids.AbstractGrid, ::Type{FT}, win, w) where {FT} = throw(ArgumentError(
    "a window tapers a field toward the edges of its domain, and a grid over a " *
    "$(nameof(typeof(FlowGeometries.Grids.grid_geometry(g)))) covers a closed surface. Apodize a mask " *
    "boundary on the field before passing it."))

# An all-uniform grid gets the zeros as samples at its own spacing. The returned `pad` scales the caller's
# `ms`.
function _padded(g::FlowGeometries.Grids.AbstractStructuredGrid{<:FlowGeometries.Geometry.AbstractCartesianGeometry},
        field::AbstractArray, pad::Float64)
    D = ndims(g)
    all(d -> FlowGeometries.Grids.isuniform(g, d) && size(g, d) >= 2, 1:D) || return _padded_points(g, field, pad)
    FT = eltype(g)
    ns = ntuple(d -> size(g, d), D)
    np = ntuple(d -> max(ns[d], round(Int, pad * ns[d])), D)
    axs = ntuple(D) do d
        ax = FlowGeometries.Grids.coordinates(g, d)
        δ = FT(ax[2]) - FT(ax[1])
        range(FT(ax[1]); step = δ, length = np[d])
    end
    per = ntuple(d -> FT(FlowGeometries.Grids.period(g, d)) * FT(np[d]) / FT(ns[d]), D)
    gp = FlowGeometries.Grids.StructuredGrid(FlowGeometries.Grids.grid_geometry(g), axs...;
        periodic = FlowGeometries.Grids.periodic_flags(g), period = per)
    batch = ntuple(i -> size(field, D + i), ndims(field) - D)
    out = zeros(eltype(field), np..., batch...)
    copyto!(view(out, ntuple(d -> 1:ns[d], D)..., ntuple(_ -> Colon(), length(batch))...),
        reshape(field, ns..., batch...))
    return (gp, out, pad)
end

# Every other Cartesian grid is transformed as a sum over its points, to which a zero adds nothing: its
# points become a node grid wrapping at `pad` times each domain length, and the field is scaled by
# `1/pad^D`, the share of the padded measure the points carry. On a uniform grid this equals appending
# zero samples.
_padded(g::FlowGeometries.Grids.AbstractGrid{<:FlowGeometries.Geometry.AbstractCartesianGeometry},
    field::AbstractArray, pad::Float64) = _padded_points(g, field, pad)

function _padded_points(g, field::AbstractArray, pad::Float64)
    FT = eltype(g)
    D = FlowGeometries.Grids.ncoordinates(g)
    pts = FlowGeometries.Grids.materialize(g)
    N = length(first(pts))
    per = ntuple(d -> FT(pad) * FT(FlowGeometries.Grids.domain_length(g, d)), D)
    gp = FlowGeometries.Grids.UnstructuredGrid(FlowGeometries.Grids.grid_geometry(g), pts,
        vec(FlowGeometries.Grids.measure_array(g)), _submask(FlowGeometries.Grids.mask(g), 1:N);
        periodic = ntuple(_ -> true, D), period = per)
    out = reshape(field, N, Grids.field_batch_shape(g, field)...)
    out .*= inv(FT(pad))^D
    return (gp, out, pad)
end

# A sphere's harmonic degrees are integers fixed by its closed surface, so it has no domain to lengthen.
_padded(g::FlowGeometries.Grids.AbstractGrid{<:Grids.SphericalHarmonicGeometry}, field::AbstractArray,
    pad::Float64) = throw(ArgumentError(
    "zero padding lengthens a Cartesian domain to refine its wavenumber spacing; the degrees of a " *
    "spherical harmonic expansion are integers, so a spherical grid has no spacing to refine."))

"""
    quadrature_weighted(grid, field) -> field

`field` scaled by the grid's per-node quadrature factor (see [`Grids.quadrature_scale`](@ref)), so a
backend's node-count normalization becomes the measure-weighted quadrature over `grid`. Returns `field`
itself where the measure is constant: every uniform grid, and every scattered grid whose nodes carry
equal weight.
"""
function quadrature_weighted(g::FlowGeometries.Grids.AbstractGrid{<:FlowGeometries.Geometry.AbstractCartesianGeometry},
        field::AbstractArray)
    D = ndims(g)
    N = prod(ntuple(d -> size(field, d), D))
    α = Grids.quadrature_scale(g, real(float(eltype(field))), N)
    α === nothing && return field
    out = similar(field, float(eltype(field)))
    B = length(field) ÷ N
    @inbounds for b in 1:B, j in 1:N
        out[j + (b - 1) * N] = field[j + (b - 1) * N] * α[j]
    end
    return out
end

# A spherical grid carries its quadrature in the transform's own weights (`Grids._sht_weights` /
# `Grids._sph_node_weights`), so its field passes through unscaled.
quadrature_weighted(::FlowGeometries.Grids.AbstractGrid{<:Grids.SphericalHarmonicGeometry},
    field::AbstractArray) = field

# =============================================================================
# Execution-backend resolution. `AutoBackend` resolves to `ThreadedBackend` when the OhMyThreads
# extension is loaded and `Threads.nthreads() > 1`, else to `SerialBackend`. GPU, Distributed and MPI
# backends need an explicit device or process context and are never selected automatically.
# =============================================================================
"""
    _backend_nthreads(exec) -> Int

Thread count for a library plan built under `exec`: `Threads.nthreads()` on a threaded backend, `1`
otherwise. Every library plan FFS builds itself receives this explicit count, since FINUFFT reads `0` as
every core on the machine. A host NUFSHT plan takes `execution = exec` and sets the same count from it;
a device one, built over device nodes, takes this count.
"""
_backend_nthreads(::ComputationalBackends.AbstractThreadedBackend) = Threads.nthreads()
_backend_nthreads(::ComputationalBackends.AbstractExecutionBackend) = 1

# The execution backends a host library plan serves.
const _HostExec = Union{ComputationalBackends.AbstractSerialBackend, ComputationalBackends.AbstractThreadedBackend}

"""
    _to_host(a) -> Array

`a` as a host `Array`, returning it unchanged when it already is one. `copyto!` covers every KA /
GPUArrays device array, so this is device-generic.
"""
_to_host(a::Array) = a
function _to_host(a::AbstractArray{T}) where {T}
    host = Array{T}(undef, size(a)...)
    copyto!(host, a)
    return host
end

_resolve_execution(backend::ComputationalBackends.AbstractExecutionBackend) = ComputationalBackends.resolve_backend(backend)
function _resolve_execution(::ComputationalBackends.AbstractAutoBackend)
    threaded = Base.get_extension(@__MODULE__, :FlowFieldSpectraOhMyThreadsExt) !== nothing
    return (threaded && Threads.nthreads() > 1) ? ComputationalBackends.ThreadedBackend() : ComputationalBackends.SerialBackend()
end

# =============================================================================
# Transform resolution. `AutoSpectralBackend` resolves to the fastest loaded transform whose
# preconditions the grid meets; a concrete tag passes through. Every candidate computes the direct sum's
# coefficients, the FFT exactly and a NUFFT / NUFSHT to its requested tolerance. The direct sum, `O(∏ms · N)`
# on a Cartesian grid and `O(L³)` on a sphere, is chosen only where no library applies.
#
# An FFT transforms a uniform axis in full, so it applies only where `ms` equals the grid's length on that
# axis; the hybrid also needs axis 1 uniform. Both are checked here.
# =============================================================================

_ext_loaded(name::Symbol) = Base.get_extension(@__MODULE__, name) !== nothing

# A distributed run's transform is chosen for the backend each worker runs locally.
_auto_exec(e::ComputationalBackends.AbstractExecutionBackend) = e
_auto_exec(e::ComputationalBackends.AbstractDistributedBackend) = ComputationalBackends.local_backend(e)
_auto_exec(e::ComputationalBackends.AbstractMPIBackend) = ComputationalBackends.local_backend(e)

# The NUFFT library available for this execution backend, or `nothing`: NonuniformFFTs, then FINUFFT. A
# device plan's arrays come from the KernelAbstractions extension.
function _auto_nufft(::ComputationalBackends.AbstractExecutionBackend)
    FTB.is_available(FTB.NonuniformFFTsBackend()) && return FTB.NonuniformFFTsBackend()
    FTB.is_available(FTB.FINUFFTBackend()) && return FTB.FINUFFTBackend()
    return nothing
end
_auto_nufft(::ComputationalBackends.AbstractGPUBackend) =
    _ext_loaded(:FlowFieldSpectraKernelAbstractionsExt) ? _auto_nufft(ComputationalBackends.SerialBackend()) :
    nothing

# The FFT provider for this execution backend; the hybrid composite's uniform-axis pass takes the same one.
_auto_fft(::ComputationalBackends.AbstractExecutionBackend) = _ext_loaded(:FlowFieldSpectraFFTWExt)
_auto_fft(::ComputationalBackends.AbstractGPUBackend) = _ext_loaded(:FlowFieldSpectraKernelAbstractionsAbstractFFTsExt)

# FastSphericalHarmonics is a host library. The device-generic SHT carries its own grid requirement and is
# selected explicitly.
_auto_fsht(::ComputationalBackends.AbstractExecutionBackend) =
    _ext_loaded(:FlowFieldSpectraFastSphericalHarmonicsExt)
_auto_fsht(::ComputationalBackends.AbstractGPUBackend) = false

_auto_nufsht(::ComputationalBackends.AbstractExecutionBackend) = _ext_loaded(:FlowFieldSpectraNUFSHTExt)
_auto_nufsht(::ComputationalBackends.AbstractGPUBackend) =
    _ext_loaded(:FlowFieldSpectraNUFSHTKernelAbstractionsExt)

"""
    _sht_applicable(grid, ms) -> Bool

Whether the FastSphericalHarmonics analysis applies to `grid` at `ms`. Its transform is addressed by
array position on FastTransforms' own Clenshaw–Curtis nodes, so only that grid at the matching size
qualifies; the extension owns the check and answers `false` until it loads.
"""
_sht_applicable(grid, ms) = false

_resolve_transform(t::SpectralBackends.AbstractSpectralBackend, grid, ms, exec) = t
_resolve_transform(::SpectralBackends.AbstractAutoSpectralBackend, grid, ms::Tuple, exec) =
    _auto_transform(grid, ms, _auto_exec(exec))

function _auto_transform(g::FlowGeometries.Grids.AbstractStructuredGrid{<:FlowGeometries.Geometry.AbstractCartesianGeometry},
        ms::Tuple, exec)
    D = ndims(g)
    length(ms) == D || return SpectralBackends.DirectSumSpectralBackend()
    umask = _uniform_mask(g, Val(D))
    full = all(ntuple(d -> !umask[d] || ms[d] == size(g, d), D))
    all(umask) && full && _auto_fft(exec) && return SpectralBackends.FFTSpectralBackend()
    nu = _auto_nufft(exec)
    # Mixed uniform/stretched: the hybrid runs one FFT over the uniform axes and a 1-D NUFFT per
    # stretched axis, so it needs both providers and axis 1 uniform.
    umask[1] && !all(umask) && full && _auto_fft(exec) && nu !== nothing &&
        return SpectralBackends.FFTSpectralBackend()
    nu === nothing || return nu
    return _auto_fallback(g, "using FFTW, NonuniformFFTs or FINUFFT")
end

_auto_transform(g::Grids.PointwiseCartesian, ms::Tuple, exec) =
    (nu = _auto_nufft(exec); nu === nothing ? _auto_fallback(g, "using NonuniformFFTs or FINUFFT") : nu)

# The direct sum's factorized spherical paths cost `O(L³)` (longitude DFT `(lmax+1)·nlon·nlat`, θ-Legendre
# contraction `nlat·L²`) against `O(L² log L)` for a library transform, so a library is taken on every
# spherical layout: FSHT on the one grid it validates, NUFSHT on any spherical point set.
function _auto_transform(g::FlowGeometries.Grids.AbstractGrid{<:Grids.SphericalHarmonicGeometry},
        ms::Tuple, exec)
    _auto_fsht(exec) && _sht_applicable(g, ms) && return SpectralBackends.FSHTSpectralBackend()
    _auto_nufsht(exec) && return SpectralBackends.NUFSHTSpectralBackend()
    return _auto_fallback(g, "using NUFSHT (any spherical grid) or FastSphericalHarmonics")
end

# Any other grid/geometry: the direct sum serves every one of them, and the backend raises where it cannot.
_auto_transform(g::FlowGeometries.Grids.AbstractGrid, ms::Tuple, exec) =
    SpectralBackends.DirectSumSpectralBackend()

# Warned once per logger: the direct sum is a correctness reference whose cost is `O(∏ms · N)`, so a
# caller landing here by default is told which package unlocks the fast path.
function _auto_fallback(g, load::String)
    @warn("no fast transform is loaded for this grid, so the O(∏ms · N) direct-sum reference will run",
        grid = nameof(typeof(g)), load, maxlog = 1)
    return SpectralBackends.DirectSumSpectralBackend()
end

"""
    calculate_spectrum(grid, field, ms::Tuple;
                       transform=AutoSpectralBackend(), execution=AutoBackend(), kwargs...)

Spectral coefficients and physical wavenumbers of `field` sampled on `grid` (a FlowGeometries grid),
whose architecture and geometry define the coordinate system. The two backend axes compose freely:

- `transform::SpectralBackends.AbstractSpectralBackend` — the spectral math
  (`AutoSpectralBackend` (default), `FFTSpectralBackend`, `NUFFTSpectralBackend`,
  `FSHTSpectralBackend`, `NUFSHTSpectralBackend`, `DirectSumSpectralBackend`).

  `AutoSpectralBackend` picks the fastest transform this grid admits among the extensions loaded, and
  only ever substitutes one computing the same coefficients: an FFT on a uniform Cartesian grid (asked
  for its own length per axis), the hybrid FFT/NUFFT composite where a grid is uniform in some directions
  and stretched in others, a NUFFT on a nonuniform or scattered Cartesian grid, the FastSphericalHarmonics
  analysis on its own Clenshaw–Curtis grid, and NUFSHT on any other spherical grid. With no fast
  transform loaded it warns once and runs the direct sum, an `O(∏ms · N)` / `O(L³)` correctness
  reference; name `DirectSumSpectralBackend()` explicitly to ask for it.
- `execution::ComputationalBackends.AbstractExecutionBackend` — where/how it runs (`SerialBackend`,
  `ThreadedBackend`, `GPUBackend`, `DistributedBackend`, `MPIBackend`, `AutoBackend` (default)).

# Data model
`field` is an `AbstractArray` shaped `(spatial…, batch…)`: the first `ndims(grid)` dims are
spatial (and must equal `size(grid)`); every trailing dim is a **batch** dim (components,
levels, time, ensemble — any number), carried through and preserved. Structured grids take an
`(N_1,…,N_D, batch…)` tensor (spherical: `(nlon, nlat, batch…)`); unstructured grids take
`(N, batch…)`. A `Tuple` of equal-shaped arrays stacks them along a new trailing batch axis.

`ms` is the spectral resolution: Cartesian `(m_1,…,m_D)`; spherical `(Nθ, Nφ)` with `lmax = Nθ-1`.
On a masked spherical grid `inactive` says how the inactive cells enter: [`ZeroFill`](@ref) (the
default) or [`Exclude`](@ref).

# Preprocessing
`preprocess::Preprocess` detrends, tapers and zero-pads the field first (see
[`preprocess_field`](@ref)); omitting it transforms the field as given, copying nothing. A taper is
scaled to unit mean square, so Parseval holds on the returned coefficients with no correction applied
afterwards. `pad > 1` lengthens the grid's axes and scales `ms` by the same factor, so the wavenumber
spacing narrows.

# Returns
`(coeffs, ks_phys)` — the coefficients and the physical wavenumber coordinates per spectral axis.

The field's element type picks the coefficient layout on both geometries. **Cartesian**: a complex field
gives the full native cube `(ms…, batch…)`. A real field gives the rfft-packed half `(b_1÷2+1, b_2…,
batch…)` of its coefficient box `b`, which is `ms` with every even `m_d` whose axis tells `+m_d/2` from
`-m_d/2` apart (a stretched axis, a point set, an axis transformed below its length) widened to `m_d + 1`
modes, `-m_d/2 … m_d/2` in FFT order (`Grids.mode_box`). The box holds the transform's modes and their
conjugate mirror, and zero at a mode in neither (`+m_d/2` with `-m_e/2` on two widened axes `d, e ≥ 2`).
Both are complex-valued. **Spherical**: `(Nθ, Nφ, batch…)`, real for a real field (the basis is the real
spherical harmonics) and complex for a complex one; see [`sph_coeff_type`](@ref).
"""
function calculate_spectrum(grid::FlowGeometries.Grids.AbstractGrid, field::AbstractArray, ms::Tuple;
        transform::SpectralBackends.AbstractSpectralBackend = SpectralBackends.AutoSpectralBackend(),
        execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
        preprocess::Union{Nothing, Preprocess} = nothing, kwargs...)
    e = _resolve_execution(execution)
    # Zero padding returns a lengthened grid, and the mode counts scale with it.
    gp, fp, scale = preprocess_field(grid, field, preprocess; _quadrature_keywords(values(kwargs))...)
    msp = scale == 1 ? ms : map(m -> max(1, round(Int, scale * m)), ms)
    return calculate_spectrum(_resolve_transform(transform, gp, msp, e), e, gp, fp, msp; kwargs...)
end

# The spherical quadrature among a transform's keywords, which the preprocessing fits run under too.
_quadrature_keywords(kw::NamedTuple) =
    (; sampling = get(kw, :sampling, nothing), weights = get(kw, :weights, nothing))

# Multi-field convenience: stack equal-shaped arrays along a new trailing batch axis.
calculate_spectrum(grid::FlowGeometries.Grids.AbstractGrid, fields::Tuple, ms::Tuple; kwargs...) =
    calculate_spectrum(grid, stack_fields(fields), ms; kwargs...)

# =============================================================================
# Canonical two-axis dispatch: calculate_spectrum(transform, execution, grid, field, ms; …). The NUFFT
# libraries are FlowTransformBindings' `FINUFFTBackend()` and `NonuniformFFTsBackend()`.
# =============================================================================

# ---- Level 1: distribution wrappers unwrap to the Distributed / MPI extension hooks ----
calculate_spectrum(t::SpectralBackends.AbstractSpectralBackend, e::ComputationalBackends.AbstractDistributedBackend, g::FlowGeometries.Grids.AbstractGrid, field::AbstractArray, ms::Tuple; kwargs...) =
    _calculate_spectrum_distributed(t, e, g, field, ms; _whole_grid_inactive(t, g, kwargs)...)
calculate_spectrum(t::SpectralBackends.AbstractSpectralBackend, e::ComputationalBackends.AbstractMPIBackend, g::FlowGeometries.Grids.AbstractGrid, field::AbstractArray, ms::Tuple; kwargs...) =
    _calculate_spectrum_mpi(t, e, g, field, ms; _whole_grid_inactive(t, g, kwargs)...)

# A point partition transforms each share as a grid of its own, so an `Exclude` quadrature is settled on
# the whole grid first: its rescaled weights, which every share reads as given.
_whole_grid_inactive(t, g::FlowGeometries.Grids.AbstractGrid, kwargs) = kwargs
function _whole_grid_inactive(t, g::FlowGeometries.Grids.AbstractGrid{<:Grids.SphericalHarmonicGeometry}, kwargs)
    s = get(kwargs, :inactive, ZeroFill())
    (s isa ZeroFill || !_point_partition(t, g, kwargs)) && return kwargs
    kw = NamedTuple(kwargs)
    w = Grids._inactive_weights(s, g, eltype(g); sampling = get(kw, :sampling, nothing),
        weights = get(kw, :weights, nothing))
    return merge(kw, (; weights = w, inactive = ZeroFill()))
end

# ---- Level 2: DirectSum on a local CPU backend (core Serial; OhMyThreads ext Threaded) ----
function calculate_spectrum(::SpectralBackends.AbstractDirectSumSpectralBackend,
        exec::Union{ComputationalBackends.AbstractSerialBackend, ComputationalBackends.AbstractThreadedBackend},
        g::FlowGeometries.Grids.AbstractGrid{<:FlowGeometries.Geometry.AbstractCartesianGeometry},
        field::AbstractArray, ms::NTuple{D, Int}; iflag::Int = 1, box = nothing, kwargs...) where {D}
    prob = TransformProblem(g, field)
    R = Val(eltype(field) <: Real)
    pms = FTB.packed_size(Grids.mode_box(g, ms, R), R)
    coeffs = zeros(Complex{eltype(g)}, pms..., batch_shape(prob)...)
    ks = _directsum_cartesian!(exec, coeffs, g, quadrature_weighted(g, field), ms, iflag, box)
    return coeffs, ks
end

function calculate_spectrum(::SpectralBackends.AbstractDirectSumSpectralBackend,
        exec::Union{ComputationalBackends.AbstractSerialBackend, ComputationalBackends.AbstractThreadedBackend},
        g::FlowGeometries.Grids.AbstractGrid{<:Grids.SphericalHarmonicGeometry},
        field::AbstractArray, ms::NTuple{2, Int}; kwargs...)
    prob = TransformProblem(g, field)
    lmax = ms[1] - 1
    coeffs = zeros(sph_coeff_type(eltype(field), eltype(g)), lmax + 1, 2 * lmax + 1,
        batch_shape(prob)...)
    ks = _directsum_spherical!(exec, coeffs, g, field, lmax; kwargs...)
    return coeffs, ks
end

# ---- Level 2: DirectSum on GPU (KernelAbstractions ext; portable on any KA device) ----
calculate_spectrum(::SpectralBackends.AbstractDirectSumSpectralBackend, exec::ComputationalBackends.AbstractGPUBackend, g::FlowGeometries.Grids.AbstractGrid{<:FlowGeometries.Geometry.AbstractCartesianGeometry}, field::AbstractArray, ms::NTuple{D, Int}; iflag::Int = 1, box = nothing, kwargs...) where {D} =
    _gpu_directsum_cartesian(exec, g, field, ms, iflag, box)
calculate_spectrum(::SpectralBackends.AbstractDirectSumSpectralBackend, exec::ComputationalBackends.AbstractGPUBackend, g::FlowGeometries.Grids.AbstractGrid{<:Grids.SphericalHarmonicGeometry}, field::AbstractArray, ms::NTuple{2, Int}; kwargs...) =
    _gpu_directsum_spherical(exec, g, field, ms[1] - 1; kwargs...)

# ---- Level 2: FFT (FFTW ext CPU; device-generic GPU FFT). Structured Cartesian. A grid uniform in
# every direction takes the pure FFT; one uniform in some directions and stretched in others takes the
# hybrid composite below. ----
function calculate_spectrum(::SpectralBackends.AbstractFFTSpectralBackend, exec::Union{ComputationalBackends.AbstractSerialBackend, ComputationalBackends.AbstractThreadedBackend}, g::FlowGeometries.Grids.AbstractStructuredGrid{<:FlowGeometries.Geometry.AbstractCartesianGeometry}, field::AbstractArray, ms::Tuple;
        nufft::SpectralBackends.AbstractSpectralBackend = SpectralBackends.AutoSpectralBackend(), kwargs...)
    D = ndims(g)
    umask = _uniform_mask(g, Val(D))
    all(umask) && return _calculate_spectrum_fft(exec, g, field, ms; kwargs...)
    any(umask) || _all_stretched_fft()
    return _calculate_spectrum_hybrid(_resolve_nufft_provider(nufft), exec, g, field,
        NTuple{D, Int}(Tuple(ms)), umask; kwargs...)
end
function calculate_spectrum(::SpectralBackends.AbstractFFTSpectralBackend, exec::ComputationalBackends.AbstractGPUBackend, g::FlowGeometries.Grids.AbstractStructuredGrid{<:FlowGeometries.Geometry.AbstractCartesianGeometry}, field::AbstractArray, ms::Tuple;
        nufft::SpectralBackends.AbstractSpectralBackend = SpectralBackends.AutoSpectralBackend(), kwargs...)
    D = ndims(g)
    umask = _uniform_mask(g, Val(D))
    all(umask) && return _calculate_spectrum_gpu_fft(exec, g, field, ms; kwargs...)
    any(umask) || _all_stretched_fft()
    return _calculate_spectrum_hybrid(_resolve_nufft_provider(nufft), exec, g, field,
        NTuple{D, Int}(Tuple(ms)), umask; kwargs...)
end

# The FFT transforms the uniform axes of a grid, the hybrid composite needing at least one.
@noinline _all_stretched_fft() = throw(ArgumentError(
    "FFTSpectralBackend needs a uniform axis in at least one direction (an AbstractRange); every axis of " *
    "this grid is stretched. Use a NUFFT (`transform = FlowTransformBindings.NonuniformFFTsBackend()` or " *
    "`FlowTransformBindings.FINUFFTBackend()`) or `DirectSumSpectralBackend()`."))

# ---- Level 2: NUFFT (`NUFFT.jl`, through FlowTransformBindings). A structured (nonuniform-gridded) grid
# takes the separable per-axis 1-D NUFFT; a point set the D-dimensional one. ----
calculate_spectrum(t::SpectralBackends.AbstractNUFFTSpectralBackend, exec::Union{ComputationalBackends.AbstractSerialBackend, ComputationalBackends.AbstractThreadedBackend}, g::Union{FlowGeometries.Grids.AbstractStructuredGrid{<:FlowGeometries.Geometry.AbstractCartesianGeometry}, Grids.PointwiseCartesian}, field::AbstractArray, ms::Tuple; kwargs...) =
    _calculate_spectrum_nufft(t, exec, g, field, ms; kwargs...)
calculate_spectrum(t::SpectralBackends.AbstractNUFFTSpectralBackend, exec::ComputationalBackends.AbstractGPUBackend, g::Union{FlowGeometries.Grids.AbstractStructuredGrid{<:FlowGeometries.Geometry.AbstractCartesianGeometry}, Grids.PointwiseCartesian}, field::AbstractArray, ms::Tuple; kwargs...) =
    _calculate_spectrum_gpu_nufft(t, exec, g, field, ms; kwargs...)

# ---- Level 2: SHT / NUFSHT (execution axis handled inside the extension) ----
calculate_spectrum(::SpectralBackends.AbstractFSHTSpectralBackend, exec::Union{ComputationalBackends.AbstractSerialBackend, ComputationalBackends.AbstractThreadedBackend}, g::FlowGeometries.Grids.AbstractStructuredGrid{<:FlowGeometries.Geometry.AbstractSphericalGeometry}, field::AbstractArray, ms::Tuple; kwargs...) =
    _calculate_spectrum_sht(exec, g, field, ms; kwargs...)
# GPU SHT: a fast device-generic transform (φ-DFT + θ-Legendre contraction) in the KernelAbstractions
# ext — no FastTransforms, runs on any KA device. Structured grid only.
calculate_spectrum(::SpectralBackends.AbstractFSHTSpectralBackend, exec::ComputationalBackends.AbstractGPUBackend, g::FlowGeometries.Grids.AbstractStructuredGrid{<:FlowGeometries.Geometry.AbstractSphericalGeometry}, field::AbstractArray, ms::Tuple; kwargs...) =
    _calculate_spectrum_gpu_sht(exec, g, field, ms; kwargs...)
calculate_spectrum(::SpectralBackends.AbstractNUFSHTSpectralBackend, exec::ComputationalBackends.AbstractExecutionBackend, g::FlowGeometries.Grids.AbstractGrid{<:Grids.SphericalHarmonicGeometry}, field::AbstractArray, ms::Tuple; kwargs...) =
    _calculate_spectrum_nufsht(exec, g, field, ms; kwargs...)

# =============================================================================
# Hybrid per-axis transform: one FFT over the uniform axes, one 1-D type-1 NUFFT along each stretched
# axis, every other spatial and batch dim riding as that pass's batch. A uniform-xy / stretched-z grid
# therefore pays the NUFFT only along z. The passes run on the execution backend's arrays: host arrays
# through FFTW, device arrays through the device's AbstractFFTs provider.
#
# Axis 1 must be uniform, at its own length. The packed publish then reads `+N₁/2` at the native `−N₁/2`,
# the same coefficient under uniform sampling. A stretched even axis `d ≥ 2` widens the coefficient box
# (`Grids.mode_box`), whose modes at `+N_d/2` are conjugate reads of computed modes with `k₁ ≤ 0`; the FFT
# pass is then complex, keeping the full `k₁` range, and otherwise it is an `rfft` halving axis 1.
#
# Every pass leaves the unnormalized transform in native order, measured from each direction's origin
# (the FFT axes' first sample, the NUFFT axes' folding origin); the `1/∏N_d` normalization is applied
# once, by the packed-layout gather the NUFFT plans publish with (`_publish!`). The per-axis sign is
# `Σ e^{-ikx}` for both providers, so `iflag = -1` is one conjugation of the whole product.
# =============================================================================

# Per-axis uniformity, answered from each axis's type. The `Val` length makes the result an
# `NTuple{D,Bool}`, so the all-uniform and mixed branches resolve at compile time.
_uniform_mask(g, ::Val{D}) where {D} = ntuple(d -> FlowGeometries.Grids.isuniform(g, d), Val(D))

# The library that transforms the stretched axes. `AutoSpectralBackend` takes NonuniformFFTs when it is
# loaded and FINUFFT when only it is; an explicit one is honoured as given.
_resolve_nufft_provider(t::SpectralBackends.AbstractNUFFTSpectralBackend) = t
function _resolve_nufft_provider(::SpectralBackends.AbstractAutoSpectralBackend)
    t = _auto_nufft(ComputationalBackends.SerialBackend())
    t === nothing || return t
    throw(ArgumentError(
        "a grid uniform in some directions and stretched in others needs a NUFFT library for its " *
        "stretched axes — run `using NonuniformFFTs` or `using FINUFFT`, or pass one as " *
        "`nufft = FlowTransformBindings.NonuniformFFTsBackend()`."))
end

"""
    _hybrid_derive(grid, ::Type{Tr}, ms, umask, R, iflag, eps) -> NamedTuple

Everything the hybrid composite needs before any transform runs: the per-axis coordinates, origins and
Fourier lengths, which dims the FFT takes and which the NUFFT takes, the coefficient box and whether the
FFT pass halves axis 1, the spectrum extents entering the publish, its sign, and the wavenumber axes.
"""
function _hybrid_derive(g, ::Type{Tr}, ms::NTuple{D, Int}, umask::NTuple{D, Bool}, R::Bool,
        iflag::Int, eps::Union{Nothing, Real}) where {Tr, D}
    umask[1] || throw(ArgumentError(
        "the hybrid FFT/NUFFT transform needs axis 1 uniform (it carries the real field's halved axis, " *
        "whose Nyquist mode coincides with the native `-N₁/2` only under uniform sampling); axis 1 of " *
        "this grid is stretched. Use a NUFFT (`transform = FlowTransformBindings.NonuniformFFTsBackend()` " *
        "or `FlowTransformBindings.FINUFFTBackend()`), whose separable path transforms every axis."))
    Ns = size(g)
    axs = ntuple(d -> Tr.(FlowGeometries.Grids.coordinates(g, d)), D)
    offs_all, ranges = Grids.axis_geometry(Tr, g, D)
    for d in 1:D
        (!umask[d] || ms[d] == Ns[d]) || throw(ArgumentError(
            "axis $d is uniform, so it transforms by FFT and needs ms[$d] == size(grid, $d) = $(Ns[d]); " *
            "got $(ms[d]). Ask for the grid's own length on every uniform axis."))
    end
    udims = Tuple(d for d in 1:D if umask[d])
    sdims = Tuple(d for d in 1:D if !umask[d])
    box = Grids.mode_box(g, ms, Val(R))
    halve = R && box == ms
    # Spectrum extents entering the publish: axis 1 is already halved by an `rfft` pass.
    nsx = ntuple(d -> (d == 1 && halve) ? ms[1] ÷ 2 + 1 : ms[d], Val(D))
    return (; Ns, npts = prod(Ns), axs, offs_all, ranges, udims, sdims, epsv = _nufft_tol(Tr, eps),
        box, halve, neg = iflag < 0, nsx, ks_phys = physical_wavenumbers(ranges, box, Val(R)))
end

# One-shot: a plan for this field's batch shape, its uniform-axis FFT planned without measuring, executed
# once and released.
function _calculate_spectrum_hybrid(t::SpectralBackends.AbstractNUFFTSpectralBackend,
        exec::Union{_HostExec, ComputationalBackends.AbstractGPUBackend},
        g::FlowGeometries.Grids.AbstractStructuredGrid{<:FlowGeometries.Geometry.AbstractCartesianGeometry},
        field::AbstractArray, ms::NTuple{D, Int}, umask::NTuple{D, Bool}; kwargs...) where {D}
    plan = _hybrid_plan(t, exec, g, float(eltype(field)), ms, umask;
        batch = Grids.field_batch_shape(g, field), measure = false, kwargs...)
    try
        coeffs = Array{Plans.coefficient_type(plan)}(undef, Plans.coefficient_size(plan)...)
        return coeffs, calculate_spectrum!(coeffs, plan, field)
    finally
        Plans.close!(plan)
    end
end

"""
    RegionFFT{P,S,O}

The hybrid composite's FFT over its uniform axes, planned once on the execution backend's arrays.
`halve` marks the real path, whose `rfft` shortens axis 1 to `n₁÷2+1`; `conj_in` conjugates a complex
input, carrying `iflag = -1` through the composite. The field is copied into `scratch` and scaled there by
the grid quadrature, so the caller's array is never mutated.
"""
struct RegionFFT{P, S, O <: Tuple}
    fwd::P
    scratch::S
    outsize::O
    halve::Bool
    conj_in::Bool
end

Base.show(io::IO, r::RegionFFT) = print(io, "RegionFFT(", r.halve ? "rfft" : "fft", " → ", r.outsize, ")")

# Output shape of the uniform-axis pass, which sizes the working array after it.
_region_fft_size(r::RegionFFT) = r.outsize

# The uniform-axis pass of `field` into `out`, the field weighted by the per-node quadrature factor `qw`
# (`nothing` for a constant measure) on its way into the scratch.
function _region_fft_exec!(out::AbstractArray, r::RegionFFT, field, qw, npts::Int, ntrans::Int)
    _stage!(r.scratch, field)
    _scale_points!(r.scratch, qw, npts, ntrans)
    r.conj_in && (r.scratch .= conj.(r.scratch))
    LinearAlgebra.mul!(out, r.fwd, r.scratch)
    return out
end

"""
    HybridPlan{T,D,R}

Reusable plan for a Cartesian grid uniform in some directions and stretched in others, on the host or a
device: the FFT over the uniform axes ([`RegionFFT`](@ref)), one 1-D type-1 NUFFT per stretched axis, the
`length(sdims) + 1` working arrays the passes write through, and the table that publishes the coefficient
box. `R` marks a real field. Execute with `calculate_spectrum!(coeffs, plan, field)`; [`close!`](@ref)
releases the NUFFT library plans.
"""
struct HybridPlan{T, D, R, NB, FP, AP, W, QW, IX, ST, KS, SD} <: Plans.AbstractSpectralPlan
    region::FP                       # the uniform-axis FFT, planned
    axes::AP                         # one axis pass per stretched dim, in `sdims` order
    sdims::SD
    work::W                          # working arrays; `work[1]` takes the region output
    qw::QW                           # (N,) grid quadrature factor on the backend, `nothing` if constant
    src::IX                          # coefficient → entry of one transform's spectrum (`_packed_src`)
    stage::ST                        # device staging for the coefficients, or `nothing` on the host
    ms::NTuple{D, Int}
    box::NTuple{D, Int}              # coefficient box
    nsx::NTuple{D, Int}              # spectrum extents entering the publish
    batch::NTuple{NB, Int}
    ntrans::Int
    npts::Int
    neg::Bool
    scale::T                         # 1/∏N_d
    ks::KS
end

Base.show(io::IO, ::HybridPlan{T, D, R}) where {T, D, R} =
    print(io, "HybridPlan{$T, $D}(", R ? "real" : "complex", ")")

# A real field publishes the packed half of its coefficient box; a complex one keeps every native mode.
Plans.coefficient_size(p::HybridPlan{T, D, R}) where {T, D, R} =
    (FTB.packed_size(p.box, Val(R))..., p.batch...)
Plans.coefficient_type(::HybridPlan{T}) where {T} = Complex{T}
Plans.wavenumbers(p::HybridPlan) = p.ks

"""
    _hybrid_plan(nufft, exec, grid, ::Type{T}, ms, umask; batch, iflag, eps, measure = true, kwargs...)

A [`HybridPlan`](@ref) on `exec`, a host or a device backend, for a grid uniform in some directions and
stretched in others. `measure` selects FFTW's measured planning for the uniform axes; the one-shot
transform passes `false`.
"""
function _hybrid_plan(nufft::SpectralBackends.AbstractSpectralBackend,
        exec::Union{_HostExec, ComputationalBackends.AbstractGPUBackend},
        g::FlowGeometries.Grids.AbstractStructuredGrid{<:FlowGeometries.Geometry.AbstractCartesianGeometry},
        ::Type{T}, ms::NTuple{D, Int}, umask::NTuple{D, Bool}; batch::Tuple = (), iflag::Int = 1,
        eps::Union{Nothing, Real} = nothing, measure::Bool = true, kwargs...) where {T, D}
    t = _resolve_nufft_provider(nufft)
    Tr = real(float(T))
    R = T <: Real
    h = _hybrid_derive(g, Tr, ms, umask, R, iflag, eps)
    bt = NTuple{length(batch), Int}(batch)
    ntrans = prod(bt; init = 1)
    region = _region_fft_plan(exec, T, (h.Ns..., bt...), h.udims, h.halve, !R && h.neg, measure)
    rsize = _region_fft_size(region)
    # Each stretched axis shortens its own dim: `outs[i]` is the size pass `i` writes and `ins[i]` the
    # size it reads.
    outs = accumulate((s, d) -> _axis_out_size(s, d, ms[d]), h.sdims; init = rsize)
    ins = (rsize, Base.front(outs)...)
    axes = _axis_passes(length(h.sdims)) do i
        d = h.sdims[i]
        _axis_nufft_plan(t, exec, Tr, ins[i], d, h.axs[d], ms[d], h.ranges[d], h.offs_all[d], h.epsv;
            kwargs...)
    end
    return _closing_on_error(axes) do
        work = (_alloc(exec, Complex{Tr}, rsize...), map(s -> _alloc(exec, Complex{Tr}, s...), outs)...)
        qwh = Grids.quadrature_scale(g, Tr, h.npts)
        qw = qwh === nothing ? nothing : _to_exec(exec, collect(Tr, qwh))
        src = _to_exec(exec, _packed_src(ms, h.box, h.nsx, Val(R)))
        stage = _on_device(exec) ? _alloc(exec, Complex{Tr}, prod(FTB.packed_size(h.box, Val(R))) * ntrans) :
            nothing
        HybridPlan{Tr, D, R, length(bt), typeof(region), typeof(axes), typeof(work), typeof(qw), typeof(src),
                typeof(stage), typeof(h.ks_phys), typeof(h.sdims)}(
            region, axes, h.sdims, work, qw, src, stage, ms, h.box, h.nsx, bt, ntrans, h.npts, h.neg,
            one(Tr) / h.npts, h.ks_phys)
    end
end

"""
    calculate_spectrum!(coeffs, plan::HybridPlan, field) -> ks

Execute a prebuilt hybrid plan in place. `field` is `(N_1…N_D, batch…)`; `coeffs` is the packed half of
the coefficient box (`Grids.mode_box`) for a real field and the full native spectrum for a complex one,
on the host or the plan's device. The uniform-axis FFT, the per-axis NUFFT plans and the working arrays
are reused across calls.
"""
function calculate_spectrum!(coeffs::AbstractArray{Complex{T}}, p::HybridPlan{T, D, R},
        field) where {T, D, R}
    length(field) == p.npts * p.ntrans || throw(DimensionMismatch(
        "field holds $(length(field)) values; this plan was built for $(p.npts * p.ntrans) — pass the " *
        "matching `batch=` to plan_spectrum"))
    Plans._check_coefficients(coeffs, p)
    W = _region_fft_exec!(first(p.work), p.region, field, p.qw, p.npts, p.ntrans)
    F = _run_stretched!(W, Base.tail(p.work), p.axes, p.sdims)
    Pn = prod(p.nsx)
    _publish!(coeffs, 0, F, 0, Pn, p.src, p.scale, prod(FTB.packed_size(p.box, Val(R))), p.ntrans, p.neg,
        p.stage)
    return p.ks
end

# =============================================================================
# The hybrid composite's synthesis, `f(x) = Σ_κ C_κ e^{+iflag·iκ·(x - x₀)}` over the native spectrum. A
# packed half is written on the host as the native series whose real part is its series over the
# coefficient box (`DirectSum.native_coefficients!`; axis 1, uniform at its own length, is never widened),
# and the series then separates: an unnormalized backward FFT over the uniform axes, then a 1-D type-2
# NUFFT along each stretched axis, every other dim a transform of the batch. The forward's `1/∏N_d` stands
# on the coefficients, so neither direction rescales. A real field is the real part of the native series.
# =============================================================================

"""
    HybridSynthesisPlan{T,D,R}

Reusable synthesis for a Cartesian grid uniform in some directions and stretched in others, the
counterpart of [`HybridPlan`](@ref) on the host or a device: the in-place backward FFT over the uniform
axes, one 1-D type-2 NUFFT per stretched axis, the working arrays the passes write through, and the host
native spectrum a packed half is written into. `R` marks a real field. Execute with
`synthesize!(out, plan, coeffs)`; [`close!`](@ref) releases the NUFFT library plans.
"""
struct HybridSynthesisPlan{T, D, R, NB, FP, AP, W, HF, HC, OB, SD, SP} <: Plans.AbstractSynthesisPlan
    region::FP                       # in-place backward FFT over the uniform axes of `work[1]`
    axes::AP                         # one type-2 axis pass per stretched dim, in `sdims` order
    sdims::SD
    work::W                          # `work[1]` the native spectrum `(ms…, ntrans)`, `work[i+1]` pass `i`'s output
    full::HF                         # host native spectrum a packed half is written into (`work[1]` on the host)
    hc::HC                           # host copy of a device packed half, or `nothing`
    outd::OB                         # a real field on the device, the last pass's real part, or `nothing`
    ms::NTuple{D, Int}
    box::NTuple{D, Int}              # coefficient box
    spatial::SP
    batch::NTuple{NB, Int}
    ntrans::Int
end

Base.show(io::IO, ::HybridSynthesisPlan{T, D, R}) where {T, D, R} =
    print(io, "HybridSynthesisPlan{$T, $D}(", R ? "real" : "complex", ")")

Plans.field_size(p::HybridSynthesisPlan) = (p.spatial..., p.batch...)
Plans.field_type(::HybridSynthesisPlan{T, D, R}) where {T, D, R} = R ? T : Complex{T}

"""
    _hybrid_synthesis_plan(nufft, exec, grid, ::Type{T}, ms, umask; batch, iflag, eps, measure = true, kwargs...)

A [`HybridSynthesisPlan`](@ref) on `exec`, a host or a device backend, for a grid uniform in some
directions and stretched in others, `T` the field's element type. `measure` selects FFTW's measured
planning for the uniform axes; a one-shot synthesis passes `false`.
"""
function _hybrid_synthesis_plan(nufft::SpectralBackends.AbstractSpectralBackend,
        exec::Union{_HostExec, ComputationalBackends.AbstractGPUBackend},
        g::FlowGeometries.Grids.AbstractStructuredGrid{<:FlowGeometries.Geometry.AbstractCartesianGeometry},
        ::Type{T}, ms::NTuple{D, Int}, umask::NTuple{D, Bool}; batch::Tuple = (), iflag::Int = 1,
        eps::Union{Nothing, Real} = nothing, measure::Bool = true, kwargs...) where {T, D}
    t = _resolve_nufft_provider(nufft)
    Tr = real(float(T))
    R = T <: Real
    h = _hybrid_derive(g, Tr, ms, umask, R, iflag, eps)
    bt = NTuple{length(batch), Int}(batch)
    ntrans = prod(bt; init = 1)
    # Each stretched axis grows its own dim from `ms[d]` modes to `Ns[d]` points: `outs[i]` is the size
    # pass `i` writes.
    s0 = (ms..., ntrans)
    outs = accumulate((s, d) -> _axis_out_size(s, d, h.Ns[d]), h.sdims; init = s0)
    # The type 2 evaluates `Σ û e^{-iflag'·iκy}`, FlowTransformBindings' sign `iflag' = -iflag`.
    axes = _axis_passes(length(h.sdims)) do i
        d = h.sdims[i]
        _axis_nufft_plan(t, exec, Tr, outs[i], d, h.axs[d], ms[d], h.ranges[d], h.offs_all[d], h.epsv;
            iflag = -iflag, kwargs...)
    end
    return _closing_on_error(axes) do
        work = (_alloc(exec, Complex{Tr}, s0...), map(s -> _alloc(exec, Complex{Tr}, s...), outs)...)
        region = _region_bfft_plan(exec, first(work), h.udims, iflag, measure)
        dev = _on_device(exec)
        full = !R ? nothing : dev ? Array{Complex{Tr}}(undef, s0...) : first(work)
        hc = (R && dev) ? Array{Complex{Tr}}(undef, FTB.packed_size(h.box, Val(true))..., ntrans) : nothing
        outd = (dev && R) ? _alloc(exec, Tr, h.Ns..., ntrans) : nothing
        HybridSynthesisPlan{Tr, D, R, length(bt), typeof(region), typeof(axes), typeof(work), typeof(full),
                typeof(hc), typeof(outd), typeof(h.sdims), typeof(h.Ns)}(
            region, axes, h.sdims, work, full, hc, outd, ms, h.box, h.Ns, bt, ntrans)
    end
end

"""
    synthesize!(out, plan::HybridSynthesisPlan, coeffs) -> out

Evaluate the series `coeffs` (the packed half of the coefficient box for a real field, the full native
spectrum for a complex one, on the host or the plan's device) at the grid's points into `out`.
"""
function Plans.synthesize!(out::AbstractArray, p::HybridSynthesisPlan{T, D, R},
        coeffs::AbstractArray) where {T, D, R}
    size(out) == Plans.field_size(p) || throw(DimensionMismatch(
        "out is $(size(out)); this plan writes $(Plans.field_size(p))"))
    sz = FTB.packed_size(p.box, Val(R))
    size(coeffs) == (sz..., p.batch...) || throw(DimensionMismatch(
        "coeffs is $(size(coeffs)); this plan reads $((sz..., p.batch...))"))
    W = first(p.work)
    if R
        half = p.hc === nothing ? reshape(coeffs, sz..., p.ntrans) : copyto!(p.hc, coeffs)
        DirectSum.native_coefficients!(p.full, half, p.ms, p.box)
        p.full === W || copyto!(W, p.full)
    else
        copyto!(W, coeffs)
    end
    p.region * W
    F = _run_stretched_synthesis!(W, Base.tail(p.work), p.axes, p.sdims)
    return _hybrid_field!(out, F, p.outd, Val(R))
end

# The field from the last pass's array `F`, its real part for a real field, taken on the device into
# `outd` where `F` lives there.
_hybrid_field!(out, F, outd, ::Val{false}) = copyto!(out, F)
function _hybrid_field!(out, F, outd, ::Val{true})
    dst = outd === nothing ? reshape(out, size(F)) : outd
    dst .= real.(F)
    outd === nothing || copyto!(out, outd)
    return out
end

# `field` scaled by the per-node factor `α` into `out`.
function quadrature_weighted_into!(out, field, α, npts::Int, ntrans::Int)
    @inbounds for b in 1:ntrans, j in 1:npts
        out[j + (b - 1) * npts] = field[j + (b - 1) * npts] * α[j]
    end
    return out
end

# ---- Least-specific catch-all: clear error for any unsupported (transform, execution, grid) ----
function calculate_spectrum(t::SpectralBackends.AbstractSpectralBackend, e::ComputationalBackends.AbstractExecutionBackend,
        g::FlowGeometries.Grids.AbstractGrid, field::AbstractArray, ms::Tuple; kwargs...)
    throw(ArgumentError(
        "$(nameof(typeof(t))) cannot act on a $(nameof(typeof(g))) over $(nameof(typeof(g.geometry))) with " *
        "$(nameof(typeof(e))). Each grid takes its transform: a uniform structured Cartesian grid the FFT " *
        "(the hybrid FFT/NUFFT where some axes are stretched), a nonuniform, scattered or curvilinear " *
        "Cartesian grid a NUFFT (FlowTransformBindings.NonuniformFFTsBackend() or FINUFFTBackend()), a " *
        "structured sphere the spherical-harmonic transform (FastSphericalHarmonics on the host, the " *
        "device-generic transform on a GPUBackend), any spherical grid NUFSHT. DirectSumSpectralBackend() " *
        "serves every Cartesian and spherical grid on every execution backend, at O(∏ms·N) cost.",
    ))
end

# =============================================================================
# synthesize — the series evaluated at the grid's points, a field `(spatial…, batch…)`.
# =============================================================================

"""
    synthesize(grid, coeffs, ms::Tuple; transform=AutoSpectralBackend(), execution=AutoBackend(),
               real_output=true, iflag=1)

The series with coefficients `coeffs` evaluated at the `grid` points: `f_j = Σ_k C_k e^{+iflag·ik·(x_j −
x₀)}` on a Cartesian grid, with `x₀` each direction's smallest coordinate, and `f_j = Σ_{l,m} C_lm
Y_lm(x_j)` on a sphere. `real_output=true` consumes the packed half a real Cartesian field transforms to,
over the coefficient box of an `ms`-mode transform on `grid` (see [`calculate_spectrum`](@ref)), and
writes a **real** array directly: the sum runs over the whole box, each stored mode `k₁ > 0` with its
conjugate partner. `real_output=false` consumes the full native spectrum `(ms…, batch…)` and returns it
complex. Spherical coefficients are `(Nθ, Nφ, batch…)` either way. Returns an array `(spatial…, batch…)`
shaped as the forward's input.

`synthesize` is the adjoint of [`calculate_spectrum`](@ref) under the grid's quadrature:
`Σ_k conj(C_k)·F(f)_k = Σ_j w_j conj(S(C)_j) f_j`, with `w` the weights the forward integrates with
(`w_j/Σw` of the grid measure on a Cartesian grid, the spherical weights totalling `4π` on a sphere).
It inverts `calculate_spectrum` where that quadrature is exact on the series: a uniform periodic grid
with `ms == size(grid)`, and FastSphericalHarmonics on its own grid. On a stretched or scattered grid
the composition is not the identity; recovering the coefficients a field was synthesized from is a
least-squares fit (`solve = true` with `NUFSHTSpectralBackend`).

`transform` selects the synthesis the same way [`calculate_spectrum`](@ref) selects the forward, and
`AutoSpectralBackend` resolves it by the same rules; the direct sum serves any grid.
"""
function synthesize(g::FlowGeometries.Grids.AbstractGrid, coeffs::AbstractArray, ms::Tuple;
        transform::SpectralBackends.AbstractSpectralBackend = SpectralBackends.AutoSpectralBackend(),
        execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
        real_output::Bool = true, iflag::Int = 1, kwargs...)
    e = _resolve_execution(execution)
    return _synthesize(_resolve_transform(transform, g, ms, e), e, g, coeffs, ms;
        real_output = real_output, iflag = iflag, kwargs...)
end

function _synthesize(::SpectralBackends.AbstractDirectSumSpectralBackend,
        exec::Union{ComputationalBackends.AbstractSerialBackend, ComputationalBackends.AbstractThreadedBackend},
        g::FlowGeometries.Grids.AbstractGrid{<:FlowGeometries.Geometry.AbstractCartesianGeometry},
        coeffs::AbstractArray, ms::NTuple{D, Int}; real_output::Bool = true, iflag::Int = 1,
        kwargs...) where {D}
    ss = size(g)
    FT = eltype(g)
    if real_output
        pms = FTB.packed_size(Grids.mode_box(g, ms, Val(true)), Val(true))
        size(coeffs)[1:D] == pms || throw(DimensionMismatch(
            "real_output=true expects the packed half $(pms) on the spectral dims; got $(size(coeffs)[1:D]). " *
            "Pass real_output=false for a full native spectrum $(ms)."))
        batch = ntuple(i -> size(coeffs, D + i), ndims(coeffs) - D)
        out = Array{FT}(undef, ss..., batch...)
        _synthesize_packed!(exec, out, g, coeffs, ms, iflag)
        return out
    end
    size(coeffs)[1:D] == ms || throw(DimensionMismatch(
        "real_output=false expects the full native spectrum $(ms) on the spectral dims; got $(size(coeffs)[1:D])."))
    batch = ntuple(i -> size(coeffs, D + i), ndims(coeffs) - D)
    out = Array{Complex{FT}}(undef, ss..., batch...)
    _synthesize_cartesian!(exec, out, g, coeffs, ms, iflag)
    return out
end

function _synthesize(::SpectralBackends.AbstractDirectSumSpectralBackend,
        exec::Union{ComputationalBackends.AbstractSerialBackend, ComputationalBackends.AbstractThreadedBackend},
        g::FlowGeometries.Grids.AbstractGrid{<:Grids.SphericalHarmonicGeometry},
        coeffs::AbstractArray, ms::NTuple{2, Int}; real_output::Bool = true, iflag::Int = 1,
        kwargs...)
    ss = size(g)
    batch = ntuple(i -> size(coeffs, 2 + i), ndims(coeffs) - 2)
    # A real coefficient array evaluates to a real field directly, so `real_output` needs no complex
    # intermediate; a complex one asked for a real output takes its real part.
    FT = eltype(g)
    if real_output && eltype(coeffs) <: Real
        out = Array{FT}(undef, ss..., batch...)
        _synthesize_spherical!(exec, out, g, coeffs, ms[1] - 1)
        return out
    end
    out = Array{Complex{FT}}(undef, ss..., batch...)
    _synthesize_spherical!(exec, out, g, coeffs, ms[1] - 1)
    return real_output ? real.(out) : out
end

"""
    DirectSumSynthesisPlan{FT}

Reusable direct-sum synthesis: the grid, resolution, sign and batch shape the series is fixed by, so
`synthesize!` writes into the caller's array and allocates nothing of its own.

The plan holds what the synthesis reads from the grid (`DirectSum.synth_setup`): on a tensor grid the
per-axis tables and the working arrays their contraction walks through, on a Cartesian point set the
positions and wavenumbers, and on a sphere the nodes and Legendre tables.
"""
struct DirectSumSynthesisPlan{FT, R, G, E, S, NB, ND} <: Plans.AbstractSynthesisPlan
    grid::G
    execution::E
    setup::S
    ms::NTuple{ND, Int}
    batch::NTuple{NB, Int}
    iflag::Int
end

Base.show(io::IO, p::DirectSumSynthesisPlan{FT, R}) where {FT, R} =
    print(io, "DirectSumSynthesisPlan{", FT, "}(", R ? "real" : "complex", ", ms=", p.ms, ")")

Plans.field_size(p::DirectSumSynthesisPlan) = (size(p.grid)..., p.batch...)
Plans.field_type(::DirectSumSynthesisPlan{FT, R}) where {FT, R} = R ? FT : Complex{FT}

function Plans.plan_synthesis(::SpectralBackends.AbstractDirectSumSpectralBackend,
        exec::Union{ComputationalBackends.AbstractSerialBackend, ComputationalBackends.AbstractThreadedBackend},
        g::FlowGeometries.Grids.AbstractGrid, ::Type{T}, ms::NTuple{ND, Int};
        batch::Tuple = (), iflag::Int = 1, box = nothing, kwargs...) where {T, ND}
    FT = real(float(eltype(g)))
    bt = NTuple{length(batch), Int}(batch)
    OT = T <: Real ? FT : Complex{FT}
    s = _synth_setup(g, OT, ms, bt, iflag, box)
    return DirectSumSynthesisPlan{FT, T <: Real, typeof(g), typeof(exec), typeof(s), length(bt), ND}(
        g, exec, s, ms, bt, iflag)
end

# `box` holds a whole grid's origins and Fourier lengths when `g` is part of it (`Grids.axis_geometry`).
_synth_setup(g::FlowGeometries.Grids.AbstractGrid{<:FlowGeometries.Geometry.AbstractCartesianGeometry},
    ::Type{OT}, ms::Tuple, batch::Tuple, iflag::Int, box) where {OT} =
    DirectSum.synth_setup(DirectSum._cart_layout(g), g, OT, ms, batch, iflag; box = box)
_synth_setup(g::FlowGeometries.Grids.AbstractGrid{<:Grids.SphericalHarmonicGeometry}, ::Type{OT},
    ms::Tuple, batch::Tuple, iflag::Int, box) where {OT} =
    DirectSum.sph_synth_setup(g, real(float(OT)), ms[1] - 1)

"""
    synthesize!(out, plan::DirectSumSynthesisPlan, coeffs) -> out

Evaluate the series `coeffs` at the plan's grid points into `out` by direct summation.
"""
function Plans.synthesize!(out::AbstractArray, plan::DirectSumSynthesisPlan{FT, R},
        coeffs::AbstractArray) where {FT, R}
    size(out) == Plans.field_size(plan) || throw(DimensionMismatch(
        "out is $(size(out)); this plan writes $(Plans.field_size(plan))"))
    return _synthesize_into!(out, plan, coeffs)
end

# Cartesian: the packed half of the coefficient box for a real field, the full native cube for a complex
# one.
function _synthesize_into!(out::AbstractArray, plan::DirectSumSynthesisPlan{FT, R, G},
        coeffs::AbstractArray) where {FT, R, G <: FlowGeometries.Grids.AbstractGrid{<:FlowGeometries.Geometry.AbstractCartesianGeometry}}
    g = plan.grid
    ms = plan.ms
    D = length(ms)
    pms = FTB.packed_size(Grids.mode_box(g, ms, Val(R)), Val(R))
    length(coeffs) == prod(pms) * prod(plan.batch; init = 1) ||
        throw(DimensionMismatch(
            "coeffs holds $(length(coeffs)) values; this plan reads the batch $(plan.batch) — pass the " *
            "matching `batch=` to plan_synthesis"))
    if R
        size(coeffs)[1:D] == pms || throw(DimensionMismatch(
            "a real-field plan expects the packed half $(pms) on the spectral dims; got $(size(coeffs)[1:D])"))
        _synthesize_packed!(plan.execution, out, g, coeffs, ms, plan.iflag, plan.setup)
    else
        size(coeffs)[1:D] == ms || throw(DimensionMismatch(
            "a complex-field plan expects the full native spectrum $(ms) on the spectral dims; got $(size(coeffs)[1:D])"))
        _synthesize_cartesian!(plan.execution, out, g, coeffs, ms, plan.iflag, plan.setup)
    end
    return out
end

# Spherical: the held nodes and Legendre tables, so a reused execution rebuilds neither. A real
# coefficient array evaluates to a real field with no complex intermediate.
function _synthesize_into!(out::AbstractArray, plan::DirectSumSynthesisPlan{FT, R, G},
        coeffs::AbstractArray) where {FT, R, G <: FlowGeometries.Grids.AbstractGrid{<:Grids.SphericalHarmonicGeometry}}
    DirectSum.sph_synth_run!(out, plan.setup, coeffs, plan.ms[1] - 1)
    return out
end

# The grids a transform here acts on.
const _TransformGrid = Union{
    FlowGeometries.Grids.AbstractGrid{<:FlowGeometries.Geometry.AbstractCartesianGeometry},
    FlowGeometries.Grids.AbstractGrid{<:Grids.SphericalHarmonicGeometry},
}

# The planning a plan built for one call takes: an FFT plans with FFTW's `ESTIMATE`, which costs no
# measurement of the transform.
_one_call(::SpectralBackends.AbstractFFTSpectralBackend) = (; measure = false)
_one_call(::SpectralBackends.AbstractSpectralBackend) = (;)

# Every other synthesis: a plan for these coefficients' batch shape, executed once and released. A pair
# with no synthesis plan raises in `plan_synthesis`.
function _synthesize(t::SpectralBackends.AbstractSpectralBackend, e::ComputationalBackends.AbstractExecutionBackend,
        g::_TransformGrid, coeffs::AbstractArray, ms::Tuple; real_output::Bool = true, iflag::Int = 1,
        kwargs...)
    FT = real(float(eltype(coeffs)))
    nsp = length(_coeff_spatial(g, ms, real_output))
    batch = ntuple(i -> size(coeffs, nsp + i), ndims(coeffs) - nsp)
    plan = Plans.plan_synthesis(t, e, g, real_output ? FT : Complex{FT}, ms; batch = batch, iflag = iflag,
        _one_call(t)..., kwargs...)
    try
        return Plans.synthesize!(Plans.allocate_field(plan), plan, coeffs)
    finally
        Plans.close!(plan)
    end
end

function _synthesize(t::SpectralBackends.AbstractSpectralBackend, e::ComputationalBackends.AbstractExecutionBackend,
        g::FlowGeometries.Grids.AbstractGrid, coeffs::AbstractArray, ms::Tuple; kwargs...)
    throw(ArgumentError(
        "$(nameof(typeof(t))) has no synthesis on a $(nameof(typeof(g))) over " *
        "$(nameof(typeof(FlowGeometries.Grids.grid_geometry(g)))): the transforms act on Cartesian and " *
        "spherical grids."))
end

# =============================================================================
# In-place calculate_spectrum! (DirectSum core; plan form lives in the extensions).
# =============================================================================

"""
    calculate_spectrum!(coeffs, grid, field, ms; transform=DirectSumSpectralBackend(), execution=AutoBackend(), kwargs...)

In-place [`calculate_spectrum`](@ref): write the coefficients into the preallocated `coeffs`, shaped as
`calculate_spectrum` returns them, and return `ks_phys`. The direct sum on `SerialBackend`/`ThreadedBackend` writes into `coeffs`
directly. Every other transform, `AutoSpectralBackend` included (resolved as `calculate_spectrum`
resolves it), runs through a plan built for this call and released after it, an FFT planned with FFTW's
`ESTIMATE` as `calculate_spectrum` plans it; a repeated caller holds one, built with
[`plan_spectrum`](@ref) and run with `calculate_spectrum!(coeffs, plan, field)`.
"""
function calculate_spectrum!(coeffs::AbstractArray{<:Number}, grid::FlowGeometries.Grids.AbstractGrid, field::AbstractArray, ms::Tuple;
        transform::SpectralBackends.AbstractSpectralBackend = SpectralBackends.DirectSumSpectralBackend(),
        execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(), kwargs...)
    e = _resolve_execution(execution)
    return _calculate_spectrum!(coeffs, _resolve_transform(transform, grid, ms, e), e, grid, field, ms;
        kwargs...)
end

function _calculate_spectrum!(coeffs::AbstractArray{Complex{T}}, ::SpectralBackends.AbstractDirectSumSpectralBackend,
        exec::Union{ComputationalBackends.AbstractSerialBackend, ComputationalBackends.AbstractThreadedBackend},
        g::FlowGeometries.Grids.AbstractGrid{<:FlowGeometries.Geometry.AbstractCartesianGeometry},
        field::AbstractArray, ms::NTuple{D, Int}; iflag::Int = 1, kwargs...) where {T, D}
    R = Val(eltype(field) <: Real)
    Plans._check_coefficients(coeffs,
        (FTB.packed_size(Grids.mode_box(g, ms, R), R)..., Grids.field_batch_shape(g, field)...))
    return _directsum_cartesian!(exec, coeffs, g, quadrature_weighted(g, field), ms, iflag)
end

function _calculate_spectrum!(coeffs::AbstractArray{<:Number}, ::SpectralBackends.AbstractDirectSumSpectralBackend,
        exec::Union{ComputationalBackends.AbstractSerialBackend, ComputationalBackends.AbstractThreadedBackend},
        g::FlowGeometries.Grids.AbstractGrid{<:Grids.SphericalHarmonicGeometry},
        field::AbstractArray, ms::NTuple{2, Int}; kwargs...)
    Plans._check_coefficients(coeffs, (ms[1], 2 * ms[1] - 1, Grids.field_batch_shape(g, field)...))
    return _directsum_spherical!(exec, coeffs, g, field, ms[1] - 1; kwargs...)
end

# Every other transform: a plan for this field's batch shape, executed into `coeffs` and released.
function _calculate_spectrum!(coeffs::AbstractArray, t::SpectralBackends.AbstractSpectralBackend,
        e::ComputationalBackends.AbstractExecutionBackend, g::FlowGeometries.Grids.AbstractGrid,
        field::AbstractArray, ms::Tuple; kwargs...)
    plan = Plans.plan_spectrum(t, e, g, float(eltype(field)), ms;
        batch = Grids.field_batch_shape(g, field), _one_call(t)..., kwargs...)
    try
        return calculate_spectrum!(coeffs, plan, field)
    finally
        Plans.close!(plan)
    end
end

"""
    plan_spectrum(grid, ::Type{T}, ms; transform=AutoSpectralBackend(), execution=AutoBackend(), batch=(), kwargs...)

Reusable [`AbstractSpectralPlan`](@ref) for the fixed geometry of `grid` at resolution `ms`,
transforming a field with trailing batch shape `batch` of element type `T` in one batched,
allocation-free execution. Reuse via `calculate_spectrum!(coeffs, plan, field)`.
"""
function Plans.plan_spectrum(grid::FlowGeometries.Grids.AbstractGrid, ::Type{T}, ms::Tuple;
        transform::SpectralBackends.AbstractSpectralBackend = SpectralBackends.AutoSpectralBackend(),
        execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
        batch::Tuple = (), kwargs...) where {T}
    e = _resolve_execution(execution)
    return Plans.plan_spectrum(_resolve_transform(transform, grid, ms, e), e, grid, T, ms; batch = batch, kwargs...)
end

"""
    plan_synthesis(grid, ::Type{T}, ms; transform=AutoSpectralBackend(), execution=AutoBackend(),
                   batch=(), kwargs...)

Reusable [`AbstractSynthesisPlan`](@ref) evaluating an `ms`-resolution series at the points of `grid`, a
field with trailing batch shape `batch` of element type `T`, as [`synthesize`](@ref) does. Reuse via
`synthesize!(out, plan, coeffs)`.
"""
function Plans.plan_synthesis(grid::FlowGeometries.Grids.AbstractGrid, ::Type{T}, ms::Tuple;
        transform::SpectralBackends.AbstractSpectralBackend = SpectralBackends.AutoSpectralBackend(),
        execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
        batch::Tuple = (), kwargs...) where {T}
    e = _resolve_execution(execution)
    return Plans.plan_synthesis(_resolve_transform(transform, grid, ms, e), e, grid, T, ms;
        batch = batch, kwargs...)
end

# Least-specific catch-all: a clear error for any (transform, execution, grid) with no synthesis plan.
function Plans.plan_synthesis(t::SpectralBackends.AbstractSpectralBackend,
        e::ComputationalBackends.AbstractExecutionBackend, grid::FlowGeometries.Grids.AbstractGrid,
        ::Type{T}, ms::Tuple; kwargs...) where {T}
    throw(ArgumentError(
        "$(nameof(typeof(t))) has no synthesis plan for a $(nameof(typeof(grid))) over " *
        "$(nameof(typeof(FlowGeometries.Grids.grid_geometry(grid)))) on $(nameof(typeof(e))): " *
        _plan_hint(t, e)))
end

# What a plan for a transform on an execution backend needs: the packages that load the extension
# implementing it, or, with that extension loaded, a grid the transform applies to.
const _NO_APPLY = "the transform does not apply to that grid. DirectSumSpectralBackend() serves every " *
    "Cartesian and spherical grid on every execution backend."
const _DEVICE_FFT = "`using KernelAbstractions` and the device's AbstractFFTs provider (`FFTW` for " *
    "`KA.CPU()`, `CUDA` for `CUDABackend`, …)"

_needs(ext::Symbol, load::String) = _ext_loaded(ext) ? _NO_APPLY : "it needs $load."

_plan_hint(::SpectralBackends.AbstractSpectralBackend, ::ComputationalBackends.AbstractExecutionBackend) =
    _NO_APPLY
_plan_hint(::SpectralBackends.AbstractFFTSpectralBackend, ::_HostExec) =
    _needs(:FlowFieldSpectraFFTWExt, "`using FFTW`")
_plan_hint(::SpectralBackends.AbstractFFTSpectralBackend, ::ComputationalBackends.AbstractGPUBackend) =
    _needs(:FlowFieldSpectraKernelAbstractionsAbstractFFTsExt, _DEVICE_FFT)
_plan_hint(::SpectralBackends.AbstractFSHTSpectralBackend, ::_HostExec) =
    _needs(:FlowFieldSpectraFastSphericalHarmonicsExt, "`using FastSphericalHarmonics`")
_plan_hint(::SpectralBackends.AbstractFSHTSpectralBackend, ::ComputationalBackends.AbstractGPUBackend) =
    _needs(:FlowFieldSpectraKernelAbstractionsAbstractFFTsExt, _DEVICE_FFT)
_plan_hint(::SpectralBackends.AbstractNUFSHTSpectralBackend, ::_HostExec) =
    _needs(:FlowFieldSpectraNUFSHTExt, "`using NUFSHT`")
_plan_hint(::SpectralBackends.AbstractNUFSHTSpectralBackend, ::ComputationalBackends.AbstractGPUBackend) =
    _needs(:FlowFieldSpectraNUFSHTKernelAbstractionsExt, "`using NUFSHT` and `using KernelAbstractions`")
_plan_hint(::SpectralBackends.AbstractDirectSumSpectralBackend, ::ComputationalBackends.AbstractGPUBackend) =
    _needs(:FlowFieldSpectraKernelAbstractionsExt, "`using KernelAbstractions`")
_plan_hint(::SpectralBackends.AbstractNUFFTSpectralBackend, ::ComputationalBackends.AbstractExecutionBackend) =
    "it names no NUFFT library. Pass FlowTransformBindings.NonuniformFFTsBackend() (`using NonuniformFFTs`) " *
    "or FlowTransformBindings.FINUFFTBackend() (`using FINUFFT`)."

"""
    DirectSumSphericalPlan{FT}

Reusable spherical direct-sum plan: the grid's materialized nodes (or its ring table, or its longitude
DFT matrix and latitude quadrature — whichever its layout reads), the Legendre recurrence tables, and the
`(lmax+1, nrings, B)` longitude buffer, all built once for a fixed grid, `lmax` and batch shape. For a
complex element type `T` the plan also holds the `exp(+imλ)` longitude transform's matrix and buffer.
Execute with `calculate_spectrum!(coeffs, plan, field)`, which runs on the plan's execution backend.
"""
struct DirectSumSphericalPlan{FT, CT, E, L, S, NB, KS} <: Plans.AbstractSpectralPlan
    execution::E                  # SerialBackend or ThreadedBackend
    layout::L                     # FlowGeometries' TensorRings / IsoLatitudeRings / NoRings
    setup::S                      # that layout's precomputation, from `DirectSum.sph_setup`
    lmax::Int
    batch::NTuple{NB, Int}
    B::Int
    ks::KS
end

Base.show(io::IO, p::DirectSumSphericalPlan{FT}) where {FT} =
    print(io, "DirectSumSphericalPlan{", FT, "}(", nameof(typeof(p.layout)), ", lmax=", p.lmax,
        ", B=", p.B, ")")

Plans.coefficient_size(p::DirectSumSphericalPlan) = (p.lmax + 1, 2 * p.lmax + 1, p.batch...)
Plans.coefficient_type(::DirectSumSphericalPlan{FT, CT}) where {FT, CT} = CT
Plans.wavenumbers(p::DirectSumSphericalPlan) = p.ks

function Plans.plan_spectrum(::SpectralBackends.AbstractDirectSumSpectralBackend,
        exec::Union{ComputationalBackends.AbstractSerialBackend, ComputationalBackends.AbstractThreadedBackend},
        g::FlowGeometries.Grids.AbstractGrid{<:Grids.SphericalHarmonicGeometry},
        ::Type{T}, ms::Tuple; batch::Tuple = (), sampling = nothing, weights = nothing,
        inactive::AbstractInactiveCells = ZeroFill(), kwargs...) where {T}
    FT = real(float(T))
    lmax = ms[1] - 1
    lmax >= 0 || throw(ArgumentError("ms[1] must be ≥ 1 (lmax = ms[1] - 1); got $(ms[1])"))
    bt = NTuple{length(batch), Int}(batch)
    B = prod(bt; init = 1)
    layout = Grids._sph_layout(g)
    CT = sph_coeff_type(T, FT)
    s = DirectSum.sph_setup(layout, g, CT, lmax, B; sampling = sampling,
        weights = Grids._inactive_weights(inactive, g, FT; sampling, weights))
    ks = (0:lmax, -lmax:lmax)
    return DirectSumSphericalPlan{FT, CT, typeof(exec), typeof(layout), typeof(s), length(bt), typeof(ks)}(
        exec, layout, s, lmax, bt, B, ks)
end

"""
    calculate_spectrum!(coeffs, plan::DirectSumSphericalPlan, field) -> ks

Fill preallocated `coeffs` `(Nθ, Nφ, batch…)` with the spherical spectrum of `field`
`(spatial…, batch…)`, reusing `plan`'s nodes / ring table / quadrature / Legendre tables.
"""
function calculate_spectrum!(coeffs::AbstractArray{<:Number}, plan::DirectSumSphericalPlan{T},
        field) where {T}
    Plans._check_coefficients(coeffs, plan)
    return _sph_run!(plan.execution, plan.layout, coeffs, plan.setup, field, plan.lmax, plan.B)
end

"""
    DirectSumCartesianPlan{FT}

Reusable Cartesian direct-sum plan: the per-axis dense DFT matrices, the working arrays the contraction
walks through, the wavenumber axes, the modes held at zero, and the quadrature-scaled field buffer, all
built once for a fixed grid, `ms`, `iflag` and batch shape. Execute with
`calculate_spectrum!(coeffs, plan, field)`, which runs on the plan's execution backend.
"""
struct DirectSumCartesianPlan{FT, E, L, S, NB, QW, QB, KS} <: Plans.AbstractSpectralPlan
    execution::E                  # SerialBackend or ThreadedBackend
    layout::L                     # TensorCartesian / CloudCartesian
    setup::S                      # that layout's precomputation, from `DirectSum.cart_setup`
    qw::QW                        # per-node quadrature factor, `nothing` where the measure is constant
    qbuf::QB                      # the scaled field this writes into, `nothing` alongside `qw`
    npts::Int
    batch::NTuple{NB, Int}
    B::Int
    ks::KS
end

Base.show(io::IO, p::DirectSumCartesianPlan{FT}) where {FT} =
    print(io, "DirectSumCartesianPlan{", FT, "}(", nameof(typeof(p.layout)), ", B=", p.B, ")")

# `setup.pms` is the packed coefficient box built for the field's realness, so it is already the full
# native size for a complex field.
Plans.coefficient_size(p::DirectSumCartesianPlan) = (p.setup.pms..., p.batch...)
Plans.coefficient_type(::DirectSumCartesianPlan{FT}) where {FT} = Complex{FT}
Plans.wavenumbers(p::DirectSumCartesianPlan) = p.ks

function Plans.plan_spectrum(::SpectralBackends.AbstractDirectSumSpectralBackend,
        exec::Union{ComputationalBackends.AbstractSerialBackend, ComputationalBackends.AbstractThreadedBackend},
        g::FlowGeometries.Grids.AbstractGrid{<:FlowGeometries.Geometry.AbstractCartesianGeometry},
        ::Type{T}, ms::NTuple{D, Int}; batch::Tuple = (), iflag::Int = 1, box = nothing,
        kwargs...) where {T, D}
    FT = real(float(T))
    bt = NTuple{length(batch), Int}(batch)
    layout = DirectSum._cart_layout(g)
    s = DirectSum.cart_setup(layout, g, T, ms, bt, iflag; box = box)
    npts = prod(size(g))
    qw = Grids.quadrature_scale(g, FT, npts)
    qbuf = qw === nothing ? nothing : Array{float(T)}(undef, npts, prod(bt; init = 1))
    return DirectSumCartesianPlan{FT, typeof(exec), typeof(layout), typeof(s), length(bt), typeof(qw),
            typeof(qbuf), typeof(s.ks)}(exec, layout, s, qw, qbuf, npts, bt, prod(bt; init = 1), s.ks)
end

"""
    calculate_spectrum!(coeffs, plan::DirectSumCartesianPlan, field) -> ks

Fill preallocated `coeffs` (the packed half of the coefficient box for a real field, full native for a
complex one) with the Cartesian spectrum of `field` `(spatial…, batch…)`, reusing `plan`'s DFT matrices
and working arrays.
"""
function calculate_spectrum!(coeffs::AbstractArray{Complex{T}}, plan::DirectSumCartesianPlan{T},
        field) where {T}
    length(field) == plan.npts * plan.B || throw(DimensionMismatch(
        "field holds $(length(field)) values; this plan was built for $(plan.npts * plan.B) — pass the " *
        "matching `batch=` to plan_spectrum"))
    Plans._check_coefficients(coeffs, plan)
    # Scaled into the plan's buffer; the caller's field is untouched.
    f = plan.qbuf === nothing ? field :
        quadrature_weighted_into!(plan.qbuf, field, plan.qw, plan.npts, plan.B)
    return _cart_run!(plan.execution, plan.layout, coeffs, plan.setup, f)
end

# Least-specific catch-all: a clear error for any (transform, execution, grid) with no spectrum plan.
function Plans.plan_spectrum(t::SpectralBackends.AbstractSpectralBackend,
        e::ComputationalBackends.AbstractExecutionBackend, grid::FlowGeometries.Grids.AbstractGrid,
        ::Type{T}, ms::Tuple; kwargs...) where {T}
    throw(ArgumentError(
        "$(nameof(typeof(t))) has no reusable plan for a $(nameof(typeof(grid))) over " *
        "$(nameof(typeof(FlowGeometries.Grids.grid_geometry(grid)))) on $(nameof(typeof(e))): " *
        _plan_hint(t, e)))
end

# =============================================================================
# Internal extension entry points — error until the relevant extension loads. Each takes the grid +
# field tensor directly (spatial/batch split from the grid).
# =============================================================================

# DirectSum forward: the setup the grid fixes, run on the execution backend by `_cart_run!` / `_sph_run!`,
# as the direct-sum plans run it. The serial runs are `DirectSum`'s; the OhMyThreads extension adds the
# threaded point-set and spherical runs, and the KernelAbstractions extension the device transform.
function _directsum_cartesian!(exec::_HostExec, coeffs::AbstractArray{Complex{FT}}, g, field, ms::Tuple,
        iflag::Int, box = nothing) where {FT}
    layout = DirectSum._cart_layout(g)
    T = eltype(field) <: Real ? FT : Complex{FT}
    s = DirectSum.cart_setup(layout, g, T, ms, Grids.field_batch_shape(g, field), iflag; box = box)
    return _cart_run!(exec, layout, coeffs, s, field)
end

function _directsum_spherical!(exec::_HostExec, coeffs::AbstractArray{C}, g, field, lmax::Int;
        sampling = nothing, weights = nothing, inactive::AbstractInactiveCells = ZeroFill()) where {C}
    layout = Grids._sph_layout(g)
    B = length(coeffs) ÷ ((lmax + 1) * (2 * lmax + 1))
    s = DirectSum.sph_setup(layout, g, C, lmax, B; sampling = sampling,
        weights = Grids._inactive_weights(inactive, g, real(float(C)); sampling, weights))
    return _sph_run!(exec, layout, coeffs, s, field, lmax, B)
end

_cart_run!(::ComputationalBackends.AbstractSerialBackend, layout, coeffs, s, field) =
    DirectSum.cart_run!(coeffs, layout, s, field)
# A tensor grid contracts through dense matrix products, which BLAS threads.
_cart_run!(::ComputationalBackends.AbstractThreadedBackend, layout::DirectSum.TensorCartesian, coeffs, s, field) =
    DirectSum.cart_run!(coeffs, layout, s, field)
_cart_run!(::ComputationalBackends.AbstractThreadedBackend, ::DirectSum.CloudCartesian, args...) =
    throw(ArgumentError("ThreadedBackend is not loaded. Run `using OhMyThreads`."))
_sph_run!(::ComputationalBackends.AbstractSerialBackend, layout, coeffs, s, field, lmax, B) =
    DirectSum.sph_run!(coeffs, layout, s, field, lmax, B)
_sph_run!(::ComputationalBackends.AbstractThreadedBackend, args...) =
    throw(ArgumentError("ThreadedBackend is not loaded. Run `using OhMyThreads`."))

# DirectSum inverse (synthesize): same Serial/Threaded split. `_synthesize_packed!` takes the packed
# half and writes a real field; `_synthesize_cartesian!` takes the full native spectrum. `geo` is the
# `DirectSum.synth_setup` a synthesis plan holds, `nothing` to build it from the grid.
_synthesize_packed!(::ComputationalBackends.AbstractSerialBackend, out, g, coeffs, ms, iflag, geo = nothing) =
    DirectSum._synthesize_packed_direct!(out, g, coeffs, ms, iflag, geo)
_synthesize_packed!(::ComputationalBackends.AbstractThreadedBackend, args...) = throw(ArgumentError("ThreadedBackend is not loaded. Run `using OhMyThreads`."))
_synthesize_cartesian!(::ComputationalBackends.AbstractSerialBackend, out, g, coeffs, ms, iflag, geo = nothing) =
    DirectSum._synthesize_cartesian_direct!(out, g, coeffs, ms, iflag, geo)
_synthesize_spherical!(::ComputationalBackends.AbstractSerialBackend, out, g, coeffs, lmax) =
    DirectSum._synthesize_spherical_direct!(out, g, coeffs, lmax)
_synthesize_cartesian!(::ComputationalBackends.AbstractThreadedBackend, args...) = throw(ArgumentError("ThreadedBackend is not loaded. Run `using OhMyThreads`."))
_synthesize_spherical!(::ComputationalBackends.AbstractThreadedBackend, args...) = throw(ArgumentError("ThreadedBackend is not loaded. Run `using OhMyThreads`."))

# GPU direct-sum (KernelAbstractions ext; portable on any KA device, incl. KA.CPU()).
_gpu_directsum_cartesian(args...; kwargs...) = throw(ArgumentError("GPUBackend is not loaded. Run `using KernelAbstractions`."))
_gpu_directsum_spherical(args...; kwargs...) = throw(ArgumentError("GPUBackend is not loaded. Run `using KernelAbstractions`."))
# Fast device-generic structured SHT (φ-DFT + θ-Legendre contraction), KernelAbstractions ext.
_calculate_spectrum_gpu_sht(args...; kwargs...) = throw(ArgumentError("GPU SHT requires `using KernelAbstractions`."))

# GPU FFT is device-generic via AbstractFFTs; the GPU NUFFT is `NUFFT.jl` on FlowTransformBindings' tags.
_calculate_spectrum_gpu_fft(args...; kwargs...) = throw(ArgumentError("GPU FFT requires `using KernelAbstractions` plus an AbstractFFTs provider (`FFTW` for `KA.CPU()`, `CUDA` for `CUDABackend`, …)."))
_calculate_spectrum_gpu_nufft(t::SpectralBackends.AbstractNUFFTSpectralBackend, args...; kwargs...) = throw(ArgumentError(
    "$(nameof(typeof(t))) names no NUFFT library — pass transform=FlowTransformBindings.NonuniformFFTsBackend() " *
    "(`using NonuniformFFTs`) or FlowTransformBindings.FINUFFTBackend() (`using FINUFFT`, `using CUDA` for CuArrays)."))

# The hybrid composite's passes, held by `HybridPlan` and `HybridSynthesisPlan`: `_region_fft_plan` builds
# the FFT over the uniform axes on the execution backend's arrays (a `RegionFFT`), `_region_bfft_plan(exec,
# A, dims, iflag, measure)` the synthesis's in-place backward FFT over them on `A`, and `_axis_nufft_plan`
# one stretched axis's 1-D NUFFT with its own buffers.
_region_fft_plan(::_HostExec, args...) = throw(ArgumentError(
    "The hybrid FFT/NUFFT transform needs an FFT provider. Run `using FFTW`."))
_region_fft_plan(::ComputationalBackends.AbstractGPUBackend, args...) = throw(ArgumentError(
    "The hybrid FFT/NUFFT transform on a GPUBackend takes its FFT through AbstractFFTs. Run $_DEVICE_FFT."))
_region_bfft_plan(exec::ComputationalBackends.AbstractExecutionBackend, args...) = _region_fft_plan(exec)
_axis_nufft_plan(t::SpectralBackends.AbstractSpectralBackend, args...; kwargs...) = throw(ArgumentError(
    "The hybrid FFT/NUFFT transform needs a NUFFT library for the stretched axes (got " *
    "$(nameof(typeof(t)))). Run `using NonuniformFFTs` or `using FINUFFT`, and pass " *
    "`nufft = FlowTransformBindings.NonuniformFFTsBackend()` or `FlowTransformBindings.FINUFFTBackend()`."))

# CPU transform libraries (extensions). A NUFFT names its library: the abstract NUFFTSpectralBackend names none.
_calculate_spectrum_fft(args...; kwargs...) = throw(ArgumentError("FFTSpectralBackend is not loaded. Run `using FFTW`."))
_calculate_spectrum_nufft(t::SpectralBackends.AbstractNUFFTSpectralBackend, args...; kwargs...) = throw(ArgumentError(
    "$(nameof(typeof(t))) names no NUFFT library — pass transform=FlowTransformBindings.NonuniformFFTsBackend() " *
    "(`using NonuniformFFTs`) or FlowTransformBindings.FINUFFTBackend() (`using FINUFFT`)."))
_calculate_spectrum_sht(args...; kwargs...) = throw(ArgumentError("FSHTSpectralBackend is not loaded. Run `using FastSphericalHarmonics`."))
_calculate_spectrum_nufsht(args...; kwargs...) = throw(ArgumentError("NUFSHTSpectralBackend is not loaded. Run `using NUFSHT`."))

# Distribution wrappers (Distributed / MPI extensions).
_calculate_spectrum_distributed(args...; kwargs...) = throw(ArgumentError("DistributedBackend is not loaded. Run `using Distributed` (and `addprocs`)."))
_calculate_spectrum_mpi(args...; kwargs...) = throw(ArgumentError("MPIBackend is not loaded. Run `using MPI` and launch under `mpiexec`."))
_plan_spectrum_distributed(args...; kwargs...) = _calculate_spectrum_distributed()
_plan_spectrum_mpi(args...; kwargs...) = _calculate_spectrum_mpi()
_plan_synthesis_distributed(args...; kwargs...) = _calculate_spectrum_distributed()
_plan_synthesis_mpi(args...; kwargs...) = _calculate_spectrum_mpi()

# Plans and synthesis divided among Distributed workers or MPI ranks, each holding the local plan of its
# share (see `_point_partition`).
Plans.plan_spectrum(t::SpectralBackends.AbstractSpectralBackend, e::ComputationalBackends.AbstractDistributedBackend,
    g::FlowGeometries.Grids.AbstractGrid, ::Type{T}, ms::Tuple; kwargs...) where {T} =
    _plan_spectrum_distributed(t, e, g, T, ms; _whole_grid_inactive(t, g, kwargs)...)
Plans.plan_spectrum(t::SpectralBackends.AbstractSpectralBackend, e::ComputationalBackends.AbstractMPIBackend,
    g::FlowGeometries.Grids.AbstractGrid, ::Type{T}, ms::Tuple; kwargs...) where {T} =
    _plan_spectrum_mpi(t, e, g, T, ms; _whole_grid_inactive(t, g, kwargs)...)
Plans.plan_synthesis(t::SpectralBackends.AbstractSpectralBackend, e::ComputationalBackends.AbstractDistributedBackend,
    g::FlowGeometries.Grids.AbstractGrid, ::Type{T}, ms::Tuple; kwargs...) where {T} =
    _plan_synthesis_distributed(t, e, g, T, ms; kwargs...)
Plans.plan_synthesis(t::SpectralBackends.AbstractSpectralBackend, e::ComputationalBackends.AbstractMPIBackend,
    g::FlowGeometries.Grids.AbstractGrid, ::Type{T}, ms::Tuple; kwargs...) where {T} =
    _plan_synthesis_mpi(t, e, g, T, ms; kwargs...)

# =============================================================================
# Distribution partition helpers (shared by the Distributed / MPI extensions). Complex coefficients
# are additive over a disjoint point partition; `α_w` compensates the per-kernel `1/N_local`.
# =============================================================================

# Node-cloud subgrid over the point indices `idx` (see `_is_scattered`), keeping the geometry,
# periodicity, per-node measure and sliced mask.
function _subgrid(g::Union{FlowGeometries.Grids.AbstractUnstructuredGrid,
            FlowGeometries.Grids.AbstractCurvilinearGrid}, idx)
    geom = FlowGeometries.Grids.grid_geometry(g)
    D = length(FlowGeometries.Grids.coordinates(g))
    coords = ntuple(d -> view(FlowGeometries.Grids.coordinates(g, d), idx), D)
    meas = view(FlowGeometries.Grids.measure_array(g), idx)
    per = FlowGeometries.Grids.periodic_flags(g)
    prd = ntuple(d -> FlowGeometries.Grids.period(g, d), D)
    return FlowGeometries.Grids.UnstructuredGrid(geom, coords, meas,
        _submask(FlowGeometries.Grids.mask(g), idx); periodic = per, period = prd)
end

_submask(::FlowGeometries.Grids.AllActive, idx) = FlowGeometries.Grids.AllActive((length(idx),))
_submask(mask::AbstractArray{Bool}, idx) = vec(mask)[idx]

"""
    _partition_weight(grid, idx, weights) -> Real

The weight the coefficients of the points `idx` carry in `C = Σ_w α_w C_w` over a disjoint point
partition. A transform on those points normalizes by their own measure total `W_w` (the Cartesian
quadrature `Σ wⱼfⱼe^{-ik·x}/Σw`, the spherical weights rescaled to `4π`), so `α_w = W_w/W`. With explicit
per-node `weights` nothing is normalized and `α_w = 1`.
"""
function _partition_weight(g::FlowGeometries.Grids.AbstractGrid, idx, ::Nothing)
    m = FlowGeometries.Grids.measure_array(g)
    return sum(view(m, idx)) / sum(m)
end
_partition_weight(::FlowGeometries.Grids.AbstractGrid, idx, weights) = 1

# A transform's keywords restricted to the points `idx` of `g`: explicit per-node `weights` sliced to
# them, and on a Cartesian grid the whole grid's origins and Fourier lengths as `box`, so every share's
# coefficients sit on the whole grid's wavenumbers and origin.
function _partition_kwargs(kwargs, g, idx)
    kw = NamedTuple(kwargs)
    w = get(kw, :weights, nothing)
    return merge(w === nothing ? kw : merge(kw, (; weights = w[idx])), _partition_box(g))
end

_partition_box(g::FlowGeometries.Grids.AbstractGrid{<:FlowGeometries.Geometry.AbstractCartesianGeometry}) =
    (; box = Grids.axis_geometry(eltype(g), g, FlowGeometries.Grids.ncoordinates(g)))
_partition_box(::FlowGeometries.Grids.AbstractGrid) = (;)

# Transforms whose coefficients are additive over a disjoint point partition.
_partitionable(::SpectralBackends.AbstractDirectSumSpectralBackend) = true
_partitionable(::SpectralBackends.AbstractNUFFTSpectralBackend) = true
_partitionable(::SpectralBackends.AbstractNUFSHTSpectralBackend) = true
_partitionable(::SpectralBackends.AbstractSpectralBackend) = false

# A grid whose cells carry their own coordinates admits a disjoint point partition. A curvilinear grid
# does too: its coordinate arrays hold one value per cell and index linearly, so a subset of its cells is
# a node cloud.
_is_scattered(::FlowGeometries.Grids.AbstractUnstructuredGrid) = true
_is_scattered(::FlowGeometries.Grids.AbstractCurvilinearGrid) = true
_is_scattered(::FlowGeometries.Grids.AbstractGrid) = false

"""
    _point_partition(transform, grid, kwargs) -> Bool

Whether a transform divided among processes splits the points of `grid` (each share's coefficients
weighted by `_partition_weight` and summed) or its batch columns (each share transforming whole
fields). The points split where the transform is additive over them and the grid is a point set; a
NUFSHT solve couples every point, so it splits the batch.
"""
_point_partition(t::SpectralBackends.AbstractSpectralBackend, g, kwargs) =
    _partitionable(t) && _is_scattered(g) && !_couples_points(t, kwargs)
_couples_points(::SpectralBackends.AbstractNUFSHTSpectralBackend, kwargs) = get(kwargs, :solve, false)
_couples_points(::SpectralBackends.AbstractSpectralBackend, kwargs) = false

# A synthesis evaluates the series at each point on its own, so it splits the points of a point set for
# any partitionable transform.
_point_partition_synthesis(t::SpectralBackends.AbstractSpectralBackend, g) = _partitionable(t) && _is_scattered(g)

# Spatial (spectral) shape of the coefficient array for a grid + `ms`, given whether the field is real:
# the coefficient box, axis 1 halved for a real Cartesian field.
function _coeff_spatial(g::FlowGeometries.Grids.AbstractGrid{<:FlowGeometries.Geometry.AbstractCartesianGeometry},
        ms, R::Bool)
    n = NTuple{length(ms), Int}(Tuple(ms))
    return FTB.packed_size(Grids.mode_box(g, n, Val(R)), Val(R))
end
_coeff_spatial(::FlowGeometries.Grids.AbstractGrid{<:Grids.SphericalHarmonicGeometry}, ms, ::Bool) = (ms[1], 2 * (ms[1] - 1) + 1)

# Element type of the coefficient array for a grid + field element type, without running a transform: a
# distributed rank holding no batch columns still allocates its share of the gather buffer.
_coeff_eltype(::FlowGeometries.Grids.AbstractGrid{<:FlowGeometries.Geometry.AbstractCartesianGeometry},
    ::Type{T}, ::Type{FT}) where {T, FT} = Complex{FT}
_coeff_eltype(::FlowGeometries.Grids.AbstractGrid{<:Grids.SphericalHarmonicGeometry},
    ::Type{T}, ::Type{FT}) where {T, FT} = sph_coeff_type(T, FT)

# Wavenumber coordinate returned alongside the coefficients.
_partition_ks(g::FlowGeometries.Grids.AbstractGrid{<:FlowGeometries.Geometry.AbstractCartesianGeometry}, ms, R::Bool) =
    physical_wavenumbers(g, NTuple{length(ms), Int}(Tuple(ms)), Val(R))
_partition_ks(::FlowGeometries.Grids.AbstractGrid{<:Grids.SphericalHarmonicGeometry}, ms, ::Bool) = (0:(ms[1] - 1), -(ms[1] - 1):(ms[1] - 1))

include("NUFFT.jl")

Plans.close!(p::Union{HybridPlan, HybridSynthesisPlan}) = (foreach(_close!, p.axes); nothing)

# =============================================================================
# Plotting stubs (CairoMakie extension).
# =============================================================================

"""
    plot_spectrum(k_ranges::Tuple, coeffs; title="Power Spectrum", u_idx=1, kwargs...)

Plot the coefficient magnitude of batch slice `u_idx`, as a line in 1D and a heatmap in 2D. Requires
`using CairoMakie`.
"""
plot_spectrum(args...; kwargs...) = throw(ArgumentError("plot_spectrum requires CairoMakie. Run `using CairoMakie`."))

"""
    compare_spectra(results; u_idx=1, title="Spectral Analysis Comparison", kwargs...)

Plot several spectra side by side, `results` an iterable of `label => (coeffs, k_ranges)`. Requires
`using CairoMakie`.
"""
compare_spectra(args...; kwargs...) = throw(ArgumentError("compare_spectra requires CairoMakie. Run `using CairoMakie`."))

"""
    compare_spectral_analysis(input_data, results; u_idx=1, title="Spectral Analysis Comparison", kwargs...)

[`compare_spectra`](@ref) with the input field `input_data = (x_coords, field)` plotted first. Requires
`using CairoMakie`.
"""
compare_spectral_analysis(args...; kwargs...) = throw(ArgumentError("compare_spectral_analysis requires CairoMakie. Run `using CairoMakie`."))

# =============================================================================
# Precompilation workload (DirectSum + reductions on the array/batch API).
# =============================================================================
PrecompileTools.@setup_workload begin
    T = Float64
    PrecompileTools.@compile_workload begin
        # Cartesian 2D DirectSum on a uniform grid with a small batch.
        ax = range(zero(T), T(10); length = 9)[1:8]
        cart = FlowGeometries.Grids.StructuredGrid(FlowGeometries.Geometry.CartesianGeometry{T}(), ax, ax;
            periodic = (true, true), period = (T(10), T(10)))
        u = rand(T, 8, 8, 2)
        c, k = calculate_spectrum(cart, u, (4, 4); transform = SpectralBackends.DirectSumSpectralBackend(), execution = ComputationalBackends.SerialBackend())
        isotropic_spectrum(k, c; num_bins = 2)
        transect_spectrum(k, c, (1,))
        # No extension is loaded during precompilation, so every call in this workload names its
        # transform and the fallback warning stays out of the build log.
        synthesize(cart, c, (4, 4); transform = SpectralBackends.DirectSumSpectralBackend(),
            execution = ComputationalBackends.SerialBackend())

        # Spherical 2D DirectSum on scattered (θ, φ) points, stored as FlowGeometries (λ, φ_lat).
        theta = rand(T, 8) .* π
        phi = rand(T, 8) .* 2π
        sph = FlowGeometries.Grids.UnstructuredGrid(FlowGeometries.Geometry.SphericalGeometry(one(T)),
            (phi, T(π) / 2 .- theta), ones(T, 8))
        fθ = rand(T, 8)
        cs, ks = calculate_spectrum(sph, fθ, (2, 3); transform = SpectralBackends.DirectSumSpectralBackend(), execution = ComputationalBackends.SerialBackend())
        spherical_energy_spectrum(cs; lmax = 1)
    end
end

end # module FlowFieldSpectra
