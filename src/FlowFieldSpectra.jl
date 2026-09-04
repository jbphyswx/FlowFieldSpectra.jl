module FlowFieldSpectra

using PrecompileTools: PrecompileTools
using ComputationalBackends: ComputationalBackends
using SpectralBackends: SpectralBackends
using FlowGeometries: FlowGeometries

include("Packing.jl")
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

using .Packing: Packing, unpacked, unpacked!
using .Grids: Grids, physical_wavenumbers
using .Preprocessing: AbstractWindow, NoWindow, Hann, Hamming, Blackman, Tukey, AbstractDetrend, NoDetrend, Demean, LinearDetrend, Preprocess, dpss
using .Normalization: AbstractSidedness, OneSided, TwoSided, AbstractScaling, DensityScaling, PowerScaling, SpectralConvention
using .Problem: Problem, TransformProblem, batch_shape, stack_fields
using .Plans: Plans, AbstractSpectralPlan, plan_spectrum
using .DirectSum: DirectSum, sph_mode_index
using .Reductions: isotropic_spectrum, isotropic_spectrum!, transect_spectrum, transect_spectrum!, spherical_energy_spectrum, spherical_energy_spectrum!, cross_spectrum, cospectrum, quadspectrum, anisotropic_spectrum
using .Operators: spectral_divergence, spectral_divergence!, spectral_vorticity, spectral_vorticity!,
    compensate, band_energy
using .Averaging: welch_power_spectrum, welch_power_spectrum!, coherence_spectrum, coherence_spectrum!
using .LombScargle: lomb_scargle, lomb_scargle!


# Preprocessing & Normalization (typed configuration)
export AbstractWindow, NoWindow, Hann, Hamming, Blackman, Tukey
export AbstractDetrend, NoDetrend, Demean, LinearDetrend, Preprocess, dpss
export AbstractSidedness, OneSided, TwoSided, AbstractScaling, DensityScaling, PowerScaling, SpectralConvention
export TransformProblem
export AbstractSpectralPlan, plan_spectrum, preprocess_field
# APIs
export calculate_spectrum, calculate_spectrum!, synthesize, isotropic_spectrum, isotropic_spectrum!, transect_spectrum, transect_spectrum!, spherical_energy_spectrum, spherical_energy_spectrum!, sph_mode_index, sph_coeff_type
export unpacked, unpacked!
export spectral_divergence, spectral_divergence!, spectral_vorticity, spectral_vorticity!,
    compensate, band_energy
export cross_spectrum, cospectrum, quadspectrum, anisotropic_spectrum
export welch_power_spectrum, welch_power_spectrum!, coherence_spectrum, coherence_spectrum!,
    lomb_scargle, lomb_scargle!
export plot_spectrum, compare_spectra, compare_spectral_analysis

"""
    sph_coeff_type(::Type{T}, ::Type{FT}) -> Type

Element type of a spherical coefficient array for a field of element type `T` on a grid of element type
`FT`: `FT` for a real field and `Complex{FT}` for a complex one.

The basis is the real spherical harmonics, so a real field's coefficients are real and a real array holds
them exactly. This mirrors the Cartesian rule, where a real field's coefficients are the packed Hermitian
half and a complex field's the full native cube.
"""
sph_coeff_type(::Type{T}, ::Type{FT}) where {T <: Real, FT} = FT
sph_coeff_type(::Type{T}, ::Type{FT}) where {T, FT} = Complex{FT}

"""
    preprocess_field(grid, field, spec::Preprocess) -> (grid′, field′, ms_scale)

`field` detrended, tapered and zero-padded per `spec`, with the grid and mode-count scaling the padding
implies. Returns `(grid, field, 1)` unchanged for a spec that is the identity, so an unpreprocessed call
copies nothing.

The caller's array is never mutated: a spec with anything to do works on a copy.

`spec.window` tapers each spatial axis (the tensor product of [`Preprocessing.axis_taper`](@ref)) and
`spec.pad > 1` extends each axis with zeros. Both read the grid's axes, so both apply to a structured
grid; a node cloud has no axis to taper or extend along, and asking for either raises. `spec.detrend`
needs no axes for `Demean`, so that applies to any grid.

Preprocessing lives outside [`plan_spectrum`](@ref): a plan is fixed by the grid and the resolution while
a taper is an estimator choice, and multitaper reuses ONE plan across `K` differently tapered copies of
the same field transformed as a batch.
"""
preprocess_field(g::FlowGeometries.Grids.AbstractGrid, field::AbstractArray, ::Nothing) = (g, field, 1)

function preprocess_field(g::FlowGeometries.Grids.AbstractGrid, field::AbstractArray,
        spec::Preprocess)
    Preprocessing.is_identity(spec) && return (g, field, 1)
    nsp = ndims(g)
    out = similar(field, float(eltype(field)))
    copyto!(out, field)
    if !(spec.detrend isa NoDetrend)
        spec.detrend isa LinearDetrend && _require_axes(g, "LinearDetrend")
        Preprocessing.detrend_spatial!(out, spec.detrend, nsp)
    end
    if !(spec.window isa NoWindow)
        _require_axes(g, "a window taper")
        FT = real(float(eltype(field)))
        tapers = ntuple(d -> Preprocessing.axis_taper(FT, spec.window, size(g, d)), nsp)
        Preprocessing.apply_window!(out, tapers, nsp)
    end
    spec.pad == 1 && return (g, out, 1)
    _require_axes(g, "zero padding")
    return _padded(g, out, spec.pad)
end

# A window, a linear detrend and zero padding are all axis-based, so they need a grid that HAS axes.
_require_axes(g::FlowGeometries.Grids.AbstractStructuredGrid, what::String) = nothing
_require_axes(g::FlowGeometries.Grids.AbstractGrid, what::String) = throw(ArgumentError(
    "$what reads the grid's per-direction axes, and a $(nameof(typeof(g))) stores one coordinate per " *
    "point. Detrend with `Demean()` (which needs no axes), or resample onto a structured grid."))

# Zero-pad each axis by `pad`, extending it at its own spacing so the samples stay uniform where they
# were. The mode count scales with the length, so the returned `ms_scale` carries `pad` to the caller's
# `ms` and the wavenumber spacing narrows by the same factor.
function _padded(g::FlowGeometries.Grids.AbstractStructuredGrid{<:FlowGeometries.Geometry.AbstractCartesianGeometry},
        field::AbstractArray, pad::Float64)
    D = ndims(g)
    FT = eltype(g)
    ns = ntuple(d -> size(g, d), D)
    np = ntuple(d -> max(ns[d], round(Int, pad * ns[d])), D)
    axs = ntuple(D) do d
        ax = FlowGeometries.Grids.coordinates(g, d)
        ns[d] >= 2 || throw(ArgumentError(
            "axis $d has $(ns[d]) sample(s); padding reads a spacing, so it needs at least 2"))
        FlowGeometries.Grids.isuniform(g, d) || throw(ArgumentError(
            "zero padding extends an axis at its own spacing, so it needs a uniform axis; axis $d of " *
            "this grid is stretched."))
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

"""
    quadrature_weighted(grid, field) -> field

`field` scaled by the grid's per-node quadrature factor (see [`Grids.quadrature_scale`](@ref)), so a
backend's node-count normalization becomes the measure-weighted quadrature over `grid`. Returns `field`
itself where the measure is constant, which is every uniform grid and every scattered grid whose nodes
carry equal weight. The scaled copy is what both the coefficients and the Nyquist twin are built from,
so they stay consistent.
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
# Execution-backend resolution (FFS-owned; ComputationalBackends deliberately errors on `AutoBackend`
# so each consumer resolves it locally). Best available LOCAL backend: `ThreadedBackend` iff the
# OhMyThreads extension is loaded and `Threads.nthreads() > 1`, else `SerialBackend`. GPU/Distributed/
# MPI are never auto-selected (they need explicit device/process context).
# =============================================================================
"""
    _backend_nthreads(exec) -> Int

Thread count for a library plan built under `exec`: `Threads.nthreads()` on a threaded backend, `1`
otherwise.

Every provider is handed this explicit count, so the thread budget follows the `-t` Julia was started
with. Several of them (FINUFFT, and NUFSHT's inner FINUFFT) read `0` as "every core on the machine",
which is a different quantity.
"""
_backend_nthreads(::ComputationalBackends.AbstractThreadedBackend) = Threads.nthreads()
_backend_nthreads(::ComputationalBackends.AbstractExecutionBackend) = 1

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
# Transform resolution. `AutoSpectralBackend` picks the fastest transform whose preconditions this grid
# meets and whose extension is loaded; a concrete tag passes through untouched.
#
# Auto substitutes only a backend computing the SAME coefficients: the FFT is exact against the direct
# sum, and a NUFFT / NUFSHT matches it to its requested tolerance. Every direct-sum path is `O(∏ms · N)`
# on a Cartesian grid and `O(L³)` on a sphere, so Auto reaches for a library wherever one applies and
# names the direct sum only where nothing else is loaded.
#
# An FFT transforms a uniform axis in full, so it applies only where `ms` asks for the grid's own length
# there; the hybrid additionally needs axis 1 uniform. Both conditions are checked here, so every backend
# Auto returns has its preconditions met.
# =============================================================================

_ext_loaded(name::Symbol) = Base.get_extension(@__MODULE__, name) !== nothing

# A distributed run's transform is chosen for the backend each worker runs locally.
_auto_exec(e::ComputationalBackends.AbstractExecutionBackend) = e
_auto_exec(e::ComputationalBackends.AbstractDistributedBackend) = ComputationalBackends.local_backend(e)
_auto_exec(e::ComputationalBackends.AbstractMPIBackend) = ComputationalBackends.local_backend(e)

# The NUFFT provider available for this execution backend, or `nothing`.
_auto_nufft(::ComputationalBackends.AbstractExecutionBackend) =
    _ext_loaded(:FlowFieldSpectraNonuniformFFTsExt) ? NonuniformFFTsBackend() :
    _ext_loaded(:FlowFieldSpectraFINUFFTExt) ? FINUFFTBackend() : nothing
_auto_nufft(::ComputationalBackends.AbstractGPUBackend) =
    _ext_loaded(:FlowFieldSpectraNonuniformFFTsKernelAbstractionsExt) ? NonuniformFFTsBackend() :
    _ext_loaded(:FlowFieldSpectracuFINUFFTExt) ? FINUFFTBackend() : nothing

_auto_fft(::ComputationalBackends.AbstractExecutionBackend) = _ext_loaded(:FlowFieldSpectraFFTWExt)
_auto_fft(::ComputationalBackends.AbstractGPUBackend) = _ext_loaded(:FlowFieldSpectraGPUFFTExt)

# The hybrid composite's FFT pass is `_region_fft`, which the FFTW extension provides for a host backend.
_auto_hybrid(::ComputationalBackends.AbstractExecutionBackend) = _ext_loaded(:FlowFieldSpectraFFTWExt)
_auto_hybrid(::ComputationalBackends.AbstractGPUBackend) = false

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
    umask[1] && !all(umask) && full && _auto_hybrid(exec) && nu !== nothing &&
        return SpectralBackends.FFTSpectralBackend()
    nu === nothing || return nu
    return _auto_fallback(g, "using FFTW, NonuniformFFTs or FINUFFT")
end

_auto_transform(g::Grids.PointwiseCartesian, ms::Tuple, exec) =
    (nu = _auto_nufft(exec); nu === nothing ? _auto_fallback(g, "using NonuniformFFTs or FINUFFT") : nu)

# The direct sum's spherical paths cost `O(L³)` even factorized — the longitude DFT is
# `(lmax+1)·nlon·nlat` and the θ-Legendre contraction `nlat·L²` — against `O(L² log L)` for a library
# transform, so Auto prefers a library on EVERY spherical layout. FSHT is exact on the one grid it
# validates; NUFSHT serves any spherical point set, tensor and ring grids included.
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

Spectral coefficients and physical wavenumbers of `field` sampled on `grid` (a FlowGeometries grid).
The grid's architecture × geometry *is* the coordinate system — no coordinate guessing. The two
backend axes compose freely:

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

# Preprocessing
`preprocess::Preprocess` detrends, tapers and zero-pads the field first (see
[`preprocess_field`](@ref)); omitting it transforms the field as given, copying nothing. A taper is
scaled to unit mean square, so Parseval holds on the returned coefficients with no correction applied
afterwards. `pad > 1` lengthens the grid's axes and scales `ms` by the same factor, so the wavenumber
spacing narrows.

# Returns
`(coeffs, ks_phys)` — the coefficients and the physical wavenumber coordinates per spectral axis.

The field's element type picks the coefficient layout on both geometries. **Cartesian**: a real field
gives the rfft-packed half `(m_1÷2+1, m_2…, batch…)`, a complex field the full native cube
`(ms…, batch…)`; both complex-valued. **Spherical**: `(Nθ, Nφ, batch…)`, REAL for a real field (the basis
is the real spherical harmonics, so the coefficients are real) and complex for a complex one — see
[`sph_coeff_type`](@ref).
"""
function calculate_spectrum(grid::FlowGeometries.Grids.AbstractGrid, field::AbstractArray, ms::Tuple;
        transform::SpectralBackends.AbstractSpectralBackend = SpectralBackends.AutoSpectralBackend(),
        execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
        preprocess::Union{Nothing, Preprocess} = nothing, kwargs...)
    e = _resolve_execution(execution)
    # Preprocessing may replace the grid (zero padding lengthens its axes), and the mode counts scale
    # with it, so the transform runs against what comes back.
    gp, fp, scale = preprocess_field(grid, field, preprocess)
    msp = scale == 1 ? ms : map(m -> max(1, round(Int, scale * m)), ms)
    return calculate_spectrum(_resolve_transform(transform, gp, msp, e), e, gp, fp, msp; kwargs...)
end

# Multi-field convenience: stack equal-shaped arrays along a new trailing batch axis.
calculate_spectrum(grid::FlowGeometries.Grids.AbstractGrid, fields::Tuple, ms::Tuple; kwargs...) =
    calculate_spectrum(grid, stack_fields(fields), ms; kwargs...)

# =============================================================================
# Canonical two-axis dispatch: calculate_spectrum(transform, execution, grid, field, ms; …).
# =============================================================================

"""
    FINUFFTBackend()
    NonuniformFFTsBackend()

The concrete NUFFT providers (`<: SpectralBackends.AbstractNUFFTSpectralBackend`). A provider is a
*library* choice, not spectral math, so these live in FlowFieldSpectra, not SpectralBackends. They are
symmetric — neither is a default: `FINUFFTBackend()` uses FINUFFT (`using FINUFFT`; GPU via cuFINUFFT),
`NonuniformFFTsBackend()` uses NonuniformFFTs (`using NonuniformFFTs`; real-data fast path, half memory).
Both may be loaded at once. The abstract `SpectralBackends.NUFFTSpectralBackend()` selects no provider —
pass one of these.
"""
struct FINUFFTBackend <: SpectralBackends.AbstractNUFFTSpectralBackend end

"See [`FINUFFTBackend`](@ref) — the NonuniformFFTs.jl NUFFT provider."
struct NonuniformFFTsBackend <: SpectralBackends.AbstractNUFFTSpectralBackend end

# ---- Level 1: distribution wrappers unwrap to the Distributed / MPI extension hooks ----
calculate_spectrum(t::SpectralBackends.AbstractSpectralBackend, e::ComputationalBackends.AbstractDistributedBackend, g::FlowGeometries.Grids.AbstractGrid, field::AbstractArray, ms::Tuple; kwargs...) =
    _calculate_spectrum_distributed(t, e, g, field, ms; kwargs...)
calculate_spectrum(t::SpectralBackends.AbstractSpectralBackend, e::ComputationalBackends.AbstractMPIBackend, g::FlowGeometries.Grids.AbstractGrid, field::AbstractArray, ms::Tuple; kwargs...) =
    _calculate_spectrum_mpi(t, e, g, field, ms; kwargs...)

# ---- Level 2: DirectSum on a local CPU backend (core Serial; OhMyThreads ext Threaded) ----
function calculate_spectrum(::SpectralBackends.AbstractDirectSumSpectralBackend,
        exec::Union{ComputationalBackends.AbstractSerialBackend, ComputationalBackends.AbstractThreadedBackend},
        g::FlowGeometries.Grids.AbstractGrid{<:FlowGeometries.Geometry.AbstractCartesianGeometry},
        field::AbstractArray, ms::NTuple{D, Int}; iflag::Int = 1, kwargs...) where {D}
    prob = TransformProblem(g, field)
    pms = Packing.packed_size(ms, Val(eltype(field) <: Real))
    coeffs = zeros(Complex{eltype(g)}, pms..., batch_shape(prob)...)
    ks = _directsum_cartesian!(exec, coeffs, g, quadrature_weighted(g, field), ms, iflag)
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
calculate_spectrum(::SpectralBackends.AbstractDirectSumSpectralBackend, exec::ComputationalBackends.AbstractGPUBackend, g::FlowGeometries.Grids.AbstractGrid{<:FlowGeometries.Geometry.AbstractCartesianGeometry}, field::AbstractArray, ms::NTuple{D, Int}; iflag::Int = 1, kwargs...) where {D} =
    _gpu_directsum_cartesian(exec, g, quadrature_weighted(g, field), ms, iflag)
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
    any(umask) || throw(ArgumentError(
        "FFTSpectralBackend needs a uniform axis in at least one direction (an AbstractRange); every axis " *
        "of this grid is stretched. Use NUFFTSpectralBackend or DirectSumSpectralBackend."))
    return _calculate_spectrum_hybrid(_resolve_nufft_provider(nufft), exec, g, field,
        NTuple{D, Int}(Tuple(ms)), umask; kwargs...)
end
function calculate_spectrum(::SpectralBackends.AbstractFFTSpectralBackend, exec::ComputationalBackends.AbstractGPUBackend, g::FlowGeometries.Grids.AbstractStructuredGrid{<:FlowGeometries.Geometry.AbstractCartesianGeometry}, field::AbstractArray, ms::Tuple;
        nufft::SpectralBackends.AbstractSpectralBackend = SpectralBackends.AutoSpectralBackend(), kwargs...)
    D = ndims(g)
    umask = _uniform_mask(g, Val(D))
    all(umask) && return _calculate_spectrum_gpu_fft(exec, g, field, ms; kwargs...)
    any(umask) || throw(ArgumentError(
        "FFTSpectralBackend needs a uniform axis in at least one direction (an AbstractRange); every axis " *
        "of this grid is stretched. Use NUFFTSpectralBackend or DirectSumSpectralBackend."))
    return _calculate_spectrum_hybrid(_resolve_nufft_provider(nufft), exec, g, field,
        NTuple{D, Int}(Tuple(ms)), umask; kwargs...)
end

# ---- Level 2: NUFFT (FINUFFT ext CPU; cuFINUFFT ext on CUDA). Structured (nonuniform-gridded) uses
# the separable per-axis 1-D NUFFT; unstructured (scattered) uses the guru NUFFT. Both are
# `_calculate_spectrum_nufft`, dispatched on the grid architecture inside the extension. ----
calculate_spectrum(t::SpectralBackends.AbstractNUFFTSpectralBackend, exec::Union{ComputationalBackends.AbstractSerialBackend, ComputationalBackends.AbstractThreadedBackend}, g::Union{FlowGeometries.Grids.AbstractStructuredGrid{<:FlowGeometries.Geometry.AbstractCartesianGeometry}, Grids.PointwiseCartesian}, field::AbstractArray, ms::Tuple; kwargs...) =
    _calculate_spectrum_nufft(t, exec, g, field, ms; kwargs...)
calculate_spectrum(t::SpectralBackends.AbstractNUFFTSpectralBackend, exec::ComputationalBackends.AbstractGPUBackend, g::Union{FlowGeometries.Grids.AbstractStructuredGrid{<:FlowGeometries.Geometry.AbstractCartesianGeometry}, Grids.PointwiseCartesian}, field::AbstractArray, ms::Tuple; kwargs...) =
    _calculate_spectrum_gpu_nufft(t, exec, g, field, ms; kwargs...)

# ---- Level 2: SHT / NUFSHT (execution axis handled inside the extension) ----
calculate_spectrum(::SpectralBackends.AbstractFSHTSpectralBackend, ::Union{ComputationalBackends.AbstractSerialBackend, ComputationalBackends.AbstractThreadedBackend}, g::FlowGeometries.Grids.AbstractStructuredGrid{<:FlowGeometries.Geometry.AbstractSphericalGeometry}, field::AbstractArray, ms::Tuple; kwargs...) =
    _calculate_spectrum_sht(g, field, ms; kwargs...)
# GPU SHT: a fast device-generic transform (φ-DFT + θ-Legendre contraction) in the KernelAbstractions
# ext — no FastTransforms, runs on any KA device. Structured grid only.
calculate_spectrum(::SpectralBackends.AbstractFSHTSpectralBackend, exec::ComputationalBackends.AbstractGPUBackend, g::FlowGeometries.Grids.AbstractStructuredGrid{<:FlowGeometries.Geometry.AbstractSphericalGeometry}, field::AbstractArray, ms::Tuple; kwargs...) =
    _calculate_spectrum_gpu_sht(exec, g, field, ms; kwargs...)
calculate_spectrum(::SpectralBackends.AbstractNUFSHTSpectralBackend, exec::ComputationalBackends.AbstractExecutionBackend, g::FlowGeometries.Grids.AbstractGrid{<:Grids.SphericalHarmonicGeometry}, field::AbstractArray, ms::Tuple; kwargs...) =
    _calculate_spectrum_nufsht(exec, g, field, ms; kwargs...)

# =============================================================================
# Hybrid per-axis transform: one FFT over the uniform axes, one 1-D type-1 NUFFT along each stretched
# axis, every other spatial and batch dim riding as that pass's batch. A uniform-xy / stretched-z grid
# therefore pays the NUFFT only along z.
#
# Axis 1 must be uniform. The packed publish is then a plain leading slice: a native even axis carries
# `−N₁/2` where the packed half wants `+N₁/2`, and those are the same coefficient under uniform sampling.
# The FFT pass is COMPLEX (not `rfft`) whenever a Nyquist twin is needed, keeping the full `k₁` range
# present so each twin is a conjugate read of a native mode; a halved axis 1 holds no `−k₁` to read.
# Where no twin is needed the pass is an `rfft` and halves axis 1 directly.
#
# Every pass leaves the raw transform in native order; the grid-offset phase (zero on the FFT axes, whose
# points are not rescaled) and the single `1/∏N_d` normalization are applied once at the end, through the
# same `Packing.offset_phase` / `publish_packed!` / `conj_twins` the separable NUFFT path uses. The
# per-axis sign is the `Σ e^{-ikx}` convention for both providers, so `iflag = -1` is one conjugation of
# the whole product.
# =============================================================================

# Per-axis uniformity. `isuniform` answers from the axis TYPE, and the `Val` length makes the result an
# `NTuple{D,Bool}`, so the all-uniform and mixed branches below resolve without a runtime-length tuple.
_uniform_mask(g, ::Val{D}) where {D} = ntuple(d -> FlowGeometries.Grids.isuniform(g, d), Val(D))

# The provider that transforms the stretched axes. `AutoSpectralBackend` takes whichever NUFFT extension
# is loaded; an explicit provider is honoured as given.
_resolve_nufft_provider(t::SpectralBackends.AbstractNUFFTSpectralBackend) = t
function _resolve_nufft_provider(::SpectralBackends.AbstractAutoSpectralBackend)
    Base.get_extension(@__MODULE__, :FlowFieldSpectraNonuniformFFTsExt) === nothing || return NonuniformFFTsBackend()
    Base.get_extension(@__MODULE__, :FlowFieldSpectraFINUFFTExt) === nothing || return FINUFFTBackend()
    throw(ArgumentError(
        "a grid uniform in some directions and stretched in others needs a NUFFT provider for its " *
        "stretched axes — run `using NonuniformFFTs` or `using FINUFFT`, or pass the provider as " *
        "`nufft = NonuniformFFTsBackend()`."))
end

"""
    _hybrid_derive(grid, ::Type{Tr}, ms, umask, R, iflag, eps, batch) -> NamedTuple

Everything the hybrid composite needs before any transform runs: the per-axis coordinates, origins and
Fourier lengths, which dims the FFT takes and which the NUFFT takes, whether a Nyquist twin is required
(and so whether the FFT pass halves axis 1), the publish extents and phase, and the wavenumber axes with
their twins.

The one-shot and the reusable plan both read this, so the two cannot disagree about the derivation.
"""
function _hybrid_derive(g, ::Type{Tr}, ms::NTuple{D, Int}, umask::NTuple{D, Bool}, R::Bool,
        iflag::Int, eps::Union{Nothing, Real}, batch::Tuple) where {Tr, D}
    umask[1] || throw(ArgumentError(
        "the hybrid FFT/NUFFT transform needs axis 1 uniform (it carries the real field's halved axis, " *
        "whose Nyquist mode coincides with the native `-N₁/2` only under uniform sampling); axis 1 of " *
        "this grid is stretched. Use NUFFTSpectralBackend, whose separable path transforms every axis."))
    Ns = size(g)
    npts = prod(Ns)
    axs = ntuple(d -> Tr.(FlowGeometries.Grids.coordinates(g, d)), D)
    offs_all, ranges = Grids.axis_geometry(Tr, g, D)
    for d in 1:D
        (!umask[d] || ms[d] == Ns[d]) || throw(ArgumentError(
            "axis $d is uniform, so it transforms by FFT and needs ms[$d] == size(grid, $d) = $(Ns[d]); " *
            "got $(ms[d]). Ask for the grid's own length on every uniform axis."))
    end
    udims = Tuple(d for d in 1:D if umask[d])
    sdims = Tuple(d for d in 1:D if !umask[d])
    epsv = eps === nothing ? Tr(1.0e-8) : eps
    # A twin is needed where an even axis ≥2 samples nonuniformly; only then must axis 1 stay full.
    need_twin = any(d -> !umask[d] && iseven(ms[d]), 2:D)
    halve = R && !need_twin
    neg = iflag < 0
    pms = Packing.packed_size(ms, Val(true))
    # Spectrum extents entering the publish: axis 1 is already halved by an `rfft` pass.
    nsx = ntuple(d -> (d == 1 && halve) ? pms[1] : ms[d], Val(D))
    # The FFT axes transform the grid's own points, so only the NUFFT axes carry a grid offset.
    offs = ntuple(d -> umask[d] ? zero(Tr) : offs_all[d], Val(D))
    # Each stretched axis is asked for `ms[d]` modes; a uniform axis keeps its own length.
    ns = ms
    if R
        phase = Packing.offset_phase(Tr, ms, offs, ranges, npts, Val(true))
        ks_phys = physical_wavenumbers(ranges, ms, Val(true))
        ks, twins = need_twin ?
            Packing.conj_twins(Tr, ks_phys, ms, nsx, offs, ranges, npts, batch) : (ks_phys, ())
    else
        phase = Packing.offset_phase(Tr, ms, offs, ranges, npts, Val(false), iflag)
        ks_phys = physical_wavenumbers(ranges, ms, Val(false))
        ks = ks_phys
        twins = ()
    end
    # `ks_phys` carries no twin; a device path builds its own device-resident twins from it.
    return (; Ns, npts, axs, offs, offs_all, ranges, udims, sdims, epsv, need_twin, halve, neg,
        pms, nsx, ns, phase, ks, ks_phys, twins)
end

function _calculate_spectrum_hybrid(t::SpectralBackends.AbstractNUFFTSpectralBackend,
        exec::ComputationalBackends.AbstractExecutionBackend,
        g::FlowGeometries.Grids.AbstractStructuredGrid{<:FlowGeometries.Geometry.AbstractCartesianGeometry},
        field::AbstractArray, ms::NTuple{D, Int}, umask::NTuple{D, Bool};
        iflag::Int = 1, eps::Union{Nothing, Real} = nothing, kwargs...) where {D}
    Tr = real(float(eltype(g)))
    R = eltype(field) <: Real
    batch = ntuple(i -> size(field, D + i), ndims(field) - D)
    ntrans = prod(batch; init = 1)
    h = _hybrid_derive(g, Tr, ms, umask, R, iflag, eps, batch)

    W = _region_fft(exec, quadrature_weighted(g, field), h.udims, h.halve, !R && h.neg)
    for d in h.sdims
        W = _axis_nufft(t, exec, W, d, h.axs[d], ms[d], h.ranges[d], h.offs_all[d], h.epsv; kwargs...)
    end
    if R
        coeffs = Array{Complex{Tr}}(undef, h.pms..., batch...)
        Packing.publish_packed!(coeffs, W, h.phase, h.nsx, h.pms, ntrans, h.neg)
        Packing.gather_conj_twins!(h.twins, W, prod(h.nsx), ntrans, h.neg)
        return coeffs, h.ks
    end
    h.neg && (W .= conj.(W))                     # closes the conjugation applied to the input
    W .*= h.phase
    return W, h.ks
end

# =============================================================================
# The hybrid composite, held for reuse. Same derivation as the one-shot above — axis 1 uniform, the FFT
# pass complex whenever a twin is needed — with the FFTW region plan, one 1-D NUFFT per stretched axis,
# and the working arrays built once. The one-shot's per-call cost is those three, so a reused execution
# writes into held buffers and allocates nothing beyond the caller's `coeffs`.
# =============================================================================

"""
    HybridPlan{T,D,R}

Reusable plan for a Cartesian grid uniform in some directions and stretched in others: the FFTW plan over
the uniform axes, one 1-D type-1 NUFFT per stretched axis (each with its own strengths/spectrum
buffers), and the `D - length(sdims) + 1` working arrays the passes write through.

`R` marks a real field, so the publish branch folds at compile time. Execute with
`calculate_spectrum!(coeffs, plan, field)`.
"""
struct HybridPlan{T, D, R, FP, AP, W, QB, PH, KS, TW, QW} <: Plans.AbstractSpectralPlan
    region::FP                       # the uniform-axis FFT, planned
    axes::AP                         # one axis plan per stretched dim, in `sdims` order
    sdims::Vector{Int}
    work::W                          # working arrays; `work[1]` takes the region output
    qbuf::QB                         # the quadrature-scaled field, or `nothing` for a constant measure
    ms::NTuple{D, Int}
    nsx::NTuple{D, Int}              # spectrum extents entering the publish
    pms::NTuple{D, Int}
    ntrans::Int
    npts::Int
    neg::Bool
    phase::PH
    ks_phys::KS
    twins::TW
    qw::QW
end

Base.show(io::IO, ::HybridPlan{T, D, R}) where {T, D, R} =
    print(io, "HybridPlan{$T, $D}(", R ? "real" : "complex", ")")

"""
    _hybrid_plan(nufft, exec, grid, ::Type{T}, ms, umask; batch, iflag, eps, kwargs...)

Build the reusable hybrid composite. The FFTW extension's `plan_spectrum` calls this where the grid is
uniform in some directions and stretched in others, the same split its forward makes.
"""
function _hybrid_plan(nufft::SpectralBackends.AbstractSpectralBackend,
        exec::ComputationalBackends.AbstractExecutionBackend,
        g::FlowGeometries.Grids.AbstractStructuredGrid{<:FlowGeometries.Geometry.AbstractCartesianGeometry},
        ::Type{T}, ms::NTuple{D, Int}, umask::NTuple{D, Bool}; batch::Tuple = (), iflag::Int = 1,
        eps::Union{Nothing, Real} = nothing, kwargs...) where {T, D}
    t = _resolve_nufft_provider(nufft)
    Tr = real(float(T))
    R = T <: Real
    h = _hybrid_derive(g, Tr, ms, umask, R, iflag, eps, batch)
    ntrans = prod(batch; init = 1)

    insize = (h.Ns..., batch...)
    region = _region_fft_plan(exec, T, insize, h.udims, h.halve, !R && h.neg)
    rsize = _region_fft_size(region)
    # Each stretched axis shortens its own dim; the working arrays follow that sequence.
    work = Vector{Any}()
    push!(work, Array{Complex{Tr}}(undef, rsize...))
    axplans = Any[]
    cur = rsize
    for d in h.sdims
        ap = _axis_nufft_plan(t, exec, Tr, cur, d, h.axs[d], h.ns[d], h.ranges[d], h.offs_all[d],
            h.epsv; kwargs...)
        push!(axplans, ap)
        cur = Packing.axis_out_size(cur, d, h.ns[d])
        push!(work, Array{Complex{Tr}}(undef, cur...))
    end
    wt = Tuple(work)
    apt = Tuple(axplans)
    qw = Grids.quadrature_scale(g, Tr, h.npts)
    # The quadrature scaling writes into a held buffer, so a reused execution copies the field once into
    # memory the plan owns and never allocates for it.
    qbuf = qw === nothing ? nothing : Array{R ? Tr : Complex{Tr}}(undef, insize...)
    return HybridPlan{Tr, D, R, typeof(region), typeof(apt), typeof(wt), typeof(qbuf), typeof(h.phase),
            typeof(h.ks), typeof(h.twins), typeof(qw)}(
        region, apt, collect(h.sdims), wt, qbuf, ms, h.nsx, h.pms, ntrans, h.npts, h.neg,
        h.phase, h.ks, h.twins, qw)
end

"""
    calculate_spectrum!(coeffs, plan::HybridPlan, field) -> ks_phys

Execute a prebuilt hybrid plan in place. `field` is `(N_1…N_D, batch…)`; `coeffs` is the packed half for
a real field and the full native spectrum for a complex one. The FFTW plan, the per-axis NUFFT plans and
the working arrays are reused across calls.
"""
function calculate_spectrum!(coeffs::AbstractArray{Complex{T}}, plan::HybridPlan{T, D, R},
        field) where {T, D, R}
    npts = plan.npts
    length(field) == npts * plan.ntrans || throw(DimensionMismatch(
        "field holds $(length(field)) values; this plan was built for $(npts * plan.ntrans) — pass the " *
        "matching `batch=` to plan_spectrum"))
    # The grid quadrature factor scales into the plan's own buffer, so the caller's field is untouched
    # and a reused execution allocates nothing for it.
    f = plan.qbuf === nothing ? field :
        quadrature_weighted_into!(plan.qbuf, field, plan.qw, npts, plan.ntrans)
    W = _region_fft_exec!(plan.work[1], plan.region, f)
    for (i, d) in enumerate(plan.sdims)
        W = _axis_nufft_exec!(plan.work[i + 1], plan.axes[i], W, d)
    end
    if R
        Packing.publish_packed!(coeffs, W, plan.phase, plan.nsx, plan.pms, plan.ntrans, plan.neg)
        Packing.gather_conj_twins!(plan.twins, W, prod(plan.nsx), plan.ntrans, plan.neg)
    else
        plan.neg && (W .= conj.(W))
        Pm = prod(plan.ms)
        @inbounds for t in 1:plan.ntrans
            o = (t - 1) * Pm
            for i in 1:Pm
                coeffs[o + i] = W[o + i] * plan.phase[i]
            end
        end
    end
    return plan.ks_phys
end

# `quadrature_weighted` allocates its own output; this writes into a caller-supplied one.
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
        "$(nameof(typeof(e))) — wrong transform for that grid's geometry/architecture. Each grid has its " *
        "transform: uniform structured " *
        "Cartesian → FFT; nonuniform-structured / scattered Cartesian → NUFFT (CPU; GPU NUFFT is " *
        "cuFINUFFT/CUDA and scattered-only); structured spherical → SHT (CPU via FastSphericalHarmonics; " *
        "GPU via the device-generic transform on a Gauss–Legendre spherical grid); scattered spherical → " *
        "NUFSHT (CPU, or GPU with `using KernelAbstractions`). DirectSumSpectralBackend runs on any " *
        "grid/backend but is an O(N·L²) correctness reference — slow, not a fast path.",
    ))
end

# =============================================================================
# synthesize — inverse transform, reconstructs a field `(spatial…, batch…)`.
# =============================================================================

"""
    synthesize(grid, coeffs, ms::Tuple; transform=AutoSpectralBackend(), execution=AutoBackend(),
               real_output=true, iflag=1)

Inverse of [`calculate_spectrum`](@ref): reconstruct field values at the `grid` points from its
coefficients. `real_output=true` consumes the packed half a real Cartesian field transforms to
(`(m_1÷2+1, m_2…, batch…)`) and writes a **real** array directly, folding each stored mode's conjugate
partner into the sum; `real_output=false` consumes the full native spectrum `(ms…, batch…)` and returns
it complex. Spherical coefficients are `(Nθ, Nφ, batch…)` either way. Returns an array
`(spatial…, batch…)` shaped as the forward's input.

`transform` selects the inverse the same way [`calculate_spectrum`](@ref) selects the forward, and
`AutoSpectralBackend` resolves it by the same rules; the direct-sum inverse serves any grid.
"""
function synthesize(g::FlowGeometries.Grids.AbstractGrid, coeffs::AbstractArray, ms::Tuple;
        transform::SpectralBackends.AbstractSpectralBackend = SpectralBackends.AutoSpectralBackend(),
        execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
        real_output::Bool = true, iflag::Int = 1, ks = nothing, kwargs...)
    e = _resolve_execution(execution)
    return _synthesize(_resolve_transform(transform, g, ms, e), e, g, coeffs, ms;
        real_output = real_output, iflag = iflag, ks = ks, kwargs...)
end

# Reconstructing a real field needs the Nyquist twin wherever index negation lands off the native axis,
# and a twin is a functional of the field, so it cannot be rebuilt from the grid alone — it arrives on
# the `ks` the forward returned. Where one is required and absent, this raises; the mirror it would
# otherwise fall back to is correct only under uniform sampling.
function _twin_for_inverse(g, ms::NTuple{D, Int}, ks) where {D}
    if ks !== nothing
        return Packing.axis_twin(ks[1])
    end
    needed = D >= 2 && !FlowGeometries.Grids.isuniform(g) && any(d -> iseven(ms[d]), 2:D)
    needed && throw(ArgumentError(
        "synthesize on this grid needs the Nyquist twin: an even axis ≥2 on a nonuniformly sampled grid " *
        "has no `+N_d/2` mode to negate into, so the `k₁ < 0` rows cannot be mirrored. Pass the `ks` " *
        "returned by calculate_spectrum: `synthesize(grid, coeffs, ms; ks = ks)`."))
    return nothing
end

function _synthesize(::SpectralBackends.AbstractDirectSumSpectralBackend,
        exec::Union{ComputationalBackends.AbstractSerialBackend, ComputationalBackends.AbstractThreadedBackend},
        g::FlowGeometries.Grids.AbstractGrid{<:FlowGeometries.Geometry.AbstractCartesianGeometry},
        coeffs::AbstractArray, ms::NTuple{D, Int}; real_output::Bool = true, iflag::Int = 1,
        ks = nothing, kwargs...) where {D}
    ss = size(g)
    FT = eltype(g)
    if real_output
        pms = Packing.packed_size(ms, Val(true))
        size(coeffs)[1:D] == pms || throw(DimensionMismatch(
            "real_output=true expects the packed half $(pms) on the spectral dims; got $(size(coeffs)[1:D]). " *
            "Pass real_output=false for a full native spectrum $(ms)."))
        twin = _twin_for_inverse(g, ms, ks)
        batch = ntuple(i -> size(coeffs, D + i), ndims(coeffs) - D)
        out = Array{FT}(undef, ss..., batch...)
        _synthesize_packed!(exec, out, g, coeffs, ms, iflag, twin)
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
        ks = nothing, kwargs...)   # spherical coefficients carry no halved axis, so no twin rides with `ks`
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

function _synthesize(t::SpectralBackends.AbstractSpectralBackend, e::ComputationalBackends.AbstractExecutionBackend, g::FlowGeometries.Grids.AbstractGrid,
        coeffs::AbstractArray, ms::Tuple; kwargs...)
    throw(ArgumentError(
        "$(nameof(typeof(t))) cannot invert on a $(nameof(typeof(g))) over $(nameof(typeof(g.geometry))) " *
        "with $(nameof(typeof(e))) — wrong transform for that grid's geometry/architecture. Each grid has " *
        "its transform, and `synthesize` mirrors `calculate_spectrum`: uniform structured Cartesian → FFT; " *
        "nonuniform-structured / scattered Cartesian → NUFFT; structured spherical → SHT; scattered " *
        "spherical → NUFSHT; DirectSumSpectralBackend runs on any grid.",
    ))
end

# =============================================================================
# In-place calculate_spectrum! (DirectSum core; plan form lives in the extensions).
# =============================================================================

"""
    calculate_spectrum!(coeffs, grid, field, ms; transform=DirectSumSpectralBackend(), execution=AutoBackend(), kwargs...)

In-place [`calculate_spectrum`](@ref): write coefficients `(ms…, batch…)` into the preallocated
`coeffs` and return `ks_phys`. Supported for `transform=DirectSumSpectralBackend()` with
`SerialBackend`/`ThreadedBackend`; for the library transforms build a reusable plan with
[`plan_spectrum`](@ref) and call `calculate_spectrum!(coeffs, plan, field)`.
"""
function calculate_spectrum!(coeffs::AbstractArray{<:Number}, grid::FlowGeometries.Grids.AbstractGrid, field::AbstractArray, ms::Tuple;
        transform::SpectralBackends.AbstractSpectralBackend = SpectralBackends.DirectSumSpectralBackend(),
        execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(), kwargs...)
    return _calculate_spectrum!(coeffs, _inplace_transform(transform), _resolve_execution(execution), grid, field, ms; kwargs...)
end

# The grid form of `calculate_spectrum!` is implemented for the direct sum, whose transform needs no plan;
# the library transforms are executed in place through `plan_spectrum`. So `AutoSpectralBackend` names the
# direct sum here.
_inplace_transform(t::SpectralBackends.AbstractSpectralBackend) = t
_inplace_transform(::SpectralBackends.AbstractAutoSpectralBackend) =
    SpectralBackends.DirectSumSpectralBackend()

function _calculate_spectrum!(coeffs::AbstractArray{Complex{T}}, ::SpectralBackends.AbstractDirectSumSpectralBackend,
        exec::Union{ComputationalBackends.AbstractSerialBackend, ComputationalBackends.AbstractThreadedBackend},
        g::FlowGeometries.Grids.AbstractGrid{<:FlowGeometries.Geometry.AbstractCartesianGeometry},
        field::AbstractArray, ms::NTuple{D, Int}; iflag::Int = 1, kwargs...) where {T, D}
    return _directsum_cartesian!(exec, coeffs, g, quadrature_weighted(g, field), ms, iflag)
end

function _calculate_spectrum!(coeffs::AbstractArray{<:Number}, ::SpectralBackends.AbstractDirectSumSpectralBackend,
        exec::Union{ComputationalBackends.AbstractSerialBackend, ComputationalBackends.AbstractThreadedBackend},
        g::FlowGeometries.Grids.AbstractGrid{<:Grids.SphericalHarmonicGeometry},
        field::AbstractArray, ms::NTuple{2, Int}; kwargs...)
    return _directsum_spherical!(exec, coeffs, g, field, ms[1] - 1; kwargs...)
end

function _calculate_spectrum!(::AbstractArray, t::SpectralBackends.AbstractSpectralBackend, e::ComputationalBackends.AbstractExecutionBackend,
        g::FlowGeometries.Grids.AbstractGrid, field::AbstractArray, ms::Tuple; kwargs...)
    throw(ArgumentError(
        "In-place calculate_spectrum! supports transform=DirectSumSpectralBackend() with " *
        "SerialBackend/ThreadedBackend (got transform=$(nameof(typeof(t))), execution=$(nameof(typeof(e)))). " *
        "Build a plan with plan_spectrum and call calculate_spectrum!(coeffs, plan, field).",
    ))
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
    DirectSumSphericalPlan{FT}

Reusable spherical direct-sum plan: the grid's materialized nodes (or its ring table, or its longitude
DFT matrix and latitude quadrature — whichever its layout reads), the Legendre recurrence tables, and the
`(lmax+1, nrings, B)` longitude buffer, all built once for a fixed grid, `lmax` and batch shape. A
complex field additionally needs the `exp(+imλ)` longitude transform, so the plan holds its matrix and
buffer too; the element type passed to [`plan_spectrum`](@ref) is what decides.

Building these is a minority of a transform's time and about three quarters of its allocations (the
latitude quadrature is a Gauss–Legendre root solve, and `materialize` is documented as an explicit
allocating call), so reuse across a time loop removes most of the heap traffic. Execute with
`calculate_spectrum!(coeffs, plan, field)`.
"""
struct DirectSumSphericalPlan{FT, L, S, NB, KS} <: Plans.AbstractSpectralPlan
    layout::L                     # TensorSphere / RingSphere / ScatteredSphere
    setup::S                      # that layout's precomputation, from `DirectSum.sph_setup`
    lmax::Int
    batch::NTuple{NB, Int}
    B::Int
    ks::KS
end

Base.show(io::IO, p::DirectSumSphericalPlan{FT}) where {FT} =
    print(io, "DirectSumSphericalPlan{", FT, "}(", nameof(typeof(p.layout)), ", lmax=", p.lmax,
        ", B=", p.B, ")")

function Plans.plan_spectrum(::SpectralBackends.AbstractDirectSumSpectralBackend,
        ::Union{ComputationalBackends.AbstractSerialBackend, ComputationalBackends.AbstractThreadedBackend},
        g::FlowGeometries.Grids.AbstractGrid{<:Grids.SphericalHarmonicGeometry},
        ::Type{T}, ms::Tuple; batch::Tuple = (), sampling = nothing, weights = nothing,
        kwargs...) where {T}
    FT = real(float(T))
    lmax = ms[1] - 1
    lmax >= 0 || throw(ArgumentError("ms[1] must be ≥ 1 (lmax = ms[1] - 1); got $(ms[1])"))
    bt = NTuple{length(batch), Int}(batch)
    B = prod(bt; init = 1)
    layout = Grids._sph_layout(g)
    s = DirectSum.sph_setup(layout, g, sph_coeff_type(T, FT), lmax, B; sampling = sampling,
        weights = weights)
    ks = (0:lmax, -lmax:lmax)
    return DirectSumSphericalPlan{FT, typeof(layout), typeof(s), length(bt), typeof(ks)}(
        layout, s, lmax, bt, B, ks)
end

"""
    calculate_spectrum!(coeffs, plan::DirectSumSphericalPlan, field) -> ks

Fill preallocated `coeffs` `(Nθ, Nφ, batch…)` with the spherical spectrum of `field`
`(spatial…, batch…)`, reusing `plan`'s nodes / ring table / quadrature / Legendre tables.
"""
function calculate_spectrum!(coeffs::AbstractArray{<:Number}, plan::DirectSumSphericalPlan{T},
        field) where {T}
    lmax = plan.lmax
    nc = (lmax + 1) * (2 * lmax + 1) * plan.B
    length(coeffs) == nc || throw(DimensionMismatch(
        "coeffs holds $(length(coeffs)) values; this plan was built for $nc — pass the matching " *
        "`batch=` to plan_spectrum"))
    return DirectSum.sph_run!(coeffs, plan.layout, plan.setup, field, lmax, plan.B)
end

"""
    DirectSumCartesianPlan{FT}

Reusable Cartesian direct-sum plan: the per-axis dense DFT matrices, the working arrays the contraction
walks through, the wavenumber axes, the Nyquist-twin storage, and the quadrature-scaled field buffer, all
built once for a fixed grid, `ms`, `iflag` and batch shape.

The matrices depend only on the grid and `ms`, and every buffer only on the shapes, so a time loop
rebuilds none of them. Execute with `calculate_spectrum!(coeffs, plan, field)`.
"""
struct DirectSumCartesianPlan{FT, L, S, NB, QW, QB, KS} <: Plans.AbstractSpectralPlan
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

function Plans.plan_spectrum(::SpectralBackends.AbstractDirectSumSpectralBackend,
        ::Union{ComputationalBackends.AbstractSerialBackend, ComputationalBackends.AbstractThreadedBackend},
        g::FlowGeometries.Grids.AbstractGrid{<:FlowGeometries.Geometry.AbstractCartesianGeometry},
        ::Type{T}, ms::NTuple{D, Int}; batch::Tuple = (), iflag::Int = 1, kwargs...) where {T, D}
    FT = real(float(T))
    bt = NTuple{length(batch), Int}(batch)
    layout = DirectSum._cart_layout(g)
    s = DirectSum.cart_setup(layout, g, T, ms, bt, iflag)
    npts = prod(size(g))
    qw = Grids.quadrature_scale(g, FT, npts)
    qbuf = qw === nothing ? nothing : Array{float(T)}(undef, npts, prod(bt; init = 1))
    ks = s.twin === nothing ? s.ks : s.twin.ks
    return DirectSumCartesianPlan{FT, typeof(layout), typeof(s), length(bt), typeof(qw), typeof(qbuf), typeof(ks)}(
        layout, s, qw, qbuf, npts, bt, prod(bt; init = 1), ks)
end

"""
    calculate_spectrum!(coeffs, plan::DirectSumCartesianPlan, field) -> ks

Fill preallocated `coeffs` (packed `(m₁÷2+1, m₂…, batch…)` for a real field, full native for a complex
one) with the Cartesian spectrum of `field` `(spatial…, batch…)`, reusing `plan`'s DFT matrices, working
arrays and twin storage.
"""
function calculate_spectrum!(coeffs::AbstractArray{Complex{T}}, plan::DirectSumCartesianPlan{T},
        field) where {T}
    length(field) == plan.npts * plan.B || throw(DimensionMismatch(
        "field holds $(length(field)) values; this plan was built for $(plan.npts * plan.B) — pass the " *
        "matching `batch=` to plan_spectrum"))
    # The quadrature factor scales into the plan's own buffer, so the caller's field is untouched.
    f = plan.qbuf === nothing ? field :
        quadrature_weighted_into!(plan.qbuf, field, plan.qw, plan.npts, plan.B)
    return DirectSum.cart_run!(coeffs, plan.layout, plan.setup, f)
end

# Every other geometry: the direct sum's spherical plan is `DirectSumSphericalPlan`, and nothing else has
# a Cartesian or spherical grid to plan against.
function Plans.plan_spectrum(t::SpectralBackends.AbstractDirectSumSpectralBackend,
        e::ComputationalBackends.AbstractExecutionBackend, grid::FlowGeometries.Grids.AbstractGrid,
        ::Type{T}, ms::Tuple; kwargs...) where {T}
    throw(ArgumentError(
        "DirectSumSpectralBackend has no reusable plan for a $(nameof(typeof(grid))) over " *
        "$(nameof(typeof(FlowGeometries.Grids.grid_geometry(grid)))) on $(nameof(typeof(e))). Call " *
        "`calculate_spectrum!(coeffs, grid, field, ms)`, which writes into your buffer directly."))
end

# =============================================================================
# Internal extension entry points — error until the relevant extension loads. Each takes the grid +
# field tensor directly (spatial/batch split from the grid).
# =============================================================================

# DirectSum forward: core Serial; OhMyThreads / KernelAbstractions extensions add Threaded / GPU.
_directsum_cartesian!(::ComputationalBackends.AbstractSerialBackend, coeffs, g, field, ms, iflag) =
    DirectSum._calculate_spectrum_cartesian_direct!(coeffs, g, field, ms, iflag)
_directsum_spherical!(::ComputationalBackends.AbstractSerialBackend, coeffs, g, field, lmax; kwargs...) =
    DirectSum._calculate_spectrum_spherical_direct!(coeffs, g, field, lmax; kwargs...)
_directsum_cartesian!(::ComputationalBackends.AbstractThreadedBackend, args...) = throw(ArgumentError("ThreadedBackend is not loaded. Run `using OhMyThreads`."))
_directsum_spherical!(::ComputationalBackends.AbstractThreadedBackend, args...; kwargs...) = throw(ArgumentError("ThreadedBackend is not loaded. Run `using OhMyThreads`."))

# DirectSum inverse (synthesize): same Serial/Threaded split. `_synthesize_packed!` takes the packed
# half and writes a real field; `_synthesize_cartesian!` takes the full native spectrum.
_synthesize_packed!(::ComputationalBackends.AbstractSerialBackend, out, g, coeffs, ms, iflag, twin) =
    DirectSum._synthesize_packed_direct!(out, g, coeffs, ms, iflag, twin)
_synthesize_packed!(::ComputationalBackends.AbstractThreadedBackend, args...) = throw(ArgumentError("ThreadedBackend is not loaded. Run `using OhMyThreads`."))
_synthesize_cartesian!(::ComputationalBackends.AbstractSerialBackend, out, g, coeffs, ms, iflag) =
    DirectSum._synthesize_cartesian_direct!(out, g, coeffs, ms, iflag)
_synthesize_spherical!(::ComputationalBackends.AbstractSerialBackend, out, g, coeffs, lmax) =
    DirectSum._synthesize_spherical_direct!(out, g, coeffs, lmax)
_synthesize_cartesian!(::ComputationalBackends.AbstractThreadedBackend, args...) = throw(ArgumentError("ThreadedBackend is not loaded. Run `using OhMyThreads`."))
_synthesize_spherical!(::ComputationalBackends.AbstractThreadedBackend, args...) = throw(ArgumentError("ThreadedBackend is not loaded. Run `using OhMyThreads`."))

# GPU direct-sum (KernelAbstractions ext; portable on any KA device, incl. KA.CPU()).
_gpu_directsum_cartesian(args...; kwargs...) = throw(ArgumentError("GPUBackend is not loaded. Run `using KernelAbstractions`."))
_gpu_directsum_spherical(args...; kwargs...) = throw(ArgumentError("GPUBackend is not loaded. Run `using KernelAbstractions`."))
# Fast device-generic structured SHT (φ-DFT + θ-Legendre contraction), KernelAbstractions ext.
_calculate_spectrum_gpu_sht(args...; kwargs...) = throw(ArgumentError("GPU SHT requires `using KernelAbstractions`."))

# GPU FFT is device-generic via AbstractFFTs; GPU NUFFT is CUDA-only (cuFINUFFT).
_calculate_spectrum_gpu_fft(args...; kwargs...) = throw(ArgumentError("GPU FFT requires `using KernelAbstractions` plus an AbstractFFTs provider (`FFTW` for `KA.CPU()`, `CUDA` for `CUDABackend`, …)."))
_calculate_spectrum_gpu_nufft(::SpectralBackends.AbstractNUFFTSpectralBackend, args...; kwargs...) = throw(ArgumentError("GPU NUFFT is CUDA-only (cuFINUFFT, via FINUFFTBackend) — run `using CUDA, FINUFFT`. For a portable GPU scattered transform use transform=DirectSumSpectralBackend()."))

# Hybrid composite passes. `_region_fft` transforms the uniform axes of a field in one call (halving
# axis 1 when asked, conjugating a complex input for `iflag < 0`); `_axis_nufft` transforms one stretched
# axis. Both return the RAW transform — the composite applies the phase and normalization once.
_region_fft(args...; kwargs...) = throw(ArgumentError("The hybrid FFT/NUFFT transform needs an FFT provider. Run `using FFTW`."))
_axis_nufft(t::SpectralBackends.AbstractSpectralBackend, args...; kwargs...) = throw(ArgumentError(
    "The hybrid FFT/NUFFT transform needs a NUFFT provider for the stretched axes (got " *
    "$(nameof(typeof(t)))) — run `using NonuniformFFTs` or `using FINUFFT` and pass " *
    "`nufft = NonuniformFFTsBackend()` / `nufft = FINUFFTBackend()`."))

# The same two passes, held for reuse by `HybridPlan`. `_region_fft_plan` builds the FFTW plan over the
# uniform axes and `_region_fft_size` reports the shape it writes; `_axis_nufft_plan` builds one axis's
# 1-D NUFFT with its own buffers. Each `_exec!` writes into a working array the plan owns.
_region_fft_plan(args...; kwargs...) = throw(ArgumentError("The hybrid FFT/NUFFT plan needs an FFT provider. Run `using FFTW`."))
_region_fft_size(args...; kwargs...) = throw(ArgumentError("The hybrid FFT/NUFFT plan needs an FFT provider. Run `using FFTW`."))
_region_fft_exec!(args...; kwargs...) = throw(ArgumentError("The hybrid FFT/NUFFT plan needs an FFT provider. Run `using FFTW`."))
_axis_nufft_plan(t::SpectralBackends.AbstractSpectralBackend, args...; kwargs...) = throw(ArgumentError(
    "The hybrid FFT/NUFFT plan needs a NUFFT provider for the stretched axes (got " *
    "$(nameof(typeof(t)))) — run `using NonuniformFFTs` or `using FINUFFT`."))
_axis_nufft_exec!(args...; kwargs...) = throw(ArgumentError(
    "The hybrid FFT/NUFFT plan needs a NUFFT provider for the stretched axes."))

# CPU transform libraries (extensions). NUFFT: pick a provider — the abstract NUFFTSpectralBackend selects none.
_calculate_spectrum_fft(args...; kwargs...) = throw(ArgumentError("FFTSpectralBackend is not loaded. Run `using FFTW`."))
_calculate_spectrum_nufft(::SpectralBackends.NUFFTSpectralBackend, args...; kwargs...) = throw(ArgumentError("NUFFTSpectralBackend selects no NUFFT provider — pass transform=FINUFFTBackend() or NonuniformFFTsBackend()."))
_calculate_spectrum_nufft(::FINUFFTBackend, args...; kwargs...) = throw(ArgumentError("FINUFFTBackend needs FINUFFT — run `using FINUFFT`."))
_calculate_spectrum_nufft(::NonuniformFFTsBackend, args...; kwargs...) = throw(ArgumentError("NonuniformFFTsBackend needs NonuniformFFTs — run `using NonuniformFFTs`."))
_calculate_spectrum_sht(args...; kwargs...) = throw(ArgumentError("FSHTSpectralBackend is not loaded. Run `using FastSphericalHarmonics`."))
_calculate_spectrum_nufsht(args...; kwargs...) = throw(ArgumentError("NUFSHTSpectralBackend is not loaded. Run `using NUFSHT`."))

# Distribution wrappers (Distributed / MPI extensions).
_calculate_spectrum_distributed(args...; kwargs...) = throw(ArgumentError("DistributedBackend is not loaded. Run `using Distributed` (and `addprocs`)."))
_calculate_spectrum_mpi(args...; kwargs...) = throw(ArgumentError("MPIBackend is not loaded. Run `using MPI` and launch under `mpiexec`."))

# =============================================================================
# Distribution partition helpers (shared by the Distributed / MPI extensions). Complex coefficients
# are additive over a disjoint point partition; `α_w` compensates the per-kernel `1/N_local`.
# =============================================================================

# Subgrid over a subset of point indices (the point-partitionable grids: a node cloud, and a curvilinear
# grid, whose coordinate arrays hold one value per cell and index linearly the same way), preserving the
# geometry, periodicity, and per-node measure (sliced). A subset of a curvilinear grid's cells keeps no
# index-space structure, so it is a node cloud either way.
function _subgrid(g::Union{FlowGeometries.Grids.AbstractUnstructuredGrid,
            FlowGeometries.Grids.AbstractCurvilinearGrid}, idx)
    geom = FlowGeometries.Grids.grid_geometry(g)
    D = length(FlowGeometries.Grids.coordinates(g))
    coords = ntuple(d -> view(FlowGeometries.Grids.coordinates(g, d), idx), D)
    meas = view(FlowGeometries.Grids.measure_array(g), idx)
    per = FlowGeometries.Grids.periodic_flags(g)
    prd = ntuple(d -> FlowGeometries.Grids.period(g, d), D)
    return FlowGeometries.Grids.UnstructuredGrid(geom, coords, meas; periodic = per, period = prd)
end

# Per-worker scalar so `coeff_global = Σ_w α_w · coeff_w`. Point-partition fires only on unstructured
# grids, whose scattered subgrids use the uniform `4π/N_local` (spherical) / `1/N_local` (Cartesian)
# weighting; `α_w = N_local/N_global` recombines them into the global `1/N_global` normalization.
_partition_alpha(::FlowGeometries.Grids.AbstractGrid, Nw, Nglob) = Nw / Nglob

# Transforms whose coefficients are additive over a disjoint point partition.
_partitionable(::SpectralBackends.AbstractDirectSumSpectralBackend) = true
_partitionable(::SpectralBackends.AbstractNUFFTSpectralBackend) = true
_partitionable(::SpectralBackends.AbstractNUFSHTSpectralBackend) = true
_partitionable(::SpectralBackends.AbstractSpectralBackend) = false

# Spatial (spectral) shape of the coefficient array for a grid + `ms`, given whether the field is real
# (a real Cartesian transform halves axis 1).
_coeff_spatial(::FlowGeometries.Grids.AbstractGrid{<:FlowGeometries.Geometry.AbstractCartesianGeometry}, ms, R::Bool) =
    Packing.packed_size(NTuple{length(ms), Int}(Tuple(ms)), Val(R))
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

# A distributed run must publish the same Nyquist twin the local transform would. Coefficients and twins
# are both linear in the field, so a disjoint point partition combines a worker's twin with the same
# `α_w` weight its coefficients carry; a batch partition concatenates along the batch axis instead, where
# each worker already sees every point and its twin is globally correct.
_combine_twins(ks::Tuple, ::Tuple{}, weights) = ks
function _combine_twins(ks::Tuple, twins::Tuple, weights)
    any(t -> t === nothing, twins) && return ks
    ref = first(twins)
    slices = ntuple(length(ref.slices)) do i
        acc = zero(ref.slices[i])
        for (t, w) in zip(twins, weights)
            acc .+= w .* t.slices[i]
        end
        acc
    end
    return (Packing.with_twin(ks[1], Packing.NyquistTwin(slices)), Base.tail(ks)...)
end

# The twin a `ks` carries on its halved axis, if any.
_ks_twin(ks::Tuple) = Packing.axis_twin(ks[1])

# Batch-partition twin assembly: each worker's slices cover its own batch columns, so they scatter into
# full-batch slices along the trailing axis. A worker slice is `(shape…, nb)` for its `nb` columns.
_alloc_batch_twins(::Nothing, B::Int) = nothing
_alloc_batch_twins(t::Packing.NyquistTwin, B::Int) =
    map(s -> zeros(eltype(s), ntuple(i -> size(s, i), ndims(s) - 1)..., B), t.slices)

# Zeroed full-batch twin buffers sized from `ms` alone, so a rank that received no batch columns agrees
# on the shapes with the ranks that did.
_alloc_twins(::Type{C}, ms::NTuple{D, Int}, B::Int) where {C, D} =
    ntuple(Packing.n_twin_slices(Val(D))) do mask
        zeros(C, Packing.twin_slice_shape(ms, mask)..., B)
    end

_scatter_batch_twins!(::Nothing, ::Any, bc, B::Int) = nothing
function _scatter_batch_twins!(dst::Tuple, t::Packing.NyquistTwin, bc, B::Int)
    for (d, s) in zip(dst, t.slices)
        isempty(d) && continue
        shape = ntuple(i -> size(d, i), ndims(d) - 1)
        @inbounds selectdim(d, ndims(d), bc) .= reshape(s, shape..., length(bc))
    end
    return nothing
end

_reshape_batch_twins(ks::Tuple, ::Nothing, batch::Tuple) = ks
function _reshape_batch_twins(ks::Tuple, dst::Tuple, batch::Tuple)
    slices = map(d -> reshape(d, ntuple(i -> size(d, i), ndims(d) - 1)..., batch...), dst)
    return (Packing.with_twin(ks[1], Packing.NyquistTwin(slices)), Base.tail(ks)...)
end

# =============================================================================
# Plotting stubs (CairoMakie extension).
# =============================================================================

"""
    plot_spectrum(ks_phys::Tuple, coeffs; title="Energy Spectrum", kwargs...)

Plot a 1D isotropic / 2D Cartesian / spherical-degree energy spectrum. Requires `using CairoMakie`.
"""
plot_spectrum(args...; kwargs...) = throw(ArgumentError("plot_spectrum requires CairoMakie. Run `using CairoMakie`."))

"""
    compare_spectra(spectra_list; labels, kwargs...)

Overlay multiple 1D energy spectra. Requires `using CairoMakie`.
"""
compare_spectra(args...; kwargs...) = throw(ArgumentError("compare_spectra requires CairoMakie. Run `using CairoMakie`."))

"""
    compare_spectral_analysis(true_coeffs, approx_coeffs; kwargs...)

Coefficient comparison + error maps. Requires `using CairoMakie`.
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
