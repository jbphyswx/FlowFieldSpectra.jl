module FlowFieldSpectra

using PrecompileTools: PrecompileTools
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
export AbstractSpectralPlan, plan_spectrum
# APIs
export calculate_spectrum, calculate_spectrum!, synthesize, isotropic_spectrum, isotropic_spectrum!, transect_spectrum, transect_spectrum!, spherical_energy_spectrum, spherical_energy_spectrum!, sph_mode_index
export spectral_divergence, spectral_divergence!, spectral_vorticity, spectral_vorticity!,
    compensate, band_energy
export cross_spectrum, cospectrum, quadspectrum, anisotropic_spectrum
export welch_power_spectrum, welch_power_spectrum!, coherence_spectrum, coherence_spectrum!,
    lomb_scargle, lomb_scargle!
export plot_spectrum, compare_spectra, compare_spectral_analysis

# =============================================================================
# Execution-backend resolution (FFS-owned; ComputationalBackends deliberately errors on `AutoBackend`
# so each consumer resolves it locally). Best available LOCAL backend: `ThreadedBackend` iff the
# OhMyThreads extension is loaded and `Threads.nthreads() > 1`, else `SerialBackend`. GPU/Distributed/
# MPI are never auto-selected (they need explicit device/process context).
# =============================================================================
_resolve_execution(backend::ComputationalBackends.AbstractExecutionBackend) = ComputationalBackends.resolve_backend(backend)
function _resolve_execution(::ComputationalBackends.AbstractAutoBackend)
    threaded = Base.get_extension(@__MODULE__, :FlowFieldSpectraOhMyThreadsExt) !== nothing
    return (threaded && Threads.nthreads() > 1) ? ComputationalBackends.ThreadedBackend() : ComputationalBackends.SerialBackend()
end

"""
    calculate_spectrum(grid, field, ms::Tuple;
                       transform=DirectSumSpectralBackend(), execution=AutoBackend(), kwargs...)

Spectral coefficients and physical wavenumbers of `field` sampled on `grid` (a FlowGeometries grid).
The grid's architecture × geometry *is* the coordinate system — no coordinate guessing. The two
backend axes compose freely:

- `transform::SpectralBackends.AbstractSpectralBackend` — the spectral math
  (`DirectSumSpectralBackend` (default), `FFTSpectralBackend`, `NUFFTSpectralBackend`,
  `FSHTSpectralBackend`, `NUFSHTSpectralBackend`).
- `execution::ComputationalBackends.AbstractExecutionBackend` — where/how it runs (`SerialBackend`,
  `ThreadedBackend`, `GPUBackend`, `DistributedBackend`, `MPIBackend`, `AutoBackend` (default)).

# Data model
`field` is an `AbstractArray` shaped `(spatial…, batch…)`: the first `ndims(grid)` dims are
spatial (and must equal `size(grid)`); every trailing dim is a **batch** dim (components,
levels, time, ensemble — any number), carried through and preserved. Structured grids take an
`(N_1,…,N_D, batch…)` tensor (spherical: `(nlon, nlat, batch…)`); unstructured grids take
`(N, batch…)`. A `Tuple` of equal-shaped arrays stacks them along a new trailing batch axis.

`ms` is the spectral resolution: Cartesian `(m_1,…,m_D)`; spherical `(Nθ, Nφ)` with `lmax = Nθ-1`.

# Returns
`(coeffs, ks_phys)` — complex coefficients `(ms…, batch…)` (Cartesian) / `(Nθ, Nφ, batch…)`
(spherical), and the physical wavenumber coordinates per spectral axis.
"""
function calculate_spectrum(grid::FlowGeometries.Grids.AbstractGrid, field::AbstractArray, ms::Tuple;
        transform::SpectralBackends.AbstractSpectralBackend = SpectralBackends.DirectSumSpectralBackend(),
        execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(), kwargs...)
    return calculate_spectrum(transform, _resolve_execution(execution), grid, field, ms; kwargs...)
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
    coeffs = zeros(Complex{eltype(g)}, ms..., batch_shape(prob)...)
    ks = _directsum_cartesian!(exec, coeffs, g, field, ms, iflag)
    return coeffs, ks
end

function calculate_spectrum(::SpectralBackends.AbstractDirectSumSpectralBackend,
        exec::Union{ComputationalBackends.AbstractSerialBackend, ComputationalBackends.AbstractThreadedBackend},
        g::FlowGeometries.Grids.AbstractGrid{<:FlowGeometries.Geometry.AbstractSphericalGeometry},
        field::AbstractArray, ms::NTuple{2, Int}; kwargs...)
    prob = TransformProblem(g, field)
    lmax = ms[1] - 1
    coeffs = zeros(Complex{eltype(g)}, lmax + 1, 2 * lmax + 1, batch_shape(prob)...)
    ks = _directsum_spherical!(exec, coeffs, g, field, lmax; kwargs...)
    return coeffs, ks
end

# ---- Level 2: DirectSum on GPU (KernelAbstractions ext; portable on any KA device) ----
calculate_spectrum(::SpectralBackends.AbstractDirectSumSpectralBackend, exec::ComputationalBackends.AbstractGPUBackend, g::FlowGeometries.Grids.AbstractGrid{<:FlowGeometries.Geometry.AbstractCartesianGeometry}, field::AbstractArray, ms::NTuple{D, Int}; iflag::Int = 1, kwargs...) where {D} =
    _gpu_directsum_cartesian(exec, g, field, ms, iflag)
calculate_spectrum(::SpectralBackends.AbstractDirectSumSpectralBackend, exec::ComputationalBackends.AbstractGPUBackend, g::FlowGeometries.Grids.AbstractGrid{<:FlowGeometries.Geometry.AbstractSphericalGeometry}, field::AbstractArray, ms::NTuple{2, Int}; kwargs...) =
    _gpu_directsum_spherical(exec, g, field, ms[1] - 1; kwargs...)

# ---- Level 2: FFT (FFTW ext CPU; device-generic GPU FFT). Structured Cartesian, uniform only. ----
function calculate_spectrum(::SpectralBackends.AbstractFFTSpectralBackend, exec::Union{ComputationalBackends.AbstractSerialBackend, ComputationalBackends.AbstractThreadedBackend}, g::FlowGeometries.Grids.AbstractStructuredGrid{<:FlowGeometries.Geometry.AbstractCartesianGeometry}, field::AbstractArray, ms::Tuple; kwargs...)
    FlowGeometries.Grids.isuniform(g) || throw(ArgumentError(
        "FFTSpectralBackend needs a uniform axis in every direction (an AbstractRange); this grid has a " *
        "stretched axis. Use NUFFTSpectralBackend or DirectSumSpectralBackend."))
    return _calculate_spectrum_fft(exec, g, field, ms; kwargs...)
end
function calculate_spectrum(::SpectralBackends.AbstractFFTSpectralBackend, exec::ComputationalBackends.AbstractGPUBackend, g::FlowGeometries.Grids.AbstractStructuredGrid{<:FlowGeometries.Geometry.AbstractCartesianGeometry}, field::AbstractArray, ms::Tuple; kwargs...)
    FlowGeometries.Grids.isuniform(g) || throw(ArgumentError(
        "FFTSpectralBackend needs a uniform axis in every direction; got a stretched axis."))
    return _calculate_spectrum_gpu_fft(exec, g, field, ms; kwargs...)
end

# ---- Level 2: NUFFT (FINUFFT ext CPU; cuFINUFFT ext on CUDA). Structured (nonuniform-gridded) uses
# the separable per-axis 1-D NUFFT; unstructured (scattered) uses the guru NUFFT. Both are
# `_calculate_spectrum_nufft`, dispatched on the grid architecture inside the extension. ----
calculate_spectrum(t::SpectralBackends.AbstractNUFFTSpectralBackend, exec::Union{ComputationalBackends.AbstractSerialBackend, ComputationalBackends.AbstractThreadedBackend}, g::Union{FlowGeometries.Grids.AbstractStructuredGrid{<:FlowGeometries.Geometry.AbstractCartesianGeometry}, FlowGeometries.Grids.AbstractUnstructuredGrid{<:FlowGeometries.Geometry.AbstractCartesianGeometry}}, field::AbstractArray, ms::Tuple; kwargs...) =
    _calculate_spectrum_nufft(t, exec, g, field, ms; kwargs...)
calculate_spectrum(t::SpectralBackends.AbstractNUFFTSpectralBackend, exec::ComputationalBackends.AbstractGPUBackend, g::Union{FlowGeometries.Grids.AbstractStructuredGrid{<:FlowGeometries.Geometry.AbstractCartesianGeometry}, FlowGeometries.Grids.AbstractUnstructuredGrid{<:FlowGeometries.Geometry.AbstractCartesianGeometry}}, field::AbstractArray, ms::Tuple; kwargs...) =
    _calculate_spectrum_gpu_nufft(t, exec, g, field, ms; kwargs...)

# ---- Level 2: SHT / NUFSHT (execution axis handled inside the extension) ----
calculate_spectrum(::SpectralBackends.AbstractFSHTSpectralBackend, ::Union{ComputationalBackends.AbstractSerialBackend, ComputationalBackends.AbstractThreadedBackend}, g::FlowGeometries.Grids.AbstractStructuredGrid{<:FlowGeometries.Geometry.AbstractSphericalGeometry}, field::AbstractArray, ms::Tuple; kwargs...) =
    _calculate_spectrum_sht(g, field, ms; kwargs...)
# GPU SHT: a fast device-generic transform (φ-DFT + θ-Legendre contraction) in the KernelAbstractions
# ext — no FastTransforms, runs on any KA device. Structured grid only.
calculate_spectrum(::SpectralBackends.AbstractFSHTSpectralBackend, exec::ComputationalBackends.AbstractGPUBackend, g::FlowGeometries.Grids.AbstractStructuredGrid{<:FlowGeometries.Geometry.AbstractSphericalGeometry}, field::AbstractArray, ms::Tuple; kwargs...) =
    _calculate_spectrum_gpu_sht(exec, g, field, ms; kwargs...)
calculate_spectrum(::SpectralBackends.AbstractNUFSHTSpectralBackend, exec::ComputationalBackends.AbstractExecutionBackend, g::FlowGeometries.Grids.AbstractGrid{<:FlowGeometries.Geometry.AbstractSphericalGeometry}, field::AbstractArray, ms::Tuple; kwargs...) =
    _calculate_spectrum_nufsht(exec, g, field, ms; kwargs...)

# ---- Least-specific catch-all: clear error for any unsupported (transform, execution, grid) ----
function calculate_spectrum(t::SpectralBackends.AbstractSpectralBackend, e::ComputationalBackends.AbstractExecutionBackend,
        g::FlowGeometries.Grids.AbstractGrid, field::AbstractArray, ms::Tuple; kwargs...)
    throw(ArgumentError(
        "$(nameof(typeof(t))) cannot act on a $(nameof(typeof(g))) with $(nameof(typeof(e))) — wrong " *
        "transform for that grid's geometry/architecture. Each grid has its transform: uniform structured " *
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
    synthesize(grid, coeffs, ms::Tuple; transform=DirectSumSpectralBackend(), execution=AutoBackend(),
               real_output=true, iflag=1)

Inverse of [`calculate_spectrum`](@ref): reconstruct field values at the `grid` points from
coefficients `coeffs` (`(ms…, batch…)` Cartesian, `(Nθ, Nφ, batch…)` spherical). Returns an array
`(spatial…, batch…)`; `real_output=true` returns its real part. Uses the direct-sum inverse (works
for any grid) with `SerialBackend`/`ThreadedBackend`.
"""
function synthesize(g::FlowGeometries.Grids.AbstractGrid, coeffs::AbstractArray, ms::Tuple;
        transform::SpectralBackends.AbstractSpectralBackend = SpectralBackends.DirectSumSpectralBackend(),
        execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
        real_output::Bool = true, iflag::Int = 1)
    return _synthesize(transform, _resolve_execution(execution), g, coeffs, ms; real_output = real_output, iflag = iflag)
end

function _synthesize(::SpectralBackends.AbstractDirectSumSpectralBackend,
        exec::Union{ComputationalBackends.AbstractSerialBackend, ComputationalBackends.AbstractThreadedBackend},
        g::FlowGeometries.Grids.AbstractGrid{<:FlowGeometries.Geometry.AbstractCartesianGeometry},
        coeffs::AbstractArray, ms::NTuple{D, Int}; real_output::Bool = true, iflag::Int = 1) where {D}
    ss = size(g)
    batch = ntuple(i -> size(coeffs, D + i), ndims(coeffs) - D)
    out = Array{Complex{eltype(g)}}(undef, ss..., batch...)
    _synthesize_cartesian!(exec, out, g, coeffs, ms, iflag)
    return real_output ? real.(out) : out
end

function _synthesize(::SpectralBackends.AbstractDirectSumSpectralBackend,
        exec::Union{ComputationalBackends.AbstractSerialBackend, ComputationalBackends.AbstractThreadedBackend},
        g::FlowGeometries.Grids.AbstractGrid{<:FlowGeometries.Geometry.AbstractSphericalGeometry},
        coeffs::AbstractArray, ms::NTuple{2, Int}; real_output::Bool = true, iflag::Int = 1)
    ss = size(g)
    batch = ntuple(i -> size(coeffs, 2 + i), ndims(coeffs) - 2)
    out = Array{Complex{eltype(g)}}(undef, ss..., batch...)
    _synthesize_spherical!(exec, out, g, coeffs, ms[1] - 1)
    return real_output ? real.(out) : out
end

function _synthesize(t::SpectralBackends.AbstractSpectralBackend, e::ComputationalBackends.AbstractExecutionBackend, g::FlowGeometries.Grids.AbstractGrid,
        coeffs::AbstractArray, ms::Tuple; kwargs...)
    throw(ArgumentError(
        "synthesize supports transform=DirectSumSpectralBackend() with SerialBackend/ThreadedBackend " *
        "(got transform=$(nameof(typeof(t))), execution=$(nameof(typeof(e)))).",
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
function calculate_spectrum!(coeffs::AbstractArray{Complex{T}}, grid::FlowGeometries.Grids.AbstractGrid, field::AbstractArray, ms::Tuple;
        transform::SpectralBackends.AbstractSpectralBackend = SpectralBackends.DirectSumSpectralBackend(),
        execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(), kwargs...) where {T}
    return _calculate_spectrum!(coeffs, transform, _resolve_execution(execution), grid, field, ms; kwargs...)
end

function _calculate_spectrum!(coeffs::AbstractArray{Complex{T}}, ::SpectralBackends.AbstractDirectSumSpectralBackend,
        exec::Union{ComputationalBackends.AbstractSerialBackend, ComputationalBackends.AbstractThreadedBackend},
        g::FlowGeometries.Grids.AbstractGrid{<:FlowGeometries.Geometry.AbstractCartesianGeometry},
        field::AbstractArray, ms::NTuple{D, Int}; iflag::Int = 1, kwargs...) where {T, D}
    return _directsum_cartesian!(exec, coeffs, g, field, ms, iflag)
end

function _calculate_spectrum!(coeffs::AbstractArray{Complex{T}}, ::SpectralBackends.AbstractDirectSumSpectralBackend,
        exec::Union{ComputationalBackends.AbstractSerialBackend, ComputationalBackends.AbstractThreadedBackend},
        g::FlowGeometries.Grids.AbstractGrid{<:FlowGeometries.Geometry.AbstractSphericalGeometry},
        field::AbstractArray, ms::NTuple{2, Int}; kwargs...) where {T}
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
    plan_spectrum(grid, ::Type{T}, ms; transform=DirectSumSpectralBackend(), execution=AutoBackend(), batch=(), kwargs...)

Reusable [`AbstractSpectralPlan`](@ref) for the fixed geometry of `grid` at resolution `ms`,
transforming a field with trailing batch shape `batch` of element type `T` in one batched,
allocation-free execution. Reuse via `calculate_spectrum!(coeffs, plan, field)`.
"""
function Plans.plan_spectrum(grid::FlowGeometries.Grids.AbstractGrid, ::Type{T}, ms::Tuple;
        transform::SpectralBackends.AbstractSpectralBackend = SpectralBackends.DirectSumSpectralBackend(),
        execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
        batch::Tuple = (), kwargs...) where {T}
    return Plans.plan_spectrum(transform, _resolve_execution(execution), grid, T, ms; batch = batch, kwargs...)
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

# DirectSum inverse (synthesize): same Serial/Threaded split.
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

# Subgrid over a subset of point indices (point-partitionable grids: unstructured), preserving the
# geometry, periodicity, and per-node measure (sliced).
function _subgrid(g::FlowGeometries.Grids.AbstractUnstructuredGrid, idx)
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

# Spatial (spectral) shape of the coefficient array for a grid + `ms`.
_coeff_spatial(::FlowGeometries.Grids.AbstractGrid{<:FlowGeometries.Geometry.AbstractCartesianGeometry}, ms) = Tuple(ms)
_coeff_spatial(::FlowGeometries.Grids.AbstractGrid{<:FlowGeometries.Geometry.AbstractSphericalGeometry}, ms) = (ms[1], 2 * (ms[1] - 1) + 1)

# Wavenumber coordinate returned alongside the coefficients.
_partition_ks(g::FlowGeometries.Grids.AbstractGrid{<:FlowGeometries.Geometry.AbstractCartesianGeometry}, ms) = physical_wavenumbers(g, NTuple{length(ms), Int}(Tuple(ms)))
_partition_ks(::FlowGeometries.Grids.AbstractGrid{<:FlowGeometries.Geometry.AbstractSphericalGeometry}, ms) = (0:(ms[1] - 1), -(ms[1] - 1):(ms[1] - 1))

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
        synthesize(cart, c, (4, 4); execution = ComputationalBackends.SerialBackend())

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
