module FlowFieldSpectra

using PrecompileTools: @setup_workload, @compile_workload

include("Types.jl")
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

using .Types: Types, AbstractSpectralBackend, DirectSumBackend, FFTBackend, NUFFTBackend, SHTBackend, NUFSHTBackend,
    AbstractExecutionBackend, SerialBackend, ThreadedBackend, GPUBackend, DistributedBackend, MPIBackend, AutoBackend
using .Grids: AbstractGrid, AbstractCartesianGrid, AbstractSphericalGrid, UniformCartesianGrid, NonuniformCartesianGrid, ScatteredCartesianGrid, StructuredSphericalGrid, ScatteredSphericalGrid, AbstractQuadrature, ClenshawCurtis, GaussLegendre, Equiangular
using .Preprocessing: AbstractWindow, NoWindow, Hann, Hamming, Blackman, Tukey, AbstractDetrend, NoDetrend, Demean, LinearDetrend, Preprocess, dpss
using .Normalization: AbstractSidedness, OneSided, TwoSided, AbstractScaling, DensityScaling, PowerScaling, SpectralConvention
using .Problem: TransformProblem
using .Plans: Plans, AbstractSpectralPlan, plan_spectrum
using .DirectSum: DirectSum, sph_mode_index
using .Reductions: isotropic_spectrum, isotropic_spectrum!, transect_spectrum, transect_spectrum!, spherical_energy_spectrum, spherical_energy_spectrum!, cross_spectrum, cospectrum, quadspectrum, anisotropic_spectrum
using .Operators: spectral_divergence, spectral_vorticity, compensate, band_energy
using .Averaging: welch_power_spectrum, coherence_spectrum
using .LombScargle: lomb_scargle

# Export transform backends (which spectral math)
export AbstractSpectralBackend, DirectSumBackend, FFTBackend, NUFFTBackend, SHTBackend, NUFSHTBackend
# Export execution backends (where/how it runs)
export AbstractExecutionBackend, SerialBackend, ThreadedBackend, GPUBackend, DistributedBackend, MPIBackend, AutoBackend

# Export Grids
export AbstractGrid, AbstractCartesianGrid, AbstractSphericalGrid, UniformCartesianGrid, NonuniformCartesianGrid, ScatteredCartesianGrid, StructuredSphericalGrid, ScatteredSphericalGrid
export AbstractQuadrature, ClenshawCurtis, GaussLegendre, Equiangular

# Export Preprocessing & Normalization (typed configuration)
export AbstractWindow, NoWindow, Hann, Hamming, Blackman, Tukey
export AbstractDetrend, NoDetrend, Demean, LinearDetrend, Preprocess, dpss
export AbstractSidedness, OneSided, TwoSided, AbstractScaling, DensityScaling, PowerScaling, SpectralConvention
export TransformProblem
export AbstractSpectralPlan, plan_spectrum

# Export APIs
export calculate_spectrum, calculate_spectrum!, synthesize, isotropic_spectrum, isotropic_spectrum!, transect_spectrum, transect_spectrum!, spherical_energy_spectrum, spherical_energy_spectrum!, sph_mode_index
export spectral_divergence, spectral_vorticity, compensate, band_energy
export cross_spectrum, cospectrum, quadspectrum, anisotropic_spectrum
export welch_power_spectrum, coherence_spectrum, lomb_scargle
export plot_spectrum, compare_spectra, compare_spectral_analysis

# =============================================================================
# AutoBackend resolution — pick the best available LOCAL execution backend.
# Uses `Base.get_extension` (never `isdefined(Main, …)`) so the default execution can only resolve
# to `ThreadedBackend` when the OhMyThreads extension is actually loaded; otherwise `SerialBackend`.
# Distribution (Distributed/MPI) and GPU are never auto-selected — they need explicit context.
# =============================================================================
function Types.resolve_backend(::AutoBackend)
    threaded = Base.get_extension(@__MODULE__, :FlowFieldSpectraOhMyThreadsExt) !== nothing
    return (threaded && Threads.nthreads() > 1) ? ThreadedBackend() : SerialBackend()
end

"""
    calculate_spectrum(grid::AbstractGrid, fields, ms::Tuple;
                       transform=DirectSumBackend(), execution=AutoBackend(), kwargs...)

Calculate the spectral coefficients and physical wavenumbers for one or more fields sampled on an
explicit `grid`. The coordinate system is determined by the grid type — there is no coordinate
guessing. The two backend axes are orthogonal and compose freely:

- `transform::AbstractSpectralBackend` — the spectral math (`DirectSumBackend` (default),
  `FFTBackend`, `NUFFTBackend`, `SHTBackend`, `NUFSHTBackend`).
- `execution::AbstractExecutionBackend` — where/how it runs (`SerialBackend`, `ThreadedBackend`,
  `GPUBackend{B}`, `DistributedBackend{Inner}`, `MPIBackend{Inner}`, or `AutoBackend` (default),
  which selects `ThreadedBackend` when OhMyThreads is loaded and `Threads.nthreads() > 1`, else
  `SerialBackend`).

# Arguments
- `grid::AbstractGrid`: the sampling grid. Construct one of:
  - `UniformCartesianGrid`, `NonuniformCartesianGrid`, `ScatteredCartesianGrid` (Cartesian), or
  - `StructuredSphericalGrid`, `ScatteredSphericalGrid` (spherical, ``(\\theta, \\phi)`` in radians).
  Domain size (Cartesian) and quadrature weights (spherical) are carried on the grid.
- `fields`: a `Tuple` of field vectors `(u, v, …)`, each of length `npoints(grid)`.
- `ms::Tuple`: target spectral resolution. Cartesian: `(mx, my, …)` modes per axis. Spherical:
  `(Nθ, Nφ)` with `lmax = Nθ - 1`.

# Keyword Arguments
- `iflag::Int`: Cartesian transform direction (`1` analysis, `-1` synthesis; default `1`).
- `tol`/`eps::Real`: accuracy for non-uniform transforms (NUFFT/NUFSHT).
- `solve::Bool`, `maxiter::Int`, `rtol::Real`: iterative-solve controls for `NUFSHTBackend`.

# Returns
- `coeffs`: complex coefficients of size `(ms..., NU)`, `NU = length(fields)`.
- `ks_phys`: physical wavenumber coordinates per axis, or `(0:lmax, -lmax:lmax)` for spherical.

# Example
```julia
using FlowFieldSpectra, FFTW

L = 2π; N = 16
x = range(0, L, N + 1)[1:N]
xv = vec([xi for xi in x, yi in x]); yv = vec([yi for xi in x, yi in x])
u = cos.(xv) .+ sin.(yv); v = zero(u)

grid = UniformCartesianGrid((xv, yv); domain_size = (L, L))
coeffs, ks = calculate_spectrum(grid, (u, v), (N, N); transform = FFTBackend())
```
"""
function calculate_spectrum(
    grid::AbstractGrid,
    fields_vecs::Tuple,
    ms::Tuple;
    transform::AbstractSpectralBackend = DirectSumBackend(),
    execution::AbstractExecutionBackend = AutoBackend(),
    kwargs...,
)
    return calculate_spectrum(transform, Types.resolve_backend(execution), grid, fields_vecs, ms; kwargs...)
end

# =============================================================================
# Canonical two-axis dispatch: calculate_spectrum(transform, execution, grid, fields, ms; …).
# The coordinate system is the grid type; the transform is the math; the execution is where/how.
# Every valid combination is a specific method below; every gap falls to the friendly-error
# catch-all (never a silent fallback).
# =============================================================================

# ---- Level 1: distribution wrappers unwrap to the Distributed / MPI extension hooks ----
calculate_spectrum(t::AbstractSpectralBackend, e::DistributedBackend, g::AbstractGrid, fields_vecs::Tuple, ms::Tuple; kwargs...) =
    _calculate_spectrum_distributed(t, e, g, fields_vecs, ms; kwargs...)
calculate_spectrum(t::AbstractSpectralBackend, e::MPIBackend, g::AbstractGrid, fields_vecs::Tuple, ms::Tuple; kwargs...) =
    _calculate_spectrum_mpi(t, e, g, fields_vecs, ms; kwargs...)

# ---- Level 2: DirectSum on a local CPU backend (core Serial; OhMyThreads ext Threaded) ----
function calculate_spectrum(::DirectSumBackend, exec::Union{SerialBackend, ThreadedBackend},
        g::AbstractCartesianGrid{FT, D}, fields_vecs::Tuple, ms::NTuple{D, Int};
        iflag::Int = 1, kwargs...) where {FT, D}
    NU = length(fields_vecs)
    coeffs = zeros(Complex{FT}, ms..., NU)
    ks = _directsum_cartesian!(exec, coeffs, g.coords, fields_vecs, ms, iflag, g.domain_size)
    return coeffs, ks
end

function calculate_spectrum(::DirectSumBackend, exec::Union{SerialBackend, ThreadedBackend},
        g::AbstractSphericalGrid{FT}, fields_vecs::Tuple, ms::NTuple{2, Int}; kwargs...) where {FT}
    NU = length(fields_vecs)
    lmax = ms[1] - 1
    coeffs = zeros(Complex{FT}, lmax + 1, 2 * lmax + 1, NU)
    ks = _directsum_spherical!(exec, coeffs, g.coords, fields_vecs, lmax, g.weights)
    return coeffs, ks
end

# ---- Level 2: DirectSum on GPU (KernelAbstractions ext; portable on any KA device) ----
function calculate_spectrum(::DirectSumBackend, exec::GPUBackend,
        g::AbstractCartesianGrid{FT, D}, fields_vecs::Tuple, ms::NTuple{D, Int};
        iflag::Int = 1, kwargs...) where {FT, D}
    return _gpu_directsum_cartesian(exec, g.coords, fields_vecs, ms, iflag, g.domain_size)
end

function calculate_spectrum(::DirectSumBackend, exec::GPUBackend,
        g::AbstractSphericalGrid{FT}, fields_vecs::Tuple, ms::NTuple{2, Int}; kwargs...) where {FT}
    return _gpu_directsum_spherical(exec, g.coords, fields_vecs, ms[1] - 1, g.weights)
end

# ---- Level 2: FFT (FFTW ext CPU; CUFFT ext on a CUDA device). Execution sets the thread count. ----
function calculate_spectrum(::FFTBackend, exec::Union{SerialBackend, ThreadedBackend},
        g::AbstractCartesianGrid, fields_vecs::Tuple, ms::Tuple; kwargs...)
    return _calculate_spectrum_fft(exec, g.coords, fields_vecs, ms; domain_size = g.domain_size, kwargs...)
end
function calculate_spectrum(::FFTBackend, exec::GPUBackend,
        g::AbstractCartesianGrid, fields_vecs::Tuple, ms::Tuple; kwargs...)
    return _calculate_spectrum_gpu_fft(exec, g.coords, fields_vecs, ms; domain_size = g.domain_size, kwargs...)
end

# ---- Level 2: NUFFT (FINUFFT ext CPU; cuFINUFFT ext on a CUDA device) ----
function calculate_spectrum(::NUFFTBackend, exec::Union{SerialBackend, ThreadedBackend},
        g::AbstractCartesianGrid, fields_vecs::Tuple, ms::Tuple; kwargs...)
    return _calculate_spectrum_nufft(exec, g.coords, fields_vecs, ms; domain_size = g.domain_size, kwargs...)
end
function calculate_spectrum(::NUFFTBackend, exec::GPUBackend,
        g::AbstractCartesianGrid, fields_vecs::Tuple, ms::Tuple; kwargs...)
    return _calculate_spectrum_gpu_nufft(exec, g.coords, fields_vecs, ms; domain_size = g.domain_size, kwargs...)
end

# ---- Level 2: SHT / NUFSHT (CPU-serial libraries; execution axis is a documented no-op) ----
function calculate_spectrum(::SHTBackend, ::Union{SerialBackend, ThreadedBackend},
        g::AbstractSphericalGrid, fields_vecs::Tuple, ms::Tuple; kwargs...)
    return _calculate_spectrum_sht(g.coords, fields_vecs, ms; kwargs...)
end
function calculate_spectrum(::NUFSHTBackend, ::Union{SerialBackend, ThreadedBackend},
        g::AbstractSphericalGrid, fields_vecs::Tuple, ms::Tuple; kwargs...)
    return _calculate_spectrum_nufsht(g.coords, fields_vecs, ms; kwargs...)
end

# ---- Least-specific catch-all: clear error for any unsupported (transform, execution, grid) ----
function calculate_spectrum(t::AbstractSpectralBackend, e::AbstractExecutionBackend,
        g::AbstractGrid, fields_vecs::Tuple, ms::Tuple; kwargs...)
    throw(ArgumentError(
        "Unsupported combination transform=$(nameof(typeof(t))), execution=$(nameof(typeof(e))), " *
        "grid=$(nameof(typeof(g))). FFT/NUFFT require a Cartesian grid; SHT/NUFSHT require a spherical " *
        "grid; fast GPU FFT/NUFFT require a CUDA device; spherical transforms have no fast GPU path " *
        "(use transform=DirectSumBackend() with a GPUBackend for a spherical direct sum on device).",
    ))
end

"""
    synthesize(grid::AbstractGrid, coeffs, ms::Tuple;
               transform=DirectSumBackend(), execution=AutoBackend(), real_output=true, iflag=1)

Inverse transform: reconstruct field values at the `grid` points from spectral coefficients
`coeffs` (shape `(ms..., NU)` Cartesian, `(Nθ, Nφ, NU)` spherical) — the inverse of
[`calculate_spectrum`](@ref). Returns a tuple of `NU` field vectors of length `npoints(grid)`.
Useful for spectral filtering (zero out modes, then synthesize) and round-trip validation. With
`real_output=true` the real part is returned (appropriate for real fields). Uses the direct-sum
inverse (so it works for any grid); `execution` may be `SerialBackend`/`ThreadedBackend`.
"""
function synthesize(g::AbstractGrid, coeffs::AbstractArray, ms::Tuple;
        transform::AbstractSpectralBackend = DirectSumBackend(),
        execution::AbstractExecutionBackend = AutoBackend(),
        real_output::Bool = true, iflag::Int = 1)
    return _synthesize(transform, Types.resolve_backend(execution), g, coeffs, ms; real_output = real_output, iflag = iflag)
end

function _synthesize(::DirectSumBackend, exec::Union{SerialBackend, ThreadedBackend},
        g::AbstractCartesianGrid{FT, D}, coeffs::AbstractArray, ms::NTuple{D, Int};
        real_output::Bool = true, iflag::Int = 1) where {FT, D}
    out = _synthesize_cartesian(exec, coeffs, g.coords, ms, iflag, g.domain_size)
    return real_output ? map(v -> real.(v), out) : out
end

function _synthesize(::DirectSumBackend, exec::Union{SerialBackend, ThreadedBackend},
        g::AbstractSphericalGrid, coeffs::AbstractArray, ms::NTuple{2, Int};
        real_output::Bool = true, iflag::Int = 1)
    out = _synthesize_spherical(exec, coeffs, g.coords, ms[1] - 1)
    return real_output ? map(v -> real.(v), out) : out
end

function _synthesize(t::AbstractSpectralBackend, e::AbstractExecutionBackend, g::AbstractGrid,
        coeffs::AbstractArray, ms::Tuple; kwargs...)
    throw(ArgumentError(
        "synthesize supports transform=DirectSumBackend() with SerialBackend/ThreadedBackend execution " *
        "(got transform=$(nameof(typeof(t))), execution=$(nameof(typeof(e)))). The direct-sum inverse " *
        "works for any grid.",
    ))
end

"""
    calculate_spectrum!(coeffs, grid::AbstractGrid, fields, ms;
                        transform=DirectSumBackend(), execution=AutoBackend(), kwargs...)

In-place version of [`calculate_spectrum`](@ref): writes coefficients into the preallocated
`coeffs` array (shape `(ms..., NU)` Cartesian, `(Nθ, Nφ, NU)` spherical) and returns `ks_phys`.
Supported in-place for `transform=DirectSumBackend()` with `SerialBackend`/`ThreadedBackend`; for
the library transforms build a reusable plan with [`plan_spectrum`](@ref) and call
`calculate_spectrum!(coeffs, plan, fields)`.

# Example
```julia
coeffs = zeros(ComplexF64, 64, 64, 2)
for t in 1:nt
    calculate_spectrum!(coeffs, grid, fields[t], (64, 64); execution = ThreadedBackend())
    # ... analyze coeffs ...
end
```
"""
function calculate_spectrum!(
    coeffs::AbstractArray{Complex{T}},
    grid::AbstractGrid,
    fields_vecs::Tuple,
    ms::Tuple;
    transform::AbstractSpectralBackend = DirectSumBackend(),
    execution::AbstractExecutionBackend = AutoBackend(),
    kwargs...,
) where {T}
    return _calculate_spectrum!(coeffs, transform, Types.resolve_backend(execution), grid, fields_vecs, ms; kwargs...)
end

function _calculate_spectrum!(coeffs::AbstractArray{Complex{T}}, ::DirectSumBackend,
        exec::Union{SerialBackend, ThreadedBackend},
        g::AbstractCartesianGrid{FT, D}, fields_vecs::Tuple, ms::NTuple{D, Int};
        iflag::Int = 1, kwargs...) where {T, FT, D}
    return _directsum_cartesian!(exec, coeffs, g.coords, fields_vecs, ms, iflag, g.domain_size)
end

function _calculate_spectrum!(coeffs::AbstractArray{Complex{T}}, ::DirectSumBackend,
        exec::Union{SerialBackend, ThreadedBackend},
        g::AbstractSphericalGrid, fields_vecs::Tuple, ms::NTuple{2, Int}; kwargs...) where {T}
    return _directsum_spherical!(exec, coeffs, g.coords, fields_vecs, ms[1] - 1, g.weights)
end

function _calculate_spectrum!(::AbstractArray, t::AbstractSpectralBackend, e::AbstractExecutionBackend,
        g::AbstractGrid, fields_vecs::Tuple, ms::Tuple; kwargs...)
    throw(ArgumentError(
        "In-place calculate_spectrum! supports transform=DirectSumBackend() with " *
        "SerialBackend/ThreadedBackend on a $(nameof(typeof(g))) (got transform=$(nameof(typeof(t))), " *
        "execution=$(nameof(typeof(e)))). Use the allocating calculate_spectrum, or build a plan with " *
        "plan_spectrum and call calculate_spectrum!(coeffs, plan, fields).",
    ))
end

"""
    plan_spectrum(grid, ::Type{T}, ms; transform=DirectSumBackend(), execution=AutoBackend(), n_transf=1, kwargs...)

Construct a reusable [`AbstractSpectralPlan`](@ref) tied to the fixed geometry of `grid` at
spectral resolution `ms`, transforming `n_transf` co-located fields/slices of element type `T` in
one batched, allocation-free execution. Reuse it across many fields/time steps via
`calculate_spectrum!(coeffs, plan, fields)`. Requires the transform's extension to be loaded
(FFTW for `FFTBackend`, FINUFFT for `NUFFTBackend`, plus CUDA for the GPU fast paths).
"""
function Plans.plan_spectrum(grid::AbstractGrid, ::Type{T}, ms::Tuple;
        transform::AbstractSpectralBackend = DirectSumBackend(),
        execution::AbstractExecutionBackend = AutoBackend(),
        n_transf::Int = 1, kwargs...) where {T}
    return Plans.plan_spectrum(transform, Types.resolve_backend(execution), grid, T, ms; n_transf = n_transf, kwargs...)
end

# ============================================================================
# Internal extension entry points — error until the relevant extension loads. Each carries the
# execution axis where the transform has genuine per-execution behaviour (direct sum, FFT threads).
# ============================================================================

# DirectSum forward: core provides Serial; the OhMyThreads / KernelAbstractions extensions add the
# ThreadedBackend / GPUBackend methods.
_directsum_cartesian!(::SerialBackend, coeffs, coords, fields, ms, iflag, ds) =
    DirectSum._calculate_spectrum_cartesian_direct!(coeffs, coords, fields, ms, iflag, ds)
_directsum_spherical!(::SerialBackend, coeffs, coords, fields, lmax, w) =
    DirectSum._calculate_spectrum_spherical_direct!(coeffs, coords, fields, lmax, w)
_directsum_cartesian!(::ThreadedBackend, args...) = throw(ArgumentError("ThreadedBackend is not loaded. Run `using OhMyThreads`."))
_directsum_spherical!(::ThreadedBackend, args...) = throw(ArgumentError("ThreadedBackend is not loaded. Run `using OhMyThreads`."))

# DirectSum inverse (synthesize): same Serial/Threaded split.
_synthesize_cartesian(::SerialBackend, coeffs, coords, ms, iflag, ds) =
    DirectSum._synthesize_cartesian_direct(coeffs, coords, ms, iflag, ds)
_synthesize_spherical(::SerialBackend, coeffs, coords, lmax) =
    DirectSum._synthesize_spherical_direct(coeffs, coords, lmax)
_synthesize_cartesian(::ThreadedBackend, args...) = throw(ArgumentError("ThreadedBackend is not loaded. Run `using OhMyThreads`."))
_synthesize_spherical(::ThreadedBackend, args...) = throw(ArgumentError("ThreadedBackend is not loaded. Run `using OhMyThreads`."))

# GPU direct-sum (KernelAbstractions ext; portable on any KA device, incl. KA.CPU()).
_gpu_directsum_cartesian(args...; kwargs...) = throw(ArgumentError("GPUBackend is not loaded. Run `using KernelAbstractions`."))
_gpu_directsum_spherical(args...; kwargs...) = throw(ArgumentError("GPUBackend is not loaded. Run `using KernelAbstractions`."))

# GPU FFT is device-generic via AbstractFFTs (FFTW on KA.CPU(), CUFFT on CUDA, rocFFT on AMDGPU, …) —
# the FlowFieldSpectraGPUFFTExt extension adds it. GPU NUFFT is genuinely CUDA-only (FINUFFT.jl's GPU
# support is cuFINUFFT — no portable equivalent exists), added by the cuFINUFFT extension.
_calculate_spectrum_gpu_fft(args...; kwargs...) = throw(ArgumentError("GPU FFT requires `using KernelAbstractions` plus an AbstractFFTs provider for the device's array type (`FFTW` for `KA.CPU()`, `CUDA` for `CUDABackend`, `AMDGPU` for `ROCBackend`, …)."))
_calculate_spectrum_gpu_nufft(args...; kwargs...) = throw(ArgumentError("GPU NUFFT is available only on a CUDA device via cuFINUFFT — run `using CUDA, FINUFFT`. FINUFFT.jl has no portable (non-CUDA) GPU NUFFT; for a portable GPU scattered transform use transform=DirectSumBackend(), or run NUFFTBackend on a CPU execution backend."))

# CPU transform libraries (extensions).
_calculate_spectrum_fft(args...; kwargs...) = throw(ArgumentError("FFTBackend is not loaded. Run `using FFTW`."))
_calculate_spectrum_nufft(args...; kwargs...) = throw(ArgumentError("NUFFTBackend is not loaded. Run `using FINUFFT`."))
_calculate_spectrum_sht(args...; kwargs...) = throw(ArgumentError("SHTBackend is not loaded. Run `using FastSphericalHarmonics`."))
_calculate_spectrum_nufsht(args...; kwargs...) = throw(ArgumentError("NUFSHTBackend is not loaded. Run `using NUFSHT`."))

# Distribution wrappers (Distributed / MPI extensions).
_calculate_spectrum_distributed(args...; kwargs...) = throw(ArgumentError("DistributedBackend is not loaded. Run `using Distributed` (and `addprocs`)."))
_calculate_spectrum_mpi(args...; kwargs...) = throw(ArgumentError("MPIBackend is not loaded. Run `using MPI` and launch under `mpiexec`."))

# ============================================================================
# Distribution partition helpers (shared by the Distributed / MPI extensions). These are pure
# functions — grid subsetting, the per-worker normalization scalar α_w, and the summable-transform
# predicate — with no dependency on Distributed/MPI, so they live in the core (DRY across the two
# extensions). See the plan/issue for the additivity proof: complex coefficients are additive over
# a disjoint point partition; `α_w` compensates the per-kernel `1/N_local` so the partials sum to
# the global result before any energy/binning step.
# ============================================================================

# Rebuild the same grid kind over a subset of point indices, preserving the GLOBAL domain_size /
# slicing the global quadrature weights (never re-infer from the subset — physical_wavenumbers
# depends on the global domain_size).
_subgrid(g::UniformCartesianGrid, idx) =
    UniformCartesianGrid(ntuple(d -> view(g.coords[d], idx), length(g.coords)); domain_size = g.domain_size)
_subgrid(g::NonuniformCartesianGrid, idx) =
    NonuniformCartesianGrid(ntuple(d -> view(g.coords[d], idx), length(g.coords)); domain_size = g.domain_size)
_subgrid(g::ScatteredCartesianGrid, idx) =
    ScatteredCartesianGrid(ntuple(d -> view(g.coords[d], idx), length(g.coords)); domain_size = g.domain_size)
_subgrid(g::ScatteredSphericalGrid, idx) =
    ScatteredSphericalGrid(view(g.coords[1], idx), view(g.coords[2], idx);
        weights = g.weights === nothing ? nothing : view(g.weights, idx))

# Per-worker scalar so `coeff_global = Σ_w α_w · coeff_w`: cartesian/NUFFT and spherical-uniform
# kernels divide by the LOCAL point count, so α_w = N_w/N_global; spherical with explicit weights
# folds no 1/N, so α_w = 1.
_partition_alpha(::AbstractCartesianGrid, Nw, Nglob) = Nw / Nglob
_partition_alpha(g::AbstractSphericalGrid, Nw, Nglob) = g.weights === nothing ? Nw / Nglob : one(Nw / Nglob)

# Transforms whose coefficients are additive over a disjoint point partition (so distribution splits
# the point axis). FFT/SHT need the full grid on one process → they distribute over the batch axis.
_partitionable(::DirectSumBackend) = true
_partitionable(::NUFFTBackend) = true
_partitionable(::NUFSHTBackend) = true
_partitionable(::AbstractSpectralBackend) = false

# Spatial shape of the coefficient array (excluding the trailing NU axis) for a grid + `ms`.
_coeff_spatial(::AbstractCartesianGrid, ms) = Tuple(ms)
_coeff_spatial(::AbstractSphericalGrid, ms) = (ms[1], 2 * (ms[1] - 1) + 1)

# Wavenumber coordinate returned alongside the coefficients (grid-independent of the point subset).
_partition_ks(g::AbstractCartesianGrid, ms) = Grids.physical_wavenumbers(g, NTuple{length(ms), Int}(Tuple(ms)))
_partition_ks(::AbstractSphericalGrid, ms) = (0:(ms[1] - 1), -(ms[1] - 1):(ms[1] - 1))

# Stubs for plotting extension
"""
    plot_spectrum(ks_phys::Tuple, coeffs::AbstractArray; title="Energy Spectrum", kwargs...)

Plot the energy spectrum of a flow field.
Supports plotting 1D isotropic spectra, 2D Cartesian spectral grids, or spherical degree spectra.
Requires `CairoMakie` to be loaded.

# Example
```julia
using CairoMakie
plot_spectrum(ks, coeffs)
```
"""
function plot_spectrum(args...; kwargs...)
    throw(ArgumentError("plot_spectrum requires CairoMakie. Run `using CairoMakie` to enable plotting."))
end

"""
    compare_spectra(spectra_list::Vector; labels::Vector{String}, title="Spectrum Comparison", kwargs...)

Plot a comparative visualization of multiple 1D energy spectra.
Each item in `spectra_list` is a tuple `(k_bins, E_k)`.
Requires `CairoMakie` to be loaded.

# Example
```julia
using CairoMakie
compare_spectra([(k_direct, E_direct), (k_fft, E_fft)]; labels=["Direct Sum", "FFT"])
```
"""
function compare_spectra(args...; kwargs...)
    throw(ArgumentError("compare_spectra requires CairoMakie. Run `using CairoMakie` to enable plotting."))
end

"""
    compare_spectral_analysis(true_coeffs, approx_coeffs; title="Spectral Coefficient Comparison", kwargs...)

Plot direct coefficient comparisons and absolute error maps between two sets of spectral coefficients.
Requires `CairoMakie` to be loaded.
"""
function compare_spectral_analysis(args...; kwargs...)
    throw(ArgumentError("compare_spectral_analysis requires CairoMakie. Run `using CairoMakie` to enable plotting."))
end

# Setup precompilation workload to reduce TTFX
@setup_workload begin
    T = Float64
    x = collect(range(0.0, stop=10.0, length=8))
    y = collect(range(0.0, stop=10.0, length=8))
    xv = vec([x_pt for x_pt in x, y_pt in y])
    yv = vec([y_pt for x_pt in x, y_pt in y])
    u = rand(T, length(xv))
    v = rand(T, length(xv))

    @compile_workload begin
        # Compile Cartesian 2D DirectSum
        cart = ScatteredCartesianGrid((xv, yv); domain_size = (10.0, 10.0))
        c, k = calculate_spectrum(cart, (u, v), (4, 4); transform = DirectSumBackend(), execution = SerialBackend())
        # Compile reductions
        isotropic_spectrum(k, c; num_bins=2)
        transect_spectrum(k, c, (1,))

        # Compile Spherical 2D DirectSum (theta, phi)
        theta = rand(T, 8) .* π
        phi = rand(T, 8) .* 2π
        sph = ScatteredSphericalGrid(theta, phi)
        cs, ks = calculate_spectrum(sph, (rand(T, 8),), (2, 3); transform = DirectSumBackend(), execution = SerialBackend())
        spherical_energy_spectrum(cs; lmax=1)
    end
end

end # module FlowFieldSpectra
