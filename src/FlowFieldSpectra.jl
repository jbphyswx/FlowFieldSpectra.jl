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
using .Grids: Grids, AbstractGrid, AbstractCartesianGrid, AbstractSphericalGrid, UniformCartesianGrid, NonuniformCartesianGrid, ScatteredCartesianGrid, StructuredSphericalGrid, ScatteredSphericalGrid, AbstractQuadrature, ClenshawCurtis, GaussLegendre, Equiangular, spatial_dims, ndims_spatial, spatial_size, npoints
using .Preprocessing: AbstractWindow, NoWindow, Hann, Hamming, Blackman, Tukey, AbstractDetrend, NoDetrend, Demean, LinearDetrend, Preprocess, dpss
using .Normalization: AbstractSidedness, OneSided, TwoSided, AbstractScaling, DensityScaling, PowerScaling, SpectralConvention
using .Problem: Problem, TransformProblem, batch_shape, stack_fields, coeff_eltype
using .Plans: Plans, AbstractSpectralPlan, plan_spectrum
using .DirectSum: DirectSum, sph_mode_index
using .Reductions: isotropic_spectrum, isotropic_spectrum!, transect_spectrum, transect_spectrum!, spherical_energy_spectrum, spherical_energy_spectrum!, cross_spectrum, cospectrum, quadspectrum, anisotropic_spectrum
using .Operators: spectral_divergence, spectral_vorticity, compensate, band_energy
using .Averaging: welch_power_spectrum, coherence_spectrum
using .LombScargle: lomb_scargle

# Transform backends (which spectral math)
export AbstractSpectralBackend, DirectSumBackend, FFTBackend, NUFFTBackend, SHTBackend, NUFSHTBackend
# Execution backends (where/how it runs)
export AbstractExecutionBackend, SerialBackend, ThreadedBackend, GPUBackend, DistributedBackend, MPIBackend, AutoBackend
# Grids
export AbstractGrid, AbstractCartesianGrid, AbstractSphericalGrid, UniformCartesianGrid, NonuniformCartesianGrid, ScatteredCartesianGrid, StructuredSphericalGrid, ScatteredSphericalGrid
export AbstractQuadrature, ClenshawCurtis, GaussLegendre, Equiangular, spatial_dims, ndims_spatial, spatial_size, npoints
# Preprocessing & Normalization (typed configuration)
export AbstractWindow, NoWindow, Hann, Hamming, Blackman, Tukey
export AbstractDetrend, NoDetrend, Demean, LinearDetrend, Preprocess, dpss
export AbstractSidedness, OneSided, TwoSided, AbstractScaling, DensityScaling, PowerScaling, SpectralConvention
export TransformProblem
export AbstractSpectralPlan, plan_spectrum
# APIs
export calculate_spectrum, calculate_spectrum!, synthesize, isotropic_spectrum, isotropic_spectrum!, transect_spectrum, transect_spectrum!, spherical_energy_spectrum, spherical_energy_spectrum!, sph_mode_index
export spectral_divergence, spectral_vorticity, compensate, band_energy
export cross_spectrum, cospectrum, quadspectrum, anisotropic_spectrum
export welch_power_spectrum, coherence_spectrum, lomb_scargle
export plot_spectrum, compare_spectra, compare_spectral_analysis

# =============================================================================
# AutoBackend resolution — best available LOCAL execution backend. `ThreadedBackend` iff the
# OhMyThreads extension is loaded and `Threads.nthreads() > 1`, else `SerialBackend`. GPU/Distributed/
# MPI are never auto-selected (they need explicit device/process context).
# =============================================================================
function Types.resolve_backend(::AutoBackend)
    threaded = Base.get_extension(@__MODULE__, :FlowFieldSpectraOhMyThreadsExt) !== nothing
    return (threaded && Threads.nthreads() > 1) ? ThreadedBackend() : SerialBackend()
end

"""
    calculate_spectrum(grid::AbstractGrid, field, ms::Tuple;
                       transform=DirectSumBackend(), execution=AutoBackend(), kwargs...)

Spectral coefficients and physical wavenumbers of `field` sampled on `grid`. The coordinate system is
the grid type — there is no coordinate guessing. The two backend axes compose freely:

- `transform::AbstractSpectralBackend` — the spectral math (`DirectSumBackend` (default),
  `FFTBackend`, `NUFFTBackend`, `SHTBackend`, `NUFSHTBackend`).
- `execution::AbstractExecutionBackend` — where/how it runs (`SerialBackend`, `ThreadedBackend`,
  `GPUBackend{B}`, `DistributedBackend{Inner}`, `MPIBackend{Inner}`, `AutoBackend` (default)).

# Data model
`field` is an `AbstractArray` shaped `(spatial…, batch…)`: the first `ndims_spatial(grid)` dims are
spatial (and must equal `spatial_size(grid)`); every trailing dim is a **batch** dim (components,
levels, time, ensemble — any number), carried through and preserved. Tensor-product grids take an
`(N_1,…,N_D, batch…)` tensor; scattered grids take `(N, batch…)`; structured spherical grids take
`(Nθ, Nφ, batch…)`. A `Tuple` of equal-shaped arrays is a convenience that stacks them along a new
trailing batch axis.

`ms` is the spectral resolution: Cartesian `(m_1,…,m_D)`; spherical `(Nθ, Nφ)` with `lmax = Nθ-1`.

# Returns
`(coeffs, ks_phys)` — complex coefficients `(ms…, batch…)` (Cartesian) / `(Nθ, Nφ, batch…)`
(spherical), and the physical wavenumber coordinates per spectral axis.

# Example
```julia
using FlowFieldSpectra, FFTW
g = UniformCartesianGrid(; domain = (2π, 2π), n = (64, 64))
u = rand(64, 64, 3)                       # (Nx, Ny) with a 3-slice batch
coeffs, ks = calculate_spectrum(g, u, (64, 64); transform = FFTBackend())  # coeffs (64,64,3)
```
"""
function calculate_spectrum(grid::AbstractGrid, field::AbstractArray, ms::Tuple;
        transform::AbstractSpectralBackend = DirectSumBackend(),
        execution::AbstractExecutionBackend = AutoBackend(), kwargs...)
    return calculate_spectrum(transform, Types.resolve_backend(execution), grid, field, ms; kwargs...)
end

# Multi-field convenience: stack equal-shaped arrays along a new trailing batch axis.
calculate_spectrum(grid::AbstractGrid, fields::Tuple, ms::Tuple; kwargs...) =
    calculate_spectrum(grid, stack_fields(fields), ms; kwargs...)

# =============================================================================
# Canonical two-axis dispatch: calculate_spectrum(transform, execution, grid, field, ms; …).
# =============================================================================

# ---- Level 1: distribution wrappers unwrap to the Distributed / MPI extension hooks ----
calculate_spectrum(t::AbstractSpectralBackend, e::DistributedBackend, g::AbstractGrid, field::AbstractArray, ms::Tuple; kwargs...) =
    _calculate_spectrum_distributed(t, e, g, field, ms; kwargs...)
calculate_spectrum(t::AbstractSpectralBackend, e::MPIBackend, g::AbstractGrid, field::AbstractArray, ms::Tuple; kwargs...) =
    _calculate_spectrum_mpi(t, e, g, field, ms; kwargs...)

# ---- Level 2: DirectSum on a local CPU backend (core Serial; OhMyThreads ext Threaded) ----
function calculate_spectrum(::DirectSumBackend, exec::Union{SerialBackend, ThreadedBackend},
        g::AbstractCartesianGrid{FT, D}, field::AbstractArray, ms::NTuple{D, Int};
        iflag::Int = 1, kwargs...) where {FT, D}
    prob = TransformProblem(g, field)
    coeffs = zeros(Complex{FT}, ms..., batch_shape(prob)...)
    ks = _directsum_cartesian!(exec, coeffs, g, field, ms, iflag)
    return coeffs, ks
end

function calculate_spectrum(::DirectSumBackend, exec::Union{SerialBackend, ThreadedBackend},
        g::AbstractSphericalGrid{FT}, field::AbstractArray, ms::NTuple{2, Int}; kwargs...) where {FT}
    prob = TransformProblem(g, field)
    lmax = ms[1] - 1
    coeffs = zeros(Complex{FT}, lmax + 1, 2 * lmax + 1, batch_shape(prob)...)
    ks = _directsum_spherical!(exec, coeffs, g, field, lmax)
    return coeffs, ks
end

# ---- Level 2: DirectSum on GPU (KernelAbstractions ext; portable on any KA device) ----
calculate_spectrum(::DirectSumBackend, exec::GPUBackend, g::AbstractCartesianGrid{FT, D}, field::AbstractArray, ms::NTuple{D, Int}; iflag::Int = 1, kwargs...) where {FT, D} =
    _gpu_directsum_cartesian(exec, g, field, ms, iflag)
calculate_spectrum(::DirectSumBackend, exec::GPUBackend, g::AbstractSphericalGrid{FT}, field::AbstractArray, ms::NTuple{2, Int}; kwargs...) where {FT} =
    _gpu_directsum_spherical(exec, g, field, ms[1] - 1)

# ---- Level 2: FFT (FFTW ext CPU; device-generic GPU FFT). Uniform grid only. ----
calculate_spectrum(::FFTBackend, exec::Union{SerialBackend, ThreadedBackend}, g::UniformCartesianGrid, field::AbstractArray, ms::Tuple; kwargs...) =
    _calculate_spectrum_fft(exec, g, field, ms; kwargs...)
calculate_spectrum(::FFTBackend, exec::GPUBackend, g::UniformCartesianGrid, field::AbstractArray, ms::Tuple; kwargs...) =
    _calculate_spectrum_gpu_fft(exec, g, field, ms; kwargs...)

# ---- Level 2: NUFFT (FINUFFT ext CPU; cuFINUFFT ext on CUDA). Scattered point clouds. ----
# (A nonuniform-but-gridded NonuniformCartesianGrid uses DirectSumBackend; a separable per-axis fast
# NUFFT for it is a planned optimization and routes to the catch-all until implemented.)
calculate_spectrum(::NUFFTBackend, exec::Union{SerialBackend, ThreadedBackend}, g::ScatteredCartesianGrid, field::AbstractArray, ms::Tuple; kwargs...) =
    _calculate_spectrum_nufft(exec, g, field, ms; kwargs...)
calculate_spectrum(::NUFFTBackend, exec::GPUBackend, g::ScatteredCartesianGrid, field::AbstractArray, ms::Tuple; kwargs...) =
    _calculate_spectrum_gpu_nufft(exec, g, field, ms; kwargs...)

# ---- Level 2: SHT / NUFSHT (execution axis handled inside the extension) ----
calculate_spectrum(::SHTBackend, ::Union{SerialBackend, ThreadedBackend}, g::StructuredSphericalGrid, field::AbstractArray, ms::Tuple; kwargs...) =
    _calculate_spectrum_sht(g, field, ms; kwargs...)
calculate_spectrum(::NUFSHTBackend, exec::AbstractExecutionBackend, g::AbstractSphericalGrid, field::AbstractArray, ms::Tuple; kwargs...) =
    _calculate_spectrum_nufsht(exec, g, field, ms; kwargs...)

# ---- Least-specific catch-all: clear error for any unsupported (transform, execution, grid) ----
function calculate_spectrum(t::AbstractSpectralBackend, e::AbstractExecutionBackend,
        g::AbstractGrid, field::AbstractArray, ms::Tuple; kwargs...)
    throw(ArgumentError(
        "Unsupported combination transform=$(nameof(typeof(t))), execution=$(nameof(typeof(e))), " *
        "grid=$(nameof(typeof(g))). FFT needs a UniformCartesianGrid; NUFFT needs a ScatteredCartesianGrid " *
        "(a nonuniform-but-gridded NonuniformCartesianGrid uses DirectSumBackend); SHT needs a " *
        "StructuredSphericalGrid; NUFSHT needs a spherical grid; fast GPU FFT/NUFFT need a CUDA device. " *
        "DirectSumBackend works on any grid (and on a GPUBackend).",
    ))
end

# =============================================================================
# synthesize — inverse transform, reconstructs a field `(spatial…, batch…)`.
# =============================================================================

"""
    synthesize(grid, coeffs, ms::Tuple; transform=DirectSumBackend(), execution=AutoBackend(),
               real_output=true, iflag=1)

Inverse of [`calculate_spectrum`](@ref): reconstruct field values at the `grid` points from
coefficients `coeffs` (`(ms…, batch…)` Cartesian, `(Nθ, Nφ, batch…)` spherical). Returns an array
`(spatial…, batch…)`; `real_output=true` returns its real part. Uses the direct-sum inverse (works
for any grid) with `SerialBackend`/`ThreadedBackend`.
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
    ss = spatial_size(g)
    batch = ntuple(i -> size(coeffs, D + i), ndims(coeffs) - D)
    out = Array{Complex{FT}}(undef, ss..., batch...)
    _synthesize_cartesian!(exec, out, g, coeffs, ms, iflag)
    return real_output ? real.(out) : out
end

function _synthesize(::DirectSumBackend, exec::Union{SerialBackend, ThreadedBackend},
        g::AbstractSphericalGrid{FT}, coeffs::AbstractArray, ms::NTuple{2, Int};
        real_output::Bool = true, iflag::Int = 1) where {FT}
    ss = spatial_size(g)
    batch = ntuple(i -> size(coeffs, 2 + i), ndims(coeffs) - 2)
    out = Array{Complex{FT}}(undef, ss..., batch...)
    _synthesize_spherical!(exec, out, g, coeffs, ms[1] - 1)
    return real_output ? real.(out) : out
end

function _synthesize(t::AbstractSpectralBackend, e::AbstractExecutionBackend, g::AbstractGrid,
        coeffs::AbstractArray, ms::Tuple; kwargs...)
    throw(ArgumentError(
        "synthesize supports transform=DirectSumBackend() with SerialBackend/ThreadedBackend " *
        "(got transform=$(nameof(typeof(t))), execution=$(nameof(typeof(e)))).",
    ))
end

# =============================================================================
# In-place calculate_spectrum! (DirectSum core; plan form lives in the extensions).
# =============================================================================

"""
    calculate_spectrum!(coeffs, grid, field, ms; transform=DirectSumBackend(), execution=AutoBackend(), kwargs...)

In-place [`calculate_spectrum`](@ref): write coefficients `(ms…, batch…)` into the preallocated
`coeffs` and return `ks_phys`. Supported for `transform=DirectSumBackend()` with
`SerialBackend`/`ThreadedBackend`; for the library transforms build a reusable plan with
[`plan_spectrum`](@ref) and call `calculate_spectrum!(coeffs, plan, field)`.
"""
function calculate_spectrum!(coeffs::AbstractArray{Complex{T}}, grid::AbstractGrid, field::AbstractArray, ms::Tuple;
        transform::AbstractSpectralBackend = DirectSumBackend(),
        execution::AbstractExecutionBackend = AutoBackend(), kwargs...) where {T}
    return _calculate_spectrum!(coeffs, transform, Types.resolve_backend(execution), grid, field, ms; kwargs...)
end

function _calculate_spectrum!(coeffs::AbstractArray{Complex{T}}, ::DirectSumBackend,
        exec::Union{SerialBackend, ThreadedBackend}, g::AbstractCartesianGrid{FT, D},
        field::AbstractArray, ms::NTuple{D, Int}; iflag::Int = 1, kwargs...) where {T, FT, D}
    return _directsum_cartesian!(exec, coeffs, g, field, ms, iflag)
end

function _calculate_spectrum!(coeffs::AbstractArray{Complex{T}}, ::DirectSumBackend,
        exec::Union{SerialBackend, ThreadedBackend}, g::AbstractSphericalGrid,
        field::AbstractArray, ms::NTuple{2, Int}; kwargs...) where {T}
    return _directsum_spherical!(exec, coeffs, g, field, ms[1] - 1)
end

function _calculate_spectrum!(::AbstractArray, t::AbstractSpectralBackend, e::AbstractExecutionBackend,
        g::AbstractGrid, field::AbstractArray, ms::Tuple; kwargs...)
    throw(ArgumentError(
        "In-place calculate_spectrum! supports transform=DirectSumBackend() with " *
        "SerialBackend/ThreadedBackend (got transform=$(nameof(typeof(t))), execution=$(nameof(typeof(e)))). " *
        "Build a plan with plan_spectrum and call calculate_spectrum!(coeffs, plan, field).",
    ))
end

"""
    plan_spectrum(grid, ::Type{T}, ms; transform=DirectSumBackend(), execution=AutoBackend(), batch=(), kwargs...)

Reusable [`AbstractSpectralPlan`](@ref) for the fixed geometry of `grid` at resolution `ms`,
transforming a field with trailing batch shape `batch` of element type `T` in one batched,
allocation-free execution. Reuse via `calculate_spectrum!(coeffs, plan, field)`.
"""
function Plans.plan_spectrum(grid::AbstractGrid, ::Type{T}, ms::Tuple;
        transform::AbstractSpectralBackend = DirectSumBackend(),
        execution::AbstractExecutionBackend = AutoBackend(),
        batch::Tuple = (), kwargs...) where {T}
    return Plans.plan_spectrum(transform, Types.resolve_backend(execution), grid, T, ms; batch = batch, kwargs...)
end

# =============================================================================
# Internal extension entry points — error until the relevant extension loads. Each takes the grid +
# field tensor directly (spatial/batch split from the grid).
# =============================================================================

# DirectSum forward: core Serial; OhMyThreads / KernelAbstractions extensions add Threaded / GPU.
_directsum_cartesian!(::SerialBackend, coeffs, g, field, ms, iflag) =
    DirectSum._calculate_spectrum_cartesian_direct!(coeffs, g, field, ms, iflag)
_directsum_spherical!(::SerialBackend, coeffs, g, field, lmax) =
    DirectSum._calculate_spectrum_spherical_direct!(coeffs, g, field, lmax)
_directsum_cartesian!(::ThreadedBackend, args...) = throw(ArgumentError("ThreadedBackend is not loaded. Run `using OhMyThreads`."))
_directsum_spherical!(::ThreadedBackend, args...) = throw(ArgumentError("ThreadedBackend is not loaded. Run `using OhMyThreads`."))

# DirectSum inverse (synthesize): same Serial/Threaded split.
_synthesize_cartesian!(::SerialBackend, out, g, coeffs, ms, iflag) =
    DirectSum._synthesize_cartesian_direct!(out, g, coeffs, ms, iflag)
_synthesize_spherical!(::SerialBackend, out, g, coeffs, lmax) =
    DirectSum._synthesize_spherical_direct!(out, g, coeffs, lmax)
_synthesize_cartesian!(::ThreadedBackend, args...) = throw(ArgumentError("ThreadedBackend is not loaded. Run `using OhMyThreads`."))
_synthesize_spherical!(::ThreadedBackend, args...) = throw(ArgumentError("ThreadedBackend is not loaded. Run `using OhMyThreads`."))

# GPU direct-sum (KernelAbstractions ext; portable on any KA device, incl. KA.CPU()).
_gpu_directsum_cartesian(args...; kwargs...) = throw(ArgumentError("GPUBackend is not loaded. Run `using KernelAbstractions`."))
_gpu_directsum_spherical(args...; kwargs...) = throw(ArgumentError("GPUBackend is not loaded. Run `using KernelAbstractions`."))

# GPU FFT is device-generic via AbstractFFTs; GPU NUFFT is CUDA-only (cuFINUFFT).
_calculate_spectrum_gpu_fft(args...; kwargs...) = throw(ArgumentError("GPU FFT requires `using KernelAbstractions` plus an AbstractFFTs provider (`FFTW` for `KA.CPU()`, `CUDA` for `CUDABackend`, …)."))
_calculate_spectrum_gpu_nufft(args...; kwargs...) = throw(ArgumentError("GPU NUFFT is CUDA-only via cuFINUFFT — run `using CUDA, FINUFFT`. For a portable GPU scattered transform use transform=DirectSumBackend()."))

# CPU transform libraries (extensions).
_calculate_spectrum_fft(args...; kwargs...) = throw(ArgumentError("FFTBackend is not loaded. Run `using FFTW`."))
_calculate_spectrum_nufft(args...; kwargs...) = throw(ArgumentError("NUFFTBackend is not loaded. Run `using FINUFFT`."))
_calculate_spectrum_sht(args...; kwargs...) = throw(ArgumentError("SHTBackend is not loaded. Run `using FastSphericalHarmonics`."))
_calculate_spectrum_nufsht(args...; kwargs...) = throw(ArgumentError("NUFSHTBackend is not loaded. Run `using NUFSHT`."))

# Distribution wrappers (Distributed / MPI extensions).
_calculate_spectrum_distributed(args...; kwargs...) = throw(ArgumentError("DistributedBackend is not loaded. Run `using Distributed` (and `addprocs`)."))
_calculate_spectrum_mpi(args...; kwargs...) = throw(ArgumentError("MPIBackend is not loaded. Run `using MPI` and launch under `mpiexec`."))

# =============================================================================
# Distribution partition helpers (shared by the Distributed / MPI extensions). Complex coefficients
# are additive over a disjoint point partition; `α_w` compensates the per-kernel `1/N_local`.
# =============================================================================

# Subgrid over a subset of point indices (point-partitionable grids: scattered), preserving the
# GLOBAL domain_size / slicing the global weights.
_subgrid(g::ScatteredCartesianGrid, idx) =
    ScatteredCartesianGrid(ntuple(d -> view(g.coords[d], idx), length(g.coords)); domain_size = g.domain_size)
_subgrid(g::ScatteredSphericalGrid, idx) =
    ScatteredSphericalGrid(view(g.coords[1], idx), view(g.coords[2], idx);
        weights = g.weights === nothing ? nothing : view(g.weights, idx))

# Per-worker scalar so `coeff_global = Σ_w α_w · coeff_w`.
_partition_alpha(::AbstractCartesianGrid, Nw, Nglob) = Nw / Nglob
_partition_alpha(g::AbstractSphericalGrid, Nw, Nglob) = g.weights === nothing ? Nw / Nglob : one(Nw / Nglob)

# Transforms whose coefficients are additive over a disjoint point partition.
_partitionable(::DirectSumBackend) = true
_partitionable(::NUFFTBackend) = true
_partitionable(::NUFSHTBackend) = true
_partitionable(::AbstractSpectralBackend) = false

# Spatial (spectral) shape of the coefficient array for a grid + `ms`.
_coeff_spatial(::AbstractCartesianGrid, ms) = Tuple(ms)
_coeff_spatial(::AbstractSphericalGrid, ms) = (ms[1], 2 * (ms[1] - 1) + 1)

# Wavenumber coordinate returned alongside the coefficients.
_partition_ks(g::AbstractCartesianGrid, ms) = Grids.physical_wavenumbers(g, NTuple{length(ms), Int}(Tuple(ms)))
_partition_ks(::AbstractSphericalGrid, ms) = (0:(ms[1] - 1), -(ms[1] - 1):(ms[1] - 1))

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
# Precompilation workload (DirectSum + reductions on the new array/batch API).
# =============================================================================
@setup_workload begin
    T = Float64
    @compile_workload begin
        # Cartesian 2D DirectSum on a uniform grid with a small batch.
        cart = UniformCartesianGrid(; domain = (10.0, 10.0), n = (8, 8))
        u = rand(T, 8, 8, 2)
        c, k = calculate_spectrum(cart, u, (4, 4); transform = DirectSumBackend(), execution = SerialBackend())
        isotropic_spectrum(k, c; num_bins = 2)
        transect_spectrum(k, c, (1,))
        synthesize(cart, c, (4, 4); execution = SerialBackend())

        # Spherical 2D DirectSum (θ, φ) scattered.
        theta = rand(T, 8) .* π
        phi = rand(T, 8) .* 2π
        sph = ScatteredSphericalGrid(theta, phi)
        fθ = rand(T, 8)
        cs, ks = calculate_spectrum(sph, fθ, (2, 3); transform = DirectSumBackend(), execution = SerialBackend())
        spherical_energy_spectrum(cs; lmax = 1)
    end
end

end # module FlowFieldSpectra
