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

using .Types: AbstractSpectralBackend, DirectSumBackend, FFTBackend, NUFFTBackend, SHTBackend, NUFSHTBackend, ThreadedBackend, GPUBackend, AutoBackend
using .Grids: AbstractGrid, AbstractCartesianGrid, AbstractSphericalGrid, UniformCartesianGrid, NonuniformCartesianGrid, ScatteredCartesianGrid, StructuredSphericalGrid, ScatteredSphericalGrid, AbstractQuadrature, ClenshawCurtis, GaussLegendre, Equiangular, physical_wavenumbers, spatial_dims, npoints
using .Preprocessing: AbstractWindow, NoWindow, Hann, Hamming, Blackman, Tukey, AbstractDetrend, NoDetrend, Demean, LinearDetrend, Preprocess
using .Normalization: AbstractSidedness, OneSided, TwoSided, AbstractScaling, Density, Power, SpectralConvention
using .Problem: TransformProblem
using .Plans: AbstractSpectralPlan, plan_spectrum
using .DirectSum: sph_mode_index
using .Reductions: isotropic_spectrum, isotropic_spectrum!, transect_spectrum, transect_spectrum!, spherical_energy_spectrum, spherical_energy_spectrum!

# Export Types
export AbstractSpectralBackend, DirectSumBackend, FFTBackend, NUFFTBackend, SHTBackend, NUFSHTBackend, ThreadedBackend, GPUBackend, AutoBackend

# Export Grids
export AbstractGrid, AbstractCartesianGrid, AbstractSphericalGrid, UniformCartesianGrid, NonuniformCartesianGrid, ScatteredCartesianGrid, StructuredSphericalGrid, ScatteredSphericalGrid
export AbstractQuadrature, ClenshawCurtis, GaussLegendre, Equiangular

# Export Preprocessing & Normalization (typed configuration)
export AbstractWindow, NoWindow, Hann, Hamming, Blackman, Tukey
export AbstractDetrend, NoDetrend, Demean, LinearDetrend, Preprocess
export AbstractSidedness, OneSided, TwoSided, AbstractScaling, Density, Power, SpectralConvention
export TransformProblem
export AbstractSpectralPlan, plan_spectrum

# Export APIs
export calculate_spectrum, calculate_spectrum!, isotropic_spectrum, isotropic_spectrum!, transect_spectrum, transect_spectrum!, spherical_energy_spectrum, spherical_energy_spectrum!, sph_mode_index
export plot_spectrum, compare_spectra, compare_spectral_analysis


"""
    calculate_spectrum(backend::AbstractSpectralBackend, grid::AbstractGrid, fields, ms::Tuple; kwargs...)
    calculate_spectrum(grid::AbstractGrid, fields, ms::Tuple; backend=DirectSumBackend(), kwargs...)

Calculate the spectral coefficients and physical wavenumbers for one or more fields sampled on
an explicit `grid`. The coordinate system is determined by the grid type — there is no
coordinate guessing.

# Arguments
- `backend::AbstractSpectralBackend`: spectral backend (default `DirectSumBackend()`).
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
coeffs, ks = calculate_spectrum(FFTBackend(), grid, (u, v), (N, N))
```
"""
function calculate_spectrum(
    grid::AbstractGrid,
    fields_vecs::Tuple,
    ms::Tuple;
    backend::AbstractSpectralBackend = DirectSumBackend(),
    kwargs...,
)
    return calculate_spectrum(backend, grid, fields_vecs, ms; kwargs...)
end

# =============================================================================
# Canonical (backend, grid) dispatch — coordinate system is the grid type.
# =============================================================================

# Per-grid keyword bundle forwarded to backend implementations.
_grid_kwargs(g::AbstractCartesianGrid) = (; domain_size = g.domain_size)
_grid_kwargs(g::AbstractSphericalGrid) = (; weights = g.weights)

# ---- DirectSum: route directly to the Cartesian / spherical kernels ----
function calculate_spectrum(
    ::DirectSumBackend,
    g::AbstractCartesianGrid{FT, D},
    fields_vecs::Tuple,
    ms::NTuple{D, Int};
    iflag::Int = 1,
    kwargs...,
) where {FT, D}
    NU = length(fields_vecs)
    coeffs = zeros(Complex{FT}, ms..., NU)
    ks = DirectSum._calculate_spectrum_cartesian_direct!(
        coeffs, g.coords, fields_vecs, ms, iflag, g.domain_size,
    )
    return coeffs, ks
end

function calculate_spectrum(
    ::DirectSumBackend,
    g::AbstractSphericalGrid{FT},
    fields_vecs::Tuple,
    ms::NTuple{2, Int};
    kwargs...,
) where {FT}
    NU = length(fields_vecs)
    lmax = ms[1] - 1
    coeffs = zeros(Complex{FT}, lmax + 1, 2 * lmax + 1, NU)
    ks = DirectSum._calculate_spectrum_spherical_direct!(
        coeffs, g.coords, fields_vecs, lmax, g.weights,
    )
    return coeffs, ks
end

# ---- Cartesian fast backends (extensions implement the `_calculate_spectrum_*`) ----
function calculate_spectrum(::FFTBackend, g::AbstractCartesianGrid, fields_vecs::Tuple, ms::Tuple; kwargs...)
    return _calculate_spectrum_fft(g.coords, fields_vecs, ms; domain_size = g.domain_size, kwargs...)
end

function calculate_spectrum(::NUFFTBackend, g::AbstractCartesianGrid, fields_vecs::Tuple, ms::Tuple; kwargs...)
    return _calculate_spectrum_nufft(g.coords, fields_vecs, ms; domain_size = g.domain_size, kwargs...)
end

# ---- Spherical fast backends ----
function calculate_spectrum(::SHTBackend, g::AbstractSphericalGrid, fields_vecs::Tuple, ms::Tuple; kwargs...)
    return _calculate_spectrum_sht(g.coords, fields_vecs, ms; kwargs...)
end

function calculate_spectrum(::NUFSHTBackend, g::AbstractSphericalGrid, fields_vecs::Tuple, ms::Tuple; kwargs...)
    return _calculate_spectrum_nufsht(g.coords, fields_vecs, ms; kwargs...)
end

# ---- Threaded (extension dispatches on grid type; no coordinate heuristic) ----
function calculate_spectrum(::ThreadedBackend, g::AbstractCartesianGrid{FT, D}, fields_vecs::Tuple, ms::NTuple{D, Int}; iflag::Int = 1, kwargs...) where {FT, D}
    NU = length(fields_vecs)
    coeffs = zeros(Complex{FT}, ms..., NU)
    ks = _calculate_spectrum_threaded_cartesian!(coeffs, g.coords, fields_vecs, ms, iflag, g.domain_size)
    return coeffs, ks
end

function calculate_spectrum(::ThreadedBackend, g::AbstractSphericalGrid{FT}, fields_vecs::Tuple, ms::NTuple{2, Int}; kwargs...) where {FT}
    NU = length(fields_vecs)
    lmax = ms[1] - 1
    coeffs = zeros(Complex{FT}, lmax + 1, 2 * lmax + 1, NU)
    ks = _calculate_spectrum_threaded_spherical!(coeffs, g.coords, fields_vecs, lmax, g.weights)
    return coeffs, ks
end

# ---- GPU (extension dispatches on grid type; no coordinate heuristic) ----
function calculate_spectrum(b::GPUBackend, g::AbstractCartesianGrid{FT, D}, fields_vecs::Tuple, ms::NTuple{D, Int}; iflag::Int = 1, kwargs...) where {FT, D}
    return _calculate_spectrum_gpu_cartesian(b, g.coords, fields_vecs, ms, iflag, g.domain_size)
end

function calculate_spectrum(b::GPUBackend, g::AbstractSphericalGrid{FT}, fields_vecs::Tuple, ms::NTuple{2, Int}; kwargs...) where {FT}
    return _calculate_spectrum_gpu_spherical(b, g.coords, fields_vecs, ms[1] - 1, g.weights)
end

# ---- AutoBackend ----
function calculate_spectrum(::AutoBackend, g::AbstractGrid, fields_vecs::Tuple, ms::Tuple; kwargs...)
    if isdefined(Main, :OhMyThreads) && Threads.nthreads() > 1
        return calculate_spectrum(ThreadedBackend(), g, fields_vecs, ms; kwargs...)
    else
        return calculate_spectrum(DirectSumBackend(), g, fields_vecs, ms; kwargs...)
    end
end

# ---- Friendly error for unsupported (backend, grid) combinations ----
function calculate_spectrum(b::AbstractSpectralBackend, g::AbstractGrid, fields_vecs::Tuple, ms::Tuple; kwargs...)
    throw(ArgumentError(
        "$(nameof(typeof(b))) does not support a $(nameof(typeof(g))). " *
        "FFTBackend/NUFFTBackend require a Cartesian grid; SHTBackend/NUFSHTBackend require a spherical grid.",
    ))
end

"""
    calculate_spectrum!(coeffs, backend, grid::AbstractGrid, fields, ms; kwargs...)

In-place version of [`calculate_spectrum`](@ref): writes coefficients into the preallocated
`coeffs` array (shape `(ms..., NU)` Cartesian, `(Nθ, Nφ, NU)` spherical) and returns `ks_phys`.
Supported in-place for `DirectSumBackend` and `ThreadedBackend`; other backends should use the
allocating `calculate_spectrum`.

# Example
```julia
coeffs = zeros(ComplexF64, 64, 64, 2)
for t in 1:nt
    calculate_spectrum!(coeffs, DirectSumBackend(), grid, fields[t], (64, 64))
    # ... analyze coeffs ...
end
```
"""
function calculate_spectrum!(
    coeffs::AbstractArray{Complex{T}},
    backend::AbstractSpectralBackend,
    grid::AbstractGrid,
    fields_vecs::Tuple,
    ms::Tuple;
    kwargs...,
) where {T}
    return _calculate_spectrum!(coeffs, backend, grid, fields_vecs, ms; kwargs...)
end

# ---- DirectSum (in-place, routes by grid type) ----
function _calculate_spectrum!(coeffs::AbstractArray{Complex{T}}, ::DirectSumBackend,
        g::AbstractCartesianGrid{FT, D}, fields_vecs::Tuple, ms::NTuple{D, Int};
        iflag::Int = 1, kwargs...) where {T, FT, D}
    return DirectSum._calculate_spectrum_cartesian_direct!(coeffs, g.coords, fields_vecs, ms, iflag, g.domain_size)
end

function _calculate_spectrum!(coeffs::AbstractArray{Complex{T}}, ::DirectSumBackend,
        g::AbstractSphericalGrid, fields_vecs::Tuple, ms::NTuple{2, Int}; kwargs...) where {T}
    return DirectSum._calculate_spectrum_spherical_direct!(coeffs, g.coords, fields_vecs, ms[1] - 1, g.weights)
end

# ---- Threaded (in-place, routes by grid type) ----
function _calculate_spectrum!(coeffs::AbstractArray{Complex{T}}, ::ThreadedBackend,
        g::AbstractCartesianGrid{FT, D}, fields_vecs::Tuple, ms::NTuple{D, Int};
        iflag::Int = 1, kwargs...) where {T, FT, D}
    return _calculate_spectrum_threaded_cartesian!(coeffs, g.coords, fields_vecs, ms, iflag, g.domain_size)
end

function _calculate_spectrum!(coeffs::AbstractArray{Complex{T}}, ::ThreadedBackend,
        g::AbstractSphericalGrid, fields_vecs::Tuple, ms::NTuple{2, Int}; kwargs...) where {T}
    return _calculate_spectrum_threaded_spherical!(coeffs, g.coords, fields_vecs, ms[1] - 1, g.weights)
end

# ---- AutoBackend (in-place) ----
function _calculate_spectrum!(coeffs::AbstractArray{Complex{T}}, ::AutoBackend,
        grid::AbstractGrid, fields_vecs::Tuple, ms::Tuple; kwargs...) where {T}
    if isdefined(Main, :OhMyThreads) && Threads.nthreads() > 1
        return _calculate_spectrum!(coeffs, ThreadedBackend(), grid, fields_vecs, ms; kwargs...)
    else
        return _calculate_spectrum!(coeffs, DirectSumBackend(), grid, fields_vecs, ms; kwargs...)
    end
end

# ---- Backends without an in-place path (use the allocating calculate_spectrum) ----
function _calculate_spectrum!(::AbstractArray, backend::AbstractSpectralBackend,
        grid::AbstractGrid, fields_vecs::Tuple, ms::Tuple; kwargs...)
    throw(ArgumentError(
        "$(nameof(typeof(backend))) does not support in-place calculate_spectrum! on a " *
        "$(nameof(typeof(grid))). Use the allocating calculate_spectrum.",
    ))
end

# ============================================================================
# Internal extension entry points — error until the relevant extension loads.
# ============================================================================
_calculate_spectrum_fft(args...; kwargs...) = throw(ArgumentError("FFTBackend is not loaded. Run `using FFTW`."))
_calculate_spectrum_nufft(args...; kwargs...) = throw(ArgumentError("NUFFTBackend is not loaded. Run `using FINUFFT`."))
_calculate_spectrum_sht(args...; kwargs...) = throw(ArgumentError("SHTBackend is not loaded. Run `using FastSphericalHarmonics`."))
_calculate_spectrum_nufsht(args...; kwargs...) = throw(ArgumentError("NUFSHTBackend is not loaded. Run `using NUFSHT`."))
_calculate_spectrum_threaded_cartesian!(args...; kwargs...) = throw(ArgumentError("ThreadedBackend is not loaded. Run `using OhMyThreads`."))
_calculate_spectrum_threaded_spherical!(args...; kwargs...) = throw(ArgumentError("ThreadedBackend is not loaded. Run `using OhMyThreads`."))
_calculate_spectrum_gpu_cartesian(args...; kwargs...) = throw(ArgumentError("GPUBackend is not loaded. Run `using KernelAbstractions`."))
_calculate_spectrum_gpu_spherical(args...; kwargs...) = throw(ArgumentError("GPUBackend is not loaded. Run `using KernelAbstractions`."))

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
        c, k = calculate_spectrum(DirectSumBackend(), cart, (u, v), (4, 4))
        # Compile reductions
        isotropic_spectrum(k, c; num_bins=2)
        transect_spectrum(k, c, (1,))

        # Compile Spherical 2D DirectSum (theta, phi)
        theta = rand(T, 8) .* π
        phi = rand(T, 8) .* 2π
        sph = ScatteredSphericalGrid(theta, phi)
        cs, ks = calculate_spectrum(DirectSumBackend(), sph, (rand(T, 8),), (2, 3))
        spherical_energy_spectrum(cs; lmax=1)
    end
end

end # module FlowFieldSpectra