module FlowFieldSpectra

using PrecompileTools: @setup_workload, @compile_workload

include("Types.jl")
include("DirectSum.jl")
include("Reductions.jl")

using .Types: AbstractSpectralBackend, DirectSumBackend, FFTBackend, NUFFTBackend, SHTBackend, NUFSHTBackend
using .DirectSum: calculate_spectrum_direct, sph_mode_index
using .Reductions: isotropic_spectrum, transect_spectrum, spherical_energy_spectrum

# Export Types
export AbstractSpectralBackend, DirectSumBackend, FFTBackend, NUFFTBackend, SHTBackend, NUFSHTBackend

# Export APIs
export calculate_spectrum, isotropic_spectrum, transect_spectrum, spherical_energy_spectrum, sph_mode_index
export plot_spectrum, compare_spectra, compare_spectral_analysis


"""
    calculate_spectrum(backend::AbstractSpectralBackend, coords_vecs::Tuple, fields_vecs::Tuple, ms::Tuple; kwargs...)
    calculate_spectrum(coords_vecs::Tuple, fields_vecs::Tuple, ms::Tuple; backend=DirectSumBackend(), kwargs...)

Calculate the spectral coefficients and physical wavenumbers for one or more fields.

# Arguments
- `backend::AbstractSpectralBackend`: The spectral backend to use (default is `DirectSumBackend()`).
- `coords_vecs::Tuple`: Tuple of coordinate vectors. 
  - For Cartesian: `(xv, yv, ...)` where each is a vector of coordinates of length `N`.
  - For Spherical: `(theta_nodes, phi_nodes)` in radians.
- `fields_vecs::Tuple`: Tuple of fields to analyze (e.g., `(u, v)`). Each field is a vector of values of length `N` matching the coordinates.
- `ms::Tuple`: Target spectral resolution.
  - For Cartesian: `(mx, my, ...)` defining the number of modes along each dimension.
  - For Spherical: `(Ntheta, Nphi)` where `lmax = Ntheta - 1`.

# Keyword Arguments
- `iflag::Int`: Direction of Cartesian Fourier transform (1 for forwards/analysis, -1 for backwards/synthesis; default is 1).
- `domain_size::Union{Nothing, Tuple}`: Physical size of the domain along each Cartesian dimension. If not specified, inferred from the bounding box of coordinates.
- `weights::Union{Nothing, AbstractVector}`: Optional quadrature/area weights for spherical transforms.
- `tol::Real`: Accuracy tolerance for non-uniform transforms (NUFFT/NUFSHT) (default is `1e-8`).
- `solve::Bool`: (For `NUFSHTBackend`) If `true`, solves the SHT as a linear system using an iterative CG solver rather than a raw adjoint projection.
- `maxiter::Int`: Maximum iterations for CG solvers (default is `500`).
- `rtol::Real`: Relative residual tolerance for CG solver convergence (default is `1e-6`).

# Returns
- `coeffs`: Array of size `(ms..., NU)` containing the complex spectral coefficients, where `NU = length(fields_vecs)`.
- `ks_phys`: A tuple of physical wavenumber coordinates along each dimension, or spherical coordinates `(0:lmax, -lmax:lmax)`.

# Example (Cartesian FFT)
```julia
using FlowFieldSpectra
using FFTW

L = 2π
N = 16
x = range(0.0, stop=L, length=N+1)[1:N]
y = range(0.0, stop=L, length=N+1)[1:N]
xv = vec([x_val for x_val in x, y_val in y])
yv = vec([y_val for x_val in x, y_val in y])

u = cos.(xv) .+ sin.(yv)
v = zeros(length(xv))

# Computes coefficients using fast FFTW backend
coeffs, ks = calculate_spectrum(FFTBackend(), (xv, yv), (u, v), (N, N); domain_size=(L, L))
```
"""
function calculate_spectrum(
    coords_vecs::Tuple,
    fields_vecs::Tuple,
    ms::Tuple;
    backend::AbstractSpectralBackend = DirectSumBackend(),
    kwargs...,
)
    return calculate_spectrum(backend, coords_vecs, fields_vecs, ms; kwargs...)
end

# Backend dispatches
function calculate_spectrum(
    ::DirectSumBackend,
    coords_vecs::Tuple,
    fields_vecs::Tuple,
    ms::Tuple;
    kwargs...,
)
    return calculate_spectrum_direct(coords_vecs, fields_vecs, ms; kwargs...)
end

# Internal stub helpers for extensions
function _calculate_spectrum_fft(args...; kwargs...)
    throw(ArgumentError("FFTBackend is not loaded. Run `using FFTW` to load the FFTW extension."))
end

function _calculate_spectrum_nufft(args...; kwargs...)
    throw(ArgumentError("NUFFTBackend is not loaded. Run `using FINUFFT` to load the FINUFFT extension."))
end

function _calculate_spectrum_sht(args...; kwargs...)
    throw(ArgumentError("SHTBackend is not loaded. Run `using FastSphericalHarmonics` to load the FastSphericalHarmonics extension."))
end

function _calculate_spectrum_nufsht(args...; kwargs...)
    throw(ArgumentError("NUFSHTBackend is not loaded. Run `using NUFSHT` to load the NUFSHT extension."))
end

function calculate_spectrum(
    ::FFTBackend,
    coords_vecs::Tuple,
    fields_vecs::Tuple,
    ms::Tuple;
    kwargs...,
)
    return _calculate_spectrum_fft(coords_vecs, fields_vecs, ms; kwargs...)
end

function calculate_spectrum(
    ::NUFFTBackend,
    coords_vecs::Tuple,
    fields_vecs::Tuple,
    ms::Tuple;
    kwargs...,
)
    return _calculate_spectrum_nufft(coords_vecs, fields_vecs, ms; kwargs...)
end

function calculate_spectrum(
    ::SHTBackend,
    coords_vecs::Tuple,
    fields_vecs::Tuple,
    ms::Tuple;
    kwargs...,
)
    return _calculate_spectrum_sht(coords_vecs, fields_vecs, ms; kwargs...)
end

function calculate_spectrum(
    ::NUFSHTBackend,
    coords_vecs::Tuple,
    fields_vecs::Tuple,
    ms::Tuple;
    kwargs...,
)
    return _calculate_spectrum_nufsht(coords_vecs, fields_vecs, ms; kwargs...)
end

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
        c, k = calculate_spectrum(DirectSumBackend(), (xv, yv), (u, v), (4, 4))
        # Compile reductions
        isotropic_spectrum(k, c; num_bins=2)
        transect_spectrum(k, c, (1,))

        # Compile Spherical 2D DirectSum (theta, phi)
        theta = rand(T, 8) .* π
        phi = rand(T, 8) .* 2π
        cs, ks = calculate_spectrum(DirectSumBackend(), (theta, phi), (rand(T, 8),), (2, 3))
        spherical_energy_spectrum(cs; lmax=1)
    end
end

end # module FlowFieldSpectra