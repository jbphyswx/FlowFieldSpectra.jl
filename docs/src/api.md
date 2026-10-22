# API Reference

## Spectra

```@docs
calculate_spectrum
calculate_spectrum!
synthesize
plan_spectrum
AbstractSpectralPlan
sph_mode_index
sph_coeff_type
```

Every transform has a reusable plan, the direct sum included: `plan_spectrum` holds what a grid fixes —
FFTW's plan, a NUFFT's point sorting, the direct sum's per-axis DFT matrices and working arrays, a
spherical grid's nodes and quadrature, FastTransforms' plans — so `calculate_spectrum!(coeffs, plan,
field)` in a time loop allocates nothing of its own on `SerialBackend`. A plan runs on the execution
backend it was built for: under `ThreadedBackend` the direct sum's threaded loops and the libraries'
threads, and under `GPUBackend` device plans for every transform, the hybrid FFT/NUFFT composite and the
direct sum included. Under `DistributedBackend` and `MPIBackend` every worker or rank holds the plan of
its share, the points of a point set or the batch columns, as `calculate_spectrum` divides the work. A
plan holds working buffers, so one task at a time executes it.

A plan reports the shape and element type of its output:

```@docs
coefficient_size
coefficient_type
wavenumbers
allocate_coefficients
close!
```

```julia
p = plan_spectrum(grid, Float64, ms; transform = FFTSpectralBackend(), batch = (nz,))
coeffs = allocate_coefficients(p)          # right shape and element type, from the plan alone
for t in times
    ks = calculate_spectrum!(coeffs, p, field_at(t))
end
```

The element type is not implied by the size: a Cartesian plan's coefficients are complex, while a
spherical plan's follow the field and are real for a real one.

Synthesis, the series evaluated at the grid's points, has the same pair, so a round trip on one grid
reuses both directions. It inverts `calculate_spectrum` on a uniform periodic grid and is its adjoint
under the grid's quadrature on any grid:

```@docs
AbstractSynthesisPlan
plan_synthesis
synthesize!
field_size
field_type
allocate_field
```

```julia
fwd = plan_spectrum(grid, Float64, ms; transform = FFTSpectralBackend(), batch = (nz,))
inv = plan_synthesis(grid, Float64, ms; transform = FFTSpectralBackend(), batch = (nz,))
coeffs = allocate_coefficients(fwd)
field  = allocate_field(inv)
for t in times
    calculate_spectrum!(coeffs, fwd, snapshot(t))
    filter!(coeffs)
    synthesize!(field, inv, coeffs)
end
```

### The packed layout

A real Cartesian field transforms to the rfft-packed half `(b_1÷2+1, b_2…, batch…)` of its coefficient box
`b`, the complete Hermitian representation of its spectrum, over the wavenumber axes
`FlowTransformBindings.RFFTAxis` (axis 1) and `FlowTransformBindings.FFTAxis`. The box is `ms`, except on
an axis whose samples tell `+m_d/2` from `-m_d/2` apart (a stretched axis, a point set, an axis transformed
below its own length), where an even `m_d` widens to the `m_d + 1` modes `-m_d/2 … m_d/2`. With two such
axes `d, e ≥ 2` the modes at `+m_d/2` with `-m_e/2` lie outside the computed modes and their mirror, and
hold zero. `FlowTransformBindings.unpacked(coeffs, b)` expands the half to the full FFT-order box, and the
reductions and the synthesis sum over the whole of it.

## Grids

The grid type defines the coordinate system, so construct the grid that matches your data. Grids come from
[FlowGeometries.jl](https://github.com/jbphyswx/FlowGeometries.jl): a Cartesian grid is a
`FG.Grids.StructuredGrid` / `FG.Grids.UnstructuredGrid` over a `FG.Geometry.CartesianGeometry`, and a
spherical grid the same over a `FG.Geometry.SphericalGeometry`. Structured vs. scattered is the grid
*architecture*, and uniform vs. non-uniform is the `FG.Grids.isuniform` value trait (an `AbstractRange`
axis is uniform, a `Vector` is not); FlowFieldSpectra defines no grid types of its own. Spherical
sampling schemes (`ClenshawCurtisSampling`, `GaussLegendreSampling`, `DriscollHealySampling`) live in
`FG.SphericalSampling`; build a structured spherical grid with `FG.Connectivity.structured_grid`.
FlowFieldSpectra reads every grid through FlowGeometries' own accessors (`size`, `ndims`, `length`,
`coordinates`, `isuniform`, `period`, …). See the FlowGeometries documentation for construction.

## Reductions

```@docs
isotropic_spectrum
isotropic_spectrum!
transect_spectrum
transect_spectrum!
spherical_energy_spectrum
spherical_energy_spectrum!
anisotropic_spectrum
```

## Cross-spectra

```@docs
cross_spectrum
cospectrum
quadspectrum
```

## Averaging (variance reduction, coherence & phase)

```@docs
welch_power_spectrum
welch_power_spectrum!
coherence_spectrum
coherence_spectrum!
lomb_scargle
lomb_scargle!
```

## Derived quantities & post-processing

```@docs
spectral_divergence
spectral_divergence!
spectral_vorticity
spectral_vorticity!
compensate
band_energy
```

## Masked spherical grids

`inactive` on a spherical transform says how a masked grid's inactive cells enter, the quadrature and the
`solve = true` fit alike, on every execution backend:

```@docs
AbstractInactiveCells
ZeroFill
Exclude
```

## Preprocessing & normalization conventions

`preprocess::Preprocess` on [`calculate_spectrum`](@ref) detrends, tapers and zero-pads the field before
transforming; [`preprocess_field`](@ref) applies the same spec explicitly, for the plan path and for
multitaper (one plan, `K` tapered copies as a batch).

```@docs
preprocess_field
Preprocess
AbstractWindow
NoWindow
Hann
Hamming
Blackman
Tukey
AbstractDetrend
NoDetrend
Demean
LinearDetrend
dpss
SpectralConvention
AbstractSidedness
OneSided
TwoSided
AbstractScaling
DensityScaling
PowerScaling
AbstractShellEstimator
ShellSum
ModeAverage
TransformProblem
```

## Transform backends (which spectral math)

The two backend axes are orthogonal and compose: pass one `transform=` and one `execution=`. See
[Backends and Extensions](@ref) for the selection matrix and profiles.

The `transform=` keyword takes a marker tag from
[SpectralBackends.jl](https://github.com/jbphyswx/SpectralBackends.jl):
`AutoSpectralBackend` (the default), `DirectSumSpectralBackend`, `FFTSpectralBackend`,
`NUFFTSpectralBackend`, `FSHTSpectralBackend`, `NUFSHTSpectralBackend` (all
`<: SpectralBackends.AbstractSpectralBackend`). Transform options such as `eps`, `tol`, `iflag`, and
`solve` are `calculate_spectrum` keyword arguments.

A NUFFT names its library with a tag from
[FlowTransformBindings.jl](https://github.com/jbphyswx/FlowTransformBindings.jl):
`FlowTransformBindings.NonuniformFFTsBackend()` (`using NonuniformFFTs`) or
`FlowTransformBindings.FINUFFTBackend()` (`using FINUFFT`), both
`<: SpectralBackends.AbstractNUFFTSpectralBackend`. `SpectralBackends.NUFFTSpectralBackend` names no
library and raises.

## Execution backends (where/how it runs)

The `execution=` keyword takes a tag from
[ComputationalBackends.jl](https://github.com/jbphyswx/ComputationalBackends.jl):
`SerialBackend`, `ThreadedBackend`, `GPUBackend`, `DistributedBackend`, `MPIBackend`, and
`AutoBackend` (the default, resolved locally to `ThreadedBackend`/`SerialBackend`), all
`<: ComputationalBackends.AbstractExecutionBackend`.

## Plotting & analysis

These require `CairoMakie` to be loaded.

```@docs
plot_spectrum
compare_spectra
compare_spectral_analysis
```
