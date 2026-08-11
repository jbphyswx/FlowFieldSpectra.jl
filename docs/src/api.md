# API Reference

## Spectra

```@docs
calculate_spectrum
calculate_spectrum!
synthesize
plan_spectrum
AbstractSpectralPlan
sph_mode_index
```

## Grids

The coordinate system is the grid type — construct the grid that matches your data. Grids come from
[FlowGeometries.jl](https://github.com/jbphyswx/FlowGeometries.jl): a Cartesian grid is a
`FG.Grids.StructuredGrid` / `FG.Grids.UnstructuredGrid` over a `FG.Geometry.CartesianGeometry`, and a
spherical grid the same over a `FG.Geometry.SphericalGeometry`. Structured vs. scattered is the grid
*architecture*, and uniform vs. non-uniform is the `FG.Grids.isuniform` value trait (an `AbstractRange`
axis is uniform, a `Vector` is not) — there are no separate FlowFieldSpectra grid types. Spherical
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
coherence_spectrum
lomb_scargle
```

## Derived quantities & post-processing

```@docs
spectral_divergence
spectral_vorticity
compensate
band_energy
```

## Preprocessing & normalization conventions

```@docs
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
TransformProblem
```

## Transform backends (which spectral math)

The two backend axes are orthogonal and compose: pass one `transform=` and one `execution=`. See
[Backends and Extensions](@ref) for the selection matrix and profiles.

The `transform=` keyword takes a marker tag from
[SpectralBackends.jl](https://github.com/jbphyswx/SpectralBackends.jl):
`DirectSumSpectralBackend` (the default), `FFTSpectralBackend`, `NUFFTSpectralBackend`,
`FSHTSpectralBackend`, `NUFSHTSpectralBackend` (all `<: SpectralBackends.AbstractSpectralBackend`).
Transform options such as `eps`, `tol`, `iflag`, and `solve` are `calculate_spectrum` keyword
arguments.

A NUFFT *provider* is a library choice, not spectral math, so FlowFieldSpectra owns two symmetric,
concrete NUFFT backends (neither is a default; `SpectralBackends.NUFFTSpectralBackend` selects neither):

```@docs
FlowFieldSpectra.FINUFFTBackend
FlowFieldSpectra.NonuniformFFTsBackend
```

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
