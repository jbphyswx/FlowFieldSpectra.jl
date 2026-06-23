# API Reference

## Spectra

```@docs
calculate_spectrum
calculate_spectrum!
plan_spectrum
AbstractSpectralPlan
sph_mode_index
```

## Grids

The coordinate system is the grid type — construct the grid that matches your data.

```@docs
AbstractGrid
AbstractCartesianGrid
AbstractSphericalGrid
UniformCartesianGrid
NonuniformCartesianGrid
ScatteredCartesianGrid
StructuredSphericalGrid
ScatteredSphericalGrid
AbstractQuadrature
ClenshawCurtis
GaussLegendre
Equiangular
```

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

## Backend types

```@docs
AbstractSpectralBackend
DirectSumBackend
FFTBackend
NUFFTBackend
SHTBackend
NUFSHTBackend
ThreadedBackend
GPUBackend
AutoBackend
```

## Plotting & analysis

These require `CairoMakie` to be loaded.

```@docs
plot_spectrum
compare_spectra
compare_spectral_analysis
```
