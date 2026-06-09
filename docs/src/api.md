# API Reference

This page documents the public-facing types, functions, and plotting utilities in `FlowFieldSpectra.jl`.

## Core API

```@docs
calculate_spectrum
sph_mode_index
```

## Reductions

```@docs
isotropic_spectrum
transect_spectrum
spherical_energy_spectrum
```

## Backend Types

```@docs
AbstractSpectralBackend
DirectSumBackend
FFTBackend
NUFFTBackend
SHTBackend
NUFSHTBackend
```

## Plotting & Analysis

These functions require `CairoMakie` to be imported before they become active.

```@docs
plot_spectrum
compare_spectra
compare_spectral_analysis
```
