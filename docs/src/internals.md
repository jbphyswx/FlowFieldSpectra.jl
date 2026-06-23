# Internals

These functions and types are not part of the public API (they are not exported), but are
documented here for contributors. They may change without notice.

## Grids

```@docs
FlowFieldSpectra.Grids.physical_wavenumbers
FlowFieldSpectra.Grids.spatial_dims
FlowFieldSpectra.Grids.npoints
```

## Transform problem & layout

```@docs
FlowFieldSpectra.Problem.n_spectral
FlowFieldSpectra.Problem.n_batch
FlowFieldSpectra.Problem.batch_size
FlowFieldSpectra.Problem.output_size
FlowFieldSpectra.Problem.coeff_eltype
FlowFieldSpectra.Problem.pack_fields
```

## Preprocessing helpers

```@docs
FlowFieldSpectra.Preprocessing.window_function
FlowFieldSpectra.Preprocessing.window_function!
FlowFieldSpectra.Preprocessing.window_correction
FlowFieldSpectra.Preprocessing.detrend!
```

## Normalization helpers

```@docs
FlowFieldSpectra.Normalization.sided_factor
```

## Spherical-harmonic kernels

```@docs
FlowFieldSpectra.SphericalKernels.LegendreTables
FlowFieldSpectra.SphericalKernels.legendre_tables
FlowFieldSpectra.SphericalKernels.fill_legendre!
FlowFieldSpectra.SphericalKernels.normalized_legendre
```
