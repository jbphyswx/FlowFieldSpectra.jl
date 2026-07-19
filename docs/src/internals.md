# Internals

These functions and types are not part of the public API (they are not exported), but are
documented here for contributors. They may change without notice.

## Execution-backend helpers

Introspection helpers for the execution axis (used internally and by the distribution extensions;
reachable as `FlowFieldSpectra.Types.<name>`).

```@docs
FlowFieldSpectra.Types.local_backend
FlowFieldSpectra.Types.is_distributed
FlowFieldSpectra.Types.resolve_backend
```

## Grids

```@docs
FlowFieldSpectra.Grids.physical_wavenumbers
FlowFieldSpectra.Grids.spatial_dims
FlowFieldSpectra.Grids.ndims_spatial
FlowFieldSpectra.Grids.spatial_size
FlowFieldSpectra.Grids.npoints
```

## Transform problem & layout

```@docs
FlowFieldSpectra.Problem.spatial_shape
FlowFieldSpectra.Problem.batch_shape
FlowFieldSpectra.Problem.n_batch
FlowFieldSpectra.Problem.batch_length
FlowFieldSpectra.Problem.coeff_output_size
FlowFieldSpectra.Problem.coeff_eltype
FlowFieldSpectra.Problem.stack_fields
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
