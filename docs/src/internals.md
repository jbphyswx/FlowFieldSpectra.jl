# Internals

These functions and types are not part of the public API (they are not exported), but are
documented here for contributors. They may change without notice.

## Execution-backend helpers

The execution-axis introspection helpers (`local_backend`, `is_distributed`, `resolve_backend`) are
provided by [ComputationalBackends.jl](https://github.com/jbphyswx/ComputationalBackends.jl) and used
internally and by the distribution extensions. The default `AutoBackend` is resolved locally through
an internal `_resolve_execution` (Threaded when `OhMyThreads` is loaded and `Threads.nthreads() > 1`,
else Serial), so precompilation never calls ComputationalBackends' `resolve_backend(::AutoBackend)`
(which errors by design).

## Grids

FlowFieldSpectra builds on FlowGeometries' grid/geometry types and reads them through FlowGeometries'
own accessors; the only grid-related internals are the spectral wavenumber grid and the spherical
`(θ, φ)` / quadrature-weight bridge (FlowGeometries stores `(λ, φ_lat)`; the transforms want
`(θ = colatitude, φ = longitude)`).

```@docs
FlowFieldSpectra.Grids.physical_wavenumbers
FlowFieldSpectra.Grids._sph_points
FlowFieldSpectra.Grids._sht_weights
```

## Transform problem & layout

```@docs
FlowFieldSpectra.Problem.spatial_shape
FlowFieldSpectra.Problem.batch_shape
FlowFieldSpectra.Problem.n_batch
FlowFieldSpectra.Problem.batch_length
FlowFieldSpectra.Problem.coeff_output_size
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
