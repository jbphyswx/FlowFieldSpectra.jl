# Internals

These functions and types are not part of the public API (they are not exported), but are
documented here for contributors. They may change without notice.

## Execution-backend helpers

The execution-axis introspection helpers (`local_backend`, `is_distributed`, `resolve_backend`) are
provided by [ComputationalBackends.jl](https://github.com/jbphyswx/ComputationalBackends.jl) and used
internally and by the distribution extensions. The default `AutoBackend` resolves through the internal
`_resolve_execution` (Threaded when `OhMyThreads` is loaded and `Threads.nthreads() > 1`, else Serial);
ComputationalBackends' own `resolve_backend(::AutoBackend)` raises.

```@docs
FlowFieldSpectra._backend_nthreads
FlowFieldSpectra._to_host
```

## Plans

```@docs
FlowFieldSpectra.DirectSumCartesianPlan
FlowFieldSpectra.DirectSumSphericalPlan
FlowFieldSpectra.DirectSumSynthesisPlan
FlowFieldSpectra.HybridPlan
FlowFieldSpectra.RegionFFT
FlowFieldSpectra._hybrid_plan
FlowFieldSpectra._hybrid_derive
FlowFieldSpectra.HybridSynthesisPlan
FlowFieldSpectra._hybrid_synthesis_plan
```

### NUFFT plans

The NUFFT plans run FlowTransformBindings' library plans in `FFTModes` order and publish the packed
layout from them: a point set's type 1, a nonuniform tensor grid's separable passes, and the type-2
inverse.

```@docs
FlowFieldSpectra.NUFFTPointPlan
FlowFieldSpectra.NUFFTSeparablePlan
FlowFieldSpectra.NUFFTAxisPass
FlowFieldSpectra.NUFFTSynthesisPlan
FlowFieldSpectra._packed_src
FlowFieldSpectra._oversampled_table
FlowFieldSpectra._gather_packed!
FlowFieldSpectra._scatter_each!
```

### The direct sum's setup / run split

Both direct-sum plans are a setup the grid fixes plus a run over a field, and each one-shot composes the
same pair. On a Cartesian grid the setup holds the per-axis dense DFT matrices, the working arrays the
contraction walks through and the modes a real field's coefficient box holds at zero; a structured grid
contracts axis by axis through `AxisPass`, and a point cloud sums directly.

```@docs
FlowFieldSpectra.DirectSum.phase_geometry
FlowFieldSpectra.DirectSum.cart_setup
FlowFieldSpectra.DirectSum.cart_run!
FlowFieldSpectra.DirectSum.synth_setup
FlowFieldSpectra.DirectSum.native_coefficients!
FlowFieldSpectra.DirectSum.AxisPass
FlowFieldSpectra.DirectSum.sph_setup
FlowFieldSpectra.DirectSum.sph_run!
FlowFieldSpectra.DirectSum.sph_synth_setup
FlowFieldSpectra.DirectSum.sph_synth_run!
```

## Grids

FlowFieldSpectra builds on FlowGeometries' grid/geometry types and reads them through FlowGeometries'
own accessors. The grid-related internals are the spectral wavenumber grid, the field-shape and
point-list readers, the spherical `(θ, φ)` / quadrature-weight bridge (FlowGeometries stores
`(λ, φ_lat)`; the transforms want `(θ = colatitude, φ = longitude)`), the spherical layout traits that
select an algorithm, and the mask/measure queries.

```@docs
FlowFieldSpectra.Grids.physical_wavenumbers
FlowFieldSpectra.Grids.PointwiseCartesian
FlowFieldSpectra.Grids.SphericalHarmonicGeometry
FlowFieldSpectra.Grids.field_batch_shape
FlowFieldSpectra.Grids.point_coordinates
FlowFieldSpectra.Grids.axis_geometry
FlowFieldSpectra.Grids.axis_range
FlowFieldSpectra.Grids.relative_coordinates
FlowFieldSpectra.Grids._sph_points
FlowFieldSpectra.Grids._colatitude
```

### Spherical layout routing

```@docs
FlowFieldSpectra.Grids._sph_layout
FlowFieldSpectra.Grids._ring_table
```

### Quadrature, band limit, masks

```@docs
FlowFieldSpectra.Grids._sht_weights
FlowFieldSpectra.Grids._sph_point_weights
FlowFieldSpectra.Grids._sph_node_weights
FlowFieldSpectra.Grids.quadrature_scale
FlowFieldSpectra.quadrature_weighted
FlowFieldSpectra.Grids._warn_bandlimit
FlowFieldSpectra.Grids._zeroed_inactive
FlowFieldSpectra.Grids._inactive_scale
FlowFieldSpectra.Grids._inactive_weights
FlowFieldSpectra.Grids._transform_nodes
```

## Transform selection

```@docs
FlowFieldSpectra._sht_applicable
```

## Packed spectral layout

```@docs
FlowFieldSpectra.Grids.mode_box
FlowFieldSpectra.Grids.series_modes
```

The packed layout and the widened axes are FlowTransformBindings' (`box_size`, `outside_modes`). The NUFFT
and hybrid plans publish the half over the coefficient box from a library's `FFTModes` output through one
index table ([`FlowFieldSpectra._packed_src`](@ref)) of direct and conjugate reads, and a real-data
NonuniformFFTs plan reads its modes at `+ms_d/2` from its oversampled spectra
([`FlowFieldSpectra._oversampled_table`](@ref)), on the host and on a device alike.

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
FlowFieldSpectra.Preprocessing.window_value
FlowFieldSpectra.Preprocessing.window_function
FlowFieldSpectra.Preprocessing.window_function!
FlowFieldSpectra.Preprocessing.window_correction
FlowFieldSpectra.Preprocessing.detrend!
FlowFieldSpectra.Preprocessing.is_identity
FlowFieldSpectra.Preprocessing.axis_taper
FlowFieldSpectra.Preprocessing.point_window
FlowFieldSpectra.Preprocessing.apply_window!
FlowFieldSpectra.Preprocessing.detrend_spatial!
```

## Radial-bin helpers

```@docs
FlowFieldSpectra.Reductions._shell_values!
```

## Spherical-harmonic kernels

```@docs
FlowFieldSpectra.SphericalKernels.LegendreTables
FlowFieldSpectra.SphericalKernels.legendre_tables
FlowFieldSpectra.SphericalKernels.fill_legendre!
FlowFieldSpectra.SphericalKernels.legendre_index
FlowFieldSpectra.SphericalKernels.legendre_count
FlowFieldSpectra.SphericalKernels.device_legendre_table
FlowFieldSpectra.SphericalKernels.normalized_legendre
```
