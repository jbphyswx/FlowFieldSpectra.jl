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

```@docs
FlowFieldSpectra._backend_nthreads
FlowFieldSpectra._to_host
```

## Plans

```@docs
FlowFieldSpectra.DirectSumCartesianPlan
FlowFieldSpectra.DirectSumSphericalPlan
FlowFieldSpectra.HybridPlan
FlowFieldSpectra._hybrid_plan
FlowFieldSpectra._hybrid_derive
```

### The direct sum's setup / run split

Both direct-sum plans are a setup the grid fixes plus a run over a field, and each one-shot composes the
same pair, so the plan and the one-shot cannot drift. On a Cartesian grid the setup holds the per-axis
dense DFT matrices, the working arrays the contraction walks through and the Nyquist-twin storage; a
structured grid contracts axis by axis through `AxisPass`, and a point cloud sums directly.

```@docs
FlowFieldSpectra.DirectSum.cart_setup
FlowFieldSpectra.DirectSum.cart_run!
FlowFieldSpectra.DirectSum.AxisPass
FlowFieldSpectra.DirectSum.sph_setup
FlowFieldSpectra.DirectSum.sph_run!
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
FlowFieldSpectra.Grids._sph_points
FlowFieldSpectra.Grids._colatitude
```

### Spherical layout routing

```@docs
FlowFieldSpectra.Grids.TensorSphere
FlowFieldSpectra.Grids.RingSphere
FlowFieldSpectra.Grids.ScatteredSphere
FlowFieldSpectra.Grids._sph_sampling
FlowFieldSpectra.Grids._sph_layout
FlowFieldSpectra.Grids._ring_table
```

### Quadrature, band limit, masks

```@docs
FlowFieldSpectra.Grids._sht_weights
FlowFieldSpectra.Grids._sph_node_weights
FlowFieldSpectra.Grids.quadrature_scale
FlowFieldSpectra.quadrature_weighted
FlowFieldSpectra.Grids._warn_bandlimit
FlowFieldSpectra.Grids._zeroed_inactive
FlowFieldSpectra.Grids.covered_area
FlowFieldSpectra.Grids.sky_fraction
```

## Transform selection

```@docs
FlowFieldSpectra._sht_applicable
```

## Packed spectral layout

`Packing` owns the packed-native layout: the half a real transform publishes, the Nyquist twins a
reduction or an inverse needs alongside it, and the per-axis gather/scatter the separable transforms
walk. [`unpacked`](@ref) and [`unpacked!`](@ref) are the public entry points.

```@docs
FlowFieldSpectra.Packing.hermitian_request_size
FlowFieldSpectra.Packing.offset_phase
FlowFieldSpectra.Packing.publish_packed!
FlowFieldSpectra.Packing.packed_half_view
FlowFieldSpectra.Packing.NyquistTwin
FlowFieldSpectra.Packing.twin_at
FlowFieldSpectra.Packing.twin_table
FlowFieldSpectra.Packing.twin_slice_shape
FlowFieldSpectra.Packing.ConjTwinSlice
FlowFieldSpectra.Packing.conj_twins
FlowFieldSpectra.Packing.gather_conj_twins!
FlowFieldSpectra.Packing.axis_layout
FlowFieldSpectra.Packing.axis_work_shape
FlowFieldSpectra.Packing.axis_chunk_offsets!
FlowFieldSpectra.Packing.gather_axis_block!
FlowFieldSpectra.Packing.scatter_axis_block!
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
FlowFieldSpectra.Preprocessing.is_identity
FlowFieldSpectra.Preprocessing.axis_taper
FlowFieldSpectra.Preprocessing.detrend_spatial!
FlowFieldSpectra.Preprocessing.apply_window!
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
