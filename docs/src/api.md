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
spherical grid's nodes and quadrature — so `calculate_spectrum!(coeffs, plan, field)` in a time loop
allocates nothing.

A plan answers for its own output, so it is sufficient to preallocate against:

```@docs
coefficient_size
coefficient_type
wavenumbers
allocate_coefficients
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

The inverse has the same pair, so a round trip on one grid reuses both directions:

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
    ks = calculate_spectrum!(coeffs, fwd, snapshot(t))
    filter!(coeffs)
    synthesize!(field, inv, coeffs; ks = ks)
end
```

`synthesize!` takes `ks` because a plan holds what the *grid* fixes, while the Nyquist twin a packed
inverse needs on a nonuniformly-sampled grid is a functional of the coefficients and arrives with them.
A uniform grid needs none.

### The packed layout

A real Cartesian field transforms to the rfft-packed half `(m_1÷2+1, m_2…, batch…)`, the complete
Hermitian representation of its spectrum. `unpacked` expands that half to the full native-order cube
when a caller wants every mode addressable.

```@docs
unpacked
unpacked!
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
