# Changelog

All notable changes to FlowFieldSpectra.jl are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Explicit grid type hierarchy (`UniformCartesianGrid`, `NonuniformCartesianGrid`,
  `ScatteredCartesianGrid`, `StructuredSphericalGrid`, `ScatteredSphericalGrid`) replacing the
  fragile coordinate-range heuristic for Cartesian-vs-spherical classification.
- Typed preprocessing (`Hann`/`Hamming`/`Blackman`/`Tukey`/`NoWindow`, `Demean`/`LinearDetrend`/
  `NoDetrend`) and normalization conventions (`OneSided`/`TwoSided`, `Density`/`Power`) — all
  dispatched on types rather than symbols.
- Shared `physical_wavenumbers` definition (previously duplicated across backends).

### Changed
- `calculate_spectrum`/`calculate_spectrum!` now dispatch on `(backend, grid, fields, ms)`. The
  coordinate system is determined by the grid type — the fragile coordinate-range heuristic that
  guessed Cartesian-vs-spherical is removed entirely (no guessing, no warnings, no fallbacks).
- In-place `calculate_spectrum!` is grid-based and supported for `DirectSumBackend`/`ThreadedBackend`;
  unsupported `(backend, grid)` combinations raise a clear error.

## [0.1.0]
- Initial implementation: `calculate_spectrum` with DirectSum/FFT/NUFFT/SHT/NUFSHT/Threaded/GPU
  backends; `isotropic_spectrum`, `transect_spectrum`, `spherical_energy_spectrum` reductions.
