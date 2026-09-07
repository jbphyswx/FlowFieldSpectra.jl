# Changelog

All notable changes to FlowFieldSpectra.jl are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Grids come from [FlowGeometries.jl](https://github.com/jbphyswx/FlowGeometries.jl) and the coordinate
  system is the grid's geometry type (`CartesianGeometry`, `SphericalGeometry`, `SpheroidGeometry`),
  replacing the fragile coordinate-range heuristic for Cartesian-vs-spherical classification. Every grid
  architecture is served: structured (uniform and stretched), unstructured point clouds, curvilinear, and
  the spherical pixelizations (HEALPix, reduced/octahedral Gaussian rings, cubed-sphere, icosahedral,
  Yin-Yang, Fibonacci).
- Typed preprocessing (`Hann`/`Hamming`/`Blackman`/`Tukey`/`NoWindow`, `Demean`/`LinearDetrend`/
  `NoDetrend`) and normalization conventions (`OneSided`/`TwoSided`, `DensityScaling`/`PowerScaling`)
  — all dispatched on types rather than symbols.
- **A plan answers for its own output**, so it is sufficient to preallocate against:
  `coefficient_size`, `coefficient_type`, `wavenumbers` and `allocate_coefficients` on every
  `AbstractSpectralPlan`. The element type is not implied by the size — a Cartesian plan's coefficients
  are complex while a spherical plan's follow the field and are real for a real one — so both accessors
  are needed to allocate, and `allocate_coefficients(plan)` does it.
- **A reusable inverse**: `plan_synthesis` and `synthesize!(out, plan, coeffs; ks)`, with `field_size`,
  `field_type` and `allocate_field` mirroring the forward's accessors. Every host backend and the device
  FFT/NUFSHT hold their backward transform and buffers, so a snapshot series inverts without rebuilding
  either. `synthesize!` takes `ks` because a plan holds what the grid fixes, and the Nyquist twin a
  packed inverse needs on a nonuniformly-sampled grid is a functional of the coefficients.
- Reusable spectral plans (`plan_spectrum`) for every transform — FFTW, the FFT/NUFFT hybrid composite,
  both NUFFT providers, NUFSHT, and the direct sum on Cartesian and spherical grids alike — with a
  trailing `batch=` shape: build once, reuse across a z/t/component loop with allocation-free steady-state
  execution (FFTW guru `mul!`, FINUFFT guru `makeplan`/`setpts!`/`exec!`, and for the direct sum the
  per-axis DFT matrices, the contraction's working arrays and the Nyquist-twin storage).
- **Two orthogonal backend axes** matching the sibling packages: a `transform=` axis
  (`AutoSpectralBackend`, `DirectSumSpectralBackend`, `FFTSpectralBackend`, `NUFFTSpectralBackend` with
  the `FINUFFTBackend()`/`NonuniformFFTsBackend()` providers, `FSHTSpectralBackend`,
  `NUFSHTSpectralBackend`) and a separate `execution=` axis (`AbstractExecutionBackend`:
  `SerialBackend`, `ThreadedBackend`, `GPUBackend{B}`, `DistributedBackend{Inner}`, `MPIBackend{Inner}`,
  `AutoBackend`) that compose freely (`transform × execution`).
- **GPU transforms** on any KernelAbstractions device: `FFTSpectralBackend`×`GPUBackend{B}` is
  device-generic via `AbstractFFTs.plan_rfft`/`plan_fft` (CUFFT on CUDA, rocFFT on AMDGPU, FFTW on
  `KA.CPU()`) through a `["KernelAbstractions","AbstractFFTs"]` extension with a reusable device plan;
  `NonuniformFFTsBackend()` is device-generic on any KA device and `FINUFFTBackend()`×
  `GPUBackend(CUDABackend())` reaches cuFINUFFT via a `["CUDA","FINUFFT"]` extension; a device-generic
  spherical harmonic transform (φ-DFT + θ-Legendre contraction) and a device-resident NUFSHT cover the
  sphere, and the portable KA direct-sum kernels cover every remaining transform×device pair.
- **Distributed and MPI execution** (`Distributed`/`MPI` extensions): point-partitionable transforms
  (DirectSum/NUFFT/NUFSHT-projection) split the point axis and sum α-weighted partial coefficients,
  combining the Nyquist twins with the same weights; FFT/SHT split the batch axis.
  `MPIBackend(GPUBackend(dev))` targets a multi-GPU cluster.
- Estimators: cross-spectrum family (`cross_spectrum`/`cospectrum`/`quadspectrum`), anisotropy-resolved
  `anisotropic_spectrum` `E(k,θ)`, derived-quantity spectra (`spectral_vorticity`/`spectral_divergence`),
  compensated/band-integrated wrappers (`compensate`/`band_energy`).
- Variance-reduction & cross-analysis: `welch_power_spectrum`, `coherence_spectrum` (magnitude-squared
  coherence + phase), `dpss` multitaper (Slepian) tapers, and `lomb_scargle` for irregular sampling.
- First-class `synthesize` (inverse transform) for Cartesian and spherical grids — filtering and
  round-trip validation.
- Shared `physical_wavenumbers` and `SphericalKernels` (cached Legendre tables) used across all backends.
- Documenter `@example` example pages (Cartesian incl. anisotropy/compensated/filtering, NUFFT +
  coastline cutout, spherical, 4D fixed-grid, cross-spectra + coherence, wavenumber–frequency
  `E(k,ω)`, irregular/windowed estimation, backends) and Internals/API docs; CI, Docs, CompatHelper,
  TagBot workflows.
- README gallery: reproducible showcase figures for every major capability (isotropic spectra,
  scattered/coastline NUFFT, anisotropy, cross-spectra + coherence/phase, derived quantities, `E(k,ω)`,
  Lomb–Scargle + multitaper, spherical degree spectrum, backend parity), generated by
  `docs/generate_assets/generate_assets.jl`.

### Fixed
- **The packed inverse boxed its wavenumber tuple once per mode per point.** `_neg_row_value` indexed
  `ks[d]` at a runtime `d` on the mixed `RFFTAxis`/`FFTAxis` tuple, so `synthesize` on a real field
  allocated in proportion to the mode count times the point count. `Val`-unrolling the index takes it
  to zero.
- **`transect_spectrum` under-reported when it KEPT the halved axis.** A real field's spectrum stores
  axis 1 as the Hermitian half, so the `−k₁` energy lives nowhere else; keeping that axis counted each
  stored mode once and left it out, so `Σ E / ∏dk` fell short of the field's folded Parseval total while
  keeping a full axis matched it — the same call on the same field meant two different things depending
  on which axis the real-input layout happened to halve. A kept halved axis now carries its
  `fold_weight` (2 interior, 1 at dc and at even-`n₁` Nyquist) and reports `|k₁|`, so the invariant holds
  whichever axes are kept.
- **A complex field's spherical transform was wrong or raised on every path.** The factorized tensor and
  ring direct sums read the longitude bins as `real(g)` and `-imag(g)` from `g[a] = Σ f e^{-i(a-1)λ}`,
  which equal `Σ f cos(mλ)` and `Σ f sin(mλ)` only when `g[-a] = conj(g[a])` — true for a real field and
  not otherwise. The general form `(g+ĝ)/2` and `i(g-ĝ)/2` now carries the `exp(+imλ)` partner
  transform, which a real field neither allocates nor computes. FastSphericalHarmonics, the device SHT
  and NUFSHT (host and device) each raised `InexactError` on a complex field instead, while
  `sph_coeff_type` promised complex support; all three now transform one real component at a time and
  recombine, the identity their own inverses already used, and the device SHT takes a full-bin `fft` for
  a complex field.
- **A direction's Fourier length is read from its `isperiodic` flag.** `period`'s contract is that it is
  meaningful only where the direction wraps, and `UnstructuredGrid`/`CurvilinearGrid` store whatever
  `period` they were given even on a direction declared non-periodic (`StructuredGrid` normalizes it to
  zero). A node cloud built with `periodic = (true, false), period = (2π, 5.0)` therefore had axis 2's
  wavenumbers scaled by `2π/5` where `2π` is correct. One reader, `Grids.axis_geometry`, replaces ~14
  duplicated sites and takes each direction's origin from `bounds`, `O(1)` on a structured axis against
  the `O(N)` `minimum` it replaces.
- **`NUFSHTSpectralBackend()` returned the unweighted adjoint `Σⱼ fⱼ Y_lm(xⱼ)`** where every other
  spherical backend returns the quadrature projection `Σⱼ wⱼ fⱼ Y_lm(xⱼ)`. Its coefficients were off by
  the grid's weights — a factor `N/4π` on equal-measure nodes, and identical for any measures at all,
  since the grid's measure was never read. The weights now reach `nusht_type1!` through the field, taken
  from the same per-point weights the direct sum uses, so a structured grid contributes its sampling's
  latitude quadrature and a masked cell a zero. `solve = true` still consumes the raw field (it fits
  `A c ≈ f`).
- NUFSHT inferred a field's batch shape as `ndims(field) - 1`, so a structured spherical grid (whose
  field is `(nlon, nlat)`) reported a phantom batch axis; and its synthesis wrote `(N, batch…)` on every
  grid. Both now come from `ndims(grid)` / `size(grid)`.
- The Nyquist twin, the NUFFT one-shots and the Distributed/MPI point partition made the same
  one-spatial-dimension assumption, which a curvilinear grid's `(N_1, N_2)` field breaks.

### Added
- **A real field's spherical coefficients are a real array.** The basis is the real spherical harmonics,
  so `calculate_spectrum` on a real field now returns `Array{FT}` (half the memory), and a complex field
  still gives `Array{Complex{FT}}` — the field's realness picks the layout, as it already did on the
  Cartesian side (packed half vs full cube). Synthesis from real coefficients writes a real field with no
  complex intermediate, `spherical_energy_spectrum` reads either layout, and the in-place and plan
  entries accept whichever buffer the caller brings.
- **`preprocess::Preprocess` is now read by the transform.** `Preprocess`, the window tapers and the
  detrend types were exported and documented while nothing in the transform path consumed them, so
  `calculate_spectrum(...; preprocess = ...)` was silently ignored. It now detrends, tapers and
  zero-pads: each axis taper is scaled to unit mean square, so the tapered field keeps the variance of
  the original and Parseval holds on the returned coefficients with no correction applied afterwards;
  `pad > 1` lengthens the grid's axes and scales `ms`, narrowing the wavenumber spacing. Omitting
  `preprocess` transforms the field as given and copies nothing. `preprocess_field` is exported for the
  plan path — a plan is fixed by the grid and resolution while a taper is an estimator choice, so
  multitaper reuses one plan across `K` tapered copies transformed as a batch. `Demean` needs no axes and
  serves any grid; a window, `LinearDetrend` and padding read the grid's axes and say so on a node cloud.
- **Spheroid grids** (`SpheroidGeometry`) take the surface spherical-harmonic expansion. The stored
  geodetic latitude is not the colatitude of a node's direction, so each node is embedded through
  `geodetic_to_cartesian` and its geocentric direction read back; the grid's own area measure supplies
  the quadrature. A structured spheroid grid's rows stay iso-latitude, so it takes the ring-factorized
  path. The radial variation is not modelled (that is the solid-harmonic expansion).
- **A reusable spherical direct-sum plan.** `plan_spectrum(grid, T, ms; transform =
  DirectSumSpectralBackend())` holds the grid's materialized nodes (or its ring table, or its longitude
  DFT matrix and latitude quadrature — whichever its layout reads), the Legendre tables and the longitude
  buffer. Building those is a minority of a transform's time and about three quarters of its allocations,
  so a reused execution is allocation-free.
- **Curvilinear Cartesian and spherical grids** (`CurvilinearGrid`) transform, synthesize, and partition.
  A curvilinear map admits no factorization over axes, so it reads as a point cloud: the direct sum on
  any execution backend, both NUFFT providers one-shot and through `plan_spectrum`, both inverses, and
  the Distributed/MPI point partition all serve it.

### Changed
- **`transform` now defaults to `AutoSpectralBackend()`**, which selects the fastest transform the grid
  admits among the loaded extensions: FFT on a uniform Cartesian grid, the hybrid FFT/NUFFT composite
  where a grid is uniform in some directions and stretched in others, a NUFFT on a nonuniform, scattered
  or curvilinear Cartesian grid, FastSphericalHarmonics on its own Clenshaw–Curtis grid, and NUFSHT on
  any other spherical grid. Each backend's precondition is checked before it is selected. With no fast
  transform loaded it warns once and runs the direct sum. The grid form of `calculate_spectrum!` resolves
  `Auto` to the direct sum, the only transform it implements.
- `plan_spectrum` raises an `ArgumentError` for `DirectSumSpectralBackend()`, which precomputes nothing a
  plan could hold, and points at `calculate_spectrum!(coeffs, grid, field, ms)`.
- **Minimum Julia is now 1.11.** The package uses Project.toml `[sources]`/`[workspace]` (added in
  Julia 1.11) to pull in the as-yet-unregistered NUFSHT.jl; on 1.10 those tables are ignored and the
  test/extension environment cannot resolve NUFSHT. Nothing in the source itself requires 1.11 — once
  NUFSHT (and the sibling packages) are registered in the General registry, the floor can drop back to
  1.10 (possibly 1.9). Users who `dev` NUFSHT locally can likely run on older Julia, but that path is
  untested.
- `calculate_spectrum`/`calculate_spectrum!`/`plan_spectrum`/`synthesize` take the two orthogonal
  `transform=` and `execution=` keywords and dispatch on `(transform, execution, grid, fields, ms)`.
  **Clean break**: the old single positional `backend` / `backend=` keyword is removed (no shim). The
  coordinate system is still the grid type — the fragile coordinate-range heuristic remains gone.
- `AutoBackend` resolves the *execution* axis via `Base.get_extension` (loaded-extension detection),
  never `isdefined(Main, …)`, and only ever to a local backend (`Serial`/`Threaded`) — GPU/Distributed/
  MPI require explicit context.
- Unsupported `(transform, execution, grid)` combinations raise a clear error instead of misrouting or
  silently falling back (e.g. fast GPU FFT/NUFFT on a non-CUDA device errors rather than downgrading).
- `transect_spectrum!` signature is now `(E_reduced, ks_phys, coeffs, dims)` — it fills only the
  reusable numeric buffer and is fully allocation-free (previously it rebuilt a `ks_reduced` vector via
  `empty!`/`push!` every call).

### Performance
- **A real field's coefficients ARE the rfft-packed half** `(m_1÷2+1, m_2…, batch…)` on every backend —
  a complete Hermitian representation, every slot computed by the transform, with no full-cube
  reconstruction pass and no scalar fill. `unpacked`/`unpacked!` expand it when a caller wants every mode
  addressable. The reductions, operators and averaging fold the missing half by mode weight, so Parseval
  holds on the packed array directly. This IS a convention change from the earlier centred full cube.
- A real transform's even-Nyquist columns carry a `NyquistTwin` on the halved axis of `ks`, supplying the
  `k₁ < 0` coefficients that index negation cannot reach on a nonuniformly sampled grid; the cutoff-free
  `transect_spectrum` reads it (it was dropping 48.9% of the Nyquist-column energy on a scattered grid),
  as does the inverse.
- GPU spherical transform rewritten to one-thread-per-output-mode with register accumulation and a
  single store — **no atomics** (was one-thread-per-point atomic accumulation).
- Every in-place / prebuilt-plan steady-state path is genuinely allocation-free, verified at two grid
  sizes (grid-independent): `isotropic_spectrum!`, `transect_spectrum!`, `spherical_energy_spectrum!`,
  serial-DirectSum `calculate_spectrum!`, and FFTW/FINUFFT plan execution. Removed hidden per-call
  allocations from the plan paths — a fused allocation-free fftshift+normalize replaces `circshift!`,
  the FINUFFT phase is stored pre-shaped so no per-call `reshape`, and batch-slice copies are type-stable
  (`enumerate` + offset `copyto!`, no `selectdim`/`view` temporaries).
- FINUFFT default tolerance is precision-aware (`1e-6` for `Float32`, `1e-8` for `Float64`), avoiding
  spurious tolerance warnings; full `Float32` support end-to-end.

### Tested
- Parseval/variance invariant and DirectSum-vs-FFT parity at `D = 1, 2, 3`; plan-reuse and batched-axis
  parity; grid-dispatch errors; explicit-import policy (ExplicitImports) and Aqua; `Float32` end-to-end;
  rfft-vs-full-FFT equivalence; GPU CPU-backend parity (no-atomics kernel).
- Execution-axis parity (Serial/Threaded/Auto and `GPUBackend(KA.CPU())`) for every transform;
  Distributed (in-process workers) and MPI (2-rank via `mpiexec`) parity vs serial; a dedicated
  allocation suite asserting **exactly zero** (grid-independent) for every in-place/plan path and
  output-proportional bounds for the must-allocate operators.
- The device-generic GPU paths — `FFTBackend` (AbstractFFTs) and `DirectSumBackend` (KA kernels) — are
  exercised on CI via `GPUBackend(KA.CPU())`. The CUDA-specific paths (CUFFT on `CuArray`, cuFINUFFT)
  are verified on real CUDA hardware via the `gpu/` project (`gpu/gpu_parity.jl`); CI has no GPU.

## [0.1.0]
- Initial implementation: `calculate_spectrum` with DirectSum/FFT/NUFFT/SHT/NUFSHT/Threaded/GPU
  backends; `isotropic_spectrum`, `transect_spectrum`, `spherical_energy_spectrum` reductions.
