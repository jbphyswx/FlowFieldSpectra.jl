# Spherical harmonic spectra

Project a field on the sphere onto spherical harmonics and read off the degree (`ℓ`) energy
spectrum. We synthesize a field from known modes `Y₂¹` and `Y₅⁻³` on a Clenshaw–Curtis grid and
recover them with the structured `SHTBackend`.

```@example sph
using FlowFieldSpectra, FastSphericalHarmonics, CairoMakie

lmax = 16
Nθ = lmax + 1
Nφ = 2 * lmax + 1

pts = FastSphericalHarmonics.sph_points(Nθ)
θnodes = vec([θ for θ in pts[1], φ in pts[2]])
φnodes = vec([φ for θ in pts[1], φ in pts[2]])

C = zeros(Nθ, Nφ)
C[FastSphericalHarmonics.sph_mode(2, 1)] = 1.0
C[FastSphericalHarmonics.sph_mode(5, -3)] = 0.5
f = vec(FastSphericalHarmonics.sph_evaluate(C))

grid = StructuredSphericalGrid(θnodes, φnodes)
coeffs, _ = calculate_spectrum(SHTBackend(), grid, (f,), (Nθ, Nφ))
deg, E_l = spherical_energy_spectrum(coeffs)

fig = Figure(size = (680, 420))
ax = Axis(fig[1, 1]; title = "Degree energy spectrum E(ℓ)", xlabel = "degree ℓ", ylabel = "E(ℓ)")
barplot!(ax, deg, E_l; color = :steelblue)
xlims!(ax, -0.5, lmax + 0.5)
fig
```

Energy appears only at degrees `ℓ = 2` and `ℓ = 5`, as expected. Scattered points on the sphere
are handled the same way with `ScatteredSphericalGrid` and the `NUFSHTBackend`.
