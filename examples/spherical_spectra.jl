using FlowFieldSpectra: FlowFieldSpectra as FFS
using FastSphericalHarmonics: FastSphericalHarmonics as FSH
using NUFSHT: NUFSHT
using CairoMakie: CairoMakie as Mke

"""
    run_spherical_example()

Spherical-harmonic degree energy spectrum recovered two ways: the **structured** transform (`SHTBackend`
on a Clenshaw–Curtis grid, exact) and the **scattered** transform (`NUFSHTBackend`, a least-squares
solve from Fibonacci-sphere points).
"""
function run_spherical_example()
    println("--- Running Spherical Grid Spectra Example ---")

    lmax = 6
    Nθ = lmax + 1
    Nφ = 2 * lmax + 1
    pts = FSH.sph_points(Nθ)                            # (θ axis, φ axis)
    θax, φax = pts[1], pts[2]
    sht_grid = FFS.StructuredSphericalGrid(θax, φax)

    C_true = zeros(Nθ, Nφ)
    C_true[FSH.sph_mode(2, 1)] = 1.0
    C_true[FSH.sph_mode(4, -2)] = 0.6
    f_val = FSH.sph_evaluate(C_true)                    # (Nθ, Nφ) tensor field

    println("Computing structured SHT via FastSphericalHarmonics...")
    c_sht, _ = FFS.calculate_spectrum(sht_grid, f_val, (Nθ, Nφ); transform = FFS.SHTBackend())
    deg, E_l = FFS.spherical_energy_spectrum(c_sht)

    println("Computing unstructured SHT via NUFSHT (CG solve, Fibonacci-sphere points)...")
    N_pts = 4 * Nθ * Nφ
    golden = π * (3 - sqrt(5))
    z_fib = [1 - 2 * (i + 0.5) / N_pts for i in 0:(N_pts-1)]
    θ_scat = acos.(clamp.(z_fib, -1.0, 1.0))
    φ_scat = mod.(golden .* (0:(N_pts-1)), 2π)
    plan = NUFSHT.make_plan(θ_scat, φ_scat, lmax; tol = 1e-10)
    f_scat = zeros(N_pts)
    NUFSHT.nusht_type2!(f_scat, C_true, plan)

    nufsht_grid = FFS.ScatteredSphericalGrid(θ_scat, φ_scat)
    c_nu, _ = FFS.calculate_spectrum(nufsht_grid, f_scat, (Nθ, Nφ);
        transform = FFS.NUFSHTBackend(), solve = true, maxiter = 3000, rtol = 1e-10)
    deg_s, E_l_s = FFS.spherical_energy_spectrum(c_nu)

    crange = maximum(abs, f_val)
    fig = Mke.Figure(size = (1200, 780))
    Mke.Label(fig[0, 1:3], "Spherical Harmonic Degree Spectrum: structured grid vs scattered points",
        fontsize = 18, font = :bold)

    ax1 = Mke.Axis(fig[1, 1]; title = "Structured: f on the Clenshaw–Curtis grid",
        xlabel = "Longitude φ", ylabel = "Colatitude θ", yreversed = true)
    Mke.heatmap!(ax1, φax, θax, f_val'; colormap = :balance, colorrange = (-crange, crange))

    ax2 = Mke.Axis(fig[1, 2]; title = "Scattered: f at $(N_pts) Fibonacci-sphere points",
        xlabel = "Longitude φ", ylabel = "Colatitude θ", yreversed = true)
    sc = Mke.scatter!(ax2, φ_scat, θ_scat; color = f_scat, colormap = :balance,
        colorrange = (-crange, crange), markersize = 6)
    Mke.Colorbar(fig[1, 3], sc; label = "f(θ, φ)")

    ax3 = Mke.Axis(fig[2, 1:3]; title = "Degree energy spectrum E(ℓ) — both samplings recover it",
        xlabel = "Degree ℓ", ylabel = "E(ℓ)")
    Mke.barplot!(ax3, deg .- 0.18, E_l; width = 0.36, color = (:steelblue, 0.9), label = "Structured (SHT, exact)")
    Mke.barplot!(ax3, deg_s .+ 0.18, E_l_s; width = 0.36, color = (:crimson, 0.9), label = "Scattered (NUFSHT solve)")
    Mke.xlims!(ax3, -0.5, lmax + 0.5)
    Mke.axislegend(ax3; position = :rt)

    outpath = joinpath(@__DIR__, "spherical_spectra.png")
    Mke.save(outpath, fig)
    println("Saved figure: ", outpath)
    println("Example run successfully!")
    return fig
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_spherical_example()
end
