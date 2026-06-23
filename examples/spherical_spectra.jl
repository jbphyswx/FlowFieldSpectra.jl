using FlowFieldSpectra
using FastSphericalHarmonics
using NUFSHT
using CairoMakie

"""
    run_spherical_example()

Spherical-harmonic degree energy spectrum of a field built from known modes, recovered two ways:
the **structured** transform (`SHTBackend`, exact on its quadrature grid) and the **scattered**
transform (`NUFSHTBackend`, a least-squares solve from jittered points). The scattered solve is
kept in a regime (modest `lmax`, ~4× oversampling) where it cleanly recovers the dominant degrees,
so the two estimates visibly agree.
"""
function run_spherical_example()
    println("--- Running Spherical Grid Spectra Example ---")

    # 1. Structured Clenshaw–Curtis grid. lmax is kept modest so the scattered least-squares
    #    solve (which becomes ill-conditioned at high degree on random points) recovers cleanly.
    lmax = 6
    Nθ = lmax + 1
    Nφ = 2 * lmax + 1

    pts = FastSphericalHarmonics.sph_points(Nθ)
    theta_grid, phi_grid = pts[1], pts[2]
    theta_nodes = vec([θ for θ in theta_grid, φ in phi_grid])
    phi_nodes = vec([φ for θ in theta_grid, φ in phi_grid])

    # Field with two known modes: Y₂¹ (amplitude 1.0) and Y₄⁻² (amplitude 0.6).
    C_true = zeros(Nθ, Nφ)
    C_true[FastSphericalHarmonics.sph_mode(2, 1)] = 1.0
    C_true[FastSphericalHarmonics.sph_mode(4, -2)] = 0.6
    f_val = vec(FastSphericalHarmonics.sph_evaluate(C_true))

    # 2. Structured SHT (exact on the quadrature grid).
    println("Computing structured SHT via FastSphericalHarmonics...")
    sht_grid = StructuredSphericalGrid(theta_nodes, phi_nodes)
    c_sht, _ = calculate_spectrum(SHTBackend(), sht_grid, (f_val,), (Nθ, Nφ))
    deg, E_l = spherical_energy_spectrum(c_sht)

    # 3. Scattered NUFSHT: ~4x more points than modes, CG least-squares solve.
    #    Use a Fibonacci-spiral sphere — a genuinely *well-distributed* scattered set. (A point
    #    layout with large gaps at scale 1/lmax leaves the SHT operator near-rank-deficient, and
    #    the minimum-norm solve then under-recovers; an even distribution recovers exactly.)
    println("Computing unstructured SHT via NUFSHT (CG solve, Fibonacci-sphere points)...")
    N_pts = 4 * Nθ * Nφ
    golden = π * (3 - sqrt(5))
    z_fib = [1 - 2 * (i + 0.5) / N_pts for i in 0:(N_pts-1)]
    theta_scat = acos.(clamp.(z_fib, -1.0, 1.0))
    phi_scat = mod.(golden .* (0:(N_pts-1)), 2π)

    plan = NUFSHT.make_plan(theta_scat, phi_scat, lmax; tol = 1e-10)
    f_scat = zeros(N_pts)
    NUFSHT.nusht_type2!(f_scat, C_true, plan)

    nufsht_grid = ScatteredSphericalGrid(theta_scat, phi_scat)
    c_nufsht, _ = calculate_spectrum(NUFSHTBackend(), nufsht_grid, (f_scat,), (Nθ, Nφ);
        solve = true, maxiter = 3000, rtol = 1e-10)
    deg_scat, E_l_scat = spherical_energy_spectrum(c_nufsht)

    # 4. Plot the TWO samplings of the same field (structured grid vs scattered points) and the
    #    degree spectra they recover. Shared colour range so the two inputs are comparable.
    crange = maximum(abs, f_val)
    fig = Figure(size = (1200, 780))
    Label(fig[0, 1:3], "Spherical Harmonic Degree Spectrum: structured grid vs scattered points",
        fontsize = 18, font = :bold)

    ax1 = Axis(fig[1, 1]; title = "Structured: f on the Clenshaw–Curtis grid",
        xlabel = "Longitude φ", ylabel = "Colatitude θ", yreversed = true)
    heatmap!(ax1, phi_grid, theta_grid, reshape(f_val, Nθ, Nφ)';
        colormap = :balance, colorrange = (-crange, crange))

    ax2 = Axis(fig[1, 2]; title = "Scattered: f at $(N_pts) Fibonacci-sphere points",
        xlabel = "Longitude φ", ylabel = "Colatitude θ", yreversed = true)
    sc = scatter!(ax2, phi_scat, theta_scat; color = f_scat, colormap = :balance,
        colorrange = (-crange, crange), markersize = 6)
    Colorbar(fig[1, 3], sc; label = "f(θ, φ)")

    ax3 = Axis(fig[2, 1:3]; title = "Degree energy spectrum E(ℓ) — both samplings recover it",
        xlabel = "Degree ℓ", ylabel = "E(ℓ)")
    barplot!(ax3, deg .- 0.18, E_l; width = 0.36, color = (:steelblue, 0.9),
        label = "Structured (SHT, exact)")
    barplot!(ax3, deg_scat .+ 0.18, E_l_scat; width = 0.36, color = (:crimson, 0.9),
        label = "Scattered (NUFSHT solve)")
    xlims!(ax3, -0.5, lmax + 0.5)
    axislegend(ax3; position = :rt)

    outpath = joinpath(@__DIR__, "spherical_spectra.png")
    save(outpath, fig)
    println("Saved figure: ", outpath)
    println("Structured  E(2)=$(round(E_l[3];sigdigits=3)),  E(4)=$(round(E_l[5];sigdigits=3))")
    println("Scattered   E(2)=$(round(E_l_scat[3];sigdigits=3)),  E(4)=$(round(E_l_scat[5];sigdigits=3))")
    println("Example run successfully!")
    return fig
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_spherical_example()
end
