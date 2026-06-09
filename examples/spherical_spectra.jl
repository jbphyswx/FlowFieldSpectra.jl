using FlowFieldSpectra
using FastSphericalHarmonics
using NUFSHT
using CairoMakie
using Statistics
using Random

function run_spherical_example()
    println("--- Running Spherical Grid Spectra Example ---")

    # 1. Setup Spherical Grid (lmax = 16)
    lmax = 16
    Nθ = lmax + 1
    Nφ = 2 * lmax + 1
    
    # Clenshaw-Curtis grid nodes
    pts = FastSphericalHarmonics.sph_points(Nθ)
    theta_grid = pts[1]
    phi_grid = pts[2]
    
    theta_nodes = vec([θ for θ in theta_grid, φ in phi_grid])
    phi_nodes = vec([φ for θ in theta_grid, φ in phi_grid])
    
    # Synthesize scalar field with specific modes: Y_2^1 (amplitude 1.0) and Y_5^-3 (amplitude 0.5)
    C_true = zeros(Nθ, Nφ)
    C_true[FastSphericalHarmonics.sph_mode(2, 1)] = 1.0
    C_true[FastSphericalHarmonics.sph_mode(5, -3)] = 0.5
    
    f_val = vec(FastSphericalHarmonics.sph_evaluate(C_true))
    
    # 2. Compute via structured SHTBackend
    println("Computing structured SHT via FastSphericalHarmonics...")
    c_sht, _ = calculate_spectrum(SHTBackend(), (theta_nodes, phi_nodes), (f_val,), (Nθ, Nφ))
    
    # Spherical degree energy spectrum
    deg, E_l = spherical_energy_spectrum(c_sht)
    
    # 3. Compute via unstructured NUFSHTBackend (on scattered nodes)
    println("Computing unstructured SHT via NUFSHT (with CG solve)...")
    N_pts = length(theta_nodes)
    Random.seed!(42)
    # Generate scattered points using latitude-band jitter
    φ_base = (2π / N_pts) .* (0:N_pts-1)
    θ_base = acos.(clamp.(2 .* ((0:N_pts-1) .+ 0.5) ./ N_pts .- 1, -1.0, 1.0))
    theta_scat = θ_base .+ (rand(N_pts) .- 0.5) .* (0.2 * π / sqrt(N_pts))
    phi_scat = mod.(φ_base .+ (rand(N_pts) .- 0.5) .* (0.2 * 2π / sqrt(N_pts)), 2π)
    theta_scat = clamp.(theta_scat, 1e-10, π - 1e-10)
    
    # Evaluate at scattered nodes using NUFSHT synthesis (type 2)
    plan = NUFSHT.make_plan(theta_scat, phi_scat, lmax; tol=1e-10)
    f_scat = zeros(N_pts)
    NUFSHT.nusht_type2!(f_scat, C_true, plan)
    
    c_nufsht, _ = calculate_spectrum(
        NUFSHTBackend(), 
        (theta_scat, phi_scat), 
        (f_scat,), 
        (Nθ, Nφ);
        solve=true,
        maxiter=500,
        rtol=1e-8
    )
    
    deg_scat, E_l_scat = spherical_energy_spectrum(c_nufsht)

    # 4. Generate Plot
    fig = Figure(size=(1200, 800))
    Label(fig[0, 1:2], "Spherical Harmonic Transform & Power Spectrum", fontsize=18, font=:bold)
    
    # Panel A: Scalar field map on (theta, phi) coordinates
    ax1 = Axis(fig[1, 1], title="Field f(θ, φ) on CC Grid", xlabel="Longitude φ", ylabel="Colatitude θ")
    hm = heatmap!(ax1, phi_grid, theta_grid, reshape(f_val, Nθ, Nφ)', colormap=:balance)
    Colorbar(fig[1, 2], hm)
    
    # Panel B: Reconstructed Spherical Harmonic degree energy spectrum
    ax2 = Axis(fig[2, 1:2], title="Degree Energy Spectrum E(ℓ)", xlabel="Degree ℓ", ylabel="E(ℓ)")
    barplot!(ax2, deg, E_l, width=0.4, color=:blue, alpha=0.7, label="Structured (FSH)")
    barplot!(ax2, deg_scat .+ 0.4, E_l_scat, width=0.4, color=:red, alpha=0.7, label="Scattered (NUFSHT)")
    xlims!(ax2, -0.5, lmax + 0.5)
    axislegend(ax2)
    
    # Save Figure
    outpath = joinpath(@__DIR__, "spherical_spectra.png")
    save(outpath, fig)
    println("Saved figure: ", outpath)
    println("Example run successfully!")
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_spherical_example()
end
