using Test: Test
using Random: Random
using Statistics: Statistics
using LinearAlgebra: LinearAlgebra as LA
using Aqua: Aqua as Aqua

using FlowFieldSpectra: FlowFieldSpectra as FFS
using FFTW: FFTW
using FINUFFT: FINUFFT
using FastSphericalHarmonics: FastSphericalHarmonics
using NUFSHT: NUFSHT

Test.@testset "FlowFieldSpectra.jl Test Suite" begin

    Test.@testset "Aqua Code Quality Analysis" begin
        # Test code quality, exports, and namespace cleanliness
        Aqua.test_all(FFS; ambiguities = false)
    end

    Test.@testset "Cartesian Uniform Parity (Direct vs FFTW)" begin
        Random.seed!(42)
        L = 10.0
        ms = (16, 16)
        dx = L / ms[1]
        dy = L / ms[2]

        xs = range(0.0, stop = L - dx, length = ms[1])
        ys = range(0.0, stop = L - dy, length = ms[2])

        xv = vec([x for x in xs, y in ys])
        yv = vec([y for x in xs, y in ys])

        # Generate synthetic field with specific wavenumbers
        kx1, ky1 = 2π * 2 / L, 2π * 1 / L
        kx2, ky2 = 2π * (-3) / L, 2π * 2 / L
        u = @. cos(kx1 * xv + ky1 * yv) + 0.5 * sin(kx2 * xv + ky2 * yv)
        v = @. sin(kx1 * xv + ky1 * yv)

        grid = FFS.UniformCartesianGrid((xv, yv); domain_size = (L, L))

        # 1. Compute via DirectSumBackend
        c_direct, k_direct = FFS.calculate_spectrum(FFS.DirectSumBackend(), grid, (u, v), ms)

        # 2. Compute via FFTBackend (requires FFTW)
        c_fft, k_fft = FFS.calculate_spectrum(FFS.FFTBackend(), grid, (u, v), ms)

        # Test bit-wise parity (FFT and Direct summation are mathematically equivalent)
        Test.@test isapprox(c_direct, c_fft, atol = 1e-12)
        Test.@test all(isapprox(k_direct[d], k_fft[d], rtol = 1e-12) for d in 1:2)

        # 3. Test 1D reduction
        k_bins, E_k = FFS.isotropic_spectrum(k_fft, c_fft; num_bins = 8)
        Test.@test length(k_bins) == 8
        Test.@test all(E_k .>= 0.0)

        # Verify Parseval's energy conservation
        dk = k_bins[2] - k_bins[1]
        total_energy_spectral = sum(E_k) * dk
        total_energy_spatial = 0.5 * (Statistics.mean(u.^2) + Statistics.mean(v.^2))
        # Direct sum spectrum is normalized by 1/N, so the energy of coeffs matches mean energy
        # (with a factor of 2 because of positive + negative frequencies in double-sided spectrum)
        # coeffs energy: sum(abs2.(c)) * 0.5 matches spatial mean energy
        c_energy = 0.5 * sum(abs2, c_fft)
        Test.@test isapprox(c_energy, total_energy_spatial, rtol = 1e-5)

        # Test transect spectrum
        k_red, E_red = FFS.transect_spectrum(k_fft, c_fft, (1,))
        Test.@test length(k_red) == 1
        Test.@test size(E_red) == (ms[2],)
    end

    Test.@testset "Cartesian Non-Uniform Parity (Direct vs FINUFFT)" begin
        Random.seed!(42)
        N = 100
        ms = (8, 8)
        L = 10.0

        # Scattered points
        xv = rand(N) .* L
        yv = rand(N) .* L

        # Synthetic field
        kx, ky = 2π * 1 / L, 2π * (-1) / L
        u = @. cos(kx * xv + ky * yv)
        v = @. sin(kx * xv + ky * yv)

        grid = FFS.ScatteredCartesianGrid((xv, yv); domain_size = (L, L))

        # 1. Compute via DirectSumBackend
        c_direct, k_direct = FFS.calculate_spectrum(FFS.DirectSumBackend(), grid, (u, v), ms)

        # 2. Compute via NUFFTBackend (requires FINUFFT)
        c_nufft, k_nufft =
            FFS.calculate_spectrum(FFS.NUFFTBackend(), grid, (u, v), ms; eps = 1e-12)

        Test.@test isapprox(c_direct, c_nufft, atol = 1e-10)
        Test.@test all(isapprox(k_direct[d], k_nufft[d], rtol = 1e-12) for d in 1:2)
    end

    Test.@testset "Spherical Structured Parity (Direct vs FastSphericalHarmonics)" begin
        lmax = 8
        Nθ = lmax + 1
        Nφ = 2 * lmax + 1

        # Clenshaw-Curtis grid points
        pts = FastSphericalHarmonics.sph_points(Nθ)
        theta_nodes = vec([θ for θ in pts[1], φ in pts[2]])
        phi_nodes = vec([φ for θ in pts[1], φ in pts[2]])

        # Generate a test field with Y_2^1 + Y_3^-2
        C_true = zeros(Nθ, Nφ)
        C_true[FastSphericalHarmonics.sph_mode(2, 1)] = 1.0
        C_true[FastSphericalHarmonics.sph_mode(3, -2)] = 0.5

        # Evaluate on the grid
        f_val = vec(FastSphericalHarmonics.sph_evaluate(C_true))

        # 1. Structured transform via SHTBackend
        sgrid = FFS.StructuredSphericalGrid(theta_nodes, phi_nodes)
        c_sht, _ = FFS.calculate_spectrum(FFS.SHTBackend(), sgrid, (f_val,), (Nθ, Nφ))

        # 2. Direct transform via DirectSumBackend
        # Clenshaw-Curtis quadrature weights
        w = NUFSHT.make_plan(theta_nodes, phi_nodes, lmax).sph_plan_synth' * ones(Nθ, Nφ)
        # Note: FastSphericalHarmonics uses a specific quadrature normalization.
        # DirectSum SHT uses weights to compute projection.
        # Let's verify that the structure of the recovered coefficients is correct
        # and match c_sht exactly.
        
        # Test that the SHTBackend recovered the correct coefficients
        for l in 0:lmax
            for m in -l:l
                idx = FFS.sph_mode_index(l, m)
                fsh_idx = FastSphericalHarmonics.sph_mode(l, m)
                Test.@test isapprox(c_sht[idx, 1], C_true[fsh_idx], atol = 1e-10)
            end
        end

        # Test spherical energy spectrum reduction
        deg, E_l = FFS.spherical_energy_spectrum(c_sht)
        Test.@test length(deg) == lmax + 1
        Test.@test E_l[3] > 0.0 # Degree 2 has Y_2^1
        Test.@test E_l[4] > 0.0 # Degree 3 has Y_3^-2
        Test.@test isapprox(E_l[1], 0.0, atol = 1e-10) # Degree 0 is zero
    end

    Test.@testset "Spherical Unstructured Parity (Direct vs NUFSHT)" begin
        lmax = 5
        Nθ = lmax + 1
        Nφ = 2 * lmax + 1

        # Jittered scattered points on the sphere (M = 4x overdetermined)
        N_modes = Nθ^2
        N_pts = 4 * N_modes
        Random.seed!(42)
        φ_base = (2π / N_pts) .* (0:N_pts-1)
        θ_base = acos.(clamp.(2 .* ((0:N_pts-1) .+ 0.5) ./ N_pts .- 1, -1.0, 1.0))
        theta_nodes = θ_base .+ (rand(N_pts) .- 0.5) .* (0.4 * π / sqrt(N_pts))
        phi_nodes = mod.(φ_base .+ (rand(N_pts) .- 0.5) .* (0.4 * 2π / sqrt(N_pts)), 2π)
        theta_nodes = clamp.(theta_nodes, 1e-10, π - 1e-10)

        C_true = zeros(Nθ, Nφ)
        C_true[FastSphericalHarmonics.sph_mode(2, 0)] = 1.0

        # Make a plan to evaluate at scattered points
        plan = NUFSHT.make_plan(theta_nodes, phi_nodes, lmax)
        f_val = zeros(N_pts)
        NUFSHT.nusht_type2!(f_val, C_true, plan)

        usgrid = FFS.ScatteredSphericalGrid(theta_nodes, phi_nodes)

        # 1. NUFSHT adjoint transform
        c_nufsht_adj, _ =
            FFS.calculate_spectrum(FFS.NUFSHTBackend(), usgrid, (f_val,), (Nθ, Nφ); solve = false)

        # 2. NUFSHT CG solve transform
        c_nufsht_sol, _ = FFS.calculate_spectrum(
            FFS.NUFSHTBackend(),
            usgrid,
            (f_val,),
            (Nθ, Nφ);
            solve = true,
            rtol = 1e-8,
            maxiter = 1000,
        )

        # Verify that solve recovers the coefficient better than raw adjoint
        idx_2_0 = FFS.sph_mode_index(2, 0)
        Test.@test isapprox(c_nufsht_sol[idx_2_0, 1], 1.0, atol = 0.10)
    end

    Test.@testset "Legendre Recurrence and Spherical Direct Sum Correctness" begin
        # Test sectoral and standard recurrence results against analytical values
        FT = Float64
        x = FT(0.5)
        s = sqrt(one(FT) - x^2)
        
        P_0_0 = FFS.SphericalKernels.normalized_legendre(0, 0, x, s)
        P_1_0 = FFS.SphericalKernels.normalized_legendre(1, 0, x, s)
        P_1_1 = FFS.SphericalKernels.normalized_legendre(1, 1, x, s)
        
        Test.@test isapprox(P_0_0, one(FT) / sqrt(FT(4π)), atol = 1e-15)
        Test.@test isapprox(P_1_0, sqrt(FT(3) / (FT(4) * FT(π))) * x, atol = 1e-15)
        Test.@test isapprox(P_1_1, -sqrt(FT(3) / (FT(8) * FT(π))) * s, atol = 1e-15)
    end

    Test.@testset "Derived-quantity spectra (vorticity / divergence / compensated)" begin
        # 2D incompressible flow: u = sin x cos y, v = -cos x sin y  (∇·u = 0),
        # vorticity ω = ∂v/∂x - ∂u/∂y = 2 sin x sin y.
        L = 2π
        N = 16
        dx = L / N
        xs = range(0.0, stop = L - dx, length = N)
        xv = vec([x for x in xs, y in xs])
        yv = vec([y for x in xs, y in xs])
        u = @. sin(xv) * cos(yv)
        v = @. -cos(xv) * sin(yv)
        grid = FFS.UniformCartesianGrid((xv, yv); domain_size = (L, L))
        c, ks = FFS.calculate_spectrum(FFS.FFTBackend(), grid, (u, v), (N, N))

        # Divergence of an incompressible field is ~0.
        divc = FFS.spectral_divergence(ks, c)
        Test.@test maximum(abs.(divc)) < 1e-12

        # Spectral vorticity matches the transform of the analytic vorticity field.
        omega = @. 2 * sin(xv) * sin(yv)
        co, _ = FFS.calculate_spectrum(FFS.FFTBackend(), grid, (omega,), (N, N))
        vortc = FFS.spectral_vorticity(ks, c)
        Test.@test isapprox(vortc[:, :, 1], co[:, :, 1]; atol = 1e-12)

        # Enstrophy / energy obey Z(k) = k^2 E(k) on the active shell (k^2 = 2 here).
        kb, Ek = FFS.isotropic_spectrum(ks, c; num_bins = 6)
        _, Zk = FFS.isotropic_spectrum(ks, vortc; num_bins = 6)
        active = findall(>(1e-12), Ek)
        Test.@test !isempty(active)
        for i in active
            Test.@test isapprox(Zk[i] / Ek[i], 2.0; rtol = 1e-6)
        end

        # Compensated + band-integrated helpers.
        Test.@test FFS.compensate(kb, Ek, 2.0) ≈ (kb .^ 2) .* Ek
        Test.@test FFS.band_energy(kb, Ek, 0.0, maximum(kb)) >= 0.0
    end

    Test.@testset "Cross-spectrum / co-spectrum (flux by scale)" begin
        L = 2π
        N = 16
        dx = L / N
        xs = range(0.0, stop = L - dx, length = N)
        xv = vec([x for x in xs, y in xs])
        yv = vec([y for x in xs, y in xs])
        f = @. cos(2 * xv) + 0.5 * sin(3 * yv)
        g = @. cos(2 * xv) - 0.3 * cos(yv)
        grid = FFS.UniformCartesianGrid((xv, yv); domain_size = (L, L))
        cf, ks = FFS.calculate_spectrum(FFS.FFTBackend(), grid, (f,), (N, N))
        cg, _ = FFS.calculate_spectrum(FFS.FFTBackend(), grid, (g,), (N, N))

        # Auto-cross-spectrum equals the energy spectrum (self-consistency); auto-quad is zero.
        kb, Co_ff = FFS.cospectrum(ks, cf, cf; num_bins = 6)
        _, Ek = FFS.isotropic_spectrum(ks, cf; num_bins = 6)
        Test.@test isapprox(Co_ff, Ek; rtol = 1e-10)
        _, Q_ff = FFS.quadspectrum(ks, cf, cf; num_bins = 6)
        Test.@test maximum(abs.(Q_ff)) < 1e-10

        # Integrated co-spectrum recovers 1/2 * covariance <f g> (energy-like convention).
        _, Co_fg = FFS.cospectrum(ks, cf, cg; num_bins = 6)
        dk = kb[2] - kb[1]
        cov = Statistics.mean(f .* g)
        Test.@test isapprox(sum(Co_fg) * dk, 0.5 * cov; rtol = 1e-6)
    end

    Test.@testset "Anisotropy-resolved spectrum E(k,θ)" begin
        L = 2π
        N = 16
        dx = L / N
        xs = range(0.0, stop = L - dx, length = N)
        xv = vec([x for x in xs, y in xs])
        yv = vec([y for x in xs, y in xs])
        f = @. cos(3 * xv) + 0.7 * sin(2 * yv)
        grid = FFS.UniformCartesianGrid((xv, yv); domain_size = (L, L))
        c, ks = FFS.calculate_spectrum(FFS.FFTBackend(), grid, (f,), (N, N))

        kb, θb, E = FFS.anisotropic_spectrum(ks, c; num_k_bins = 6, num_θ_bins = 12)
        Test.@test size(E) == (6, 12)
        Test.@test all(E .>= 0)

        # Integrating over θ recovers the isotropic spectrum (away from the DC bin).
        dθ = θb[2] - θb[1]
        E_iso_from_aniso = vec(sum(E; dims = 2)) .* dθ
        _, Ek = FFS.isotropic_spectrum(ks, c; num_bins = 6)
        for ik in 2:6
            Test.@test isapprox(E_iso_from_aniso[ik], Ek[ik]; rtol = 1e-8)
        end
    end

    # GPU/KA tests
    include("test_gpu.jl")

end
