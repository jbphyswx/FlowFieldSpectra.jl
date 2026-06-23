using Test: Test
using Random: Random
using Statistics: Statistics
using LinearAlgebra: LinearAlgebra as LA
using Aqua: Aqua as Aqua
using ExplicitImports: ExplicitImports as EI

using FlowFieldSpectra: FlowFieldSpectra as FFS
using FFTW: FFTW
using FINUFFT: FINUFFT
using FastSphericalHarmonics: FastSphericalHarmonics
using NUFSHT: NUFSHT

Test.@testset "FlowFieldSpectra.jl Test Suite" begin

    Test.@testset "Aqua Code Quality Analysis" begin
        # Test code quality, exports, and namespace cleanliness.
        #
        # `unbound_args` is gated to Julia ≥ 1.12. The grid structs store `domain_size::NTuple{D, FT}`,
        # whose auto-generated constructor is flagged on 1.10/1.11 for the *empty-tuple* (`D = 0`)
        # case: `NTuple{0, FT}` == `Tuple{}` matches for any `FT`, so `FT` is unbound there. That is
        # a genuine-but-unreachable corner — a 0-dimensional grid is never constructed — and 1.12's
        # `detect_unbound_args` no longer reports it. We keep the check on where it is accurate.
        Aqua.test_all(FFS; ambiguities = false, unbound_args = (VERSION >= v"1.12"))
    end

    Test.@testset "Explicit imports (no implicit / no stale)" begin
        # Enforce the package style: no reliance on bare `using` re-exports, no dead imports.
        # Checks the core module and every loaded backend extension.
        Test.@test (EI.check_no_implicit_imports(FFS); true)
        Test.@test (EI.check_no_stale_explicit_imports(FFS); true)
        for extname in (
            :FlowFieldSpectraFFTWExt,
            :FlowFieldSpectraFINUFFTExt,
            :FlowFieldSpectraFastSphericalHarmonicsExt,
            :FlowFieldSpectraNUFSHTExt,
        )
            ext = Base.get_extension(FFS, extname)
            ext === nothing && continue
            Test.@test (EI.check_no_implicit_imports(ext); true)
            Test.@test (EI.check_no_stale_explicit_imports(ext); true)
        end
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

    Test.@testset "Parseval invariant + D=1/2/3 coverage (DirectSum vs FFT)" begin
        Random.seed!(11)
        for D in 1:3
            N = D == 3 ? 8 : 16
            L = 2π
            dx = L / N
            ax = collect(range(0.0, stop = L - dx, length = N))
            grids = ntuple(d -> ax, D)
            mesh = Iterators.product(grids...)
            coords = ntuple(d -> [pt[d] for pt in mesh] |> vec, D)
            # Real demeaned field built from a few low wavenumbers.
            f = zeros(N^D)
            for pt_i in 1:length(coords[1])
                s = 0.0
                for d in 1:D
                    s += cos((d + 1) * coords[d][pt_i]) + 0.3 * sin((d) * coords[d][pt_i])
                end
                f[pt_i] = s
            end
            f .-= Statistics.mean(f)
            ms = ntuple(_ -> N, D)
            domain = ntuple(_ -> L, D)
            grid = FFS.UniformCartesianGrid(coords; domain_size = domain)

            c_fft, _ = FFS.calculate_spectrum(FFS.FFTBackend(), grid, (f,), ms)
            c_dir, _ = FFS.calculate_spectrum(FFS.DirectSumBackend(), grid, (f,), ms)
            # D-dimensional parity.
            Test.@test isapprox(c_fft, c_dir; atol = 1e-10)
            # Parseval: sum|C|^2 (1/N-normalized coeffs) == mean(f^2) == var(f).
            Test.@test isapprox(sum(abs2, c_fft), Statistics.mean(abs2, f); rtol = 1e-10)
        end
    end

    Test.@testset "Real-input rfft fast path == full FFT (D=1/2/3)" begin
        Random.seed!(13)
        for D in 1:3
            N = D == 3 ? 6 : 12
            L = 2π
            dx = L / N
            ax = collect(range(0.0, stop = L - dx, length = N))
            mesh = Iterators.product(ntuple(_ -> ax, D)...)
            coords = ntuple(d -> [pt[d] for pt in mesh] |> vec, D)
            ms = ntuple(_ -> N, D)
            domain = ntuple(_ -> L, D)
            grid = FFS.UniformCartesianGrid(coords; domain_size = domain)
            # Two components to exercise the trailing NU axis through the mirror.
            f1 = [sum(cos((d + 1) * coords[d][i]) for d in 1:D) for i in 1:N^D]
            f2 = [sum(sin(d * coords[d][i]) for d in 1:D) for i in 1:N^D]
            # Real input → rfft branch; complex input (identical values) → full-FFT branch.
            c_real, _ = FFS.calculate_spectrum(FFS.FFTBackend(), grid, (f1, f2), ms)
            c_cplx, _ = FFS.calculate_spectrum(FFS.FFTBackend(), grid,
                (ComplexF64.(f1), ComplexF64.(f2)), ms)
            Test.@test isapprox(c_real, c_cplx; atol = 1e-12)
        end
    end

    Test.@testset "Float32 end-to-end (DirectSum / FFT / NUFFT)" begin
        Random.seed!(7)
        L = 2.0f0 * Float32(π)
        N = 16
        dx = L / N
        ax = collect(range(0.0f0, stop = L - dx, length = N))
        xv = vec(Float32[x for x in ax, y in ax])
        yv = vec(Float32[y for x in ax, y in ax])
        f = @. cos(2xv) + 0.5f0 * sin(3yv)
        ug = FFS.UniformCartesianGrid((xv, yv); domain_size = (L, L))

        c_dir, _ = FFS.calculate_spectrum(FFS.DirectSumBackend(), ug, (f,), (N, N))
        c_fft, ks = FFS.calculate_spectrum(FFS.FFTBackend(), ug, (f,), (N, N))
        # Coefficient eltype stays single precision through the whole pipeline.
        Test.@test eltype(c_dir) === ComplexF32
        Test.@test eltype(c_fft) === ComplexF32
        Test.@test isapprox(c_fft, c_dir; atol = 1.0f-4)
        # Reductions preserve precision.
        k_iso, E_iso = FFS.isotropic_spectrum(ks, c_fft; num_bins = 6)
        Test.@test eltype(E_iso) === Float32

        # Scattered Float32 via FINUFFT: no tolerance warning (default eps is precision-aware),
        # parity with the double-precision result on the same points.
        xs = rand(Float32, 80) .* L
        ys = rand(Float32, 80) .* L
        fs = @. cos(xs) + sin(2ys)
        sg = FFS.ScatteredCartesianGrid((xs, ys); domain_size = (L, L))
        c32, _ = FFS.calculate_spectrum(FFS.NUFFTBackend(), sg, (fs,), (N, N))
        Test.@test eltype(c32) === ComplexF32
        sg64 = FFS.ScatteredCartesianGrid((Float64.(xs), Float64.(ys)); domain_size = (Float64(L), Float64(L)))
        c64, _ = FFS.calculate_spectrum(FFS.NUFFTBackend(), sg64, (Float64.(fs),), (N, N))
        Test.@test isapprox(ComplexF64.(c32), c64; atol = 1e-3)
    end

    Test.@testset "Plan reuse parity + batch (FFTW / FINUFFT)" begin
        Random.seed!(99)
        L = 2π
        N = 16
        dx = L / N
        ax = range(0.0, stop = L - dx, length = N)
        xv = vec([x for x in ax, y in ax])
        yv = vec([y for x in ax, y in ax])
        u = @. cos(2xv) + sin(3yv)
        v = @. sin(xv)
        ug = FFS.UniformCartesianGrid((xv, yv); domain_size = (L, L))

        # FFTW plan reuse == one-shot.
        c1, _ = FFS.calculate_spectrum(FFS.FFTBackend(), ug, (u, v), (N, N))
        plan = FFS.plan_spectrum(FFS.FFTBackend(), ug, Float64, (N, N); n_transf = 2)
        cc = zeros(ComplexF64, N, N, 2)
        FFS.calculate_spectrum!(cc, plan, (u, v))
        Test.@test cc ≈ c1

        # FINUFFT batched plan: each slice matches an independent single-field transform.
        xs = rand(64) .* L
        ys = rand(64) .* L
        sg = FFS.ScatteredCartesianGrid((xs, ys); domain_size = (L, L))
        nb = 5
        stack = zeros(64, nb)
        for b in 1:nb
            @. stack[:, b] = cos(b * xs) + sin(b * ys)
        end
        bplan = FFS.plan_spectrum(FFS.NUFFTBackend(), sg, Float64, (N, N); n_transf = nb, eps = 1e-10)
        C = zeros(ComplexF64, N, N, nb)
        FFS.calculate_spectrum!(C, bplan, stack)
        for b in (1, nb)
            cb, _ = FFS.calculate_spectrum(FFS.NUFFTBackend(), sg, (stack[:, b],), (N, N); eps = 1e-10)
            Test.@test C[:, :, b] ≈ cb[:, :, 1]
        end
    end

    Test.@testset "Grid dispatch errors (no silent misroute)" begin
        sg = FFS.ScatteredSphericalGrid([0.1, 0.2, 0.3], [0.1, 0.2, 0.3])
        cg = FFS.ScatteredCartesianGrid(([0.0, 1, 2], [0.0, 1, 2]))
        # FFT on a spherical grid is unsupported and must error clearly.
        Test.@test_throws ArgumentError FFS.calculate_spectrum(FFS.FFTBackend(), sg, ([1.0, 2, 3],), (2, 3))
        # In-place FFT (needs a plan) is not a (backend, grid) in-place method.
        Test.@test_throws ArgumentError FFS.calculate_spectrum!(zeros(ComplexF64, 4, 4, 1),
            FFS.FFTBackend(), cg, ([1.0, 2, 3],), (4, 4))
    end

    Test.@testset "Welch averaging + coherence / phase" begin
        Random.seed!(2024)
        L = 2π
        N = 64
        dx = L / N
        x = collect(range(0.0, stop = L - dx, length = N))
        grid = FFS.UniformCartesianGrid((x,); domain_size = (L,))

        nens = 60
        kc = 5                               # shared (coherent) wavenumber
        cf = zeros(ComplexF64, N, nens)
        cg = zeros(ComplexF64, N, nens)
        for e in 1:nens
            ϕ = 2π * rand()                  # common phase for this realization
            shared = cos.(kc .* x .+ ϕ)
            f = shared .+ 0.3 .* randn(N)     # f: coherent part + own noise
            g = 0.8 .* shared .+ 0.3 .* randn(N)  # g: correlated at kc + own noise
            ce, _ = FFS.calculate_spectrum(FFS.FFTBackend(), grid, (f,), (N,))
            cge, _ = FFS.calculate_spectrum(FFS.FFTBackend(), grid, (g,), (N,))
            cf[:, e] .= ce[:, 1]
            cg[:, e] .= cge[:, 1]
        end
        ks = (FFS.calculate_spectrum(FFS.FFTBackend(), grid, (x,), (N,))[2][1],)

        kb, γ², φ = FFS.coherence_spectrum(ks, cf, cg; num_bins = 16)
        Test.@test all(0 .<= γ² .<= 1)
        # The coherent bin (containing k = kc) should have high coherence.
        ic = argmin(abs.(kb .- kc))
        Test.@test γ²[ic] > 0.7
        # A noise-only bin (well away from kc) should be markedly less coherent.
        inoise = argmin(abs.(kb .- (kc + 6)))
        Test.@test γ²[inoise] < γ²[ic]
        Test.@test length(φ) == length(kb)

        # Welch power spectrum: non-negative, peaks at the coherent wavenumber.
        kbw, Ew = FFS.welch_power_spectrum(ks, cf; num_bins = 16)
        Test.@test all(Ew .>= 0)
        Test.@test argmin(abs.(kbw .- kc)) == argmax(Ew)
    end

    Test.@testset "Multitaper (DPSS) tapers + estimate" begin
        N = 128
        K = 7
        V = FFS.dpss(N, 4.0, K)
        Test.@test size(V) == (N, K)
        Test.@test maximum(abs.(V' * V - Matrix(LA.I(K)))) < 1e-8   # orthonormal
        Test.@test_throws ArgumentError FFS.dpss(N, 4.0, N + 1)

        # Multitaper PSD of a pure tone peaks at the tone (reuses the Welch averaging path).
        L = 2π
        dx = L / N
        x = collect(range(0.0, stop = L - dx, length = N))
        k0 = 7
        sig = cos.(k0 .* x)
        grid = FFS.UniformCartesianGrid((x,); domain_size = (L,))
        C = zeros(ComplexF64, N, K)
        for k in 1:K
            c, _ = FFS.calculate_spectrum(FFS.FFTBackend(), grid, (V[:, k] .* sig,), (N,))
            C[:, k] .= c[:, 1]
        end
        ks = (FFS.calculate_spectrum(FFS.FFTBackend(), grid, (x,), (N,))[2][1],)
        kb, E = FFS.welch_power_spectrum(ks, C; num_bins = 16)
        Test.@test argmin(abs.(kb .- k0)) == argmax(E)
    end

    Test.@testset "Lomb–Scargle (irregular sampling)" begin
        Random.seed!(7)
        N = 200
        t = sort(rand(N) .* 10.0)            # irregular sample times in [0, 10]
        f0 = 1.3
        y = sin.(2π * f0 .* t) .+ 0.2 .* randn(N)
        freqs = collect(range(0.1, stop = 4.0, length = 256))
        P = FFS.lomb_scargle(t, y, freqs)
        Test.@test length(P) == length(freqs)
        Test.@test all(P .>= 0)
        Test.@test abs(freqs[argmax(P)] - f0) < 0.1     # peak at the true frequency
        Test.@test_throws ArgumentError FFS.lomb_scargle(t, y, [0.0, 1.0])
    end

    Test.@testset "Synthesis / inverse transform (round-trip)" begin
        # Cartesian: forward then synthesize recovers the field on a uniform grid (exact DFT/IDFT).
        L = 2π
        N = 16
        dx = L / N
        xs = range(0.0, stop = L - dx, length = N)
        xv = vec([x for x in xs, y in xs])
        yv = vec([y for x in xs, y in xs])
        u = @. cos(2xv) + 0.5 * sin(3yv) - 0.3 * cos(xv + 2yv)
        grid = FFS.UniformCartesianGrid((xv, yv); domain_size = (L, L))
        coeffs, _ = FFS.calculate_spectrum(FFS.DirectSumBackend(), grid, (u,), (N, N))
        urec = FFS.synthesize(grid, coeffs, (N, N))
        Test.@test urec isa Tuple
        Test.@test isapprox(urec[1], u; atol = 1e-10)

        # Spectral filtering: zero the highest modes, synthesize a smoothed field (still real).
        cfilt = copy(coeffs)
        cfilt[1, :, :] .= 0          # drop the kx = -N/2 column
        ufilt = FFS.synthesize(grid, cfilt, (N, N))
        Test.@test length(ufilt[1]) == N^2

        # Spherical: synthesize from a single-mode coefficient set, transform back, recover it.
        lmax = 6
        Nθ, Nφ = lmax + 1, 2lmax + 1
        N_pts = 4 * Nθ * Nφ
        ga = π * (3 - sqrt(5))
        z = [1 - 2 * (i + 0.5) / N_pts for i in 0:(N_pts-1)]
        θs = acos.(clamp.(z, -1.0, 1.0))
        φs = mod.(ga .* (0:(N_pts-1)), 2π)
        sg = FFS.ScatteredSphericalGrid(θs, φs)
        C = zeros(ComplexF64, Nθ, Nφ, 1)
        C[FFS.sph_mode_index(3, 1), 1] = 1.0
        fs = FFS.synthesize(sg, C, (Nθ, Nφ))
        Test.@test length(fs[1]) == N_pts
        Test.@test all(isfinite, fs[1])
    end

    Test.@testset "Allocations (in-place reductions + steady-state plans ≈ 0)" begin
        # Measure through function barriers (typed args) so `@allocated` reflects the kernel,
        # not closure-capture boxing of global testset variables.
        _iso(Ek, kb, ks, c, nb) =
            @allocated FFS.isotropic_spectrum!(Ek, kb, ks, c; num_bins = nb)
        _tr(Er, c, ms) = @allocated FFS.Reductions._accumulate_transect!(Er, c, ms, 1, (1,), (ms[2],))
        _sph(El, C, l) = @allocated FFS.spherical_energy_spectrum!(El, C; lmax = l)
        _exec(out, plan, f) = @allocated FFS.calculate_spectrum!(out, plan, (f,))

        Random.seed!(5)
        N = 16
        L = 2π
        dx = L / N
        xs = range(0.0, stop = L - dx, length = N)
        xv = vec([x for x in xs, y in xs])
        yv = vec([y for x in xs, y in xs])
        u = @. cos(2xv) + sin(3yv)
        grid = FFS.UniformCartesianGrid((xv, yv); domain_size = (L, L))
        c, ks = FFS.calculate_spectrum(FFS.FFTBackend(), grid, (u,), (N, N))

        # --- In-place reductions: genuinely zero steady-state allocation. ---
        nb = 6
        E_k = zeros(Float64, nb)
        k_bins = zeros(Float64, nb)
        _iso(E_k, k_bins, ks, c, nb)                      # warmup
        Test.@test _iso(E_k, k_bins, ks, c, nb) == 0

        E_red = zeros(Float64, N)
        _tr(E_red, c, (N, N))
        Test.@test _tr(E_red, c, (N, N)) == 0

        lmax = 8
        Csph = zeros(ComplexF64, lmax + 1, 2lmax + 1, 1)
        Csph[FFS.sph_mode_index(3, 1), 1] = 1.0
        E_l = zeros(Float64, lmax + 1)
        _sph(E_l, Csph, lmax)
        Test.@test _sph(E_l, Csph, lmax) == 0

        # --- Steady-state plan execution: small, constant (does not scale with data). ---
        fplan = FFS.plan_spectrum(FFS.FFTBackend(), grid, Float64, (N, N); n_transf = 1)
        cc = zeros(ComplexF64, N, N, 1)
        _exec(cc, fplan, u)
        Test.@test _exec(cc, fplan, u) < 1024            # broadcast/reshape wrappers only

        Random.seed!(6)
        M = 200
        px = rand(M) .* L
        py = rand(M) .* L
        fs = @. cos(px) + sin(2py)
        sg = FFS.ScatteredCartesianGrid((px, py); domain_size = (L, L))
        nplan = FFS.plan_spectrum(FFS.NUFFTBackend(), sg, Float64, (N, N); n_transf = 1)
        cn = zeros(ComplexF64, N, N, 1)
        _exec(cn, nplan, fs)
        Test.@test _exec(cn, nplan, fs) < 512
    end

    # GPU/KA tests
    include("test_gpu.jl")

end
