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
using OhMyThreads: OhMyThreads
using KernelAbstractions: KernelAbstractions as KA

Test.@testset "FlowFieldSpectra.jl Test Suite" begin

    Test.@testset "Aqua Code Quality Analysis" begin
        Aqua.test_all(FFS; ambiguities = false, unbound_args = (VERSION >= v"1.12"))
    end

    Test.@testset "Explicit imports (no implicit / no stale)" begin
        Test.@test (EI.check_no_implicit_imports(FFS); true)
        Test.@test (EI.check_no_stale_explicit_imports(FFS); true)
        for extname in (
            :FlowFieldSpectraFFTWExt,
            :FlowFieldSpectraFINUFFTExt,
            :FlowFieldSpectraFastSphericalHarmonicsExt,
            :FlowFieldSpectraNUFSHTExt,
            :FlowFieldSpectraOhMyThreadsExt,
            :FlowFieldSpectraKernelAbstractionsExt,
            :FlowFieldSpectraGPUFFTExt,
        )
            ext = Base.get_extension(FFS, extname)
            ext === nothing && continue
            Test.@test (EI.check_no_implicit_imports(ext); true)
            Test.@test (EI.check_no_stale_explicit_imports(ext); true)
        end
    end

    Test.@testset "Cartesian Uniform Parity (Direct vs FFTW)" begin
        L = 10.0
        ms = (16, 16)
        g = FFS.UniformCartesianGrid(; domain = (L, L), n = ms)
        xs, ys = g.axes
        kx1, ky1 = 2π * 2 / L, 2π * 1 / L
        kx2, ky2 = 2π * (-3) / L, 2π * 2 / L
        u = [cos(kx1 * x + ky1 * y) + 0.5 * sin(kx2 * x + ky2 * y) for x in xs, y in ys]   # (16,16) tensor
        v = [sin(kx1 * x + ky1 * y) for x in xs, y in ys]

        c_direct, k_direct = FFS.calculate_spectrum(g, (u, v), ms; transform = FFS.DirectSumBackend())
        c_fft, k_fft = FFS.calculate_spectrum(g, (u, v), ms; transform = FFS.FFTBackend())
        Test.@test size(c_fft) == (ms..., 2)
        Test.@test isapprox(c_direct, c_fft, atol = 1e-12)
        Test.@test all(isapprox(k_direct[d], k_fft[d], rtol = 1e-12) for d in 1:2)

        # 1D isotropic energy, folding the 2 velocity components (dim 3) into the energy.
        k_bins, E_k = FFS.isotropic_spectrum(k_fft, c_fft; num_bins = 8, dims = 3)
        Test.@test length(k_bins) == 8
        Test.@test all(E_k .>= 0.0)

        # Parseval: Σ|C|² (1/N-normalized) == mean(u²)+mean(v²).
        Test.@test isapprox(sum(abs2, c_fft), Statistics.mean(abs2, u) + Statistics.mean(abs2, v); rtol = 1e-5)

        # Transect: integrate out dim 1, keep dim 2 + the component batch → (my, 2).
        k_red, E_red = FFS.transect_spectrum(k_fft, c_fft, (1,))
        Test.@test length(k_red) == 1
        Test.@test size(E_red) == (ms[2], 2)
    end

    Test.@testset "Cartesian Non-Uniform Parity (Direct vs FINUFFT)" begin
        Random.seed!(42)
        N = 100
        ms = (8, 8)
        L = 10.0
        xv = rand(N) .* L
        yv = rand(N) .* L
        kx, ky = 2π * 1 / L, 2π * (-1) / L
        u = @. cos(kx * xv + ky * yv)          # (N,) scattered field
        v = @. sin(kx * xv + ky * yv)
        g = FFS.ScatteredCartesianGrid((xv, yv); domain_size = (L, L))

        c_direct, k_direct = FFS.calculate_spectrum(g, (u, v), ms; transform = FFS.DirectSumBackend())
        c_nufft, k_nufft = FFS.calculate_spectrum(g, (u, v), ms; transform = FFS.NUFFTBackend(), eps = 1e-12)
        Test.@test size(c_nufft) == (ms..., 2)
        Test.@test isapprox(c_direct, c_nufft, atol = 1e-10)
        Test.@test all(isapprox(k_direct[d], k_nufft[d], rtol = 1e-12) for d in 1:2)
    end

    Test.@testset "Nonuniform-gridded Parity (separable NUFFT vs Direct)" begin
        # NonuniformCartesianGrid: a tensor-product grid with nonuniform axis spacing. The NUFFT path is
        # a separable sweep of 1-D transforms (FINUFFTExt._nufft_axis) — no ∏N_d coordinate blob. Because
        # every transform is 1-D it also covers D=3 (FINUFFT's per-call D≤3 cap is never hit).
        Random.seed!(7)
        L = (2π, 2π)
        xax = sort(rand(13)) .* L[1]
        yax = sort(rand(11)) .* L[2]
        g = FFS.NonuniformCartesianGrid((xax, yax); domain_size = L)
        ms = (8, 7)
        f = [cos(2x) + 0.5sin(3y) + 0.3cos(x + 2y) for x in xax, y in yax]

        c_d, k_d = FFS.calculate_spectrum(g, f, ms; transform = FFS.DirectSumBackend())
        c_n, k_n = FFS.calculate_spectrum(g, f, ms; transform = FFS.NUFFTBackend(), eps = 1e-13)
        Test.@test size(c_n) == ms
        Test.@test isapprox(c_d, c_n, atol = 1e-10)
        Test.@test all(isapprox(k_d[d], k_n[d], rtol = 1e-12) for d in 1:2)

        # trailing (nz, nt) batch preserved and batched through the separable sweep
        fb = cat(cat(f, 2f, -0.5f; dims = 3), 0.7 .* cat(f, 2f, -0.5f; dims = 3); dims = 4)  # (13,11,3,2)
        cb_d, _ = FFS.calculate_spectrum(g, fb, ms; transform = FFS.DirectSumBackend())
        cb_n, _ = FFS.calculate_spectrum(g, fb, ms; transform = FFS.NUFFTBackend(), eps = 1e-13)
        Test.@test size(cb_n) == (ms..., 3, 2)
        Test.@test isapprox(cb_d, cb_n, atol = 1e-10)

        # D = 3 (separable = all 1-D, so FINUFFT's D≤3 per-call cap is irrelevant here)
        zax = sort(rand(6)) .* 2π
        g3 = FFS.NonuniformCartesianGrid((xax, yax, zax); domain_size = (2π, 2π, 2π))
        ms3 = (5, 4, 3)
        f3 = [cos(x) + sin(2y) + 0.4cos(z) for x in xax, y in yax, z in zax]
        c3_d, _ = FFS.calculate_spectrum(g3, f3, ms3; transform = FFS.DirectSumBackend())
        c3_n, _ = FFS.calculate_spectrum(g3, f3, ms3; transform = FFS.NUFFTBackend(), eps = 1e-13)
        Test.@test isapprox(c3_d, c3_n, atol = 1e-9)
    end

    Test.@testset "Spherical Structured Parity (Direct vs FastSphericalHarmonics)" begin
        lmax = 8
        Nθ = lmax + 1
        Nφ = 2 * lmax + 1
        pts = FastSphericalHarmonics.sph_points(Nθ)          # (θ axis, φ axis)
        g = FFS.StructuredSphericalGrid(pts[1], pts[2])
        C_true = zeros(Nθ, Nφ)
        C_true[FastSphericalHarmonics.sph_mode(2, 1)] = 1.0
        C_true[FastSphericalHarmonics.sph_mode(3, -2)] = 0.5
        f = FastSphericalHarmonics.sph_evaluate(C_true)      # (Nθ, Nφ) tensor field

        c_sht, _ = FFS.calculate_spectrum(g, f, (Nθ, Nφ); transform = FFS.SHTBackend())
        Test.@test size(c_sht) == (Nθ, Nφ)
        for l in 0:lmax, m in -l:l
            Test.@test isapprox(c_sht[FFS.sph_mode_index(l, m)], C_true[FastSphericalHarmonics.sph_mode(l, m)], atol = 1e-10)
        end

        deg, E_l = FFS.spherical_energy_spectrum(c_sht)
        Test.@test length(deg) == lmax + 1
        Test.@test E_l[3] > 0.0
        Test.@test E_l[4] > 0.0
        Test.@test isapprox(E_l[1], 0.0, atol = 1e-10)
    end

    Test.@testset "Spherical Unstructured Parity (Direct vs NUFSHT)" begin
        lmax = 5
        Nθ = lmax + 1
        Nφ = 2 * lmax + 1
        N_pts = 4 * Nθ^2
        Random.seed!(42)
        ga = π * (3 - sqrt(5))
        zf = [1 - 2 * (i + 0.5) / N_pts for i in 0:(N_pts-1)]
        θ_nodes = acos.(clamp.(zf, -1.0, 1.0))
        φ_nodes = mod.(ga .* (0:(N_pts-1)), 2π)

        C_true = zeros(Nθ, Nφ)
        C_true[FastSphericalHarmonics.sph_mode(2, 0)] = 1.0
        plan = NUFSHT.make_plan(θ_nodes, φ_nodes, lmax)
        f_val = zeros(N_pts)
        NUFSHT.nusht_type2!(f_val, C_true, plan)

        g = FFS.ScatteredSphericalGrid(θ_nodes, φ_nodes)
        c_sol, _ = FFS.calculate_spectrum(g, f_val, (Nθ, Nφ); transform = FFS.NUFSHTBackend(), solve = true, rtol = 1e-10, maxiter = 2000)
        Test.@test isapprox(c_sol[FFS.sph_mode_index(2, 0)], 1.0, atol = 0.05)
    end

    Test.@testset "DirectSum spherical convention == FastSphericalHarmonics" begin
        # DirectSum spherical uses the real-SH (FSH) convention; its synthesis must equal
        # FastSphericalHarmonics.sph_evaluate on the same grid (verifies the s(m)=(-1)^|m|√2 map).
        lmax = 6
        Nθ = lmax + 1
        Nφ = 2 * lmax + 1
        pts = FastSphericalHarmonics.sph_points(Nθ)
        g = FFS.StructuredSphericalGrid(pts[1], pts[2])
        modes = ((0, 0, 0.8), (1, 1, -0.5), (2, -2, 0.6), (4, -3, 0.3), (6, 4, -0.2))
        Cr = zeros(Nθ, Nφ)
        Cc = zeros(ComplexF64, Nθ, Nφ)
        for (l, m, v) in modes
            Cr[FastSphericalHarmonics.sph_mode(l, m)] = v
            Cc[FFS.sph_mode_index(l, m)] = v
        end
        f_ffs = real(FFS.synthesize(g, Cc, (Nθ, Nφ)))
        f_fsh = FastSphericalHarmonics.sph_evaluate(Cr)
        Test.@test isapprox(f_ffs, f_fsh; atol = 1e-11)
    end

    Test.@testset "Gauss-Legendre exact SHT + fast GPU SHT (KA.CPU)" begin
        # On a Gauss-Legendre grid the direct/GPU spherical transform is EXACT (quadrature exact to
        # degree 2lmax+1). The fast device-generic GPU SHT (φ-FFT + Legendre) matches it to round-off.
        lmax = 8
        Nθ = lmax + 1
        Nφ = 2 * lmax + 1
        g = FFS.gauss_legendre_sphere(lmax)
        modes = ((0, 0, 0.8), (1, 0, 1.0), (2, 1, 0.6), (3, -2, 0.5), (4, 0, 0.3), (5, 3, -0.4), (6, -1, 0.7), (8, 4, 0.2))
        C = zeros(ComplexF64, Nθ, Nφ)
        for (l, m, v) in modes
            C[FFS.sph_mode_index(l, m)] = v
        end
        f = real(FFS.synthesize(g, C, (Nθ, Nφ)))

        c_dir, _ = FFS.calculate_spectrum(g, f, (Nθ, Nφ); transform = FFS.DirectSumBackend())
        for (l, m, v) in modes
            Test.@test isapprox(real(c_dir[FFS.sph_mode_index(l, m)]), v; atol = 1e-10)
        end
        Test.@test maximum(abs, imag(c_dir)) < 1e-10

        c_gsht, _ = FFS.calculate_spectrum(g, f, (Nθ, Nφ); transform = FFS.SHTBackend(), execution = FFS.GPUBackend(KA.CPU()))
        c_gds, _ = FFS.calculate_spectrum(g, f, (Nθ, Nφ); transform = FFS.DirectSumBackend(), execution = FFS.GPUBackend(KA.CPU()))
        Test.@test isapprox(c_gsht, c_dir; atol = 1e-10)
        Test.@test isapprox(c_gds, c_dir; atol = 1e-10)

        fb = cat(f, 2 .* f; dims = 3)                          # (Nθ, Nφ, 2) batch
        cb, _ = FFS.calculate_spectrum(g, fb, (Nθ, Nφ); transform = FFS.SHTBackend(), execution = FFS.GPUBackend(KA.CPU()))
        Test.@test size(cb) == (Nθ, Nφ, 2)
        Test.@test isapprox(cb[:, :, 1], c_dir; atol = 1e-10)
        Test.@test isapprox(cb[:, :, 2], 2 .* c_dir; atol = 1e-10)
    end

    Test.@testset "Legendre Recurrence" begin
        FT = Float64
        x = FT(0.5)
        s = sqrt(one(FT) - x^2)
        Test.@test isapprox(FFS.SphericalKernels.normalized_legendre(0, 0, x, s), one(FT) / sqrt(FT(4π)), atol = 1e-15)
        Test.@test isapprox(FFS.SphericalKernels.normalized_legendre(1, 0, x, s), sqrt(FT(3) / (FT(4) * FT(π))) * x, atol = 1e-15)
        Test.@test isapprox(FFS.SphericalKernels.normalized_legendre(1, 1, x, s), -sqrt(FT(3) / (FT(8) * FT(π))) * s, atol = 1e-15)
    end

    Test.@testset "Execution-axis parity (Serial / Threaded / Auto / GPU-CPU)" begin
        Random.seed!(314)
        L = 2π
        N = 16
        ms = (N, N)
        ug = FFS.UniformCartesianGrid(; domain = (L, L), n = ms)
        xs, ys = ug.axes
        u = [cos(2x) + 0.5 * sin(3y) for x in xs, y in ys]
        xv = vec([x for x in xs, y in ys])
        yv = vec([y for x in xs, y in ys])
        uv_scat = vec(u)
        sc = FFS.ScatteredCartesianGrid((xv, yv); domain_size = (L, L))

        ref, kref = FFS.calculate_spectrum(sc, uv_scat, ms; transform = FFS.DirectSumBackend(), execution = FFS.SerialBackend())
        for exec in (FFS.ThreadedBackend(), FFS.AutoBackend(), FFS.GPUBackend(KA.CPU()))
            c, k = FFS.calculate_spectrum(sc, uv_scat, ms; transform = FFS.DirectSumBackend(), execution = exec)
            Test.@test isapprox(c, ref; atol = 1e-12)
            Test.@test all(isapprox(collect(k[d]), collect(kref[d]); rtol = 1e-12) for d in 1:2)
        end

        cf_s, _ = FFS.calculate_spectrum(ug, u, ms; transform = FFS.FFTBackend(), execution = FFS.SerialBackend())
        cf_t, _ = FFS.calculate_spectrum(ug, u, ms; transform = FFS.FFTBackend(), execution = FFS.ThreadedBackend())
        cd, _ = FFS.calculate_spectrum(ug, u, ms; transform = FFS.DirectSumBackend())
        cf_gpu, _ = FFS.calculate_spectrum(ug, u, ms; transform = FFS.FFTBackend(), execution = FFS.GPUBackend(KA.CPU()))
        Test.@test isapprox(cf_s, cf_t; atol = 1e-12)
        Test.@test isapprox(cf_s, cd; atol = 1e-12)
        Test.@test isapprox(cf_s, cf_gpu; atol = 1e-12)

        cn_s, _ = FFS.calculate_spectrum(sc, uv_scat, ms; transform = FFS.NUFFTBackend(), execution = FFS.SerialBackend(), eps = 1e-12)
        cn_t, _ = FFS.calculate_spectrum(sc, uv_scat, ms; transform = FFS.NUFFTBackend(), execution = FFS.ThreadedBackend(), eps = 1e-12)
        Test.@test isapprox(cn_s, cn_t; atol = 1e-10)

        c_ip_s = zeros(ComplexF64, ms...)
        c_ip_t = zeros(ComplexF64, ms...)
        FFS.calculate_spectrum!(c_ip_s, sc, uv_scat, ms; execution = FFS.SerialBackend())
        FFS.calculate_spectrum!(c_ip_t, sc, uv_scat, ms; execution = FFS.ThreadedBackend())
        Test.@test isapprox(c_ip_s, c_ip_t; atol = 1e-12)
        Test.@test isapprox(c_ip_s, ref; atol = 1e-12)

        Random.seed!(20)
        θ = rand(60) .* π
        φ = rand(60) .* 2π
        fθ = rand(60)
        sph = FFS.ScatteredSphericalGrid(θ, φ)
        cs_s, _ = FFS.calculate_spectrum(sph, fθ, (6, 11); transform = FFS.DirectSumBackend(), execution = FFS.SerialBackend())
        for exec in (FFS.ThreadedBackend(), FFS.GPUBackend(KA.CPU()))
            cs, _ = FFS.calculate_spectrum(sph, fθ, (6, 11); transform = FFS.DirectSumBackend(), execution = exec)
            Test.@test isapprox(cs_s, cs; atol = 1e-10)
        end

        lmaxs = 4
        pts = FastSphericalHarmonics.sph_points(lmaxs + 1)
        sgs = FFS.StructuredSphericalGrid(pts[1], pts[2])
        Ct = zeros(lmaxs + 1, 2lmaxs + 1)
        Ct[FastSphericalHarmonics.sph_mode(2, 1)] = 1.0
        fv = FastSphericalHarmonics.sph_evaluate(Ct)
        csht_s, _ = FFS.calculate_spectrum(sgs, fv, (lmaxs + 1, 2lmaxs + 1); transform = FFS.SHTBackend(), execution = FFS.SerialBackend())
        csht_t, _ = FFS.calculate_spectrum(sgs, fv, (lmaxs + 1, 2lmaxs + 1); transform = FFS.SHTBackend(), execution = FFS.ThreadedBackend())
        Test.@test csht_s == csht_t

        coeffs, _ = FFS.calculate_spectrum(ug, u, ms; transform = FFS.DirectSumBackend())
        rs = FFS.synthesize(ug, coeffs, ms; execution = FFS.SerialBackend())
        rt = FFS.synthesize(ug, coeffs, ms; execution = FFS.ThreadedBackend())
        Test.@test isapprox(rs, rt; atol = 1e-12)
        Test.@test isapprox(rs, u; atol = 1e-10)

        Test.@test_throws ArgumentError FFS.calculate_spectrum(sc, uv_scat, ms; transform = FFS.NUFFTBackend(), execution = FFS.GPUBackend(KA.CPU()))
    end

    Test.@testset "Derived-quantity spectra (vorticity / divergence / compensated)" begin
        L = 2π
        N = 16
        g = FFS.UniformCartesianGrid(; domain = (L, L), n = (N, N))
        xs, ys = g.axes
        u = [sin(x) * cos(y) for x in xs, y in ys]
        v = [-cos(x) * sin(y) for x in xs, y in ys]
        c, ks = FFS.calculate_spectrum(g, (u, v), (N, N); transform = FFS.FFTBackend())   # (N,N,2), components on dim 3

        divc = FFS.spectral_divergence(ks, c)
        Test.@test maximum(abs.(divc)) < 1e-12

        omega = [2 * sin(x) * sin(y) for x in xs, y in ys]
        co, _ = FFS.calculate_spectrum(g, omega, (N, N); transform = FFS.FFTBackend())
        vortc = FFS.spectral_vorticity(ks, c)
        Test.@test isapprox(vortc[:, :, 1], co; atol = 1e-12)

        kb, Ek = FFS.isotropic_spectrum(ks, c; num_bins = 6, dims = 3)
        _, Zk = FFS.isotropic_spectrum(ks, vortc; num_bins = 6, dims = 3)
        active = findall(>(1e-12), Ek)
        Test.@test !isempty(active)
        for i in active
            Test.@test isapprox(Zk[i] / Ek[i], 2.0; rtol = 1e-6)
        end
        Test.@test FFS.compensate(kb, Ek, 2.0) ≈ (kb .^ 2) .* Ek
        Test.@test FFS.band_energy(kb, Ek, 0.0, maximum(kb)) >= 0.0
    end

    Test.@testset "Cross-spectrum / co-spectrum" begin
        L = 2π
        N = 16
        g = FFS.UniformCartesianGrid(; domain = (L, L), n = (N, N))
        xs, ys = g.axes
        f = [cos(2x) + 0.5 * sin(3y) for x in xs, y in ys]
        h = [cos(2x) - 0.3 * cos(y) for x in xs, y in ys]
        cf, ks = FFS.calculate_spectrum(g, f, (N, N); transform = FFS.FFTBackend())
        cg, _ = FFS.calculate_spectrum(g, h, (N, N); transform = FFS.FFTBackend())

        kb, Co_ff = FFS.cospectrum(ks, cf, cf; num_bins = 6)
        _, Ek = FFS.isotropic_spectrum(ks, cf; num_bins = 6)
        Test.@test isapprox(Co_ff, Ek; rtol = 1e-10)
        _, Q_ff = FFS.quadspectrum(ks, cf, cf; num_bins = 6)
        Test.@test maximum(abs.(Q_ff)) < 1e-10

        _, Co_fg = FFS.cospectrum(ks, cf, cg; num_bins = 6)
        dk = kb[2] - kb[1]
        Test.@test isapprox(sum(Co_fg) * dk, 0.5 * Statistics.mean(f .* h); rtol = 1e-6)
    end

    Test.@testset "Anisotropy-resolved spectrum E(k,θ)" begin
        L = 2π
        N = 16
        g = FFS.UniformCartesianGrid(; domain = (L, L), n = (N, N))
        xs, ys = g.axes
        f = [cos(3x) + 0.7 * sin(2y) for x in xs, y in ys]
        c, ks = FFS.calculate_spectrum(g, f, (N, N); transform = FFS.FFTBackend())
        kb, θb, E = FFS.anisotropic_spectrum(ks, c; num_k_bins = 6, num_θ_bins = 12)
        Test.@test size(E) == (6, 12)
        Test.@test all(E .>= 0)
        dθ = θb[2] - θb[1]
        E_iso = vec(sum(E; dims = 2)) .* dθ
        _, Ek = FFS.isotropic_spectrum(ks, c; num_bins = 6)
        for ik in 2:6
            Test.@test isapprox(E_iso[ik], Ek[ik]; rtol = 1e-8)
        end
    end

    Test.@testset "Parseval + D=1/2/3 coverage (DirectSum vs FFT)" begin
        Random.seed!(11)
        for D in 1:3
            N = D == 3 ? 8 : 16
            L = 2π
            g = FFS.UniformCartesianGrid(; domain = ntuple(_ -> L, D), n = ntuple(_ -> N, D))
            axs = g.axes
            f = collect([sum(cos((d + 1) * pt[d]) + 0.3 * sin(d * pt[d]) for d in 1:D) for pt in Iterators.product(axs...)])
            f = f .- Statistics.mean(f)
            ms = ntuple(_ -> N, D)
            c_fft, _ = FFS.calculate_spectrum(g, f, ms; transform = FFS.FFTBackend())
            c_dir, _ = FFS.calculate_spectrum(g, f, ms; transform = FFS.DirectSumBackend())
            Test.@test isapprox(c_fft, c_dir; atol = 1e-10)
            Test.@test isapprox(sum(abs2, c_fft), Statistics.mean(abs2, f); rtol = 1e-10)
        end
    end

    Test.@testset "Real-input rfft fast path == full FFT (D=1/2/3)" begin
        Random.seed!(13)
        for D in 1:3
            N = D == 3 ? 6 : 12
            L = 2π
            g = FFS.UniformCartesianGrid(; domain = ntuple(_ -> L, D), n = ntuple(_ -> N, D))
            axs = g.axes
            f = collect([sum(cos((d + 1) * pt[d]) for d in 1:D) for pt in Iterators.product(axs...)])
            ms = ntuple(_ -> N, D)
            c_real, _ = FFS.calculate_spectrum(g, f, ms; transform = FFS.FFTBackend())
            c_cplx, _ = FFS.calculate_spectrum(g, ComplexF64.(f), ms; transform = FFS.FFTBackend())
            Test.@test isapprox(c_real, c_cplx; atol = 1e-12)
        end
    end

    Test.@testset "Float32 end-to-end (DirectSum / FFT / NUFFT)" begin
        Random.seed!(7)
        L = 2.0f0 * Float32(π)
        N = 16
        g = FFS.UniformCartesianGrid(; domain = (L, L), n = (N, N))
        xs, ys = g.axes
        f = Float32[cos(2x) + 0.5f0 * sin(3y) for x in xs, y in ys]
        c_dir, _ = FFS.calculate_spectrum(g, f, (N, N); transform = FFS.DirectSumBackend())
        c_fft, ks = FFS.calculate_spectrum(g, f, (N, N); transform = FFS.FFTBackend())
        Test.@test eltype(c_dir) === ComplexF32
        Test.@test eltype(c_fft) === ComplexF32
        Test.@test isapprox(c_fft, c_dir; atol = 1.0f-4)
        _, E_iso = FFS.isotropic_spectrum(ks, c_fft; num_bins = 6)
        Test.@test eltype(E_iso) === Float32

        xs32 = rand(Float32, 80) .* L
        ys32 = rand(Float32, 80) .* L
        fs = @. cos(xs32) + sin(2ys32)
        sg = FFS.ScatteredCartesianGrid((xs32, ys32); domain_size = (L, L))
        c32, _ = FFS.calculate_spectrum(sg, fs, (N, N); transform = FFS.NUFFTBackend())
        Test.@test eltype(c32) === ComplexF32
    end

    Test.@testset "Plan reuse parity + batch (FFTW / FINUFFT)" begin
        Random.seed!(99)
        L = 2π
        N = 16
        g = FFS.UniformCartesianGrid(; domain = (L, L), n = (N, N))
        xs, ys = g.axes
        u = [cos(2x) + sin(3y) for x in xs, y in ys]
        v = [sin(x) for x in xs, y in ys]

        c1, _ = FFS.calculate_spectrum(g, (u, v), (N, N); transform = FFS.FFTBackend())
        plan = FFS.plan_spectrum(g, Float64, (N, N); transform = FFS.FFTBackend(), batch = (2,))
        cc = zeros(ComplexF64, N, N, 2)
        stack = cat(u, v; dims = 3)
        FFS.calculate_spectrum!(cc, plan, stack)
        Test.@test cc ≈ c1

        xv = rand(64) .* L
        yv = rand(64) .* L
        sg = FFS.ScatteredCartesianGrid((xv, yv); domain_size = (L, L))
        nb = 5
        fstack = zeros(64, nb)
        for b in 1:nb
            fstack[:, b] .= cos.(b .* xv) .+ sin.(b .* yv)
        end
        bplan = FFS.plan_spectrum(sg, Float64, (N, N); transform = FFS.NUFFTBackend(), batch = (nb,), eps = 1e-10)
        C = zeros(ComplexF64, N, N, nb)
        FFS.calculate_spectrum!(C, bplan, fstack)
        for b in (1, nb)
            cb, _ = FFS.calculate_spectrum(sg, fstack[:, b], (N, N); transform = FFS.NUFFTBackend(), eps = 1e-10)
            Test.@test C[:, :, b] ≈ cb
        end
    end

    Test.@testset "Grid dispatch errors (no silent misroute)" begin
        sg = FFS.ScatteredSphericalGrid([0.1, 0.2, 0.3], [0.1, 0.2, 0.3])
        cg = FFS.ScatteredCartesianGrid(([0.0, 1, 2], [0.0, 1, 2]))
        Test.@test_throws ArgumentError FFS.calculate_spectrum(sg, [1.0, 2, 3], (2, 3); transform = FFS.FFTBackend())
        Test.@test_throws ArgumentError FFS.calculate_spectrum!(zeros(ComplexF64, 4, 4), cg, [1.0, 2, 3], (4, 4); transform = FFS.FFTBackend())
    end

    Test.@testset "Welch averaging + coherence / phase" begin
        Random.seed!(2024)
        L = 2π
        N = 64
        g = FFS.UniformCartesianGrid(; domain = (L,), n = (N,))
        x = collect(g.axes[1])
        nens = 60
        kc = 5
        cf = zeros(ComplexF64, N, nens)
        cg = zeros(ComplexF64, N, nens)
        for e in 1:nens
            ϕ = 2π * rand()
            shared = cos.(kc .* x .+ ϕ)
            fe = shared .+ 0.3 .* randn(N)
            ge = 0.8 .* shared .+ 0.3 .* randn(N)
            cf[:, e] .= FFS.calculate_spectrum(g, fe, (N,); transform = FFS.FFTBackend())[1]
            cg[:, e] .= FFS.calculate_spectrum(g, ge, (N,); transform = FFS.FFTBackend())[1]
        end
        ks = (FFS.calculate_spectrum(g, x, (N,); transform = FFS.FFTBackend())[2][1],)

        kb, γ², φ = FFS.coherence_spectrum(ks, cf, cg; num_bins = 16)
        Test.@test all(0 .<= γ² .<= 1)
        Test.@test γ²[argmin(abs.(kb .- kc))] > 0.7
        Test.@test length(φ) == length(kb)

        kbw, Ew = FFS.welch_power_spectrum(ks, cf; num_bins = 16)
        Test.@test all(Ew .>= 0)
        Test.@test argmin(abs.(kbw .- kc)) == argmax(Ew)
    end

    Test.@testset "Multitaper (DPSS)" begin
        N = 128
        K = 7
        V = FFS.dpss(N, 4.0, K)
        Test.@test size(V) == (N, K)
        Test.@test maximum(abs.(V' * V - Matrix(LA.I(K)))) < 1e-8
        Test.@test_throws ArgumentError FFS.dpss(N, 4.0, N + 1)

        L = 2π
        g = FFS.UniformCartesianGrid(; domain = (L,), n = (N,))
        x = collect(g.axes[1])
        k0 = 7
        sig = cos.(k0 .* x)
        C = zeros(ComplexF64, N, K)
        for k in 1:K
            C[:, k] .= FFS.calculate_spectrum(g, V[:, k] .* sig, (N,); transform = FFS.FFTBackend())[1]
        end
        ks = (FFS.calculate_spectrum(g, x, (N,); transform = FFS.FFTBackend())[2][1],)
        kb, E = FFS.welch_power_spectrum(ks, C; num_bins = 16)
        Test.@test argmin(abs.(kb .- k0)) == argmax(E)
    end

    Test.@testset "Lomb–Scargle (irregular sampling)" begin
        Random.seed!(7)
        N = 200
        t = sort(rand(N) .* 10.0)
        f0 = 1.3
        y = sin.(2π * f0 .* t) .+ 0.2 .* randn(N)
        freqs = collect(range(0.1, stop = 4.0, length = 256))
        P = FFS.lomb_scargle(t, y, freqs)
        Test.@test length(P) == length(freqs)
        Test.@test all(P .>= 0)
        Test.@test abs(freqs[argmax(P)] - f0) < 0.1
        Test.@test_throws ArgumentError FFS.lomb_scargle(t, y, [0.0, 1.0])
    end

    Test.@testset "Synthesis / inverse transform (round-trip)" begin
        L = 2π
        N = 16
        g = FFS.UniformCartesianGrid(; domain = (L, L), n = (N, N))
        xs, ys = g.axes
        u = [cos(2x) + 0.5 * sin(3y) - 0.3 * cos(x + 2y) for x in xs, y in ys]
        coeffs, _ = FFS.calculate_spectrum(g, u, (N, N); transform = FFS.DirectSumBackend())
        urec = FFS.synthesize(g, coeffs, (N, N))
        Test.@test isapprox(urec, u; atol = 1e-10)

        lmax = 6
        Nθ, Nφ = lmax + 1, 2lmax + 1
        N_pts = 4 * Nθ * Nφ
        ga = π * (3 - sqrt(5))
        z = [1 - 2 * (i + 0.5) / N_pts for i in 0:(N_pts-1)]
        θs = acos.(clamp.(z, -1.0, 1.0))
        φs = mod.(ga .* (0:(N_pts-1)), 2π)
        sg = FFS.ScatteredSphericalGrid(θs, φs)
        C = zeros(ComplexF64, Nθ, Nφ)
        C[FFS.sph_mode_index(3, 1)] = 1.0
        fs = FFS.synthesize(sg, C, (Nθ, Nφ))
        Test.@test length(fs) == N_pts
        Test.@test all(isfinite, fs)
    end

    include("test_allocs.jl")
    include("test_gpu.jl")
    include("test_distributed.jl")
    include("test_mpi.jl")
end
