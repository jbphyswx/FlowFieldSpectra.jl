using Test: Test
using Random: Random
using Statistics: Statistics
using LinearAlgebra: LinearAlgebra as LA
using Aqua: Aqua as Aqua
using ExplicitImports: ExplicitImports as EI

using FlowFieldSpectra: FlowFieldSpectra as FFS
using FFTW: FFTW
using FINUFFT: FINUFFT
using NonuniformFFTs: NonuniformFFTs
using FastSphericalHarmonics: FastSphericalHarmonics as FSH
using NUFSHT: NUFSHT
using OhMyThreads: OhMyThreads
using KernelAbstractions: KernelAbstractions as KA

using ComputationalBackends: ComputationalBackends as CB
using SpectralBackends: SpectralBackends as SB
using FlowGeometries: FlowGeometries as FG

# =============================================================================
# Test-local grid constructors (test utilities, not FFS API — FFS builds directly on FlowGeometries).
# FFS grids ARE FlowGeometries grids: a Cartesian grid over a CartesianGeometry, a spherical grid over a
# SphericalGeometry. Spectral transforms need a periodic domain, so these mark the grid periodic with an
# explicit wrap length. Spherical fields are `(nlon, nlat)`; `(θ, φ)` = (colatitude, longitude) map to
# FlowGeometries `(λ, φ_lat)` = (longitude, geographic latitude) as `λ = φ`, `φ_lat = π/2 − θ`.
# =============================================================================
_cg(::Type{T}) where {T} = FG.Geometry.CartesianGeometry{T}()

# Packed coefficient size for a Cartesian transform of full sizes `ms`: a real field halves axis 1.
pks(ms::Tuple, real::Bool = true) = FFS.Packing.packed_size(NTuple{length(ms), Int}(ms), Val(real))

# Two-sided Parseval sum over a packed spectrum: each stored mode counts with its fold weight, so this
# equals `mean(abs2, field)` under the 1/∏N normalization on any layout.
parseval(ks::Tuple, c::AbstractArray) =
    sum(I -> FFS.Packing.mode_fold(ks, I) * abs2(c[I]), CartesianIndices(c))

# Uniform Cartesian grid from per-axis point counts + periodic domain lengths.
function ucg(::Type{T}, domain::Tuple, n::Tuple) where {T}
    D = length(n)
    axes = ntuple(d -> range(zero(T), T(domain[d]); length = n[d] + 1)[1:n[d]], D)
    return FG.Grids.StructuredGrid(_cg(T), axes...; periodic = ntuple(_ -> true, D), period = T.(domain))
end
ucg(domain::Tuple, n::Tuple) = ucg(Float64, domain, n)
ucg_axis(::Type{T}, domain, n) where {T} = range(zero(T), T(domain); length = n + 1)[1:n]  # matching axis values

# Nonuniform-gridded Cartesian (stretched axes + periodic domain).
nucg(::Type{T}, axes::Tuple, domain::Tuple) where {T} =
    FG.Grids.StructuredGrid(_cg(T), ntuple(d -> collect(T, axes[d]), length(axes))...;
        periodic = ntuple(_ -> true, length(axes)), period = T.(domain))
nucg(axes::Tuple, domain::Tuple) = nucg(Float64, axes, domain)

# Scattered Cartesian point cloud (unused per-node measure = ones; periodic domain).
scg(::Type{T}, coords::Tuple, domain::Tuple) where {T} =
    FG.Grids.UnstructuredGrid(_cg(T), ntuple(d -> collect(T, coords[d]), length(coords)),
        ones(T, length(coords[1])); periodic = ntuple(_ -> true, length(coords)), period = T.(domain))
scg(coords::Tuple, domain::Tuple) = scg(Float64, coords, domain)

# Structured spherical grids: Clenshaw–Curtis matches FastSphericalHarmonics' grid (SHT); Gauss–Legendre
# gives exact weighted-quadrature (DirectSum / GPU SHT with sampling=GaussLegendreSampling()).
sph_cc(lmax) = FG.Connectivity.structured_grid(Float64, FG.SphericalSampling.ClenshawCurtisSampling(), lmax + 1)
sph_gl(lmax) = FG.Connectivity.structured_grid(Float64, FG.SphericalSampling.GaussLegendreSampling(), lmax + 1)
const CC = FG.SphericalSampling.ClenshawCurtisSampling()
const GL = FG.SphericalSampling.GaussLegendreSampling()
# Scattered sphere from FFS (θ=colatitude, φ=longitude).
sph_scat(θ, φ) = FG.Grids.UnstructuredGrid(FG.Geometry.SphericalGeometry(1.0),
    (collect(Float64, φ), Float64(π) / 2 .- collect(Float64, θ)), ones(Float64, length(θ)))

Test.@testset "FlowFieldSpectra.jl Test Suite" begin

    Test.@testset "Aqua Code Quality Analysis" begin
        Aqua.test_all(FFS; ambiguities = false, unbound_args = (VERSION >= v"1.12"))
    end

    Test.@testset "Explicit imports (no implicit / no stale)" begin
        Test.@test (EI.check_no_implicit_imports(FFS); true)
        Test.@test (EI.check_no_stale_explicit_imports(FFS); true)
        for extname in (
            :FlowFieldSpectraFFTWExt, :FlowFieldSpectraFINUFFTExt,
            :FlowFieldSpectraFastSphericalHarmonicsExt, :FlowFieldSpectraNonuniformFFTsExt, :FlowFieldSpectraNUFSHTExt,
            :FlowFieldSpectraOhMyThreadsExt, :FlowFieldSpectraKernelAbstractionsExt,
            :FlowFieldSpectraGPUFFTExt, :FlowFieldSpectraNUFSHTKernelAbstractionsExt,
        )
            ext = Base.get_extension(FFS, extname)
            ext === nothing && continue
            Test.@test (EI.check_no_implicit_imports(ext); true)
            Test.@test (EI.check_no_stale_explicit_imports(ext); true)
        end
    end

    Test.@testset "AutoBackend resolves locally (loadable, honest)" begin
        # ComputationalBackends.resolve_backend(AutoBackend) errors by design; FFS resolves it locally.
        rb = FFS._resolve_execution(CB.AutoBackend())
        Test.@test rb isa Union{CB.SerialBackend, CB.ThreadedBackend}
        Test.@test FFS._resolve_execution(CB.SerialBackend()) === CB.SerialBackend()
    end

    Test.@testset "Cartesian Uniform Parity (Direct vs FFTW)" begin
        L = 10.0
        ms = (16, 16)
        xs = ucg_axis(Float64, L, 16); ys = ucg_axis(Float64, L, 16)
        g = ucg((L, L), ms)
        kx1, ky1 = 2π * 2 / L, 2π * 1 / L
        kx2, ky2 = 2π * (-3) / L, 2π * 2 / L
        u = [cos(kx1 * x + ky1 * y) + 0.5 * sin(kx2 * x + ky2 * y) for x in xs, y in ys]
        v = [sin(kx1 * x + ky1 * y) for x in xs, y in ys]

        c_direct, k_direct = FFS.calculate_spectrum(g, (u, v), ms; transform = SB.DirectSumSpectralBackend())
        c_fft, k_fft = FFS.calculate_spectrum(g, (u, v), ms; transform = SB.FFTSpectralBackend())
        Test.@test size(c_fft) == (pks(ms)..., 2)                # real field ⇒ axis 1 halved
        Test.@test isapprox(c_direct, c_fft, atol = 1e-12)
        Test.@test all(isapprox(k_direct[d], k_fft[d], rtol = 1e-12) for d in 1:2)
        # `unpacked` restores the full native cube, which the complex transform produces directly.
        c_cplx, _ = FFS.calculate_spectrum(g, ComplexF64.(cat(u, v; dims = 3)), ms; transform = SB.FFTSpectralBackend())
        Test.@test size(c_cplx) == (ms..., 2)
        Test.@test isapprox(FFS.unpacked(c_fft, ms), c_cplx; atol = 1e-12)

        k_bins, E_k = FFS.isotropic_spectrum(k_fft, c_fft; num_bins = 8, dims = 3)
        Test.@test length(k_bins) == 8
        Test.@test all(E_k .>= 0.0)
        Test.@test isapprox(parseval(k_fft, c_fft), Statistics.mean(abs2, u) + Statistics.mean(abs2, v); rtol = 1e-5)

        k_red, E_red = FFS.transect_spectrum(k_fft, c_fft, (1,))
        Test.@test length(k_red) == 1
        Test.@test size(E_red) == (ms[2], 2)

        # A transect recovers the folded Parseval total whichever axis it keeps: a kept FULL axis carries
        # both signs of its wavenumber, and a kept HALVED axis reports |k₁| with the −k₁ energy folded in.
        # So the answer cannot depend on which axis the real-input layout happened to halve.
        dk1 = Float64(k_fft[1][2] - k_fft[1][1])
        dk2 = Float64(k_fft[2][2] - k_fft[2][1])
        total = 0.5 * parseval(k_fft, c_fft)
        _, E_keep_full = FFS.transect_spectrum(k_fft, c_fft, (1,))       # keep axis 2 (full, signed)
        _, E_keep_half = FFS.transect_spectrum(k_fft, c_fft, (2,))       # keep axis 1 (halved)
        Test.@test isapprox(sum(E_keep_full) / dk1, total; rtol = 1e-12)
        Test.@test isapprox(sum(E_keep_half) / dk2, total; rtol = 1e-12)
        # A real field's spectrum is symmetric on a kept full axis, so ±k report the same energy.
        Test.@test maximum(i -> abs(E_keep_full[i, 1] - E_keep_full[mod(ms[2] - i + 1, ms[2]) + 1, 1]),
            2:ms[2]) < 1e-14
    end

    Test.@testset "Cartesian Scattered Parity (Direct vs FINUFFT)" begin
        Random.seed!(42)
        N = 100; ms = (8, 8); L = 10.0
        xv = rand(N) .* L; yv = rand(N) .* L
        kx, ky = 2π * 1 / L, 2π * (-1) / L
        u = @. cos(kx * xv + ky * yv)
        v = @. sin(kx * xv + ky * yv)
        g = scg((xv, yv), (L, L))
        c_direct, k_direct = FFS.calculate_spectrum(g, (u, v), ms; transform = SB.DirectSumSpectralBackend())
        c_nufft, k_nufft = FFS.calculate_spectrum(g, (u, v), ms; transform = FFS.FINUFFTBackend(), eps = 1e-12)
        Test.@test size(c_nufft) == (pks(ms)..., 2)
        Test.@test isapprox(c_direct, c_nufft, atol = 1e-10)
        Test.@test all(isapprox(k_direct[d], k_nufft[d], rtol = 1e-12) for d in 1:2)
    end

    Test.@testset "Nonuniform-gridded Parity (separable NUFFT vs Direct)" begin
        Random.seed!(7)
        L = (2π, 2π)
        xax = sort(rand(13)) .* L[1]; yax = sort(rand(11)) .* L[2]
        g = nucg((xax, yax), L)
        ms = (8, 7)
        f = [cos(2x) + 0.5sin(3y) + 0.3cos(x + 2y) for x in xax, y in yax]
        c_d, k_d = FFS.calculate_spectrum(g, f, ms; transform = SB.DirectSumSpectralBackend())
        c_n, k_n = FFS.calculate_spectrum(g, f, ms; transform = FFS.FINUFFTBackend(), eps = 1e-13)
        Test.@test size(c_n) == pks(ms)
        Test.@test isapprox(c_d, c_n, atol = 1e-10)
        Test.@test all(isapprox(k_d[d], k_n[d], rtol = 1e-12) for d in 1:2)

        fb = cat(cat(f, 2f, -0.5f; dims = 3), 0.7 .* cat(f, 2f, -0.5f; dims = 3); dims = 4)  # (13,11,3,2)
        cb_d, _ = FFS.calculate_spectrum(g, fb, ms; transform = SB.DirectSumSpectralBackend())
        cb_n, _ = FFS.calculate_spectrum(g, fb, ms; transform = FFS.FINUFFTBackend(), eps = 1e-13)
        Test.@test size(cb_n) == (pks(ms)..., 3, 2)
        Test.@test isapprox(cb_d, cb_n, atol = 1e-10)

        zax = sort(rand(6)) .* 2π
        g3 = nucg((xax, yax, zax), (2π, 2π, 2π))
        ms3 = (5, 4, 3)
        f3 = [cos(x) + sin(2y) + 0.4cos(z) for x in xax, y in yax, z in zax]
        c3_d, _ = FFS.calculate_spectrum(g3, f3, ms3; transform = SB.DirectSumSpectralBackend())
        c3_n, _ = FFS.calculate_spectrum(g3, f3, ms3; transform = FFS.FINUFFTBackend(), eps = 1e-13)
        Test.@test isapprox(c3_d, c3_n, atol = 1e-9)
    end

    Test.@testset "Spherical Structured SHT round-trip (FastSphericalHarmonics)" begin
        # FastSphericalHarmonics is an exact analysis on the Clenshaw–Curtis grid, inverse to its
        # synthesis. DirectSum synthesis uses the same real-SH convention, so SHT ∘ DirectSum-synth = id.
        lmax = 8; Nθ = lmax + 1; Nφ = 2 * lmax + 1
        g = sph_cc(lmax)
        Test.@test size(g) == (Nφ, Nθ)                               # (nlon, nlat)
        C = zeros(ComplexF64, Nθ, Nφ)
        for (l, m, v) in ((2, 1, 1.0), (3, -2, 0.5), (0, 0, 0.7), (5, 4, -0.3))
            C[FFS.sph_mode_index(l, m)] = v
        end
        f = real(FFS.synthesize(g, C, (Nθ, Nφ); transform = SB.DirectSumSpectralBackend()))
        c_sht, _ = FFS.calculate_spectrum(g, f, (Nθ, Nφ); transform = SB.FSHTSpectralBackend())
        Test.@test size(c_sht) == (Nθ, Nφ)
        for l in 0:lmax, m in -l:l
            Test.@test isapprox(c_sht[FFS.sph_mode_index(l, m)], C[FFS.sph_mode_index(l, m)], atol = 1e-9)
        end
        deg, E_l = FFS.spherical_energy_spectrum(c_sht)
        Test.@test length(deg) == lmax + 1
        Test.@test E_l[3] > 0.0 && E_l[4] > 0.0
    end

    Test.@testset "Spherical Scattered NUFSHT solve" begin
        lmax = 5; Nθ = lmax + 1; Nφ = 2 * lmax + 1
        N_pts = 4 * Nθ^2
        Random.seed!(42)
        ga = π * (3 - sqrt(5))
        zf = [1 - 2 * (i + 0.5) / N_pts for i in 0:(N_pts - 1)]
        θ_nodes = acos.(clamp.(zf, -1.0, 1.0))
        φ_nodes = mod.(ga .* (0:(N_pts - 1)), 2π)
        C_true = zeros(Nθ, Nφ); C_true[FSH.sph_mode(2, 0)] = 1.0
        plan = NUFSHT.make_plan(Float64, θ_nodes, φ_nodes, lmax)
        f_val = zeros(N_pts); NUFSHT.nusht_type2!(f_val, C_true, plan)
        g = sph_scat(θ_nodes, φ_nodes)
        c_sol, _ = FFS.calculate_spectrum(g, f_val, (Nθ, Nφ); transform = SB.NUFSHTSpectralBackend(), solve = true, rtol = 1e-10, maxiter = 2000)
        Test.@test isapprox(c_sol[FFS.sph_mode_index(2, 0)], 1.0, atol = 0.05)
    end

    Test.@testset "NUFSHT reusable plan (scattered-spherical)" begin
        Random.seed!(11)
        lmax = 6; Nθ = lmax + 1; Nφ = 2 * lmax + 1; ms = (Nθ, Nφ)
        N = 4 * Nθ^2
        θ = rand(N) .* (0.8π) .+ 0.1π
        φ = rand(N) .* 2π
        g = sph_scat(θ, φ)
        # Forward type-1: one plan matches the one-shot AND is reusable across many fields (no re-planning).
        plan = FFS.plan_spectrum(g, Float64, ms; transform = SB.NUFSHTSpectralBackend())
        cbuf = zeros(ComplexF64, Nθ, Nφ)
        for _ in 1:3
            f = randn(N)
            cref, _ = FFS.calculate_spectrum(g, f, ms; transform = SB.NUFSHTSpectralBackend())
            ks = FFS.calculate_spectrum!(cbuf, plan, f)
            Test.@test isapprox(cbuf, cref; atol = 1e-12)
            Test.@test ks == (0:lmax, -lmax:lmax)
        end
        # Batched (ntrans = B).
        B = 3; fb = randn(N, B)
        cb_ref, _ = FFS.calculate_spectrum(g, fb, ms; transform = SB.NUFSHTSpectralBackend())
        planb = FFS.plan_spectrum(g, Float64, ms; transform = SB.NUFSHTSpectralBackend(), batch = (B,))
        cb = zeros(ComplexF64, Nθ, Nφ, B); FFS.calculate_spectrum!(cb, planb, fb)
        Test.@test isapprox(cb, cb_ref; atol = 1e-12)
        # CG solve: reusable plan (persistent workspace) matches the one-shot solve.
        C_true = zeros(Nθ, Nφ); C_true[FSH.sph_mode(2, 0)] = 1.0
        p2 = NUFSHT.make_plan(Float64, θ, φ, lmax); fv = zeros(N); NUFSHT.nusht_type2!(fv, C_true, p2)
        cs_ref, _ = FFS.calculate_spectrum(g, fv, ms; transform = SB.NUFSHTSpectralBackend(), solve = true, rtol = 1e-10, maxiter = 2000)
        psolve = FFS.plan_spectrum(g, Float64, ms; transform = SB.NUFSHTSpectralBackend(), solve = true, rtol = 1e-10, maxiter = 2000)
        cs = zeros(ComplexF64, Nθ, Nφ); FFS.calculate_spectrum!(cs, psolve, fv)
        Test.@test isapprox(cs, cs_ref; atol = 1e-9)
        # `nufft=` selects NUFSHT's internal NUFFT engine: the real-data NonuniformFFTs engine yields the
        # same SHT coefficients as the default (FINUFFT) engine, via both the one-shot and a reusable plan.
        fn = randn(N)
        c_fin, _ = FFS.calculate_spectrum(g, fn, ms; transform = SB.NUFSHTSpectralBackend())
        c_nff, _ = FFS.calculate_spectrum(g, fn, ms; transform = SB.NUFSHTSpectralBackend(), nufft = NUFSHT.NonuniformFFTsBackend())
        Test.@test isapprox(c_nff, c_fin; atol = 1e-7)
        planN = FFS.plan_spectrum(g, Float64, ms; transform = SB.NUFSHTSpectralBackend(), nufft = NUFSHT.NonuniformFFTsBackend())
        cN = zeros(ComplexF64, Nθ, Nφ); FFS.calculate_spectrum!(cN, planN, fn)
        Test.@test isapprox(cN, c_fin; atol = 1e-7)
    end

    Test.@testset "Gauss-Legendre exact DirectSum + fast GPU SHT (KA.CPU)" begin
        lmax = 8; Nθ = lmax + 1; Nφ = 2 * lmax + 1
        g = sph_gl(lmax)
        modes = ((0, 0, 0.8), (1, 0, 1.0), (2, 1, 0.6), (3, -2, 0.5), (4, 0, 0.3), (5, 3, -0.4), (6, -1, 0.7), (8, 4, 0.2))
        C = zeros(ComplexF64, Nθ, Nφ)
        for (l, m, v) in modes
            C[FFS.sph_mode_index(l, m)] = v
        end
        f = real(FFS.synthesize(g, C, (Nθ, Nφ); transform = SB.DirectSumSpectralBackend()))
        c_dir, _ = FFS.calculate_spectrum(g, f, (Nθ, Nφ); transform = SB.DirectSumSpectralBackend(), sampling = GL)
        for (l, m, v) in modes
            Test.@test isapprox(real(c_dir[FFS.sph_mode_index(l, m)]), v; atol = 1e-10)
        end
        Test.@test maximum(abs, imag(c_dir)) < 1e-10

        gpu = CB.GPUBackend(KA.CPU())
        c_gsht, _ = FFS.calculate_spectrum(g, f, (Nθ, Nφ); transform = SB.FSHTSpectralBackend(), execution = gpu, sampling = GL)
        c_gds, _ = FFS.calculate_spectrum(g, f, (Nθ, Nφ); transform = SB.DirectSumSpectralBackend(), execution = gpu, sampling = GL)
        Test.@test isapprox(c_gsht, c_dir; atol = 1e-10)
        Test.@test isapprox(c_gds, c_dir; atol = 1e-10)

        fb = cat(f, 2 .* f; dims = 3)                                # (nlon, nlat, 2) batch
        cb, _ = FFS.calculate_spectrum(g, fb, (Nθ, Nφ); transform = SB.FSHTSpectralBackend(), execution = gpu, sampling = GL)
        Test.@test size(cb) == (Nθ, Nφ, 2)
        Test.@test isapprox(cb[:, :, 1], c_dir; atol = 1e-10)
        Test.@test isapprox(cb[:, :, 2], 2 .* c_dir; atol = 1e-10)
    end

    Test.@testset "GPU NUFSHT parity (KA.CPU)" begin
        lmax = 5; Nθ = lmax + 1; Nφ = 2 * lmax + 1
        N_pts = 4 * Nθ^2
        ga = π * (3 - sqrt(5))
        zf = [1 - 2 * (i + 0.5) / N_pts for i in 0:(N_pts - 1)]
        θ = acos.(clamp.(zf, -1.0, 1.0)); φ = mod.(ga .* (0:(N_pts - 1)), 2π)
        Ct = zeros(Nθ, Nφ); Ct[FSH.sph_mode(2, 0)] = 1.0
        plan = NUFSHT.make_plan(Float64, θ, φ, lmax)
        f = zeros(N_pts); NUFSHT.nusht_type2!(f, Ct, plan)
        g = sph_scat(θ, φ); gpu = CB.GPUBackend(KA.CPU())
        cs, _ = FFS.calculate_spectrum(g, f, (Nθ, Nφ); transform = SB.NUFSHTSpectralBackend(), execution = CB.SerialBackend())
        cg, _ = FFS.calculate_spectrum(g, f, (Nθ, Nφ); transform = SB.NUFSHTSpectralBackend(), execution = gpu)
        Test.@test size(cg) == (Nθ, Nφ)
        Test.@test isapprox(cs, cg; atol = 1e-10)
        cs2, _ = FFS.calculate_spectrum(g, f, (Nθ, Nφ); transform = SB.NUFSHTSpectralBackend(), execution = CB.SerialBackend(), solve = true, rtol = 1e-10, maxiter = 2000)
        cg2, _ = FFS.calculate_spectrum(g, f, (Nθ, Nφ); transform = SB.NUFSHTSpectralBackend(), execution = gpu, solve = true, rtol = 1e-10, maxiter = 2000)
        Test.@test isapprox(cs2, cg2; atol = 1e-8)
        Test.@test isapprox(cg2[FFS.sph_mode_index(2, 0)], 1.0; atol = 0.05)
        fb = hcat(f, 2 .* f)
        cgb, _ = FFS.calculate_spectrum(g, fb, (Nθ, Nφ); transform = SB.NUFSHTSpectralBackend(), execution = gpu)
        Test.@test size(cgb) == (Nθ, Nφ, 2)
        Test.@test isapprox(cgb[:, :, 1], cg; atol = 1e-10)
        # Reusable device plan: build once, execute in-place — matches the one-shot GPU path.
        gplan = FFS.plan_spectrum(g, Float64, (Nθ, Nφ); transform = SB.NUFSHTSpectralBackend(), execution = gpu)
        cgp = zeros(ComplexF64, Nθ, Nφ); FFS.calculate_spectrum!(cgp, gplan, f)
        Test.@test isapprox(cgp, cg; atol = 1e-10)
        gplanb = FFS.plan_spectrum(g, Float64, (Nθ, Nφ); transform = SB.NUFSHTSpectralBackend(), execution = gpu, batch = (2,))
        cgpb = zeros(ComplexF64, Nθ, Nφ, 2); FFS.calculate_spectrum!(cgpb, gplanb, fb)
        Test.@test isapprox(cgpb, cgb; atol = 1e-10)
    end

    Test.@testset "Legendre Recurrence" begin
        FT = Float64; x = FT(0.5); s = sqrt(one(FT) - x^2)
        Test.@test isapprox(FFS.SphericalKernels.normalized_legendre(0, 0, x, s), one(FT) / sqrt(FT(4π)), atol = 1e-15)
        Test.@test isapprox(FFS.SphericalKernels.normalized_legendre(1, 0, x, s), sqrt(FT(3) / (FT(4) * FT(π))) * x, atol = 1e-15)
        Test.@test isapprox(FFS.SphericalKernels.normalized_legendre(1, 1, x, s), -sqrt(FT(3) / (FT(8) * FT(π))) * s, atol = 1e-15)
    end

    Test.@testset "Execution-axis parity (Serial / Threaded / Auto / GPU-CPU)" begin
        Random.seed!(314)
        L = 2π; N = 16; ms = (N, N)
        xs = ucg_axis(Float64, L, N); ys = ucg_axis(Float64, L, N)
        ug = ucg((L, L), ms)
        u = [cos(2x) + 0.5 * sin(3y) for x in xs, y in ys]
        xv = vec([x for x in xs, y in ys]); yv = vec([y for x in xs, y in ys])
        uv_scat = vec(u)
        sc = scg((xv, yv), (L, L))
        gpu = CB.GPUBackend(KA.CPU())

        ref, kref = FFS.calculate_spectrum(sc, uv_scat, ms; transform = SB.DirectSumSpectralBackend(), execution = CB.SerialBackend())
        for exec in (CB.ThreadedBackend(), CB.AutoBackend(), gpu)
            c, k = FFS.calculate_spectrum(sc, uv_scat, ms; transform = SB.DirectSumSpectralBackend(), execution = exec)
            Test.@test isapprox(c, ref; atol = 1e-12)
            Test.@test all(isapprox(collect(k[d]), collect(kref[d]); rtol = 1e-12) for d in 1:2)
        end

        cf_s, _ = FFS.calculate_spectrum(ug, u, ms; transform = SB.FFTSpectralBackend(), execution = CB.SerialBackend())
        cf_t, _ = FFS.calculate_spectrum(ug, u, ms; transform = SB.FFTSpectralBackend(), execution = CB.ThreadedBackend())
        cd, _ = FFS.calculate_spectrum(ug, u, ms; transform = SB.DirectSumSpectralBackend())
        cf_gpu, _ = FFS.calculate_spectrum(ug, u, ms; transform = SB.FFTSpectralBackend(), execution = gpu)
        Test.@test isapprox(cf_s, cf_t; atol = 1e-12)
        Test.@test isapprox(cf_s, cd; atol = 1e-12)
        Test.@test isapprox(cf_s, cf_gpu; atol = 1e-12)

        cn_s, _ = FFS.calculate_spectrum(sc, uv_scat, ms; transform = FFS.FINUFFTBackend(), execution = CB.SerialBackend(), eps = 1e-12)
        cn_t, _ = FFS.calculate_spectrum(sc, uv_scat, ms; transform = FFS.FINUFFTBackend(), execution = CB.ThreadedBackend(), eps = 1e-12)
        Test.@test isapprox(cn_s, cn_t; atol = 1e-10)

        c_ip_s = zeros(ComplexF64, pks(ms)...); c_ip_t = zeros(ComplexF64, pks(ms)...)
        FFS.calculate_spectrum!(c_ip_s, sc, uv_scat, ms; execution = CB.SerialBackend())
        FFS.calculate_spectrum!(c_ip_t, sc, uv_scat, ms; execution = CB.ThreadedBackend())
        Test.@test isapprox(c_ip_s, c_ip_t; atol = 1e-12)
        Test.@test isapprox(c_ip_s, ref; atol = 1e-12)

        Random.seed!(20)
        θ = rand(60) .* π; φ = rand(60) .* 2π; fθ = rand(60)
        sph = sph_scat(θ, φ)
        cs_s, _ = FFS.calculate_spectrum(sph, fθ, (6, 11); transform = SB.DirectSumSpectralBackend(), execution = CB.SerialBackend())
        for exec in (CB.ThreadedBackend(), gpu)
            cs, _ = FFS.calculate_spectrum(sph, fθ, (6, 11); transform = SB.DirectSumSpectralBackend(), execution = exec)
            Test.@test isapprox(cs_s, cs; atol = 1e-10)
        end

        lmaxs = 4; gsc = sph_cc(lmaxs)
        Cc = zeros(ComplexF64, lmaxs + 1, 2lmaxs + 1); Cc[FFS.sph_mode_index(2, 1)] = 1.0
        fv = real(FFS.synthesize(gsc, Cc, (lmaxs + 1, 2lmaxs + 1); transform = SB.DirectSumSpectralBackend()))
        csht_s, _ = FFS.calculate_spectrum(gsc, fv, (lmaxs + 1, 2lmaxs + 1); transform = SB.FSHTSpectralBackend(), execution = CB.SerialBackend())
        csht_t, _ = FFS.calculate_spectrum(gsc, fv, (lmaxs + 1, 2lmaxs + 1); transform = SB.FSHTSpectralBackend(), execution = CB.ThreadedBackend())
        Test.@test csht_s == csht_t

        coeffs, kcoef = FFS.calculate_spectrum(ug, u, ms; transform = SB.DirectSumSpectralBackend())
        # This compares the SERIAL and THREADED direct-sum inverses, so both name that transform.
        rs = FFS.synthesize(ug, coeffs, ms; transform = SB.DirectSumSpectralBackend(),
            execution = CB.SerialBackend(), ks = kcoef)
        rt = FFS.synthesize(ug, coeffs, ms; transform = SB.DirectSumSpectralBackend(),
            execution = CB.ThreadedBackend(), ks = kcoef)
        Test.@test isapprox(rs, rt; atol = 1e-12)
        Test.@test isapprox(rs, u; atol = 1e-10)

        Test.@test_throws ArgumentError FFS.calculate_spectrum(sc, uv_scat, ms; transform = FFS.FINUFFTBackend(), execution = gpu)
    end

    Test.@testset "Derived-quantity spectra (vorticity / divergence / compensated)" begin
        L = 2π; N = 16
        xs = ucg_axis(Float64, L, N); ys = ucg_axis(Float64, L, N)
        g = ucg((L, L), (N, N))
        u = [sin(x) * cos(y) for x in xs, y in ys]
        v = [-cos(x) * sin(y) for x in xs, y in ys]
        c, ks = FFS.calculate_spectrum(g, (u, v), (N, N); transform = SB.FFTSpectralBackend())
        divc = FFS.spectral_divergence(ks, c)
        Test.@test maximum(abs.(divc)) < 1e-12
        omega = [2 * sin(x) * sin(y) for x in xs, y in ys]
        co, _ = FFS.calculate_spectrum(g, omega, (N, N); transform = SB.FFTSpectralBackend())
        vortc = FFS.spectral_vorticity(ks, c)
        Test.@test isapprox(vortc[:, :, 1], co; atol = 1e-12)
        # in-place variants: bit-identical to the allocating form, and a mis-shaped `out` is rejected
        divc2 = similar(divc); FFS.spectral_divergence!(divc2, ks, c)
        Test.@test divc2 == divc
        vortc2 = similar(vortc); FFS.spectral_vorticity!(vortc2, ks, c)
        Test.@test vortc2 == vortc
        Test.@test_throws DimensionMismatch FFS.spectral_divergence!(similar(c), ks, c)
        Test.@test_throws DimensionMismatch FFS.spectral_vorticity!(similar(c), ks, c)
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
        L = 2π; N = 16
        xs = ucg_axis(Float64, L, N); ys = ucg_axis(Float64, L, N)
        g = ucg((L, L), (N, N))
        f = [cos(2x) + 0.5 * sin(3y) for x in xs, y in ys]
        h = [cos(2x) - 0.3 * cos(y) for x in xs, y in ys]
        cf, ks = FFS.calculate_spectrum(g, f, (N, N); transform = SB.FFTSpectralBackend())
        cg, _ = FFS.calculate_spectrum(g, h, (N, N); transform = SB.FFTSpectralBackend())
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
        L = 2π; N = 16
        xs = ucg_axis(Float64, L, N); ys = ucg_axis(Float64, L, N)
        g = ucg((L, L), (N, N))
        f = [cos(3x) + 0.7 * sin(2y) for x in xs, y in ys]
        c, ks = FFS.calculate_spectrum(g, f, (N, N); transform = SB.FFTSpectralBackend())
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
            N = D == 3 ? 8 : 16; L = 2π
            g = ucg(ntuple(_ -> L, D), ntuple(_ -> N, D))
            axs = ntuple(d -> ucg_axis(Float64, L, N), D)
            f = collect([sum(cos((d + 1) * pt[d]) + 0.3 * sin(d * pt[d]) for d in 1:D) for pt in Iterators.product(axs...)])
            f = f .- Statistics.mean(f)
            ms = ntuple(_ -> N, D)
            c_fft, k_fft = FFS.calculate_spectrum(g, f, ms; transform = SB.FFTSpectralBackend())
            c_dir, _ = FFS.calculate_spectrum(g, f, ms; transform = SB.DirectSumSpectralBackend())
            Test.@test isapprox(c_fft, c_dir; atol = 1e-10)
            Test.@test isapprox(parseval(k_fft, c_fft), Statistics.mean(abs2, f); rtol = 1e-10)
        end
    end

    Test.@testset "Real-input packed half unpacks to the full FFT (D=1/2/3)" begin
        Random.seed!(13)
        for D in 1:3
            N = D == 3 ? 6 : 12; L = 2π
            g = ucg(ntuple(_ -> L, D), ntuple(_ -> N, D))
            axs = ntuple(d -> ucg_axis(Float64, L, N), D)
            f = collect([sum(cos((d + 1) * pt[d]) for d in 1:D) for pt in Iterators.product(axs...)])
            ms = ntuple(_ -> N, D)
            c_real, _ = FFS.calculate_spectrum(g, f, ms; transform = SB.FFTSpectralBackend())
            c_cplx, _ = FFS.calculate_spectrum(g, ComplexF64.(f), ms; transform = SB.FFTSpectralBackend())
            Test.@test size(c_real) == pks(ms)
            Test.@test size(c_cplx) == ms
            Test.@test isapprox(FFS.unpacked(c_real, ms), c_cplx; atol = 1e-12)
        end
    end

    Test.@testset "Float32 end-to-end (DirectSum / FFT / NUFFT)" begin
        Random.seed!(7)
        L = 2.0f0 * Float32(π); N = 16
        xs = ucg_axis(Float32, L, N); ys = ucg_axis(Float32, L, N)
        g = ucg(Float32, (L, L), (N, N))
        f = Float32[cos(2x) + 0.5f0 * sin(3y) for x in xs, y in ys]
        c_dir, _ = FFS.calculate_spectrum(g, f, (N, N); transform = SB.DirectSumSpectralBackend())
        c_fft, ks = FFS.calculate_spectrum(g, f, (N, N); transform = SB.FFTSpectralBackend())
        Test.@test eltype(c_dir) === ComplexF32
        Test.@test eltype(c_fft) === ComplexF32
        Test.@test isapprox(c_fft, c_dir; atol = 1.0f-4)
        _, E_iso = FFS.isotropic_spectrum(ks, c_fft; num_bins = 6)
        Test.@test eltype(E_iso) === Float32

        xs32 = rand(Float32, 80) .* L; ys32 = rand(Float32, 80) .* L
        fs = @. cos(xs32) + sin(2ys32)
        sg = scg(Float32, (xs32, ys32), (L, L))
        c32, _ = FFS.calculate_spectrum(sg, fs, (N, N); transform = FFS.FINUFFTBackend())
        Test.@test eltype(c32) === ComplexF32
    end

    Test.@testset "Plan reuse parity + batch (FFTW / FINUFFT)" begin
        Random.seed!(99)
        L = 2π; N = 16
        xs = ucg_axis(Float64, L, N); ys = ucg_axis(Float64, L, N)
        g = ucg((L, L), (N, N))
        u = [cos(2x) + sin(3y) for x in xs, y in ys]
        v = [sin(x) for x in xs, y in ys]
        c1, _ = FFS.calculate_spectrum(g, (u, v), (N, N); transform = SB.FFTSpectralBackend())
        plan = FFS.plan_spectrum(g, Float64, (N, N); transform = SB.FFTSpectralBackend(), batch = (2,))
        cc = zeros(ComplexF64, pks((N, N))..., 2)
        FFS.calculate_spectrum!(cc, plan, cat(u, v; dims = 3))
        Test.@test cc ≈ c1

        xv = rand(64) .* L; yv = rand(64) .* L
        sg = scg((xv, yv), (L, L))
        nb = 5
        fstack = zeros(64, nb)
        for b in 1:nb
            fstack[:, b] .= cos.(b .* xv) .+ sin.(b .* yv)
        end
        bplan = FFS.plan_spectrum(sg, Float64, (N, N); transform = FFS.FINUFFTBackend(), batch = (nb,), eps = 1e-10)
        C = zeros(ComplexF64, pks((N, N))..., nb)
        FFS.calculate_spectrum!(C, bplan, fstack)
        for b in (1, nb)
            cb, _ = FFS.calculate_spectrum(sg, fstack[:, b], (N, N); transform = FFS.FINUFFTBackend(), eps = 1e-10)
            Test.@test C[:, :, b] ≈ cb
        end
    end

    Test.@testset "Grid dispatch errors (no silent misroute)" begin
        sg = sph_scat([0.1, 0.2, 0.3], [0.1, 0.2, 0.3])
        cg = scg(([0.0, 1, 2], [0.0, 1, 2]), (3.0, 3.0))
        Test.@test_throws ArgumentError FFS.calculate_spectrum(sg, [1.0, 2, 3], (2, 3); transform = SB.FFTSpectralBackend())
        Test.@test_throws ArgumentError FFS.calculate_spectrum!(zeros(ComplexF64, 4, 4), cg, [1.0, 2, 3], (4, 4); transform = SB.FFTSpectralBackend())
    end

    Test.@testset "Welch averaging + coherence / phase" begin
        Random.seed!(2024)
        L = 2π; N = 64
        x = ucg_axis(Float64, L, N)
        g = ucg((L,), (N,))
        nens = 60; kc = 5
        nh = pks((N,))[1]                                        # real 1-D field ⇒ halved axis
        cf = zeros(ComplexF64, nh, nens); cg = zeros(ComplexF64, nh, nens)
        for e in 1:nens
            ϕ = 2π * rand()
            shared = cos.(kc .* x .+ ϕ)
            fe = shared .+ 0.3 .* randn(N); ge = 0.8 .* shared .+ 0.3 .* randn(N)
            cf[:, e] .= FFS.calculate_spectrum(g, fe, (N,); transform = SB.FFTSpectralBackend())[1]
            cg[:, e] .= FFS.calculate_spectrum(g, ge, (N,); transform = SB.FFTSpectralBackend())[1]
        end
        ks = (FFS.calculate_spectrum(g, collect(x), (N,); transform = SB.FFTSpectralBackend())[2][1],)
        kb, γ², φ = FFS.coherence_spectrum(ks, cf, cg; num_bins = 16)
        Test.@test all(0 .<= γ² .<= 1)
        Test.@test γ²[argmin(abs.(kb .- kc))] > 0.7
        Test.@test length(φ) == length(kb)
        kbw, Ew = FFS.welch_power_spectrum(ks, cf; num_bins = 16)
        Test.@test all(Ew .>= 0)
        Test.@test argmin(abs.(kbw .- kc)) == argmax(Ew)
        # in-place variants match the allocating form
        γ²2 = similar(γ²); φ2 = similar(φ); kb2 = similar(kb)
        FFS.coherence_spectrum!(γ²2, φ2, kb2, ks, cf, cg; num_bins = 16)
        Test.@test γ²2 == γ² && φ2 == φ && kb2 == kb
        Ew2 = similar(Ew); kbw2 = similar(kbw)
        FFS.welch_power_spectrum!(Ew2, kbw2, ks, cf; num_bins = 16)
        Test.@test Ew2 == Ew && kbw2 == kbw
    end

    Test.@testset "Multitaper (DPSS)" begin
        N = 128; K = 7
        V = FFS.dpss(N, 4.0, K)
        Test.@test size(V) == (N, K)
        Test.@test maximum(abs.(V' * V - Matrix(LA.I(K)))) < 1e-8
        Test.@test_throws ArgumentError FFS.dpss(N, 4.0, N + 1)
        L = 2π
        x = ucg_axis(Float64, L, N)
        g = ucg((L,), (N,))
        k0 = 7; sig = cos.(k0 .* x)
        C = zeros(ComplexF64, pks((N,))[1], K)                   # real 1-D field ⇒ halved axis
        for k in 1:K
            C[:, k] .= FFS.calculate_spectrum(g, V[:, k] .* sig, (N,); transform = SB.FFTSpectralBackend())[1]
        end
        ks = (FFS.calculate_spectrum(g, collect(x), (N,); transform = SB.FFTSpectralBackend())[2][1],)
        kb, E = FFS.welch_power_spectrum(ks, C; num_bins = 16)
        Test.@test argmin(abs.(kb .- k0)) == argmax(E)
    end

    Test.@testset "Lomb–Scargle (irregular sampling)" begin
        Random.seed!(7)
        N = 200; t = sort(rand(N) .* 10.0); f0 = 1.3
        y = sin.(2π * f0 .* t) .+ 0.2 .* randn(N)
        freqs = collect(range(0.1, stop = 4.0, length = 256))
        P = FFS.lomb_scargle(t, y, freqs)
        Test.@test length(P) == length(freqs)
        Test.@test all(P .>= 0)
        Test.@test abs(freqs[argmax(P)] - f0) < 0.1
        Test.@test_throws ArgumentError FFS.lomb_scargle(t, y, [0.0, 1.0])
        P2 = similar(P); FFS.lomb_scargle!(P2, t, y, freqs)
        Test.@test P2 == P
        Test.@test_throws DimensionMismatch FFS.lomb_scargle!(similar(P, length(freqs) + 1), t, y, freqs)
    end

    Test.@testset "Synthesis / inverse transform (round-trip)" begin
        L = 2π; N = 16
        xs = ucg_axis(Float64, L, N); ys = ucg_axis(Float64, L, N)
        g = ucg((L, L), (N, N))
        u = [cos(2x) + 0.5 * sin(3y) - 0.3 * cos(x + 2y) for x in xs, y in ys]
        # The packed half inverts directly to a real field; the full native spectrum takes
        # `real_output=false` and returns complex.
        # This testset gates the DIRECT-SUM inverse, so every call here names it.
        ds = SB.DirectSumSpectralBackend()
        coeffs, kk = FFS.calculate_spectrum(g, u, (N, N); transform = ds)
        urec = FFS.synthesize(g, coeffs, (N, N); transform = ds, ks = kk)
        Test.@test eltype(urec) === Float64
        Test.@test isapprox(urec, u; atol = 1e-10)
        ufull = FFS.synthesize(g, FFS.unpacked(coeffs, (N, N)), (N, N); transform = ds, real_output = false)
        Test.@test isapprox(real.(ufull), u; atol = 1e-10)
        Test.@test_throws DimensionMismatch FFS.synthesize(g, FFS.unpacked(coeffs, (N, N)), (N, N); transform = ds)

        lmax = 6; Nθ, Nφ = lmax + 1, 2lmax + 1
        N_pts = 4 * Nθ * Nφ
        ga = π * (3 - sqrt(5))
        z = [1 - 2 * (i + 0.5) / N_pts for i in 0:(N_pts - 1)]
        θs = acos.(clamp.(z, -1.0, 1.0)); φs = mod.(ga .* (0:(N_pts - 1)), 2π)
        sg = sph_scat(θs, φs)
        C = zeros(ComplexF64, Nθ, Nφ); C[FFS.sph_mode_index(3, 1)] = 1.0
        fs = FFS.synthesize(sg, C, (Nθ, Nφ); transform = ds)
        Test.@test length(fs) == N_pts
        Test.@test all(isfinite, fs)
    end

    Test.@testset "NonuniformFFTs NUFFT provider (vs DirectSum)" begin
        # Distinct backend from FINUFFT — both are loaded here, no collision. Fail loud if the ext
        # didn't load (rather than silently skipping).
        Test.@test Base.get_extension(FFS, :FlowFieldSpectraNonuniformFFTsExt) !== nothing
        Random.seed!(11); L = 2π; M = 1500
        nu(g, f, ms; kw...) = FFS.calculate_spectrum(g, f, ms; transform = FFS.NonuniformFFTsBackend(), execution = CB.SerialBackend(), eps = 1.0e-10, kw...)[1]
        ds(g, f, ms; kw...) = FFS.calculate_spectrum(g, f, ms; transform = SB.DirectSumSpectralBackend(), execution = CB.SerialBackend(), kw...)[1]
        # Real scattered fields, EVEN and ODD ms (the Nyquist edge), D = 1, 2, 3.
        for (ms, _name) in [((12,), "1D even"), ((11,), "1D odd"), ((12, 10), "2D even"),
                ((11, 9), "2D odd"), ((8, 6, 10), "3D even"), ((7, 5, 9), "3D odd")]
            D = length(ms)
            coords = ntuple(_ -> rand(M) .* L, D)
            f = [sum(cos(2 * c[i]) for c in coords) for i in 1:M]
            g = scg(coords, ntuple(_ -> L, D))
            Test.@test isapprox(nu(g, f, ms), ds(g, f, ms); atol = 1.0e-7)
        end
        coords = (rand(M) .* L, rand(M) .* L); g = scg(coords, (L, L))
        fc = @. cis(2 * coords[1]) + 0.4 * cis(3 * coords[2])                    # complex field
        Test.@test isapprox(nu(g, fc, (12, 12)), ds(g, fc, (12, 12)); atol = 1.0e-7)
        fb = hcat((cos.((1 + b) .* coords[1]) for b in 1:4)...)                  # batch (12,12,4)
        Test.@test isapprox(nu(g, fb, (12, 12)), ds(g, fb, (12, 12)); atol = 1.0e-7)
        fr = cos.(2 .* coords[1])                                                # iflag = -1
        Test.@test isapprox(nu(g, fr, (12, 12); iflag = -1), ds(g, fr, (12, 12); iflag = -1); atol = 1.0e-7)
        c32 = (rand(Float32, M) .* Float32(L), rand(Float32, M) .* Float32(L))   # Float32 end-to-end
        f32 = Float32.(cos.(2 .* c32[1])); g32 = scg(Float32, c32, (L, L))
        cn32 = FFS.calculate_spectrum(g32, f32, (10, 10); transform = FFS.NonuniformFFTsBackend(), execution = CB.SerialBackend(), eps = 1.0f-5)[1]
        Test.@test eltype(cn32) == ComplexF32
        cd32 = FFS.calculate_spectrum(g32, f32, (10, 10); transform = SB.DirectSumSpectralBackend(), execution = CB.SerialBackend())[1]
        Test.@test isapprox(cn32, cd32; atol = 1.0f-3)
        # NUFFTSpectralBackend selects no provider — it errors, directing to a concrete one.
        Test.@test_throws ArgumentError FFS.calculate_spectrum(g, coords[1], (12, 12); transform = SB.NUFFTSpectralBackend())

        # Device-generic path (GPUBackend): the KA ext threads the backend into PlanNUFFT and
        # reconstructs with broadcasts / a KA kernel. Exercised on KA.CPU (device path on host arrays);
        # it must equal the CPU path bit-for-bit-close (same plan + math, scalar loop vs kernel).
        Test.@test Base.get_extension(FFS, :FlowFieldSpectraNonuniformFFTsKernelAbstractionsExt) !== nothing
        gpu = CB.GPUBackend(KA.CPU())
        nug(gg, ff, mss; kw...) = FFS.calculate_spectrum(gg, ff, mss; transform = FFS.NonuniformFFTsBackend(), execution = gpu, eps = 1.0e-10, kw...)[1]
        for ms in [(12, 10), (11, 9)]                                            # 2D even (Nyquist kernel) + odd (mirror)
            cd = (rand(M) .* L, rand(M) .* L); gd = scg(cd, (L, L))
            fd = @. cos(2 * cd[1]) + 0.5 * sin(3 * cd[2])
            Test.@test isapprox(nug(gd, fd, ms), nu(gd, fd, ms); atol = 1.0e-10)
        end
        cdev = (rand(M) .* L, rand(M) .* L); gdev = scg(cdev, (L, L))
        fcd = @. cis(2 * cdev[1]) + 0.4 * cis(3 * cdev[2])                        # complex
        Test.@test isapprox(nug(gdev, fcd, (12, 12)), nu(gdev, fcd, (12, 12)); atol = 1.0e-10)
        fbd = hcat((cos.((1 + b) .* cdev[1]) for b in 1:3)...)                    # batch
        Test.@test isapprox(nug(gdev, fbd, (12, 12)), nu(gdev, fbd, (12, 12)); atol = 1.0e-10)
        frd = cos.(2 .* cdev[1])
        Test.@test isapprox(nug(gdev, frd, (12, 12); iflag = -1), nu(gdev, frd, (12, 12); iflag = -1); atol = 1.0e-10)
        plang = FFS.plan_spectrum(gdev, Float64, (12, 12); transform = FFS.NonuniformFFTsBackend(), execution = gpu, eps = 1.0e-10)  # reusable device plan
        cpg = zeros(ComplexF64, pks((12, 12))...); FFS.calculate_spectrum!(cpg, plang, frd)
        Test.@test isapprox(cpg, nug(gdev, frd, (12, 12)); atol = 1.0e-10)
    end

    include("test_spherical_layouts.jl")
    include("test_spherical_parity.jl")
    include("test_curvilinear.jl")
    include("test_spheroid.jl")
    include("test_auto_transform.jl")
    include("test_dispatch_matrix.jl")
    include("test_preprocessing.jl")
    include("test_hybrid_plan.jl")
    include("test_real_spherical.jl")
    include("test_reference_parity.jl")
    include("test_directsum_plan.jl")
    include("test_device_genericity.jl")
    include("test_allocs.jl")
    include("test_gpu.jl")
    include("test_distributed.jl")
    include("test_mpi.jl")
end
