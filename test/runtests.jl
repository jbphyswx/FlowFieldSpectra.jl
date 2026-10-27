using Test: Test
using Random: Random
using Statistics: Statistics
using LinearAlgebra: LinearAlgebra as LA
using Aqua: Aqua as Aqua
using ExplicitImports: ExplicitImports as EI

using FlowFieldSpectra: FlowFieldSpectra as FFS
using FlowTransformBindings: FlowTransformBindings as FTB
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
# Test-local grid constructors. FFS grids are FlowGeometries grids: a Cartesian grid over a
# CartesianGeometry, a spherical grid over a SphericalGeometry. Spectral transforms need a periodic
# domain, so these mark the grid periodic with an explicit wrap length. Spherical fields are
# `(nlon, nlat)`; `(θ, φ)` = (colatitude, longitude) map to FlowGeometries `(λ, φ_lat)` = (longitude,
# geographic latitude) as `λ = φ`, `φ_lat = π/2 − θ`.
# =============================================================================
_cg(::Type{T}) where {T} = FG.Geometry.CartesianGeometry{T}()

# Two-sided Parseval sum over a packed spectrum: each stored mode counts with its fold weight, so this
# equals `mean(abs2, field)` under the 1/∏N normalization on any layout.
parseval(ks::Tuple, c::AbstractArray) =
    sum(I -> FTB.mode_fold(ks, I) * abs2(c[I]), CartesianIndices(c))

# Every direction periodic.
_periodic(n) = ntuple(_ -> FG.Grids.Periodic(), length(n))

# Uniform Cartesian grid from per-axis point counts + periodic domain lengths.
function ucg(::Type{T}, domain::Tuple, n::Tuple) where {T}
    axes = map((L, m) -> range(zero(T), T(L); length = m + 1)[1:m], domain, n)
    return FG.Grids.StructuredGrid(_cg(T), axes...; topology = _periodic(n), period = T.(domain))
end
ucg(domain::Tuple, n::Tuple) = ucg(Float64, domain, n)
ucg_axis(::Type{T}, domain, n) where {T} = range(zero(T), T(domain); length = n + 1)[1:n]  # matching axis values

# Nonuniform-gridded Cartesian (stretched axes + periodic domain).
nucg(::Type{T}, axes::Tuple, domain::Tuple) where {T} =
    FG.Grids.StructuredGrid(_cg(T), map(a -> collect(T, a), axes)...; topology = _periodic(axes),
        period = T.(domain))
nucg(axes::Tuple, domain::Tuple) = nucg(Float64, axes, domain)

# Scattered Cartesian point cloud (unit per-node measure; periodic domain).
scg(::Type{T}, coords::Tuple, domain::Tuple) where {T} =
    FG.Grids.UnstructuredGrid(_cg(T), map(c -> collect(T, c), coords), ones(T, length(coords[1]));
        topology = _periodic(coords), period = T.(domain))
scg(coords::Tuple, domain::Tuple) = scg(Float64, coords, domain)

# Structured spherical grids: Fejér's first rule is FastSphericalHarmonics' grid (SHT); Gauss–Legendre
# gives exact weighted-quadrature (DirectSum / GPU SHT with sampling=GaussLegendreSampling()).
sph_cc(lmax) = FG.Connectivity.structured_grid(Float64, FG.SphericalSampling.Fejer1Sampling(), lmax + 1)
sph_gl(lmax) = FG.Connectivity.structured_grid(Float64, FG.SphericalSampling.GaussLegendreSampling(), lmax + 1)
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
            :FlowFieldSpectraAbstractFFTsExt, :FlowFieldSpectraFFTWExt, :FlowFieldSpectraFastSphericalHarmonicsExt,
            :FlowFieldSpectraNUFSHTExt, :FlowFieldSpectraOhMyThreadsExt, :FlowFieldSpectraKernelAbstractionsExt,
            :FlowFieldSpectraKernelAbstractionsAbstractFFTsExt, :FlowFieldSpectraNUFSHTKernelAbstractionsExt,
            :FlowFieldSpectraGPUArraysCoreExt,
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
        xs = ucg_axis(Float64, L, 16); ys = ucg_axis(Float64, L, 16)
        g = ucg((L, L), ms)
        kx1, ky1 = 2π * 2 / L, 2π * 1 / L
        kx2, ky2 = 2π * (-3) / L, 2π * 2 / L
        u = [cos(kx1 * x + ky1 * y) + 0.5 * sin(kx2 * x + ky2 * y) for x in xs, y in ys]
        v = [sin(kx1 * x + ky1 * y) for x in xs, y in ys]

        c_direct, k_direct = FFS.calculate_spectrum(g, (u, v), ms; transform = SB.DirectSumSpectralBackend())
        c_fft, k_fft = FFS.calculate_spectrum(g, (u, v), ms; transform = SB.FFTSpectralBackend())
        Test.@test isapprox(c_direct, c_fft, atol = 1e-12)
        Test.@test all(isapprox(k_direct[d], k_fft[d], rtol = 1e-12) for d in 1:2)
        Test.@test isapprox(parseval(k_fft, c_fft), Statistics.mean(abs2, u) + Statistics.mean(abs2, v); rtol = 1e-12)

        # A transect recovers the folded Parseval total whichever axis it keeps: a kept full axis carries
        # both signs of its wavenumber, and a kept halved axis reports |k₁| with the −k₁ energy folded in.
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

    Test.@testset "Both NUFFT providers equal the direct sum" begin
        Random.seed!(11)
        L = 2π
        M = 200
        gpu = CB.GPUBackend(KA.CPU())
        ds(g, f, ms; kw...) = FFS.calculate_spectrum(g, f, ms; transform = SB.DirectSumSpectralBackend(),
            execution = CB.SerialBackend(), kw...)[1]
        cloud = scg((rand(M) .* L, rand(M) .* L), (L, L))
        str = [L * (i - 1) / 8 + 0.04 * L * sinpi(2 * (i - 1) / 8) for i in 1:8]
        tensor = nucg((str, str), (L, L))
        for nu in (FTB.FINUFFTBackend(), FTB.NonuniformFFTsBackend())
            nuf(g, f, ms, exec; kw...) = FFS.calculate_spectrum(g, f, ms; transform = nu, execution = exec,
                eps = 1.0e-12, kw...)[1]
            for ms in ((12,), (11,), (12, 10), (11, 9), (8, 6, 10), (7, 5, 9))
                D = length(ms)
                coords = ntuple(_ -> rand(M) .* L, D)
                f = [sum(cos(2 * c[i]) for c in coords) for i in 1:M]
                g = scg(coords, ntuple(_ -> L, D))
                Test.@test isapprox(nuf(g, f, ms, CB.SerialBackend()), ds(g, f, ms); atol = 1.0e-10)
            end
            coords = (rand(M) .* L, rand(M) .* L)
            g = scg(coords, (L, L))
            for (f, kw) in ((@.(cis(2 * coords[1]) + 0.4 * cis(3 * coords[2])), (;)),
                            (hcat((cos.((1 + b) .* coords[1]) for b in 1:3)...), (;)),
                            (cos.(2 .* coords[1]), (; iflag = -1)))
                ref = ds(g, f, (12, 10); kw...)
                for exec in (CB.SerialBackend(), gpu)
                    Test.@test isapprox(nuf(g, f, (12, 10), exec; kw...), ref; atol = 1.0e-10)
                end
            end
            c32 = (rand(Float32, M) .* Float32(L), rand(Float32, M) .* Float32(L))
            g32 = scg(Float32, c32, (L, L))
            f32 = Float32.(cos.(2 .* c32[1]))
            cn32 = FFS.calculate_spectrum(g32, f32, (10, 10); transform = nu, execution = CB.SerialBackend(),
                eps = 1.0f-5)[1]
            Test.@test eltype(cn32) === ComplexF32
            Test.@test isapprox(cn32, ds(g32, f32, (10, 10)); atol = 1.0f-4)

            # Even axes on a point set and a stretched grid, so the coefficient box holds both `±4` on each.
            for (g, f) in ((cloud, randn(M, 3)), (tensor, randn(8, 8, 3))), exec in (CB.SerialBackend(), gpu)
                cd, ksd = FFS.calculate_spectrum(g, f, (8, 8); transform = SB.DirectSumSpectralBackend(),
                    execution = CB.SerialBackend())
                p = FFS.plan_spectrum(g, Float64, (8, 8); transform = nu, execution = exec, batch = (3,),
                    eps = 1.0e-13, batch_chunk = 2)
                c = FFS.allocate_coefficients(p)
                ks = FFS.calculate_spectrum!(c, p, f)
                Test.@test size(c) == size(cd) == (5, 9, 3)
                Test.@test isapprox(c, cd; atol = 1.0e-10)
                Test.@test all(d -> collect(ks[d]) ≈ collect(ksd[d]), 1:2)
                FFS.close!(p)
            end
        end
    end

    Test.@testset "Nonuniform-gridded Parity (separable NUFFT vs Direct)" begin
        Random.seed!(7)
        L = (2π, 2π)
        xax = sort(rand(13)) .* L[1]; yax = sort(rand(11)) .* L[2]
        g = nucg((xax, yax), L)
        ms = (8, 7)
        f = [cos(2x) + 0.5sin(3y) + 0.3cos(x + 2y) for x in xax, y in yax]
        fb = cat(cat(f, 2f, -0.5f; dims = 3), 0.7 .* cat(f, 2f, -0.5f; dims = 3); dims = 4)  # (13,11,3,2)
        cb_d, _ = FFS.calculate_spectrum(g, fb, ms; transform = SB.DirectSumSpectralBackend())
        cb_n, _ = FFS.calculate_spectrum(g, fb, ms; transform = FTB.FINUFFTBackend(), eps = 1e-13)
        Test.@test isapprox(cb_d, cb_n, atol = 1e-10)

        zax = sort(rand(6)) .* 2π
        g3 = nucg((xax, yax, zax), (2π, 2π, 2π))
        ms3 = (5, 4, 3)
        f3 = [cos(x) + sin(2y) + 0.4cos(z) for x in xax, y in yax, z in zax]
        c3_d, _ = FFS.calculate_spectrum(g3, f3, ms3; transform = SB.DirectSumSpectralBackend())
        c3_n, _ = FFS.calculate_spectrum(g3, f3, ms3; transform = FTB.FINUFFTBackend(), eps = 1e-13)
        Test.@test isapprox(c3_d, c3_n, atol = 1e-10)
    end

    Test.@testset "Spherical Structured SHT round-trip (FastSphericalHarmonics)" begin
        # FastSphericalHarmonics is an exact analysis on the Clenshaw–Curtis grid, inverse to its
        # synthesis. DirectSum synthesis uses the same real-SH convention, so SHT ∘ DirectSum-synth = id,
        # and the degree spectrum is `E(ℓ) = ½ Σ_m |C_ℓm|²`.
        lmax = 8; Nθ = lmax + 1; Nφ = 2 * lmax + 1
        g = sph_cc(lmax)
        C = zeros(Nθ, Nφ)
        for (l, m, v) in ((2, 1, 1.0), (3, -2, 0.5), (0, 0, 0.7), (5, 4, -0.3))
            C[FFS.sph_mode_index(l, m)] = v
        end
        f = FFS.synthesize(g, C, (Nθ, Nφ); transform = SB.DirectSumSpectralBackend())
        c_sht, _ = FFS.calculate_spectrum(g, f, (Nθ, Nφ); transform = SB.FSHTSpectralBackend())
        Test.@test maximum(abs, c_sht .- C) < 1e-12
        _, E_l = FFS.spherical_energy_spectrum(c_sht)
        Test.@test maximum(abs, E_l .- [0.5 * sum(abs2(C[FFS.sph_mode_index(l, m)]) for m in -l:l)
                                        for l in 0:lmax]) < 1e-12
    end

    Test.@testset "Gauss-Legendre exact DirectSum + fast GPU SHT (KA.CPU)" begin
        lmax = 8; Nθ = lmax + 1; Nφ = 2 * lmax + 1
        g = sph_gl(lmax)
        C = zeros(Nθ, Nφ)
        for (l, m, v) in ((0, 0, 0.8), (1, 0, 1.0), (2, 1, 0.6), (3, -2, 0.5), (4, 0, 0.3), (5, 3, -0.4),
                          (6, -1, 0.7), (8, 4, 0.2))
            C[FFS.sph_mode_index(l, m)] = v
        end
        f = FFS.synthesize(g, C, (Nθ, Nφ); transform = SB.DirectSumSpectralBackend())
        gpu = CB.GPUBackend(KA.CPU())
        for (tr, ex) in ((SB.DirectSumSpectralBackend(), CB.SerialBackend()), (SB.FSHTSpectralBackend(), gpu))
            c, _ = FFS.calculate_spectrum(g, f, (Nθ, Nφ); transform = tr, execution = ex, sampling = GL)
            Test.@test maximum(abs, c .- C) < 1e-12
            cb, _ = FFS.calculate_spectrum(g, cat(f, 2 .* f; dims = 3), (Nθ, Nφ); transform = tr,
                execution = ex, sampling = GL)
            Test.@test maximum(abs, cb .- cat(C, 2 .* C; dims = 3)) < 1e-12
        end
    end

    Test.@testset "Legendre Recurrence" begin
        FT = Float64; x = FT(0.5); s = sqrt(one(FT) - x^2)
        Test.@test isapprox(FFS.SphericalKernels.normalized_legendre(0, 0, x, s), one(FT) / sqrt(FT(4π)), atol = 1e-15)
        Test.@test isapprox(FFS.SphericalKernels.normalized_legendre(1, 0, x, s), sqrt(FT(3) / (FT(4) * FT(π))) * x, atol = 1e-15)
        Test.@test isapprox(FFS.SphericalKernels.normalized_legendre(1, 1, x, s), -sqrt(FT(3) / (FT(8) * FT(π))) * s, atol = 1e-15)
    end

    Test.@testset "Execution-axis parity (Serial / Threaded / Auto / GPU-CPU)" begin
        L = 2π; N = 16; ms = (N, N)
        xs = ucg_axis(Float64, L, N); ys = ucg_axis(Float64, L, N)
        ug = ucg((L, L), ms)
        u = [cos(2x) + 0.5 * sin(3y) for x in xs, y in ys]
        sc = scg((vec([x for x in xs, y in ys]), vec([y for x in xs, y in ys])), (L, L))
        ds = SB.DirectSumSpectralBackend()

        ref, _ = FFS.calculate_spectrum(sc, vec(u), ms; transform = ds, execution = CB.SerialBackend())
        ct, _ = FFS.calculate_spectrum(sc, vec(u), ms; transform = ds, execution = CB.ThreadedBackend())
        Test.@test isapprox(ct, ref; atol = 1e-12)

        cd, _ = FFS.calculate_spectrum(ug, u, ms; transform = ds)
        cf_gpu, _ = FFS.calculate_spectrum(ug, u, ms; transform = SB.FFTSpectralBackend(),
            execution = CB.GPUBackend(KA.CPU()))
        Test.@test isapprox(cf_gpu, cd; atol = 1e-12)
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
        Test.@test_throws DimensionMismatch FFS.spectral_divergence!(similar(c), ks, c)
        Test.@test_throws DimensionMismatch FFS.spectral_vorticity!(similar(c), ks, c)
        # One wavenumber |k| = √2 carries all the energy, so enstrophy is |k|² times energy in its bin.
        _, Ek = FFS.isotropic_spectrum(ks, c; num_bins = 6, dims = 3)
        _, Zk = FFS.isotropic_spectrum(ks, vortc; num_bins = 6, dims = 3)
        active = findall(>(1e-12), Ek)
        Test.@test length(active) == 1
        Test.@test isapprox(Zk[only(active)] / Ek[only(active)], 2.0; rtol = 1e-12)
        # The trapezoid band integral is exact on a linear spectrum.
        kl = collect(range(0.5, 5.5; length = 6))
        Test.@test FFS.band_energy(kl, 2 .* kl .+ 1, kl[1], kl[end]) ≈ (kl[end]^2 - kl[1]^2) + (kl[end] - kl[1])
    end

    Test.@testset "Derived quantities on a stretched grid unpack to the operator on the whole box" begin
        # Every even axis of a stretched grid holds both `±N_d/2`, each at its own signed wavenumber, so the
        # derived packed half unpacks to the operator applied mode by mode to the velocity's spectrum over
        # the whole coefficient box, at its FFT-order wavenumbers.
        Random.seed!(23)
        L = 2π
        n = 10
        str = [L * (i - 1) / n + 0.04 * L * sinpi(2 * (i - 1) / n) for i in 1:n]
        ds = SB.DirectSumSpectralBackend()
        full(ks) = (collect(FTB.FFTAxis(FTB.full_length(ks[1]), ks[1][2])), map(collect, Base.tail(ks))...)

        g = nucg((str, str), (L, L))
        c, ks = FFS.calculate_spectrum(g, randn(n, n, 2), (8, 8); transform = ds)
        box = FTB.full_size(ks)
        Test.@test box == (9, 9)
        U = FTB.unpacked(c, box)
        kx, ky = full(ks)
        vort = FFS.spectral_vorticity(ks, c)
        dv = FFS.spectral_divergence(ks, c)
        Test.@test isapprox(FTB.unpacked(vort, box)[:, :, 1],
            [im * (kx[i] * U[i, j, 2] - ky[j] * U[i, j, 1]) for i in 1:box[1], j in 1:box[2]]; atol = 1e-12)
        Test.@test isapprox(FTB.unpacked(dv, box)[:, :, 1],
            [im * (kx[i] * U[i, j, 1] + ky[j] * U[i, j, 2]) for i in 1:box[1], j in 1:box[2]]; atol = 1e-12)

        n3 = 7
        str3 = [L * (i - 1) / n3 + 0.04 * L * sinpi(2 * (i - 1) / n3) for i in 1:n3]
        g3 = nucg((str3, str3, str3), (L, L, L))
        c3, ks3 = FFS.calculate_spectrum(g3, randn(n3, n3, n3, 3), (6, 6, 6); transform = ds)
        box3 = FTB.full_size(ks3)
        Test.@test box3 == (7, 7, 7)
        U3 = FTB.unpacked(c3, box3)
        k1, k2, k3 = full(ks3)
        W3 = FTB.unpacked(FFS.spectral_vorticity(ks3, c3), box3)
        I3 = CartesianIndices(box3)
        curl = (I -> im * (k2[I[2]] * U3[I, 3] - k3[I[3]] * U3[I, 2]),
                I -> im * (k3[I[3]] * U3[I, 1] - k1[I[1]] * U3[I, 3]),
                I -> im * (k1[I[1]] * U3[I, 2] - k2[I[2]] * U3[I, 1]))
        for a in 1:3
            Test.@test isapprox(W3[:, :, :, a], map(curl[a], I3); atol = 1e-12)
        end
    end

    Test.@testset "Cross-spectrum / co-spectrum" begin
        L = 2π; N = 16
        xs = ucg_axis(Float64, L, N); ys = ucg_axis(Float64, L, N)
        g = ucg((L, L), (N, N))
        f = [cos(2x) + 0.5 * sin(3y) for x in xs, y in ys]
        h = [cos(2x) - 0.3 * cos(y) for x in xs, y in ys]
        cf, ks = FFS.calculate_spectrum(g, f, (N, N); transform = SB.FFTSpectralBackend())
        cg, _ = FFS.calculate_spectrum(g, h, (N, N); transform = SB.FFTSpectralBackend())
        # The co-spectrum integrates to the covariance and a field's own quadrature spectrum vanishes.
        kb, Co_fg = FFS.cospectrum(ks, cf, cg; num_bins = 6)
        Test.@test isapprox(sum(Co_fg) * (kb[2] - kb[1]), 0.5 * Statistics.mean(f .* h); rtol = 1e-12)
        _, Q_ff = FFS.quadspectrum(ks, cf, cf; num_bins = 6)
        Test.@test maximum(abs.(Q_ff)) < 1e-14
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
            Test.@test isapprox(c_fft, c_dir; atol = 1e-12)
            Test.@test isapprox(parseval(k_fft, c_fft), Statistics.mean(abs2, f); rtol = 1e-12)
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
            Test.@test isapprox(FTB.unpacked(c_real, ms), c_cplx; atol = 1e-12)
        end
    end

    Test.@testset "Float32 end-to-end (DirectSum / FFT)" begin
        L = 2.0f0 * Float32(π); N = 16
        xs = ucg_axis(Float32, L, N); ys = ucg_axis(Float32, L, N)
        g = ucg(Float32, (L, L), (N, N))
        f = Float32[cos(2x) + 0.5f0 * sin(3y) for x in xs, y in ys]
        c_dir, _ = FFS.calculate_spectrum(g, f, (N, N); transform = SB.DirectSumSpectralBackend())
        c_fft, ks = FFS.calculate_spectrum(g, f, (N, N); transform = SB.FFTSpectralBackend())
        c64, _ = FFS.calculate_spectrum(ucg((Float64(L), Float64(L)), (N, N)), Float64.(f), (N, N);
            transform = SB.FFTSpectralBackend())
        Test.@test eltype(c_dir) === ComplexF32
        Test.@test eltype(c_fft) === ComplexF32
        Test.@test isapprox(c_fft, c64; atol = 1.0f-6)
        Test.@test isapprox(c_dir, c64; atol = 1.0f-5)
        _, E_iso = FFS.isotropic_spectrum(ks, c_fft; num_bins = 6)
        Test.@test eltype(E_iso) === Float32
    end

    Test.@testset "Coherence / phase" begin
        # A field against a negative multiple of itself is coherent in every bin, at phase π.
        Random.seed!(2024)
        L = 2π; N = 32; nens = 4
        g = ucg((L,), (N,))
        cf, ks = FFS.calculate_spectrum(g, randn(N, nens), (N,); transform = SB.FFTSpectralBackend())
        kb, γ², φ = FFS.coherence_spectrum(ks, cf, -0.7 .* cf; num_bins = 8)
        Test.@test maximum(abs, γ² .- 1) < 1e-12
        Test.@test maximum(abs, abs.(φ) .- π) < 1e-12
    end

    Test.@testset "Multitaper (DPSS)" begin
        # The tapers are orthogonal with unit mean square, and each is an eigenvector of the
        # time–bandwidth concentration kernel `A[i, j] = sin(2πW(i−j)) / (π(i−j))`, its eigenvalue the
        # fraction of energy inside `|f| < W`, largest for order 0.
        N = 64; NW = 4.0; K = 7
        V = FFS.dpss(N, NW, K)
        Test.@test maximum(abs.(V' * V - N * Matrix(LA.I(K)))) < 1e-10 * N
        W = NW / N
        A = [i == j ? 2W : sinpi(2W * (i - j)) / (π * (i - j)) for i in 1:N, j in 1:N]
        λ = [V[:, k]' * A * V[:, k] / N for k in 1:K]
        Test.@test maximum(k -> LA.norm(A * V[:, k] .- λ[k] .* V[:, k]), 1:K) < 1e-10 * sqrt(N)
        Test.@test issorted(λ; rev = true) && λ[1] < 1
        V32 = Test.@inferred FFS.dpss(Float32, N, NW, K)
        Test.@test eltype(V32) === Float32 && maximum(abs, V32 .- V) < N * eps(Float32)
    end

    Test.@testset "Lomb–Scargle (irregular sampling)" begin
        # A tone at `f₀` lies in the span of the orthogonalized cosine and sine, so its periodogram there
        # is half its squared norm; shifting the time origin leaves the periodogram unchanged.
        Random.seed!(7)
        N = 60; t = sort(rand(N) .* 10.0); f0 = 1.3
        y = 0.8 .* cos.(2π * f0 .* t .+ 0.4)
        freqs = [0.7, f0, 2.1]
        P = FFS.lomb_scargle(t, y, freqs; center = false)
        Test.@test isapprox(P[2], 0.5 * sum(abs2, y); rtol = 1e-12)
        Pc = FFS.lomb_scargle(t, y, freqs)
        Test.@test maximum(abs, FFS.lomb_scargle(t .+ 3.7, y, freqs) .- Pc) < 1e-10 * maximum(Pc)
        Test.@test_throws DimensionMismatch FFS.lomb_scargle!(similar(Pc, length(freqs) + 1), t, y, freqs)
    end

    Test.@testset "Synthesis / inverse transform (round-trip)" begin
        L = 2π; N = 16
        xs = ucg_axis(Float64, L, N); ys = ucg_axis(Float64, L, N)
        g = ucg((L, L), (N, N))
        u = [cos(2x) + 0.5 * sin(3y) - 0.3 * cos(x + 2y) for x in xs, y in ys]
        # The packed half inverts directly to a real field; the full native spectrum is a complex field's
        # series. Both are the direct-sum inverse.
        ds = SB.DirectSumSpectralBackend()
        coeffs, _ = FFS.calculate_spectrum(g, u, (N, N); transform = ds)
        urec = FFS.synthesize(g, coeffs, (N, N); transform = ds)
        Test.@test eltype(urec) === Float64
        Test.@test isapprox(urec, u; atol = 1e-12)
        ufull = FFS.synthesize(ComplexF64, g, FTB.unpacked(coeffs, (N, N)), (N, N); transform = ds)
        Test.@test isapprox(real.(ufull), u; atol = 1e-12)
        Test.@test_throws DimensionMismatch FFS.synthesize(g, FTB.unpacked(coeffs, (N, N)), (N, N); transform = ds)

        # The real spherical harmonic `Y₃¹ = s(1)·P̄₃¹(cos θ)·cos φ`, `s(1) = −√2`, in closed form at scattered
        # nodes: `(√2/8)·√(21/π)·sin θ·(5cos²θ − 1)·cos φ`.
        lmax = 6; Nθ, Nφ = lmax + 1, 2lmax + 1
        N_pts = 4 * Nθ * Nφ
        ga = π * (3 - sqrt(5))
        z = [1 - 2 * (i + 0.5) / N_pts for i in 0:(N_pts - 1)]
        θs = acos.(clamp.(z, -1.0, 1.0)); φs = mod.(ga .* (0:(N_pts - 1)), 2π)
        sg = sph_scat(θs, φs)
        C = zeros(Nθ, Nφ); C[FFS.sph_mode_index(3, 1)] = 1.0
        fs = FFS.synthesize(sg, C, (Nθ, Nφ); transform = ds)
        Y31 = (sqrt(2) / 8) * sqrt(21 / π) .* sin.(θs) .* (5 .* cos.(θs) .^ 2 .- 1) .* cos.(φs)
        Test.@test maximum(abs, fs .- Y31) < 1e-13
    end

    include("test_reductions.jl")
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
    include("test_synthesis_plan.jl")
    include("test_device_genericity.jl")
    include("test_allocs.jl")
    include("test_jet.jl")
    include("test_distributed.jl")
    include("test_mpi.jl")
end
