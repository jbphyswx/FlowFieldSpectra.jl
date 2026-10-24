# Cross-backend agreement on the sphere.
#
# `calculate_spectrum` returns the quadrature projection `C[l,m] = Σⱼ wⱼ fⱼ Y_lm(xⱼ)` with the grid's own
# weights totalling `4π`, whichever spherical backend computes it; NUFSHT's quadrature analysis takes the
# grid's weights. A grid whose per-node measures vary is the case that separates a weighted transform from
# an unweighted one.
#
# `solve = true` is the least-squares fit `A c ≈ f`, recovering the coefficients a field was synthesized
# from.

using Test: Test
using FlowFieldSpectra: FlowFieldSpectra as FFS
using FlowTransformBindings: FlowTransformBindings as FTB
using FlowGeometries: FlowGeometries as FG
using SpectralBackends: SpectralBackends as SB
using ComputationalBackends: ComputationalBackends as CB
using FastSphericalHarmonics: FastSphericalHarmonics as FSH
using NUFSHT: NUFSHT
using FINUFFT: FINUFFT
using NonuniformFFTs: NonuniformFFTs
using KernelAbstractions: KernelAbstractions as KA
using Logging: Logging
using Random: Random

const SP_DS = SB.DirectSumSpectralBackend()
const SP_NU = SB.NUFSHTSpectralBackend()
const SP_SER = CB.SerialBackend()
sp_rel(a, b) = maximum(abs, a .- b) / max(maximum(abs, b), eps())

# Fibonacci-like nodes: a genuine point cloud with no iso-latitude structure.
function sp_nodes(N)
    ga = π * (3 - sqrt(5))
    λ = [mod(ga * i, 2π) for i in 1:N]
    z = [-1 + 2 * (i - 0.5) / N for i in 1:N]
    return λ, asin.(z), z
end

Test.@testset "NUFSHT forward is the quadrature projection" begin
    Random.seed!(4242)
    lmax = 4
    ms = (lmax + 1, 2 * lmax + 1)
    N = 300
    λ, φlat, z = sp_nodes(N)
    f = [1.0 + 0.5 * sin(λ[i]) + 0.3 * z[i] + 0.2 * cos(2 * λ[i]) * sqrt(1 - z[i]^2) for i in 1:N]

    # Equal-measure nodes: the weights are the constant 4π/N.
    ge = FG.Grids.UnstructuredGrid(FG.Geometry.SphericalGeometry(1.0), (λ, φlat), fill(4π / N, N))
    cd, _ = FFS.calculate_spectrum(ge, f, ms; transform = SP_DS, execution = SP_SER)
    cn, _ = FFS.calculate_spectrum(ge, f, ms; transform = SP_NU, tol = 1e-12)
    Test.@test sp_rel(cn, cd) < 1e-10

    # Varying measures move the projection, and NUFSHT follows them with either NUFFT library.
    w = [1.0 + 0.8 * sin(λ[i])^2 for i in 1:N]
    gv = FG.Grids.UnstructuredGrid(FG.Geometry.SphericalGeometry(1.0), (λ, φlat), w)
    cdv, _ = FFS.calculate_spectrum(gv, f, ms; transform = SP_DS, execution = SP_SER)
    Test.@test sp_rel(cdv, cd) > 1e-3
    for nufft in (FTB.FINUFFTBackend(), FTB.NonuniformFFTsBackend())
        cnv, _ = FFS.calculate_spectrum(gv, f, ms; transform = SP_NU, nufft = nufft, tol = 1e-12)
        Test.@test sp_rel(cnv, cdv) < 1e-10
    end

    fb = hcat(f, reverse(f), 0.5 .* f)
    cnb, _ = FFS.calculate_spectrum(gv, fb, ms; transform = SP_NU, tol = 1e-12)
    cdb, _ = FFS.calculate_spectrum(gv, fb, ms; transform = SP_DS, execution = SP_SER)
    Test.@test sp_rel(cnb, cdb) < 1e-10

    # `batch_chunk` splits the batch across executions; a chunk of 4 leaves a partial final chunk of 2.
    Random.seed!(31)
    lmaxb = 4
    msb = (lmaxb + 1, 2 * lmaxb + 1)
    Nb = 120
    λb, φb, _ = sp_nodes(Nb)
    gb = FG.Grids.UnstructuredGrid(FG.Geometry.SphericalGeometry(1.0), (λb, φb), fill(4π / Nb, Nb))
    B = 6
    fr = randn(Nb, B)
    fc = complex.(fr, reverse(fr; dims = 2))
    gpu = CB.GPUBackend(KA.CPU())
    for (T, ff) in ((Float64, fr), (ComplexF64, fc))
        r, _ = FFS.calculate_spectrum(gb, ff, msb; transform = SP_DS, execution = SP_SER)
        for bc in (1, 4, 8), exec in (SP_SER, gpu)
            pb = FFS.plan_spectrum(gb, T, msb; transform = SP_NU, execution = exec, batch = (B,),
                batch_chunk = bc, tol = 1e-12)
            buf = FFS.allocate_coefficients(pb)
            for _ in 1:2
                FFS.calculate_spectrum!(buf, pb, ff)
                Test.@test sp_rel(buf, r) < 1e-10
            end
        end
    end
end

Test.@testset "NUFSHT solve consumes the raw field" begin
    # A field synthesized from known coefficients; the solve returns them for any node measures.
    lmax = 3
    Nθ, Nφ = lmax + 1, 2 * lmax + 1
    N = 400
    λ, φlat, _ = sp_nodes(N)
    Ct = zeros(Nθ, Nφ)
    Ct[FSH.sph_mode(0, 0)] = 1.0
    Ct[FSH.sph_mode(2, 1)] = -0.4
    Ct[FSH.sph_mode(3, -2)] = 0.7
    Cref = zeros(Nθ, Nφ)
    for l in 0:lmax, m in -l:l
        Cref[FFS.sph_mode_index(l, m)] = Ct[FSH.sph_mode(l, m)]
    end
    plan = NUFSHT.make_plan(Float64, Float64(π) / 2 .- φlat, λ, lmax)
    fv = zeros(N)
    NUFSHT.nusht_type2!(fv, Ct, plan)
    for w in (fill(4π / N, N), [1.0 + 0.8 * sin(λ[i])^2 for i in 1:N])
        g = FG.Grids.UnstructuredGrid(FG.Geometry.SphericalGeometry(1.0), (λ, φlat), w)
        for exec in (SP_SER, CB.GPUBackend(KA.CPU()))
            cs, _ = FFS.calculate_spectrum(g, fv, (Nθ, Nφ); transform = SP_NU, execution = exec, solve = true,
                rtol = 1.0e-10, maxiter = 2000)
            Test.@test maximum(abs, cs .- Cref) < 1e-10
        end
    end
end

Test.@testset "NUFSHT on a structured spherical grid" begin
    # A structured grid's field has two spatial dims; the batch and the synthesis shape both come from
    # the grid, and the weights are the sampling's own latitude quadrature.
    lmax = 4
    Nθ, Nφ = lmax + 1, 2 * lmax + 1
    g = FG.Connectivity.structured_grid(Float64, FG.SphericalSampling.GaussLegendreSampling(), Nθ)
    f = [1.0 + 0.3 * sin(2 * λ) * cos(φ) for λ in FG.Grids.coordinates(g, 1),
        φ in FG.Grids.coordinates(g, 2)]

    cn, _ = FFS.calculate_spectrum(g, f, (Nθ, Nφ); transform = SP_NU)
    cd, _ = FFS.calculate_spectrum(g, f, (Nθ, Nφ); transform = SP_DS, execution = SP_SER)
    Test.@test sp_rel(cn, cd) < 1e-8
    cnb, _ = FFS.calculate_spectrum(g, cat(f, 2 .* f; dims = 3), (Nθ, Nφ); transform = SP_NU)
    Test.@test sp_rel(cnb, cat(cd, 2 .* cd; dims = 3)) < 1e-8

    out = FFS.synthesize(g, cn, (Nθ, Nφ); transform = SP_NU)
    Test.@test sp_rel(out, FFS.synthesize(g, cn, (Nθ, Nφ); transform = SP_DS, execution = SP_SER)) < 1e-8
end

# Real (or, with `cplx`, complex) coefficients on every `(l, m)` slot up to `lmax`, `B` sets of them.
function sp_coeffs(lmax, B; cplx = false)
    C = zeros(cplx ? ComplexF64 : Float64, lmax + 1, 2 * lmax + 1, B)
    for b in 1:B, l in 0:lmax, m in -l:l
        C[FFS.sph_mode_index(l, m), b] = cplx ? randn(ComplexF64) : randn()
    end
    return C
end

Test.@testset "A FastSphericalHarmonics plan recovers the coefficients a field was synthesized from" begin
    # The analysis is exact on its own Clenshaw–Curtis grid, so a field the direct sum synthesizes from
    # known coefficients returns them, on one thread and on `Threads.nthreads()`.
    Random.seed!(91)
    lmax = 5
    ms = (lmax + 1, 2 * lmax + 1)
    B = 3
    gcc = FG.Connectivity.structured_grid(Float64, FG.SphericalSampling.ClenshawCurtisSampling(), lmax + 1)
    for cplx in (false, true), exec in (SP_SER, CB.ThreadedBackend())
        C = sp_coeffs(lmax, B; cplx = cplx)
        f = FFS.synthesize(gcc, C, ms; transform = SP_DS, execution = SP_SER, real_output = !cplx)
        p = FFS.plan_spectrum(gcc, eltype(f), ms; transform = SB.FSHTSpectralBackend(), execution = exec,
            batch = (B,))
        buf = FFS.allocate_coefficients(p)
        for _ in 1:2
            FFS.calculate_spectrum!(buf, p, f)
            Test.@test maximum(abs, buf .- C) < 1e-12
        end
        # A buffer or a field of another shape is refused.
        Test.@test_throws DimensionMismatch FFS.calculate_spectrum!(zeros(eltype(buf), ms...), p, f)
        Test.@test_throws DimensionMismatch FFS.calculate_spectrum!(buf, p, f[:, :, 1])
    end
end

Test.@testset "A FastSphericalHarmonics synthesis plan evaluates the series at the nodes" begin
    # Against the direct sum, which evaluates `Σ C_lm Y_lm` point by point: a real and a complex series,
    # batched, reused, on one thread and on `Threads.nthreads()`.
    Random.seed!(94)
    lmax = 5
    ms = (lmax + 1, 2 * lmax + 1)
    B = 3
    gcc = FG.Connectivity.structured_grid(Float64, FG.SphericalSampling.ClenshawCurtisSampling(), lmax + 1)
    for cplx in (false, true), exec in (SP_SER, CB.ThreadedBackend())
        p = FFS.plan_synthesis(gcc, cplx ? ComplexF64 : Float64, ms; transform = SB.FSHTSpectralBackend(),
            execution = exec, batch = (B,))
        out = FFS.allocate_field(p)
        for _ in 1:2
            C = sp_coeffs(lmax, B; cplx = cplx)
            fd = FFS.synthesize(gcc, C, ms; transform = SP_DS, execution = SP_SER, real_output = !cplx)
            FFS.synthesize!(out, p, C)
            Test.@test sp_rel(out, fd) < 1e-12
        end
    end
end

Test.@testset "NUFSHT solve on a masked grid, inactive cells zero-filled or excluded" begin
    # A field synthesized from known coefficients holds NaN under the mask. `ZeroFill` fits the field
    # extended by zero, as the unmasked grid fits it; `Exclude` fits the active nodes alone, returning
    # those coefficients as the fit on a grid of the active nodes does.
    Random.seed!(95)
    lmax = 3
    ms = (lmax + 1, 2 * lmax + 1)
    N = 400
    λ, φlat, z = sp_nodes(N)
    land = [z[i] > 0.4 && λ[i] < π for i in 1:N]
    sea = .!land
    geo = FG.Geometry.SphericalGeometry(1.0)
    C = sp_coeffs(lmax, 1)[:, :, 1]
    gu = FG.Grids.UnstructuredGrid(geo, (λ, φlat), fill(4π / N, N))
    f = FFS.synthesize(gu, C, ms; transform = SP_DS, execution = SP_SER)
    fn = copy(f)
    fn[land] .= NaN
    fz = copy(f)
    fz[land] .= 0
    gm = FG.Grids.UnstructuredGrid(geo, (λ, φlat), fill(4π / N, N), sea)
    gs = FG.Grids.UnstructuredGrid(geo, (λ[sea], φlat[sea]), fill(4π / N, count(sea)))
    kw = (; transform = SP_NU, solve = true, rtol = 1.0e-12, maxiter = 2000)
    for exec in (SP_SER, CB.GPUBackend(KA.CPU()))
        cz, _ = FFS.calculate_spectrum(gm, fn, ms; execution = exec, kw...)
        cu, _ = FFS.calculate_spectrum(gu, fz, ms; execution = exec, kw...)
        Test.@test maximum(abs, cz .- cu) < 1e-10
        cm, _ = FFS.calculate_spectrum(gm, fn, ms; execution = exec, inactive = FFS.Exclude(), kw...)
        cs, _ = FFS.calculate_spectrum(gs, f[sea], ms; execution = exec, kw...)
        Test.@test maximum(abs, cm .- C) < 1e-10
        Test.@test maximum(abs, cm .- cs) < 1e-10
    end
end

Test.@testset "The device SHT reads every order's longitude bin" begin
    # Gauss–Legendre latitudes and `nlon` uniform longitudes, against the direct sum's exact longitude
    # sums. Below `2lmax + 1` longitudes the orders alias, and the device transform reads each order's
    # aliased bin, as the direct sum evaluates it.
    Random.seed!(92)
    lmax = 6
    ms = (lmax + 1, 2 * lmax + 1)
    GL = FG.SphericalSampling.GaussLegendreSampling()
    gpu = CB.GPUBackend(KA.CPU())
    for nlon in (2 * lmax - 1, 2 * lmax + 1, 2 * lmax + 4)
        g = FG.Connectivity.structured_grid(Float64, GL, lmax + 1; nlon = nlon)
        for f in (randn(nlon, lmax + 1, 2), randn(ComplexF64, nlon, lmax + 1))
            cd, _ = Logging.with_logger(Logging.NullLogger()) do
                FFS.calculate_spectrum(g, f, ms; transform = SP_DS, execution = SP_SER)
            end
            p = Logging.with_logger(Logging.NullLogger()) do
                FFS.plan_spectrum(g, eltype(f), ms; transform = SB.FSHTSpectralBackend(), execution = gpu,
                    batch = size(f)[3:end])
            end
            buf = FFS.allocate_coefficients(p)
            for _ in 1:2
                FFS.calculate_spectrum!(buf, p, f)
                Test.@test sp_rel(buf, cd) < 1e-12
            end
        end
    end

    # A masked grid: the device transform zeroes the inactive cells, `NaN` included, and the direct sum's
    # ring path skips them.
    nlat = lmax + 1
    mask = trues(2 * lmax + 1, nlat)
    mask[3:5, 2] .= false
    mask[1, nlat] = false
    gm = FG.Connectivity.structured_grid(Float64, GL, nlat; mask = mask)
    fm = randn(size(gm)...)
    fm[.!mask] .= NaN
    cdm, _ = FFS.calculate_spectrum(gm, fm, ms; transform = SP_DS, execution = SP_SER)
    cgm, _ = FFS.calculate_spectrum(gm, fm, ms; transform = SB.FSHTSpectralBackend(), execution = gpu)
    Test.@test sp_rel(cgm, cdm) < 1e-12

    # The device Legendre table matches closed forms at low degree and is orthonormal,
    # `2π Σ_j w_j P̄_ℓ^m P̄_ℓ'^m = δ_ℓℓ'`, under the Gauss–Legendre rule, exact at these degrees.
    SK = FFS.SphericalKernels
    g = FG.Connectivity.structured_grid(Float64, GL, nlat; nlon = 2 * lmax + 1)
    θ = π / 2 .- collect(FG.Grids.coordinates(g, 2))
    row(l, m) = SK.legendre_index(l, m, lmax)
    P = SK.device_legendre_table(KA.CPU(), Float64, lmax, θ)
    Test.@test maximum(abs, P[row(0, 0), :] .- 1 / sqrt(4π)) < 1e-15
    Test.@test maximum(abs, P[row(1, 0), :] .- sqrt(3 / 4π) .* cos.(θ)) < 1e-14
    Test.@test maximum(abs, P[row(1, 1), :] .+ sqrt(3 / 8π) .* sin.(θ)) < 1e-14
    Test.@test maximum(abs, P[row(2, 2), :] .- sqrt(15 / 32π) .* sin.(θ) .^ 2) < 1e-14
    wq = FG.SphericalSampling.latitude_weights(Float64, GL, nlat)
    gram = [2π * sum(wq .* P[row(l, m), :] .* P[row(l2, m), :]) - (l == l2)
            for m in 0:lmax for l in m:lmax for l2 in m:lmax]
    Test.@test maximum(abs, gram) < 1e-13
end

# `P̄_ℓ^m` up to lmax = 512 against the three-term recurrence in 256-bit `BigFloat`, whose exponent range
# holds `sinᵐθ` at every order: rows next to both poles, mid-latitudes and the equator, host and device,
# `Float64` and `Float32`. Each order's column lies within `lmax eps` of its largest value, plus the
# `√((2lmax+1)/4π) 2^{−3K/2}` below which a value is stored as zero. At `θ = 0.467`, `m = 200` starts
# at `sin²⁰⁰θ ≈ 10⁻⁶⁹`, below the `Float32` range, and the column reaches order one by `ℓ ≈ 444`.
Test.@testset "normalized Legendre values at lmax = 512 against BigFloat" begin
    SK = FFS.SphericalKernels
    lmax = 512
    ref_column(m, θ) = setprecision(BigFloat, 256) do
        x = cos(BigFloat(θ)); s = sin(BigFloat(θ))
        col = zeros(BigFloat, lmax - m + 1)
        col[1] = prod(k -> -sqrt(BigFloat(2k + 1) / (2k)) * s, 1:m; init = 1 / sqrt(4 * BigFloat(π)))
        m < lmax && (col[2] = x * sqrt(BigFloat(2m + 3)) * col[1])
        for l in (m + 2):lmax
            a = sqrt(BigFloat(4l^2 - 1) / (l^2 - m^2))
            b = sqrt(BigFloat(2l + 1) * ((l - 1)^2 - m^2) / ((2l - 3) * (l^2 - m^2)))
            col[l - m + 1] = x * a * col[l - m] - b * col[l - m - 1]
        end
        col
    end
    for FT in (Float64, Float32)
        θs = FT[2π / 1030, 0.467, 1.2, π / 2, 2.5, π - 2π / 1030]
        t = SK.legendre_tables(FT, lmax)
        Plm = zeros(FT, lmax + 1, lmax + 1)
        Pd = Array(SK.device_legendre_table(KA.CPU(), FT, lmax, θs))
        floor = sqrt((2lmax + 1) / 4π) * ldexp(1.0, -3 * SK._span(FT) ÷ 2)
        for (j, θ) in enumerate(θs)
            SK.fill_legendre!(Plm, t, cos(θ), sin(θ), lmax)
            for m in (0, 3, 100, 200, 300, 450, 512)
                r = ref_column(m, θ)
                tol = lmax * eps(FT) * Float64(maximum(abs, r)) + floor
                host = Plm[(m + 1):(lmax + 1), m + 1]
                dev = [Pd[SK.legendre_index(l, m, lmax), j] for l in m:lmax]
                Test.@test maximum(abs, Float64.(host) .- Float64.(r)) <= tol
                Test.@test maximum(abs, Float64.(dev) .- Float64.(r)) <= tol
            end
        end
    end
end

Test.@testset "The device SHT synthesis evaluates the series at the nodes" begin
    # Against the direct sum, which evaluates `Σ C_lm Y_lm` point by point: every longitude count, a real
    # and a complex series, batched. On a Gauss–Legendre grid with `2lmax + 1` longitudes the quadrature is
    # exact, so the device analysis returns the coefficients.
    Random.seed!(93)
    lmax = 6
    ms = (lmax + 1, 2 * lmax + 1)
    GL = FG.SphericalSampling.GaussLegendreSampling()
    gpu = CB.GPUBackend(KA.CPU())
    for nlon in (2 * lmax - 1, 2 * lmax + 1, 2 * lmax + 4), cplx in (false, true)
        g = FG.Connectivity.structured_grid(Float64, GL, lmax + 1; nlon = nlon)
        C = sp_coeffs(lmax, 2; cplx = cplx)
        fd = FFS.synthesize(g, C, ms; transform = SP_DS, execution = SP_SER, real_output = !cplx)
        p = FFS.plan_synthesis(g, cplx ? ComplexF64 : Float64, ms; transform = SB.FSHTSpectralBackend(),
            execution = gpu, batch = (2,))
        out = FFS.allocate_field(p)
        for _ in 1:2
            FFS.synthesize!(out, p, C)
            Test.@test sp_rel(out, fd) < 1e-12
        end
        if nlon == 2 * lmax + 1
            cg, _ = FFS.calculate_spectrum(g, out, ms; transform = SB.FSHTSpectralBackend(), execution = gpu)
            Test.@test sp_rel(cg, C) < 1e-12
        end
    end
end
