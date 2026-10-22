# Each transform against a reference that shares no code with it: the factorized direct sum and every
# library over a widened coefficient box against an inline naive sum, the FFT and FastSphericalHarmonics
# against their own inverse pairs, and every synthesis against an inline evaluation of its mode sum.

using Test: Test
using FlowFieldSpectra: FlowFieldSpectra as FFS
using FlowTransformBindings: FlowTransformBindings as FTB
using FlowGeometries: FlowGeometries as FG
using SpectralBackends: SpectralBackends as SB
using ComputationalBackends: ComputationalBackends as CB
using FFTW: FFTW
using NonuniformFFTs: NonuniformFFTs
using FINUFFT: FINUFFT
using FastSphericalHarmonics: FastSphericalHarmonics
using NUFSHT: NUFSHT
using Random: Random

const RP_SER = CB.SerialBackend()
const RP_DS = SB.DirectSumSpectralBackend()
const RP_FFT = SB.FFTSpectralBackend()
const RP_CART = FG.Geometry.CartesianGeometry{Float64}()
rp_rel(a, b) = maximum(abs, a .- b) / max(maximum(abs, b), eps())
rp_uni(L, N) = range(0.0, L; length = N + 1)[1:N]
rp_str(L, N) = [L * (i - 1) / N + 0.04 * L * sinpi(2 * (i - 1) / N) for i in 1:N]
# The integer frequencies of an `m + 1`-mode widened axis, `m` even, in FFT order.
rp_wide(m) = [0:(m ÷ 2); -(m ÷ 2):-1]
# The libraries a Cartesian transform or synthesis runs through, each with its tolerance.
const RP_NUFFTS = ((FTB.FINUFFTBackend(), (; eps = 1e-13), 1e-10), (FTB.NonuniformFFTsBackend(), (; eps = 1e-13), 1e-10))

Test.@testset "Factorized direct sum equals an inline naive sum" begin
    # `C[k] = (1/N) Σ_j f_j exp(-i k·x_j)` over the packed half, written out here. On a uniform grid the
    # per-node measure is constant, so the quadrature factor is exactly one and drops out.
    Random.seed!(31)
    L = 2π
    N = 6
    xs = rp_uni(L, N)
    g = FG.Grids.StructuredGrid(RP_CART, xs, xs; periodic = (true, true), period = (L, L))
    f = randn(N, N)
    ms = (N, N)
    c, ks = FFS.calculate_spectrum(g, f, ms; transform = RP_DS, execution = RP_SER)

    pms = FTB.packed_size(ms, Val(true))
    naive = zeros(ComplexF64, pms...)
    for i1 in 1:pms[1], i2 in 1:pms[2]
        k1 = ks[1][i1]
        k2 = ks[2][i2]
        acc = zero(ComplexF64)
        for a in 1:N, b in 1:N
            acc += f[a, b] * cis(-(k1 * xs[a] + k2 * xs[b]))
        end
        naive[i1, i2] = acc / (N * N)
    end
    Test.@test rp_rel(c, naive) < 1e-13

    # The same on a stretched grid, where the quadrature factor is not one: the naive sum carries the
    # grid's own per-node measure, `C[k] = Σ w f e^{-ikx} / Σw`. Both even axes tell `±N/2` apart there, so
    # the box holds `k₁ = 0 … N/2` and `k₂ = -N/2 … N/2`.
    ax = rp_str(L, N)
    gs = FG.Grids.StructuredGrid(RP_CART, ax, ax; periodic = (true, true), period = (L, L))
    cs, kss = FFS.calculate_spectrum(gs, f, ms; transform = RP_DS, execution = RP_SER)
    w = collect(FG.Grids.measure(gs))
    k1s = 0:(N ÷ 2)
    k2s = rp_wide(N)
    Test.@test collect(kss[1]) ≈ k1s .* (2π / L)
    Test.@test collect(kss[2]) ≈ k2s .* (2π / L)
    naives = [sum(w[I] * f[I] * cis(-(k1 * ax[I[1]] + k2 * ax[I[2]]) * 2π / L) for I in CartesianIndices((N, N))) /
              sum(w) for k1 in k1s, k2 in k2s]
    Test.@test rp_rel(cs, naives) < 1e-13
end

Test.@testset "Every transform of a point set equals the naive sum over its coefficient box" begin
    # Every even axis of a point set tells `±m/2` apart, so the box holds `-m/2 … m/2` on it. A real field's
    # transform computes the packed half of the `m`-mode box and, by conjugate symmetry, its mirror; with
    # two axes `d ≥ 2` widened, a mode at `+m_d/2` with `-m_e/2` lies in neither and holds zero. Every other
    # mode is the naive sum `(1/M) Σ_j f_j exp(-iflag·i k·(x_j - x₀))`.
    Random.seed!(35)
    L = 2π
    M = 120
    for ms in ((6, 6), (4, 6, 4))
        D = length(ms)
        xs = ntuple(_ -> L .* rand(M), D)
        g = FG.Grids.UnstructuredGrid(RP_CART, xs, fill(1.0, M); periodic = ntuple(_ -> true, D),
            period = ntuple(_ -> L, D))
        x0 = map(minimum, xs)
        f = randn(M)
        axk = (0:(ms[1] ÷ 2), ntuple(d -> rp_wide(ms[d + 1]), D - 1)...)
        outside(k) = D == 3 && (k[2], k[3]) in ((ms[2] ÷ 2, -(ms[3] ÷ 2)), (-(ms[2] ÷ 2), ms[3] ÷ 2))
        for iflag in (1, -1)
            naive = map(Iterators.product(axk...)) do k
                outside(k) ? zero(ComplexF64) :
                    sum(f[j] * cis(-iflag * sum(k[d] * (xs[d][j] - x0[d]) for d in 1:D)) for j in 1:M) / M
            end
            for (t, kw, tol) in ((RP_DS, (; execution = RP_SER), 1e-13), RP_NUFFTS...)
                c, _ = FFS.calculate_spectrum(g, f, ms; transform = t, iflag, kw...)
                Test.@test size(c) == size(naive)
                Test.@test rp_rel(c, naive) < tol
            end
        end
    end
end

Test.@testset "Two widened axes of a stretched grid: zero outside the computed modes and their mirror" begin
    # Uniform along axis 1 at its own length and stretched along axes 2 and 3, whose `±m/2` the box holds.
    # The modes at `+m₂/2` with `-m₃/2`, and at `-m₂/2` with `+m₃/2`, are zero; every other mode is the naive
    # sum `Σ w f e^{-ik·(x - x₀)} / Σw`, through the direct sum, the separable NUFFT of each library and the
    # hybrid FFT/NUFFT composite.
    Random.seed!(37)
    L = 2π
    n = 6
    x1 = rp_uni(L, n)
    ax = rp_str(L, n)
    g = FG.Grids.StructuredGrid(RP_CART, x1, ax, ax; periodic = (true, true, true), period = (L, L, L))
    ms = (n, 6, 4)
    f = randn(n, n, n)
    w = collect(FG.Grids.measure(g))
    axk = (0:(n ÷ 2), rp_wide(ms[2]), rp_wide(ms[3]))
    naive = map(Iterators.product(axk...)) do k
        (k[2], k[3]) in ((3, -2), (-3, 2)) ? zero(ComplexF64) :
            sum(w[I] * f[I] * cis(-(k[1] * x1[I[1]] + k[2] * ax[I[2]] + k[3] * ax[I[3]])) for I in CartesianIndices(f)) /
            sum(w)
    end
    Test.@test count(iszero, naive) == 2 * length(axk[1])
    hybrid = map(((t, kw, tol),) -> (RP_FFT, (; execution = RP_SER, nufft = t, kw...), tol), RP_NUFFTS)
    for (t, kw, tol) in ((RP_DS, (; execution = RP_SER), 1e-13), RP_NUFFTS..., hybrid...)
        c, _ = FFS.calculate_spectrum(g, f, ms; transform = t, kw...)
        Test.@test size(c) == size(naive)
        Test.@test rp_rel(c, naive) < tol
    end
end

Test.@testset "Synthesis from the coefficient box is the real series over the whole box" begin
    # `f_j = Σ_κ C_κ exp(+iflag·iκ·(x_j - x₀))` over every mode of the box, the packed half unpacked by
    # conjugate symmetry, evaluated inline at each point. The series is real, and every synthesis of the
    # packed half gives it: on a grid stretched along both axes, uniform along axis 1 and stretched along
    # axis 2, and point sets in two and three dimensions.
    Random.seed!(36)
    L = 2π
    M = 50
    ax = rp_str(L, 8)
    hybrid = map(((t, kw, tol),) -> (RP_FFT, (; nufft = t, kw...), tol), RP_NUFFTS)
    cases = (
        (FG.Grids.StructuredGrid(RP_CART, ax, ax; periodic = (true, true), period = (L, L)), (8, 8), ()),
        (FG.Grids.StructuredGrid(RP_CART, rp_uni(L, 8), ax; periodic = (true, true), period = (L, L)), (8, 8),
            hybrid),
        (FG.Grids.UnstructuredGrid(RP_CART, (L .* rand(M), L .* rand(M)), fill(1.0, M);
            periodic = (true, true), period = (L, L)), (6, 8), ()),
        (FG.Grids.UnstructuredGrid(RP_CART, (L .* rand(M), L .* rand(M), L .* rand(M)), fill(1.0, M);
            periodic = (true, true, true), period = (L, L, L)), (4, 6, 4), ()),
    )
    for (g, ms, extra) in cases, iflag in (1, -1)
        D = length(ms)
        pts = FG.Grids.materialize(g)
        xs = ntuple(d -> vec(collect(pts[d])), D)
        x0 = map(minimum, xs)
        c, ks = FFS.calculate_spectrum(g, randn(size(g)...), ms; transform = RP_DS, execution = RP_SER, iflag)
        box = FTB.full_size(ks)
        full = FTB.unpacked(c, box)
        kf = ntuple(d -> [FTB.mode_frequency(FTB.FFTAxis(box[d], 1.0), i) for i in 1:box[d]], D)
        ref = [sum(full[I] * cis(iflag * sum(kf[d][I[d]] * (xs[d][j] - x0[d]) for d in 1:D))
                   for I in CartesianIndices(box)) for j in eachindex(xs[1])]
        Test.@test maximum(abs ∘ imag, ref) < 1e-13 * maximum(abs, ref)
        for (t, kw, tol) in ((RP_DS, (; execution = RP_SER), 1e-12), RP_NUFFTS..., extra...)
            out = FFS.synthesize(g, c, ms; transform = t, iflag, kw...)
            Test.@test eltype(out) === Float64
            Test.@test rp_rel(vec(out), real.(ref)) < tol
        end
    end
end

Test.@testset "Synthesis reproduces the band-limited part of a field" begin
    # A uniform grid transformed below its own length: the transform of each mode of the box is exact, so
    # synthesis of a field's coefficients is its projection `Σ_κ a_κ exp(iκ·x)` onto the box, with
    # `a_κ = (1/N²) Σ_j f_j exp(-iκ·x_j)`, and a field in that span returns unchanged.
    Random.seed!(38)
    L = 2π
    N = 12
    xs = rp_uni(L, N)
    g = FG.Grids.StructuredGrid(RP_CART, xs, xs; periodic = (true, true), period = (L, L))
    ms = (6, 6)
    k = -(ms[1] ÷ 2):(ms[1] ÷ 2)
    f = randn(N, N)
    a = [sum(f[i, j] * cis(-(k1 * xs[i] + k2 * xs[j])) for i in 1:N, j in 1:N) / N^2 for k1 in k, k2 in k]
    proj = [sum(a[p, q] * cis(k[p] * xs[i] + k[q] * xs[j]) for p in eachindex(k), q in eachindex(k)) for i in 1:N, j in 1:N]
    Test.@test maximum(abs ∘ imag, proj) < 1e-13 * maximum(abs, proj)
    proj = real.(proj)
    for (t, kw, tol) in ((RP_DS, (; execution = RP_SER), 1e-12), RP_NUFFTS...)
        c, _ = FFS.calculate_spectrum(g, f, ms; transform = t, kw...)
        Test.@test rp_rel(FFS.synthesize(g, c, ms; transform = t, kw...), proj) < tol
        cp, _ = FFS.calculate_spectrum(g, proj, ms; transform = t, kw...)
        Test.@test rp_rel(FFS.synthesize(g, cp, ms; transform = t, kw...), proj) < tol
    end
end

Test.@testset "Inverses that use no direct sum" begin
    Random.seed!(33)
    L = 2π
    N = 12
    xs = rp_uni(L, N)
    g = FG.Grids.StructuredGrid(RP_CART, xs, xs; periodic = (true, true), period = (L, L))
    f = randn(N, N)
    ms = (N, N)

    # FFT: its own forward/inverse pair round-trips the field.
    c, _ = FFS.calculate_spectrum(g, f, ms; transform = RP_FFT, execution = RP_SER)
    fr = FFS.synthesize(g, c, ms; transform = RP_FFT, execution = RP_SER)
    Test.@test rp_rel(fr, f) < 1e-12
    # And through the complex route.
    cc, _ = FFS.calculate_spectrum(g, ComplexF64.(f), ms; transform = RP_FFT, execution = RP_SER)
    fcr = FFS.synthesize(g, cc, ms; transform = RP_FFT, execution = RP_SER, real_output = false)
    Test.@test rp_rel(real.(fcr), f) < 1e-12

    # FastSphericalHarmonics: its analysis and synthesis are an exact pair in the coefficient direction.
    # Its Clenshaw–Curtis grid carries more samples than the degree it resolves, so analysis of an
    # arbitrary grid field is a band-limited projection; the identity is synthesis followed by analysis.
    lmax = 5
    sms = (lmax + 1, 2 * lmax + 1)
    gcc = FG.Connectivity.structured_grid(Float64, FG.SphericalSampling.ClenshawCurtisSampling(), lmax + 1)
    C = zeros(Float64, sms...)
    for l in 0:lmax, m in -l:l
        C[FFS.sph_mode_index(l, m)] = randn()
    end
    fsyn = FFS.synthesize(gcc, C, sms; transform = SB.FSHTSpectralBackend(), execution = RP_SER)
    C2 = FFS.calculate_spectrum(gcc, fsyn, sms; transform = SB.FSHTSpectralBackend(), execution = RP_SER)[1]
    Test.@test maximum(abs, C2 .- C) < 1e-12
    # A band-limited field then round-trips in the field direction too.
    Test.@test rp_rel(FFS.synthesize(gcc, C2, sms; transform = SB.FSHTSpectralBackend(), execution = RP_SER),
        fsyn) < 1e-11
end

Test.@testset "A NUFSHT type-2 evaluates the harmonic sum" begin
    lmax = 3
    sms = (lmax + 1, 2 * lmax + 1)
    Ns = 60
    ga = π * (3 - sqrt(5))
    λ = [mod(ga * i, 2π) for i in 1:Ns]
    φ = asin.([-1 + 2 * (i - 0.5) / Ns for i in 1:Ns])
    gs = FG.Grids.UnstructuredGrid(FG.Geometry.SphericalGeometry(1.0), (λ, φ), fill(4π / Ns, Ns))
    C = zeros(Float64, sms...)
    C[FFS.sph_mode_index(2, 1)] = 1.0
    C[FFS.sph_mode_index(3, -2)] = -0.6
    outn = FFS.synthesize(gs, C, sms; transform = SB.NUFSHTSpectralBackend())
    outd = FFS.synthesize(gs, C, sms; transform = RP_DS, execution = RP_SER)
    Test.@test rp_rel(outn, outd) < 1e-8
end
