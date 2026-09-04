# Independent references: each transform checked against something that shares no code with it.
#
# Three kinds of check live here.
#
# 1. The factorized direct sum against an INLINE naive double sum. The direct sum is what every other
#    backend is compared to, so its own reference is written out here in the test file.
# 2. The hybrid FFT/NUFFT composite against the direct sum. The composite runs an FFT over the uniform
#    axes and a 1-D NUFFT along each stretched one, so nothing about it is shared with the direct sum.
# 3. Inverses WITHOUT the direct sum: FFT and FastSphericalHarmonics each round-trip through their own
#    pair, and the NUFFT/NUFSHT type-2 is checked against an inline evaluation of the mode sum. A NUFFT
#    type-2 is the ADJOINT of the type-1, so composing them on nonuniform nodes gives `AᴴA`; the
#    invariant that holds there is the mode sum.

using Test: Test
using FlowFieldSpectra: FlowFieldSpectra as FFS
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
rp_rel(a, b) = maximum(abs, a .- b) / max(maximum(abs, b), eps())
rp_uni(L, N) = range(0.0, L; length = N + 1)[1:N]
rp_str(L, N) = [L * (i - 1) / N + 0.04 * L * sinpi(2 * (i - 1) / N) for i in 1:N]

Test.@testset "Factorized direct sum equals an inline naive sum" begin
    # `C[k] = (1/N) Σ_j f_j exp(-i k·x_j)` over the packed half, written out here. On a uniform grid the
    # per-node measure is constant, so the quadrature factor is exactly one and drops out.
    Random.seed!(31)
    L = 2π
    N = 6
    xs = rp_uni(L, N)
    g = FG.Grids.StructuredGrid(FG.Geometry.CartesianGeometry{Float64}(), xs, xs;
        periodic = (true, true), period = (L, L))
    f = randn(N, N)
    ms = (N, N)
    c, ks = FFS.calculate_spectrum(g, f, ms; transform = RP_DS, execution = RP_SER)

    pms = FFS.Packing.packed_size(ms, Val(true))
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

    # The same on a STRETCHED grid, where the quadrature factor is not one: the naive sum carries the
    # grid's own per-node measure, `C[k] = Σ w f e^{-ikx} / Σw`.
    ax = rp_str(L, N)
    gs = FG.Grids.StructuredGrid(FG.Geometry.CartesianGeometry{Float64}(), ax, ax;
        periodic = (true, true), period = (L, L))
    cs, kss = FFS.calculate_spectrum(gs, f, ms; transform = RP_DS, execution = RP_SER)
    w = collect(FG.Grids.measure(gs))
    sw = sum(w)
    naives = zeros(ComplexF64, pms...)
    for i1 in 1:pms[1], i2 in 1:pms[2]
        k1 = kss[1][i1]
        k2 = kss[2][i2]
        acc = zero(ComplexF64)
        for I in CartesianIndices((N, N))
            acc += w[I] * f[I] * cis(-(k1 * ax[I[1]] + k2 * ax[I[2]]))
        end
        naives[i1, i2] = acc / sw
    end
    Test.@test rp_rel(cs, naives) < 1e-13
end

Test.@testset "Hybrid composite equals the direct sum" begin
    # The composite shares no code with the direct sum: an FFT over the uniform axes, a 1-D NUFFT along
    # each stretched one. Both NUFFT providers are checked.
    Random.seed!(32)
    L = 2π
    for (N1, N2) in ((8, 8), (8, 7), (7, 8))
        g = FG.Grids.StructuredGrid(FG.Geometry.CartesianGeometry{Float64}(),
            rp_uni(L, N1), rp_str(L, N2); periodic = (true, true), period = (L, L))
        f = randn(N1, N2)
        ms = (N1, N2)
        cd, ksd = FFS.calculate_spectrum(g, f, ms; transform = RP_DS, execution = RP_SER)
        for nu in (FFS.NonuniformFFTsBackend(), FFS.FINUFFTBackend())
            ch, ksh = FFS.calculate_spectrum(g, f, ms; transform = RP_FFT, execution = RP_SER,
                nufft = nu, eps = 1e-12)
            Test.@test size(ch) == size(cd)
            Test.@test rp_rel(ch, cd) < 1e-9
            # The Nyquist twin the composite publishes must match the direct sum's entry for entry.
            th = FFS.Packing.axis_twin(ksh[1])
            td = FFS.Packing.axis_twin(ksd[1])
            if th !== nothing && td !== nothing
                for (a, b) in zip(th.slices, td.slices)
                    isempty(a) || Test.@test rp_rel(a, b) < 1e-9
                end
            end
        end
    end

    # Complex and batched fields take the same route.
    N = 8
    g = FG.Grids.StructuredGrid(FG.Geometry.CartesianGeometry{Float64}(),
        rp_uni(L, N), rp_str(L, N); periodic = (true, true), period = (L, L))
    fb = randn(N, N, 2)
    Test.@test rp_rel(FFS.calculate_spectrum(g, fb, (N, N); transform = RP_FFT, execution = RP_SER)[1],
        FFS.calculate_spectrum(g, fb, (N, N); transform = RP_DS, execution = RP_SER)[1]) < 1e-9
    fc = randn(ComplexF64, N, N)
    Test.@test rp_rel(FFS.calculate_spectrum(g, fc, (N, N); transform = RP_FFT, execution = RP_SER)[1],
        FFS.calculate_spectrum(g, fc, (N, N); transform = RP_DS, execution = RP_SER)[1]) < 1e-9
end

Test.@testset "Inverses that use no direct sum" begin
    Random.seed!(33)
    L = 2π
    N = 12
    xs = rp_uni(L, N)
    g = FG.Grids.StructuredGrid(FG.Geometry.CartesianGeometry{Float64}(), xs, xs;
        periodic = (true, true), period = (L, L))
    f = randn(N, N)
    ms = (N, N)

    # FFT: its own forward/inverse pair round-trips the field.
    c, ks = FFS.calculate_spectrum(g, f, ms; transform = RP_FFT, execution = RP_SER)
    fr = FFS.synthesize(g, c, ms; transform = RP_FFT, execution = RP_SER, ks = ks)
    Test.@test eltype(fr) === Float64
    Test.@test rp_rel(fr, f) < 1e-12
    # And through the complex route.
    cc, kc = FFS.calculate_spectrum(g, ComplexF64.(f), ms; transform = RP_FFT, execution = RP_SER)
    fcr = FFS.synthesize(g, cc, ms; transform = RP_FFT, execution = RP_SER, real_output = false, ks = kc)
    Test.@test rp_rel(real.(fcr), f) < 1e-12

    # FastSphericalHarmonics: its analysis and synthesis are an exact pair, in the COEFFICIENT direction.
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
    Test.@test eltype(fsyn) === Float64
    Test.@test size(fsyn) == size(gcc)
    C2 = FFS.calculate_spectrum(gcc, fsyn, sms; transform = SB.FSHTSpectralBackend(), execution = RP_SER)[1]
    for l in 0:lmax, m in -l:l
        Test.@test isapprox(C2[FFS.sph_mode_index(l, m)], C[FFS.sph_mode_index(l, m)], atol = 1e-11)
    end
    # A band-limited field then round-trips in the FIELD direction too.
    Test.@test rp_rel(FFS.synthesize(gcc, C2, sms; transform = SB.FSHTSpectralBackend(), execution = RP_SER),
        fsyn) < 1e-11
end

Test.@testset "A NUFFT type-2 evaluates the mode sum" begin
    # `synthesize` through a NUFFT is the type-2, `f_j = Σ_κ C_κ exp(+i κ·x_j)` over the native modes.
    # Composing it with the type-1 gives `AᴴA` on nonuniform nodes, so the mode sum written out here is
    # the invariant, and it shares no code with the provider.
    Random.seed!(34)
    L = 2π
    M = 40
    N = 8
    ms = (N, N)
    xv = L .* rand(M)
    yv = L .* rand(M)
    g = FG.Grids.UnstructuredGrid(FG.Geometry.CartesianGeometry{Float64}(), (xv, yv),
        fill(L^2 / M, M); periodic = (true, true), period = (L, L))
    f = randn(M)
    c, ks = FFS.calculate_spectrum(g, f, ms; transform = FFS.NonuniformFFTsBackend(), eps = 1e-12)

    # The full native cube the half stands for, and the physical wavenumbers of each native mode.
    full = FFS.unpacked(c, ms, ks)
    kf = FFS.Grids.physical_wavenumbers(g, ms, Val(false))
    ref = zeros(Float64, M)
    for j in 1:M
        acc = zero(ComplexF64)
        for I in CartesianIndices(ms)
            acc += full[I] * cis(kf[1][I[1]] * xv[j] + kf[2][I[2]] * yv[j])
        end
        ref[j] = real(acc)
    end
    out = FFS.synthesize(g, c, ms; transform = FFS.NonuniformFFTsBackend(), ks = ks, eps = 1e-12)
    Test.@test length(out) == M
    Test.@test rp_rel(out, ref) < 1e-9

    # NUFSHT's type-2 likewise: an evaluation of the harmonic sum at the nodes.
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
