# Windowing, detrending and zero padding, wired into the transform.
#
# `Preprocess` and the window/detrend types were exported and documented while nothing in the transform
# path read them, so `calculate_spectrum(...; preprocess = ...)` was swallowed by `kwargs...` and the
# taper never reached the field. These testsets pin the wiring and the invariant that makes a taper safe
# to apply automatically: each axis taper is scaled to unit mean square, so the tapered field keeps the
# variance of the original and Parseval holds on the coefficients with nothing applied afterwards.

using Test: Test
using FlowFieldSpectra: FlowFieldSpectra as FFS
using FlowGeometries: FlowGeometries as FG
using SpectralBackends: SpectralBackends as SB
using ComputationalBackends: ComputationalBackends as CB
using FFTW: FFTW
using NonuniformFFTs: NonuniformFFTs
using Random: Random

const PP_SER = CB.SerialBackend()
const PP_DS = SB.DirectSumSpectralBackend()

pp_unif(L, N) = range(0.0, L; length = N + 1)[1:N]
pp_cart(L, N) = FG.Grids.StructuredGrid(FG.Geometry.CartesianGeometry{Float64}(),
    pp_unif(L, N), pp_unif(L, N); periodic = (true, true), period = (L, L))

# Folded Parseval on the packed half: `Σ_k mode_fold·|C|²` is `mean|f|²`.
function pp_parseval(ks, c)
    s = 0.0
    for I in CartesianIndices(size(c))
        s += FFS.Packing.mode_fold(ks, I) * abs2(c[I])
    end
    return s
end

Test.@testset "A taper preserves variance, so Parseval survives it" begin
    Random.seed!(9090)
    L = 2π
    N = 16
    g = pp_cart(L, N)
    xs = pp_unif(L, N)
    # A field with no mean, so demeaning is a no-op and the taper is what is under test.
    f = [sin(3x) * cos(2y) + 0.4 * sin(x + y) for x in xs, y in xs]

    for win in (FFS.NoWindow(), FFS.Hann(), FFS.Hamming(), FFS.Blackman(), FFS.Tukey(0.4))
        spec = FFS.Preprocess(; detrend = FFS.NoDetrend(), window = win)
        c, ks = FFS.calculate_spectrum(g, f, (N, N); preprocess = spec, execution = PP_SER)
        Test.@test size(c) == FFS.Packing.packed_size((N, N), Val(true))
        # The tapered field's own mean square, which the coefficients must reproduce.
        _, ftap, _ = FFS.preprocess_field(g, f, spec)
        Test.@test isapprox(pp_parseval(ks, c), sum(abs2, ftap) / length(ftap); rtol = 1e-10)
        # And each axis taper is unit mean square by construction.
        w = FFS.Preprocessing.axis_taper(Float64, win, N)
        Test.@test isapprox(sum(abs2, w) / N, 1.0; rtol = 1e-12)
    end

    # `NoWindow` is an exact no-op, so an unpreprocessed call and a rectangular one agree bit for bit.
    c0, _ = FFS.calculate_spectrum(g, f, (N, N); execution = PP_SER)
    c1, _ = FFS.calculate_spectrum(g, f, (N, N); execution = PP_SER,
        preprocess = FFS.Preprocess(; detrend = FFS.NoDetrend(), window = FFS.NoWindow()))
    Test.@test c0 == c1
    # Omitting `preprocess` copies nothing: the same array comes back.
    gr, fr, sc = FFS.preprocess_field(g, f, nothing)
    Test.@test fr === f && gr === g && sc == 1
end

Test.@testset "A taper lowers the sidelobes of a non-periodic tone" begin
    L = 1.0
    N = 128
    xs = pp_unif(L, N)
    g1 = FG.Grids.StructuredGrid(FG.Geometry.CartesianGeometry{Float64}(), xs;
        periodic = (true,), period = (L,))
    # A frequency that is NOT an integer number of cycles over the window, so the rectangular transform
    # leaks across the whole band.
    f = [sin(2π * 10.5 * x) for x in xs]

    rect, ks = FFS.calculate_spectrum(g1, f, (N,); transform = PP_DS, execution = PP_SER,
        preprocess = FFS.Preprocess(; detrend = FFS.NoDetrend(), window = FFS.NoWindow()))
    hann, _ = FFS.calculate_spectrum(g1, f, (N,); transform = PP_DS, execution = PP_SER,
        preprocess = FFS.Preprocess(; detrend = FFS.NoDetrend(), window = FFS.Hann()))

    # Far from the peak (which sits between bins 10 and 11), the Hann taper must be well below the
    # rectangular window's leakage.
    far = [i for i in 1:length(ks[1]) if abs(i - 11) > 8]
    Test.@test maximum(abs, hann[far]) < 0.05 * maximum(abs, rect[far])
end

Test.@testset "Detrending" begin
    Random.seed!(9091)
    L = 2π
    N = 16
    g = pp_cart(L, N)
    xs = pp_unif(L, N)
    f = [5.0 + 2.0 * x + 0.5 * sin(3y) for x in xs, y in xs]     # offset + linear ramp + a tone

    # `Demean` removes the DC mode; `NoDetrend` leaves the offset there.
    cd, _ = FFS.calculate_spectrum(g, f, (N, N); execution = PP_SER,
        preprocess = FFS.Preprocess(; detrend = FFS.Demean()))
    Test.@test abs(cd[1, 1]) < 1e-12
    cn, _ = FFS.calculate_spectrum(g, f, (N, N); execution = PP_SER)
    Test.@test abs(cn[1, 1]) > 1.0

    # `LinearDetrend` removes the ramp along each axis, leaving less low-wavenumber power than `Demean`.
    cl, ksl = FFS.calculate_spectrum(g, f, (N, N); execution = PP_SER,
        preprocess = FFS.Preprocess(; detrend = FFS.LinearDetrend()))
    low(c) = sum(abs2, c[1:3, 1:3])
    Test.@test low(cl) < low(cd)

    # `Demean` needs no axes, so it serves a point cloud; the axis-based operations say why they cannot.
    M = 50
    cloud = FG.Grids.UnstructuredGrid(FG.Geometry.CartesianGeometry{Float64}(),
        (L .* rand(M), L .* rand(M)), fill(L^2 / M, M); periodic = (true, true), period = (L, L))
    fc = randn(M) .+ 3.0
    cc, _ = FFS.calculate_spectrum(cloud, fc, (8, 8); transform = PP_DS, execution = PP_SER,
        preprocess = FFS.Preprocess(; detrend = FFS.Demean()))
    Test.@test abs(cc[1, 1]) < 1e-12
    Test.@test_throws ArgumentError FFS.preprocess_field(cloud, fc,
        FFS.Preprocess(; window = FFS.Hann()))
    Test.@test_throws ArgumentError FFS.preprocess_field(cloud, fc,
        FFS.Preprocess(; detrend = FFS.LinearDetrend()))
end

Test.@testset "Zero padding narrows the wavenumber spacing" begin
    L = 2π
    N = 16
    g = pp_cart(L, N)
    xs = pp_unif(L, N)
    f = [sin(3x) + 0.5 * cos(2y) for x in xs, y in xs]

    c0, ks0 = FFS.calculate_spectrum(g, f, (N, N); execution = PP_SER)
    c2, ks2 = FFS.calculate_spectrum(g, f, (N, N); execution = PP_SER,
        preprocess = FFS.Preprocess(; detrend = FFS.NoDetrend(), pad = 2.0))

    # Twice the modes per axis, at half the spacing, over the same wavenumber range.
    Test.@test size(c2) == FFS.Packing.packed_size((2N, 2N), Val(true))
    dk0 = ks0[1][2] - ks0[1][1]
    dk2 = ks2[1][2] - ks2[1][1]
    Test.@test isapprox(dk2, dk0 / 2; rtol = 1e-12)
    Test.@test isapprox(maximum(abs, ks2[1]), maximum(abs, ks0[1]); rtol = 0.05)

    # The padded grid holds the original samples and zeros beyond them.
    gp, fp, sc = FFS.preprocess_field(g, f, FFS.Preprocess(; detrend = FFS.NoDetrend(), pad = 2.0))
    Test.@test sc == 2.0
    Test.@test size(gp) == (2N, 2N)
    Test.@test fp[1:N, 1:N] == f
    Test.@test all(iszero, fp[(N + 1):end, :])
    Test.@test isapprox(FG.Grids.period(gp, 1), 2 * L; rtol = 1e-12)

    # Zero padding extends an axis at its own spacing, so a stretched axis raises.
    gs = FG.Grids.StructuredGrid(FG.Geometry.CartesianGeometry{Float64}(),
        [L * (i - 1) / N + 0.05 * L * sinpi(2 * (i - 1) / N) for i in 1:N], pp_unif(L, N);
        periodic = (true, true), period = (L, L))
    Test.@test_throws ArgumentError FFS.preprocess_field(gs, f,
        FFS.Preprocess(; detrend = FFS.NoDetrend(), pad = 2.0))

    # `pad < 1` is rejected at construction.
    Test.@test_throws ArgumentError FFS.Preprocess(; pad = 0.5)
end

Test.@testset "SpectralConvention is read by the reductions" begin
    # `SpectralConvention` was exported and documented while the reductions read none of it. The default
    # reproduces the previous numbers exactly; `PowerScaling` drops the `1/dk`.
    Random.seed!(31)
    L = 2π
    N = 16
    g = pp_cart(L, N)
    xs = pp_unif(L, N)
    f = [sin(3x) + 0.5 * cos(2y) for x in xs, y in xs]
    c, ks = FFS.calculate_spectrum(g, f, (N, N); execution = PP_SER)

    kb0, E0 = FFS.isotropic_spectrum(ks, c)
    _, E1 = FFS.isotropic_spectrum(ks, c; convention = FFS.SpectralConvention())
    Test.@test E0 == E1

    dk = kb0[2] - kb0[1]
    _, Ep = FFS.isotropic_spectrum(ks, c;
        convention = FFS.SpectralConvention(; scaling = FFS.PowerScaling()))
    Test.@test isapprox(Ep, E0 .* dk; rtol = 1e-12)
    # Per-bin power sums to the folded variance (the field has zero mean).
    Test.@test isapprox(sum(Ep), 0.5 * sum(abs2, f) / length(f); rtol = 1e-10)

    # The in-place form honours it too.
    Ei = zeros(length(kb0))
    kbi = zeros(length(kb0))
    FFS.isotropic_spectrum!(Ei, kbi, ks, c;
        convention = FFS.SpectralConvention(; scaling = FFS.PowerScaling()))
    Test.@test isapprox(Ei, Ep; rtol = 1e-12)
    FFS.isotropic_spectrum!(Ei, kbi, ks, c)
    Test.@test isapprox(Ei, E0; rtol = 1e-12)

    # A radial bin sums over every direction, so it has no signed axis: `TwoSided` names a spectrum this
    # reduction does not produce, and says so.
    Test.@test_throws ArgumentError FFS.isotropic_spectrum(ks, c;
        convention = FFS.SpectralConvention(; sided = FFS.TwoSided()))
    Test.@test_throws ArgumentError FFS.isotropic_spectrum!(Ei, kbi, ks, c;
        convention = FFS.SpectralConvention(; sided = FFS.TwoSided()))
end

Test.@testset "Preprocessing composes with the batch axis and the backends" begin
    Random.seed!(9092)
    L = 2π
    N = 16
    g = pp_cart(L, N)
    xs = pp_unif(L, N)
    fb = cat([1.0 + sin(3x) + 0.2y for x in xs, y in xs],
        [2.0 + cos(2x) for x in xs, y in xs]; dims = 3)
    spec = FFS.Preprocess(; detrend = FFS.Demean(), window = FFS.Hann())

    c, _ = FFS.calculate_spectrum(g, fb, (N, N); preprocess = spec, execution = PP_SER)
    Test.@test size(c) == (FFS.Packing.packed_size((N, N), Val(true))..., 2)
    # Detrending runs before the taper, and a taper reweights the samples, so a demeaned field carries a
    # small mean again afterwards. What the pairing gives is a DC mode far below the untrended one.
    cw, _ = FFS.calculate_spectrum(g, fb, (N, N); execution = PP_SER,
        preprocess = FFS.Preprocess(; detrend = FFS.NoDetrend(), window = FFS.Hann()))
    # Each slice is detrended and tapered on its own.
    for b in 1:2
        c1, _ = FFS.calculate_spectrum(g, fb[:, :, b], (N, N); preprocess = spec, execution = PP_SER)
        Test.@test maximum(abs, c[:, :, b] .- c1) < 1e-12
        Test.@test abs(c[1, 1, b]) < 0.05 * abs(cw[1, 1, b])
    end

    # The same spec through a different transform gives the same coefficients.
    cf, _ = FFS.calculate_spectrum(g, fb, (N, N); preprocess = spec,
        transform = SB.FFTSpectralBackend(), execution = PP_SER)
    Test.@test maximum(abs, cf .- c) < 1e-11

    # The caller's array is never mutated.
    keep = copy(fb)
    FFS.calculate_spectrum(g, fb, (N, N); preprocess = spec, execution = PP_SER)
    Test.@test fb == keep
end
