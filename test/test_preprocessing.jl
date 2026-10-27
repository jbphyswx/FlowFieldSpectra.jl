# Windowing, detrending and zero padding through `calculate_spectrum(...; preprocess = ...)`. A window is
# scaled to unit mean square under the grid's measure, so the windowed field keeps the variance of the
# original and Parseval holds on the coefficients with nothing applied afterwards.

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
const PP_CART = FG.Geometry.CartesianGeometry{Float64}()

pp_unif(L, N) = range(0.0, L; length = N + 1)[1:N]
pp_cart(L, N) = FG.Grids.StructuredGrid(PP_CART, pp_unif(L, N), pp_unif(L, N);
    topology = (FG.Grids.Periodic(), FG.Grids.Periodic()), period = (L, L))
pp_str(L, N, a) = [L * (i - 1) / N + a * L / N * sinpi(2 * (i - 1) / N) for i in 1:N]

# The Hann window at each point's fractional position across the extent of `x` and `y`.
pp_hann(x, y) = sinpi.((x .- minimum(x)) ./ (maximum(x) - minimum(x))) .^ 2 .*
                sinpi.((y .- minimum(y)) ./ (maximum(y) - minimum(y))) .^ 2

# The residual of the least-squares fit of `A`'s columns to `f` under the weights `w`, solved by QR.
pp_lsq_residual(A, f, w) = f .- A * ((sqrt.(w) .* A) \ (sqrt.(w) .* f))

Test.@testset "A window is evaluated at each point's position and keeps the variance" begin
    Random.seed!(9090)
    L = 2π
    N = 16
    g = pp_cart(L, N)
    xs = pp_unif(L, N)
    f = [sin(3x) * cos(2y) + 0.4 * sin(x + y) for x in xs, y in xs]
    for win in (FFS.NoWindow(), FFS.Hann(), FFS.Hamming(), FFS.Blackman(), FFS.Tukey(0.4))
        spec = FFS.Preprocess(; detrend = FFS.NoDetrend(), window = win)
        c, ks = FFS.calculate_spectrum(g, f, (N, N); preprocess = spec, execution = PP_SER)
        _, ftap, _ = FFS.preprocess_field(g, f, spec)
        # On a uniform grid the window is the tensor product of the per-axis tapers, each of unit mean
        # square, and Parseval holds on the coefficients of the windowed field.
        w = FFS.Preprocessing.axis_taper(Float64, win, N)
        Test.@test isapprox(sum(abs2, w) / N, 1.0; rtol = 1e-12)
        Test.@test maximum(abs, ftap .- f .* (w * w')) < 1e-14 * maximum(abs, f)
        Test.@test isapprox(parseval(ks, c), sum(abs2, ftap) / length(ftap); rtol = 1e-12)
    end

    # On a stretched grid and a node cloud the window reads each point's position, and its mean square
    # under the measure is one.
    xst = pp_str(L, 16, 0.3)
    yst = [L * ((j - 1) / 12)^1.5 for j in 1:12]
    gs = FG.Grids.StructuredGrid(PP_CART, xst, yst)
    M = 60
    xc, yc = L .* rand(M), L .* rand(M)
    wc = 0.5 .+ rand(M)
    gc = FG.Grids.UnstructuredGrid(PP_CART, (xc, yc), wc; topology = (FG.Grids.Periodic(), FG.Grids.Periodic()), period = (L, L))
    hann = FFS.Preprocess(; detrend = FFS.NoDetrend(), window = FFS.Hann())
    for (gg, x, y, m) in ((gs, vec([x for x in xst, y in yst]), vec([y for x in xst, y in yst]),
                           vec(FG.Grids.measure_array(gs))), (gc, xc, yc, wc))
        _, ft, _ = FFS.preprocess_field(gg, ones(size(gg)...), hann)
        h = pp_hann(x, y)
        inner = h .> 1e-3
        r = vec(ft)[inner] ./ h[inner]
        Test.@test maximum(r) - minimum(r) < 1e-12 * maximum(r)
        Test.@test sum(m .* vec(ft) .^ 2) / sum(m) ≈ 1.0 rtol = 1e-12
    end
end

Test.@testset "Both detrends are least-squares fits under the transform's weights" begin
    Random.seed!(9091)
    L = 2π
    N = 16
    xs = pp_unif(L, N)
    demean = FFS.Preprocess(; detrend = FFS.Demean())
    linear = FFS.Preprocess(; detrend = FFS.LinearDetrend())

    # The DC coefficient is the measure-weighted mean, so `Demean` zeroes it on a uniform grid, a
    # stretched grid, a node cloud of unequal measures, and a sphere under its quadrature: equal-measure
    # nodes and the Gaussian weights of a Gauss–Legendre grid.
    gu = pp_cart(L, N)
    xst = pp_str(L, N, 0.25)
    gs = FG.Grids.StructuredGrid(PP_CART, xst, xs; topology = (FG.Grids.Periodic(), FG.Grids.Periodic()), period = (L, L))
    M = 60
    xc, yc = L .* rand(M), L .* rand(M)
    wc = 0.5 .+ rand(M)
    gc = FG.Grids.UnstructuredGrid(PP_CART, (xc, yc), wc; topology = (FG.Grids.Periodic(), FG.Grids.Periodic()), period = (L, L))
    for (g, f) in ((gu, [5.0 + 2.0 * x + 0.5 * sin(3y) for x in xs, y in xs]),
                   (gs, [3.0 + sin(x) + 0.5 * cos(2y) + x^2 / 10 for x in xst, y in xs]),
                   (gc, 3.0 .+ randn(M)))
        c, _ = FFS.calculate_spectrum(g, f, (8, 8); preprocess = demean, transform = PP_DS, execution = PP_SER)
        Test.@test abs(c[1, 1]) < 1e-13
    end
    Ns = 80
    ga = π * (3 - sqrt(5))
    gsc = FG.Grids.UnstructuredGrid(FG.Geometry.SphericalGeometry(1.0),
        ([mod(ga * i, 2π) for i in 1:Ns], asin.([-1 + 2 * (i - 0.5) / Ns for i in 1:Ns])), 0.5 .+ rand(Ns))
    ggl = FG.Connectivity.structured_grid(Float64, FG.SphericalSampling.GaussLegendreSampling(), 6)
    for (g, ms) in ((gsc, (4, 7)), (ggl, (6, 11)))
        c, _ = FFS.calculate_spectrum(g, 2.0 .+ randn(size(g)...), ms; preprocess = demean, transform = PP_DS,
            execution = PP_SER)
        Test.@test abs(c[FFS.sph_mode_index(0, 0)]) < 1e-13
    end
    # `sin(3y)` has zero mean over the period, so `Demean` returns it unchanged.
    fsin = [sin(3y) for x in xs, y in xs]
    Test.@test maximum(abs, FFS.preprocess_field(gu, fsin, demean)[2] .- fsin) < 1e-14

    # `LinearDetrend` removes the weighted least-squares plane `a + b·x` in the point coordinates: a plane
    # exactly, and from any other field the residual QR leaves, on a stretched grid and a node cloud.
    yst = [L * ((j - 1) / 12)^1.5 for j in 1:12]
    gs2 = FG.Grids.StructuredGrid(PP_CART, pp_str(L, 16, 0.3), yst)
    xv = vec([x for x in pp_str(L, 16, 0.3), y in yst])
    yv = vec([y for x in pp_str(L, 16, 0.3), y in yst])
    for (g, x, y, w) in ((gs2, xv, yv, vec(FG.Grids.measure_array(gs2))), (gc, xc, yc, wc))
        sz = size(g)
        plane = reshape(2.0 .+ 0.7 .* x .- 1.3 .* y, sz)
        Test.@test maximum(abs, FFS.preprocess_field(g, plane, linear)[2]) < 1e-12 * maximum(abs, plane)
        fn = reshape(x .^ 3 .+ y .^ 3 .+ sin.(x .* y), sz)
        r = FFS.preprocess_field(g, fn, linear)[2]
        Test.@test maximum(abs, vec(r) .- pp_lsq_residual([ones(length(x)) x y], vec(fn), w)) <
                   1e-10 * maximum(abs, fn)
    end
    # On a uniform grid symmetric about the origin, `cos(3y)` is orthogonal to `1`, `x` and `y` (`N` is not
    # a multiple of 3, and the grid is even in `y`), so `LinearDetrend` returns it unchanged.
    xcen = [-π + (j - 0.5) * 2π / N for j in 1:N]
    gcen = FG.Grids.StructuredGrid(PP_CART, xcen, xcen; topology = (FG.Grids.Periodic(), FG.Grids.Periodic()), period = (L, L))
    fcos = [cos(3y) for x in xcen, y in xcen]
    Test.@test maximum(abs, FFS.preprocess_field(gcen, fcos, linear)[2] .- fcos) < 1e-14

    # On a sphere the coordinates are the unit vector `n`, whose components span the degree-1 harmonics:
    # `a + b·n` is removed exactly, `P₂(cos θ)` is orthogonal to it under the Gauss–Legendre quadrature and
    # returned unchanged, and any field's residual has no coefficient of degree `l ≤ 1`.
    gsph = FG.Connectivity.structured_grid(Float64, FG.SphericalSampling.GaussLegendreSampling(), 5)
    θ, λ = FFS.Grids._sph_points(gsph)
    ex, ey, ez = sin.(θ) .* cos.(λ), sin.(θ) .* sin.(λ), cos.(θ)
    f1 = reshape(1.5 .+ 0.3 .* ex .- 0.8 .* ey .+ 0.6 .* ez, size(gsph))
    Test.@test maximum(abs, FFS.preprocess_field(gsph, f1, linear)[2]) < 1e-13
    p2 = reshape((3 .* ez .^ 2 .- 1) ./ 2, size(gsph))
    Test.@test maximum(abs, FFS.preprocess_field(gsph, p2, linear)[2] .- p2) < 1e-13
    c, _ = FFS.calculate_spectrum(gsph, randn(size(gsph)...), (5, 9); preprocess = linear, transform = PP_DS,
        execution = PP_SER)
    Test.@test maximum(abs, [c[FFS.sph_mode_index(l, m)] for l in 0:1 for m in -l:l]) < 1e-13
end

Test.@testset "Zero padding lengthens the domain with zeros" begin
    L = 2π
    N = 16
    g = pp_cart(L, N)
    xs = pp_unif(L, N)
    f = [sin(3x) + 0.5 * cos(2y) for x in xs, y in xs]
    nopad = FFS.Preprocess(; detrend = FFS.NoDetrend(), pad = 2.0)

    # A uniform grid gains zero samples at its own spacing: twice the modes per axis at half the spacing,
    # over the same wavenumber range.
    c0, ks0 = FFS.calculate_spectrum(g, f, (N, N); execution = PP_SER)
    c2, ks2 = FFS.calculate_spectrum(g, f, (N, N); execution = PP_SER, preprocess = nopad)
    Test.@test isapprox(ks2[1][2] - ks2[1][1], (ks0[1][2] - ks0[1][1]) / 2; rtol = 1e-12)
    Test.@test isapprox(maximum(abs, ks2[1]), maximum(abs, ks0[1]); rtol = 1e-12)
    gp, fp, _ = FFS.preprocess_field(g, f, nopad)
    Test.@test fp[1:N, 1:N] == f
    Test.@test all(iszero, fp[(N + 1):end, :]) && all(iszero, fp[:, (N + 1):end])
    Test.@test isapprox(FG.Grids.period(gp, 1), 2 * L; rtol = 1e-12)

    # The same points as a node cloud take no new samples, and give the coefficients the zeros do at every
    # frequency of the padded grid's box. The cloud's box widens its even axis 2 to both `±N`, which the
    # points sample alike.
    cloud = FG.Grids.UnstructuredGrid(PP_CART, FG.Grids.materialize(g), vec(FG.Grids.measure_array(g)))
    cz, _ = FFS.calculate_spectrum(g, f, (N, N); preprocess = nopad, transform = PP_DS, execution = PP_SER)
    cp, _ = FFS.calculate_spectrum(cloud, vec(f), (N, N); preprocess = nopad, transform = PP_DS,
        execution = PP_SER)
    Np = 2N
    Test.@test size(cp) == (N + 1, Np + 1)
    at = [k >= 0 ? k + 1 : Np + k + 2 for k in Int.(FFTW.fftfreq(Np, Np))]
    Test.@test maximum(abs, cp[:, at] .- cz) < 1e-13
    Test.@test maximum(abs, cp[:, N + 1] .- cp[:, N + 2]) < 1e-13

    # A stretched grid is padded the same way: its coefficients are the measure-weighted sum
    # `Σ w f e^{-ik·(x - x₀)} / (pad² Σ w)` on wavenumbers `2πn / (pad L)`, evaluated here directly.
    xst = pp_str(L, N, 0.05)
    gs = FG.Grids.StructuredGrid(PP_CART, xst, xs; topology = (FG.Grids.Periodic(), FG.Grids.Periodic()), period = (L, L))
    fs = [sin(3x) + 0.5 * cos(2y) for x in xst, y in xs]
    cs, kss = FFS.calculate_spectrum(gs, fs, (N, N); preprocess = nopad, transform = PP_DS, execution = PP_SER)
    Test.@test isapprox(kss[1][2] - kss[1][1], 2π / (2L); rtol = 1e-12)
    wst = FG.Grids.measure_array(gs)
    ref = [sum(wst[i, j] * fs[i, j] * cis(-(kss[1][a] * (xst[i] - xst[1]) + kss[2][b] * xs[j]))
               for i in 1:N, j in 1:N) / (4 * sum(wst)) for a in axes(cs, 1), b in axes(cs, 2)]
    Test.@test maximum(abs, cs .- ref) < 1e-13
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
    keep = copy(fb)
    c, _ = FFS.calculate_spectrum(g, fb, (N, N); preprocess = spec, execution = PP_SER)
    Test.@test fb == keep                                     # the caller's array is never mutated
    # Each slice is detrended and windowed on its own.
    for b in 1:2
        c1, _ = FFS.calculate_spectrum(g, fb[:, :, b], (N, N); preprocess = spec, execution = PP_SER)
        Test.@test maximum(abs, c[:, :, b] .- c1) < 1e-12
    end
    # The same spec through a different transform gives the same coefficients.
    cf, _ = FFS.calculate_spectrum(g, fb, (N, N); preprocess = spec,
        transform = SB.FFTSpectralBackend(), execution = PP_SER)
    Test.@test maximum(abs, cf .- c) < 1e-12
end
