# The inverse's reusable pair: `plan_synthesis` once, `synthesize!` per coefficient set. Allocation is
# gated in `test_allocs.jl`.

using Test: Test
using Random: Random
using FFTW: FFTW
using FINUFFT: FINUFFT
using NonuniformFFTs: NonuniformFFTs
using NUFSHT: NUFSHT
using KernelAbstractions: KernelAbstractions as KA
using OhMyThreads: OhMyThreads
using FlowFieldSpectra: FlowFieldSpectra as FFS
using FlowTransformBindings: FlowTransformBindings as FTB
using FlowGeometries: FlowGeometries as FG
using SpectralBackends: SpectralBackends as SB
using ComputationalBackends: ComputationalBackends as CB

const SP_SER = CB.SerialBackend()
sp_cart() = FG.Geometry.CartesianGeometry{Float64}()
sp_uni(L, N) = range(0.0, L; length = N + 1)[1:N]
sp_str(L, N) = collect(sp_uni(L, N)) .+ (0.03 * L / N) .* sinpi.(2 .* (0:(N - 1)) ./ N)

# The plan against the allocating `synthesize`, executed twice, leaving the caller's coefficients intact.
function sp_agrees(g, f, ms, T; batch = (), kw...)
    c, _ = FFS.calculate_spectrum(g, f, ms; kw...)
    ref = FFS.synthesize(T, g, c, ms; kw...)
    p = FFS.plan_synthesis(g, T, ms; batch = batch, kw...)
    out = FFS.allocate_field(p)
    keep = copy(c)
    FFS.synthesize!(out, p, c)
    Test.@test isapprox(out, ref; rtol = 1e-8)
    Test.@test c == keep
    first = copy(out)
    FFS.synthesize!(out, p, c)
    Test.@test out == first
    return p
end

Test.@testset "Cartesian synthesis plans" begin
    Random.seed!(81)
    L = 2π
    N = 8
    gu = FG.Grids.StructuredGrid(sp_cart(), sp_uni(L, N), sp_uni(L, N);
        topology = (FG.Grids.Periodic(), FG.Grids.Periodic()), period = (L, L))
    gs = FG.Grids.StructuredGrid(sp_cart(), sp_str(L, N), sp_str(L, N);
        topology = (FG.Grids.Periodic(), FG.Grids.Periodic()), period = (L, L))

    Test.@testset "FFTW" begin
        for (f, batch, T) in ((randn(N, N), (), Float64), (randn(N, N, 3), (3,), Float64),
                              (randn(ComplexF64, N, N), (), ComplexF64))
            sp_agrees(gu, f, (N, N), T; batch = batch, transform = SB.FFTSpectralBackend())
        end
    end

    Test.@testset "direct sum" begin
        # A tensor grid's series contracts one axis at a time; a node grid over the same points sums it
        # point by point, both over the same coefficient box. The two agree on even and odd resolutions.
        gp = FG.Grids.UnstructuredGrid(sp_cart(), FG.Grids.materialize(gs), vec(FG.Grids.measure_array(gs));
            topology = (FG.Grids.Periodic(), FG.Grids.Periodic()), period = (L, L))
        for (f, ms) in ((randn(N, N, 2), (N - 1, N + 2)), (randn(N, N), (N, N)), (randn(ComplexF64, N, N), (N, N - 1)))
            R = eltype(f) <: Real
            c, _ = FFS.calculate_spectrum(gs, f, ms; transform = SB.DirectSumSpectralBackend(), execution = SP_SER)
            ref = FFS.synthesize(eltype(f), gp, c, ms; transform = SB.DirectSumSpectralBackend(),
                execution = SP_SER)
            out = FFS.synthesize(eltype(f), gs, c, ms; transform = SB.DirectSumSpectralBackend(),
                execution = SP_SER)
            Test.@test maximum(abs, vec(out) .- vec(ref)) < 1e-12 * maximum(abs, ref)
        end
    end
end

Test.@testset "FFT synthesis against the direct-sum synthesis" begin
    # A real field over two or more dims runs a c2c along `2:D` then the c2r along dim 1; one dim and a
    # complex field run the whole backward transform. Coefficients in a view take the staged path.
    Random.seed!(87)
    L = 2π
    for ns in ((10,), (8, 6), (6, 4, 5)), T in (Float64, ComplexF64), iflag in (1, -1)
        D = length(ns)
        g = FG.Grids.StructuredGrid(sp_cart(), map(n -> sp_uni(L, n), ns)...;
            topology = ntuple(_ -> FG.Grids.Periodic(), D), period = ntuple(_ -> L, D))
        c, _ = FFS.calculate_spectrum(g, randn(T, ns..., 2), ns; transform = SB.DirectSumSpectralBackend(),
            execution = SP_SER, iflag)
        ref = FFS.synthesize(T, g, c, ns; transform = SB.DirectSumSpectralBackend(), execution = SP_SER, iflag)
        p = FFS.plan_synthesis(g, T, ns; transform = SB.FFTSpectralBackend(), batch = (2,), iflag)
        out = FFS.allocate_field(p)
        keep = copy(c)
        for coeffs in (c, view(c, ntuple(_ -> Colon(), ndims(c))...))
            FFS.synthesize!(out, p, coeffs)
            Test.@test isapprox(out, ref; rtol = 1e-12)
        end
        Test.@test c == keep
    end
end

Test.@testset "Spherical synthesis plans" begin
    Random.seed!(82)
    lmax = 4
    ms = (lmax + 1, 2 * lmax + 1)
    Ns = 80
    ga = π * (3 - sqrt(5))
    λ = [mod(ga * i, 2π) for i in 1:Ns]
    φ = asin.([-1 + 2 * (i - 0.5) / Ns for i in 1:Ns])
    gsc = FG.Grids.UnstructuredGrid(FG.Geometry.SphericalGeometry(1.0), (λ, φ), fill(4π / Ns, Ns))
    for (f, batch) in ((randn(Ns), ()), (randn(Ns, 2), (2,)))
        sp_agrees(gsc, f, ms, Float64; batch = batch, transform = SB.NUFSHTSpectralBackend())
    end
end

Test.@testset "Device synthesis plans" begin
    Random.seed!(85)
    dev = CB.GPUBackend(KA.CPU())
    L = 2π
    N = 8
    gu = FG.Grids.StructuredGrid(sp_cart(), sp_uni(L, N), sp_uni(L, N);
        topology = (FG.Grids.Periodic(), FG.Grids.Periodic()), period = (L, L))
    for (f, batch, T) in ((randn(N, N), (), Float64), (randn(N, N, 2), (2,), Float64),
                          (randn(ComplexF64, N, N), (), ComplexF64))
        sp_agrees(gu, f, (N, N), T; batch = batch,
            transform = SB.FFTSpectralBackend(), execution = dev)
    end
end

Test.@testset "The threaded direct sum on point sets and spheres equals the serial one" begin
    # The OhMyThreads extension splits point clouds and per-point spheres over blocks of points with its
    # own kernels; the serial direct sum is a separate implementation in `src`.
    Random.seed!(86)
    L = 2π
    M = 70
    gc = FG.Grids.UnstructuredGrid(sp_cart(), (L .* rand(M), L .* rand(M)), 0.5 .+ rand(M);
        topology = (FG.Grids.Periodic(), FG.Grids.Periodic()), period = (L, L))
    lmax = 5
    sms = (lmax + 1, 2 * lmax + 1)
    Ns = 90
    ga = π * (3 - sqrt(5))
    gsc = FG.Grids.UnstructuredGrid(FG.Geometry.SphericalGeometry(1.0),
        ([mod(ga * i, 2π) for i in 1:Ns], asin.([-1 + 2 * (i - 0.5) / Ns for i in 1:Ns])), 0.5 .+ rand(Ns))
    ds = SB.DirectSumSpectralBackend()
    close_to(a, b) = maximum(abs, a .- b) <= 1e-12 * maximum(abs, b)
    for (g, ms, fields) in ((gc, (8, 9), (randn(M), randn(M, 2), randn(ComplexF64, M))),
                            (gsc, sms, (randn(Ns), randn(ComplexF64, Ns))))
        for f in fields
            R = eltype(f) <: Real
            cs, _ = FFS.calculate_spectrum(g, f, ms; transform = ds, execution = SP_SER)
            ct, _ = FFS.calculate_spectrum(g, f, ms; transform = ds, execution = CB.ThreadedBackend())
            Test.@test close_to(ct, cs)
            ss = FFS.synthesize(eltype(f), g, cs, ms; transform = ds, execution = SP_SER)
            st = FFS.synthesize(eltype(f), g, cs, ms; transform = ds, execution = CB.ThreadedBackend())
            Test.@test close_to(st, ss)
        end
    end
end

Test.@testset "A synthesis plan is fixed to its shape" begin
    Random.seed!(83)
    L = 2π
    N = 8
    g = FG.Grids.StructuredGrid(sp_cart(), sp_uni(L, N), sp_uni(L, N);
        topology = (FG.Grids.Periodic(), FG.Grids.Periodic()), period = (L, L))
    p = FFS.plan_synthesis(g, Float64, (N, N); transform = SB.FFTSpectralBackend())
    out = FFS.allocate_field(p)
    c, _ = FFS.calculate_spectrum(g, randn(N, N), (N, N); transform = SB.FFTSpectralBackend())

    Test.@test_throws DimensionMismatch FFS.synthesize!(zeros(Float64, N + 1, N), p, c)
    Test.@test_throws DimensionMismatch FFS.synthesize!(out, p, zeros(ComplexF64, N, N))

    # A round trip through both plans returns the field the forward consumed.
    f = randn(N, N)
    fp = FFS.plan_spectrum(g, Float64, (N, N); transform = SB.FFTSpectralBackend())
    coeffs = FFS.allocate_coefficients(fp)
    FFS.calculate_spectrum!(coeffs, fp, f)
    FFS.synthesize!(out, p, coeffs)
    Test.@test isapprox(out, f; rtol = 1e-12)
end

Test.@testset "Synthesis is the adjoint of the forward under the grid's quadrature" begin
    # `Σ_k conj(C_k)·F(f)_k = Σ_j w_j conj(S(C)_j) f_j`, with `w_j` the grid measure normalized to sum
    # to 1 on a Cartesian grid and to `4π` on a sphere. The grids carry varying measures.
    Random.seed!(84)
    L = 2π
    N = 8
    ms = (N, N)
    gs = FG.Grids.StructuredGrid(sp_cart(), sp_str(L, N), sp_str(L, N);
        topology = (FG.Grids.Periodic(), FG.Grids.Periodic()), period = (L, L))
    M = 50
    gc = FG.Grids.UnstructuredGrid(sp_cart(), (L .* rand(M), L .* rand(M)), 0.5 .+ rand(M);
        topology = (FG.Grids.Periodic(), FG.Grids.Periodic()), period = (L, L))
    ds = (; transform = SB.DirectSumSpectralBackend(), execution = SP_SER)
    for (g, kw, tol) in ((gs, ds, 1e-12), (gc, ds, 1e-12),
                         (gc, (; transform = FTB.FINUFFTBackend(), eps = 1e-13), 1e-9),
                         (gc, (; transform = FTB.NonuniformFFTsBackend(), eps = 1e-13), 1e-9))
        f = randn(ComplexF64, size(g)...)
        C = randn(ComplexF64, ms...)
        F, _ = FFS.calculate_spectrum(g, f, ms; kw...)
        S = FFS.synthesize(ComplexF64, g, C, ms; kw...)
        w = vec(FG.Grids.measure_array(g))
        w = w ./ sum(w)
        Test.@test isapprox(sum(conj.(C) .* F), sum(w .* conj.(vec(S)) .* vec(f)); rtol = tol)
    end

    lmax = 4
    sms = (lmax + 1, 2 * lmax + 1)
    Ns = 80
    ga = π * (3 - sqrt(5))
    λ = [mod(ga * i, 2π) for i in 1:Ns]
    φ = asin.([-1 + 2 * (i - 0.5) / Ns for i in 1:Ns])
    gsc = FG.Grids.UnstructuredGrid(FG.Geometry.SphericalGeometry(1.0), (λ, φ), 0.5 .+ rand(Ns))
    for (kw, tol) in ((ds, 1e-12), ((; transform = SB.NUFSHTSpectralBackend(), tol = 1e-13), 1e-9))
        f = randn(Ns)
        C = zeros(ComplexF64, sms...)
        for l in 0:lmax, m in -l:l
            C[FFS.sph_mode_index(l, m)] = randn(ComplexF64)
        end
        F, _ = FFS.calculate_spectrum(gsc, f, sms; kw...)
        S = FFS.synthesize(ComplexF64, gsc, C, sms; kw...)
        m = FG.Grids.measure_array(gsc)
        w = 4π .* m ./ sum(m)
        Test.@test isapprox(sum(conj.(C) .* F), sum(w .* conj.(S) .* f); rtol = tol)
    end
end
