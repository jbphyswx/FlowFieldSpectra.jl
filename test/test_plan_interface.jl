# A plan answers for its own output: `coefficient_size`, `coefficient_type`, `wavenumbers`.
#
# `calculate_spectrum!(coeffs, plan, field)` needs a preallocated `coeffs`, and the shape and element
# type are backend-specific — a real Cartesian field's spectrum is the packed half and a complex one's
# the full native cube, while a spherical plan's coefficients are `(lmax+1, 2lmax+1, batch…)` and REAL
# for a real field. The reference these testsets hold every accessor to is the allocating
# `calculate_spectrum` on the same grid and field: whatever it returns is what the plan must promise.

using Test: Test
using Random: Random
using FFTW: FFTW
using FINUFFT: FINUFFT
using NonuniformFFTs: NonuniformFFTs
using FastSphericalHarmonics: FastSphericalHarmonics
using NUFSHT: NUFSHT
using KernelAbstractions: KernelAbstractions as KA
using FlowFieldSpectra: FlowFieldSpectra as FFS
using FlowGeometries: FlowGeometries as FG
using SpectralBackends: SpectralBackends as SB
using ComputationalBackends: ComputationalBackends as CB

const PI_SER = CB.SerialBackend()
pi_cart() = FG.Geometry.CartesianGeometry{Float64}()
pi_uni(L, N) = range(0.0, L; length = N + 1)[1:N]
pi_str(L, N) = collect(pi_uni(L, N)) .+ (0.03 * L / N) .* sinpi.(2 .* (0:(N - 1)) ./ N)

# The plan's promises against what the allocating call actually returns.
function pi_agrees(name, g, f, ms; batch = (), kw...)
    T = eltype(f) <: Real ? Float64 : ComplexF64
    c, ks = FFS.calculate_spectrum(g, f, ms; kw...)
    p = FFS.plan_spectrum(g, T, ms; batch = batch, kw...)

    Test.@test FFS.coefficient_size(p) == size(c)
    Test.@test FFS.coefficient_type(p) === eltype(c)

    kp = FFS.wavenumbers(p)
    Test.@test length(kp) == length(ks)
    for d in eachindex(ks)
        Test.@test collect(kp[d]) ≈ collect(ks[d])
    end

    # `allocate_coefficients` is the point of the pair: a plan alone is enough to preallocate against,
    # with no reach into `Packing` and no throwaway allocating call to learn the shape.
    buf = FFS.allocate_coefficients(p)
    Test.@test size(buf) == size(c)
    Test.@test eltype(buf) === eltype(c)
    Test.@test all(iszero, buf)
    FFS.calculate_spectrum!(buf, p, f)
    Test.@test isapprox(buf, c; rtol = 1e-8)
    return nothing
end

Test.@testset "A Cartesian plan answers for its own coefficients" begin
    Random.seed!(71)
    L = 2π
    N = 8
    gu = FG.Grids.StructuredGrid(pi_cart(), pi_uni(L, N), pi_uni(L, N);
        periodic = (true, true), period = (L, L))
    gs = FG.Grids.StructuredGrid(pi_cart(), pi_str(L, N), pi_str(L, N);
        periodic = (true, true), period = (L, L))
    gh = FG.Grids.StructuredGrid(pi_cart(), pi_uni(L, N), pi_str(L, N);
        periodic = (true, true), period = (L, L))
    M = 60
    gc = FG.Grids.UnstructuredGrid(pi_cart(), (L .* rand(M), L .* rand(M)), fill(L^2 / M, M);
        periodic = (true, true), period = (L, L))

    # FFTW, the hybrid composite, both NUFFT providers, and the direct sum — real, complex and batched.
    for (nm, g, f, batch, kw) in (
            ("fftw real", gu, randn(N, N), (), (; transform = SB.FFTSpectralBackend())),
            ("fftw batched", gu, randn(N, N, 3), (3,), (; transform = SB.FFTSpectralBackend())),
            ("fftw complex", gu, randn(ComplexF64, N, N), (), (; transform = SB.FFTSpectralBackend())),
            ("hybrid real", gh, randn(N, N), (), (; transform = SB.FFTSpectralBackend())),
            ("hybrid batched", gh, randn(N, N, 2), (2,), (; transform = SB.FFTSpectralBackend())),
            ("hybrid complex", gh, randn(ComplexF64, N, N), (), (; transform = SB.FFTSpectralBackend())),
            ("nufft cloud", gc, randn(M), (), (; transform = FFS.NonuniformFFTsBackend())),
            ("nufft cloud batched", gc, randn(M, 3), (3,), (; transform = FFS.NonuniformFFTsBackend())),
            ("nufft cloud complex", gc, randn(ComplexF64, M), (), (; transform = FFS.NonuniformFFTsBackend())),
            ("nufft separable", gs, randn(N, N), (), (; transform = FFS.NonuniformFFTsBackend())),
            ("finufft cloud", gc, randn(M), (), (; transform = FFS.FINUFFTBackend())),
            ("finufft separable", gs, randn(N, N), (), (; transform = FFS.FINUFFTBackend())),
            ("finufft complex", gc, randn(ComplexF64, M), (), (; transform = FFS.FINUFFTBackend())),
            ("directsum tensor", gs, randn(N, N), (), (; transform = SB.DirectSumSpectralBackend(), execution = PI_SER)),
            ("directsum cloud", gc, randn(M), (), (; transform = SB.DirectSumSpectralBackend(), execution = PI_SER)),
            ("directsum batched", gs, randn(N, N, 2), (2,), (; transform = SB.DirectSumSpectralBackend(), execution = PI_SER)),
            ("directsum complex", gs, randn(ComplexF64, N, N), (), (; transform = SB.DirectSumSpectralBackend(), execution = PI_SER)))
        Test.@testset "$nm" begin
            pi_agrees(nm, g, f, (N, N); batch = batch, kw...)
        end
    end

    # A halved axis carries its Nyquist twin, so the axes a reduction needs come from the plan too.
    p = FFS.plan_spectrum(gs, Float64, (N, N); transform = SB.DirectSumSpectralBackend(), execution = PI_SER)
    Test.@test FFS.Packing.axis_twin(FFS.wavenumbers(p)[1]) !== nothing
end

Test.@testset "A spherical plan reports real coefficients for a real field" begin
    Random.seed!(72)
    lmax = 4
    ms = (lmax + 1, 2 * lmax + 1)
    ggl = FG.Connectivity.structured_grid(Float64, FG.SphericalSampling.GaussLegendreSampling(), lmax + 1)
    Ns = 80
    ga = π * (3 - sqrt(5))
    λ = [mod(ga * i, 2π) for i in 1:Ns]
    φ = asin.([-1 + 2 * (i - 0.5) / Ns for i in 1:Ns])
    gsc = FG.Grids.UnstructuredGrid(FG.Geometry.SphericalGeometry(1.0), (λ, φ), fill(4π / Ns, Ns))

    for (nm, g, f, batch, kw) in (
            ("directsum real", ggl, randn(size(ggl)...), (), (; transform = SB.DirectSumSpectralBackend(), execution = PI_SER)),
            ("directsum batched", ggl, randn(size(ggl)..., 2), (2,), (; transform = SB.DirectSumSpectralBackend(), execution = PI_SER)),
            ("directsum complex", ggl, randn(ComplexF64, size(ggl)...), (), (; transform = SB.DirectSumSpectralBackend(), execution = PI_SER)),
            ("nufsht real", gsc, randn(Ns), (), (; transform = SB.NUFSHTSpectralBackend())),
            ("nufsht batched", gsc, randn(Ns, 2), (2,), (; transform = SB.NUFSHTSpectralBackend())),
            ("nufsht complex", gsc, randn(ComplexF64, Ns), (), (; transform = SB.NUFSHTSpectralBackend())))
        Test.@testset "$nm" begin
            pi_agrees(nm, g, f, ms; batch = batch, kw...)
        end
    end

    # The real/complex split is the part a caller cannot derive from the size, so assert it directly.
    pr = FFS.plan_spectrum(ggl, Float64, ms; transform = SB.DirectSumSpectralBackend(), execution = PI_SER)
    pc = FFS.plan_spectrum(ggl, ComplexF64, ms; transform = SB.DirectSumSpectralBackend(), execution = PI_SER)
    Test.@test FFS.coefficient_type(pr) === Float64
    Test.@test FFS.coefficient_type(pc) === ComplexF64
    Test.@test FFS.coefficient_size(pr) == FFS.coefficient_size(pc)
    Test.@test sizeof(FFS.allocate_coefficients(pr)) * 2 == sizeof(FFS.allocate_coefficients(pc))
end

Test.@testset "A device plan answers the same way" begin
    Random.seed!(73)
    dev = CB.GPUBackend(KA.CPU())
    L = 2π
    N = 8
    gu = FG.Grids.StructuredGrid(pi_cart(), pi_uni(L, N), pi_uni(L, N);
        periodic = (true, true), period = (L, L))
    M = 60
    gc = FG.Grids.UnstructuredGrid(pi_cart(), (L .* rand(M), L .* rand(M)), fill(L^2 / M, M);
        periodic = (true, true), period = (L, L))
    Ns = 80
    ga = π * (3 - sqrt(5))
    λ = [mod(ga * i, 2π) for i in 1:Ns]
    φ = asin.([-1 + 2 * (i - 0.5) / Ns for i in 1:Ns])
    gsc = FG.Grids.UnstructuredGrid(FG.Geometry.SphericalGeometry(1.0), (λ, φ), fill(4π / Ns, Ns))

    for (nm, g, f, batch, kw) in (
            ("gpu fft real", gu, randn(N, N), (), (; transform = SB.FFTSpectralBackend(), execution = dev)),
            ("gpu fft complex", gu, randn(ComplexF64, N, N), (), (; transform = SB.FFTSpectralBackend(), execution = dev)),
            ("gpu nufft real", gc, randn(M), (), (; transform = FFS.NonuniformFFTsBackend(), execution = dev)),
            ("gpu nufft complex", gc, randn(ComplexF64, M), (), (; transform = FFS.NonuniformFFTsBackend(), execution = dev)),
            ("gpu nufsht real", gsc, randn(Ns), (), (; transform = SB.NUFSHTSpectralBackend(), execution = dev)),
            ("gpu nufsht complex", gsc, randn(ComplexF64, Ns), (), (; transform = SB.NUFSHTSpectralBackend(), execution = dev)))
        Test.@testset "$nm" begin
            pi_agrees(nm, g, f, nm == "gpu nufsht real" || nm == "gpu nufsht complex" ? (5, 9) : (N, N);
                batch = batch, kw...)
        end
    end
end
