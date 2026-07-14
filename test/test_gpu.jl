# GPU/KernelAbstractions Tests for FlowFieldSpectra.jl
#
# CI has no GPU, so these exercise the portable KA direct-sum path on `GPUBackend(KA.CPU())` for
# parity with the serial CPU direct sum. Real-device (cuFINUFFT / CUFFT / KA-on-CUDA) numerical
# parity lives in `gpu/` and is verified only on an actual CUDA device — never on CI.

using Test: Test
using Random: Random
using Statistics: Statistics
using KernelAbstractions: KernelAbstractions as KA
using FlowFieldSpectra: FlowFieldSpectra as FFS

Test.@testset "GPU Backend Parity via KernelAbstractions.CPU()" begin
    # 1. Cartesian Parity (2D)
    Random.seed!(42)
    T = Float64
    L = 10.0
    ms = (16, 16)
    dx = L / ms[1]
    dy = L / ms[2]

    xs = range(0.0, stop = L - dx, length = ms[1])
    ys = range(0.0, stop = L - dy, length = ms[2])

    xv = vec([x for x in xs, y in ys])
    yv = vec([y for x in xs, y in ys])

    kx1, ky1 = 2π * 2 / L, 2π * 1 / L
    u = @. cos(kx1 * xv + ky1 * yv)
    v = @. sin(kx1 * xv + ky1 * yv)

    cgrid = FFS.UniformCartesianGrid((xv, yv); domain_size = (L, L))

    # Serial DirectSum
    c_cpu, k_cpu = FFS.calculate_spectrum(cgrid, (u, v), ms; transform = FFS.DirectSumBackend(), execution = FFS.SerialBackend())

    # KA CPU Backend (DirectSum on GPUBackend(KA.CPU()))
    c_ka, k_ka = FFS.calculate_spectrum(cgrid, (u, v), ms; transform = FFS.DirectSumBackend(), execution = FFS.GPUBackend(KA.CPU()))

    Test.@test isapprox(c_cpu, c_ka, atol = 1e-12)
    Test.@test all(isapprox(k_cpu[d], k_ka[d], rtol = 1e-12) for d in 1:2)

    # 1b. Cartesian parity at D = 1 and D = 3 (covers the flat-index kernel decode).
    for D in (1, 3)
        Nd = D == 3 ? 6 : 12
        Ld = 2π
        dxd = Ld / Nd
        ax = collect(range(0.0, stop = Ld - dxd, length = Nd))
        mesh = Iterators.product(ntuple(_ -> ax, D)...)
        coords = ntuple(d -> vec([pt[d] for pt in mesh]), D)
        msd = ntuple(_ -> Nd, D)
        f = [sum(cos((d + 1) * coords[d][i]) for d in 1:D) for i in 1:Nd^D]
        gd = FFS.UniformCartesianGrid(coords; domain_size = ntuple(_ -> Ld, D))
        cs, _ = FFS.calculate_spectrum(gd, (f,), msd; transform = FFS.DirectSumBackend(), execution = FFS.SerialBackend())
        cg, _ = FFS.calculate_spectrum(gd, (f,), msd; transform = FFS.DirectSumBackend(), execution = FFS.GPUBackend(KA.CPU()))
        Test.@test isapprox(cs, cg, atol = 1e-12)
    end

    # 2. Spherical Parity
    lmax = 4
    Nθ = lmax + 1
    Nφ = 2 * lmax + 1
    N_pts = 30

    # Scattered points
    Random.seed!(123)
    θ_nodes = rand(T, N_pts) .* (0.8π) .+ 0.1π
    φ_nodes = rand(T, N_pts) .* 2π
    f_val = rand(T, N_pts)

    sgrid = FFS.ScatteredSphericalGrid(θ_nodes, φ_nodes)

    # Serial DirectSum Spherical
    c_sph_cpu, k_sph_cpu =
        FFS.calculate_spectrum(sgrid, (f_val,), (Nθ, Nφ); transform = FFS.DirectSumBackend(), execution = FFS.SerialBackend())

    # KA CPU Backend Spherical
    c_sph_ka, k_sph_ka =
        FFS.calculate_spectrum(sgrid, (f_val,), (Nθ, Nφ); transform = FFS.DirectSumBackend(), execution = FFS.GPUBackend(KA.CPU()))

    Test.@test isapprox(c_sph_cpu, c_sph_ka, atol = 1e-10)
end
