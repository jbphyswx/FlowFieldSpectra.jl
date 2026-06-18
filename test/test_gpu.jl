# GPU/KernelAbstractions Tests for FlowFieldSpectra.jl

using Test: Test
using Random: Random
using Statistics: Statistics
using KernelAbstractions: KernelAbstractions as KA
using FlowFieldSpectra: FlowFieldSpectra as FFS, GPUBackend

Test.@testset "GPU Backend Parity via KernelAbstractions.CPU()" begin
    # 1. Cartesian Parity
    Random.seed!(42)
    T = Float64
    L = 10.0
    N = 16
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

    # Serial DirectSum
    c_cpu, k_cpu = FFS.calculate_spectrum(
        FFS.DirectSumBackend(),
        (xv, yv),
        (u, v),
        ms;
        domain_size = (L, L),
    )

    # KA CPU Backend
    c_ka, k_ka = FFS.calculate_spectrum(
        GPUBackend(KA.CPU()),
        (xv, yv),
        (u, v),
        ms;
        domain_size = (L, L),
    )

    Test.@test isapprox(c_cpu, c_ka, atol = 1e-12)
    Test.@test all(isapprox(k_cpu[d], k_ka[d], rtol = 1e-12) for d in 1:2)

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

    # Serial DirectSum Spherical
    c_sph_cpu, k_sph_cpu = FFS.calculate_spectrum(
        FFS.DirectSumBackend(),
        (θ_nodes, φ_nodes),
        (f_val,),
        (Nθ, Nφ),
    )

    # KA CPU Backend Spherical
    c_sph_ka, k_sph_ka = FFS.calculate_spectrum(
        GPUBackend(KA.CPU()),
        (θ_nodes, φ_nodes),
        (f_val,),
        (Nθ, Nφ),
    )

    Test.@test isapprox(c_sph_cpu, c_sph_ka, atol = 1e-12)
end
