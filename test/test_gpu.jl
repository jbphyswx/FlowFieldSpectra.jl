# GPU/KernelAbstractions parity on `GPUBackend(KA.CPU())` (CI has no GPU). Exercises the portable KA
# direct-sum kernels for tensor + scattered Cartesian and spherical grids against the serial reference.
# Real-device (CUFFT / cuFINUFFT / KA-on-CUDA) parity lives in `gpu/` and runs only on a CUDA device.

using Test: Test
using Random: Random
using KernelAbstractions: KernelAbstractions as KA
using FlowFieldSpectra: FlowFieldSpectra as FFS

Test.@testset "GPU Backend Parity via KernelAbstractions.CPU()" begin
    dev = FFS.GPUBackend(KA.CPU())

    # Cartesian tensor grid (uniform) + scattered point cloud (2D).
    Random.seed!(42)
    L = 10.0
    ms = (16, 16)
    ug = FFS.UniformCartesianGrid(; domain = (L, L), n = ms)
    xs, ys = ug.axes
    kx1, ky1 = 2π * 2 / L, 2π * 1 / L
    u = [cos(kx1 * x + ky1 * y) for x in xs, y in ys]
    ct_s, k_s = FFS.calculate_spectrum(ug, u, ms; transform = FFS.DirectSumBackend(), execution = FFS.SerialBackend())
    ct_g, k_g = FFS.calculate_spectrum(ug, u, ms; transform = FFS.DirectSumBackend(), execution = dev)
    Test.@test isapprox(ct_s, ct_g; atol = 1e-12)
    Test.@test all(isapprox(k_s[d], k_g[d]; rtol = 1e-12) for d in 1:2)

    xv = vec([x for x in xs, y in ys])
    yv = vec([y for x in xs, y in ys])
    fv = vec(u)
    sc = FFS.ScatteredCartesianGrid((xv, yv); domain_size = (L, L))
    cs_s, _ = FFS.calculate_spectrum(sc, fv, ms; transform = FFS.DirectSumBackend(), execution = FFS.SerialBackend())
    cs_g, _ = FFS.calculate_spectrum(sc, fv, ms; transform = FFS.DirectSumBackend(), execution = dev)
    Test.@test isapprox(cs_s, cs_g; atol = 1e-12)

    # D = 1 and D = 3 tensor-grid coverage (flat-index kernel decode).
    for D in (1, 3)
        Nd = D == 3 ? 6 : 12
        Ld = 2π
        gd = FFS.UniformCartesianGrid(; domain = ntuple(_ -> Ld, D), n = ntuple(_ -> Nd, D))
        axs = gd.axes
        f = collect([sum(cos((d + 1) * pt[d]) for d in 1:D) for pt in Iterators.product(axs...)])
        msd = ntuple(_ -> Nd, D)
        cs, _ = FFS.calculate_spectrum(gd, f, msd; transform = FFS.DirectSumBackend(), execution = FFS.SerialBackend())
        cg, _ = FFS.calculate_spectrum(gd, f, msd; transform = FFS.DirectSumBackend(), execution = dev)
        Test.@test isapprox(cs, cg; atol = 1e-12)
    end

    # Spherical scattered (with a batch axis).
    Random.seed!(123)
    lmax = 4
    Nθ = lmax + 1
    Nφ = 2 * lmax + 1
    θ = rand(30) .* (0.8π) .+ 0.1π
    φ = rand(30) .* 2π
    fval = rand(30, 2)
    sgrid = FFS.ScatteredSphericalGrid(θ, φ)
    csph_s, _ = FFS.calculate_spectrum(sgrid, fval, (Nθ, Nφ); transform = FFS.DirectSumBackend(), execution = FFS.SerialBackend())
    csph_g, _ = FFS.calculate_spectrum(sgrid, fval, (Nθ, Nφ); transform = FFS.DirectSumBackend(), execution = dev)
    Test.@test isapprox(csph_s, csph_g; atol = 1e-10)
end
