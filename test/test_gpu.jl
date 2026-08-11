# GPU/KernelAbstractions parity on `GPUBackend(KA.CPU())` (CI has no GPU). Exercises the portable KA
# direct-sum kernels for tensor + scattered Cartesian and spherical grids against the serial reference.
# Real-device (CUFFT / cuFINUFFT / KA-on-CUDA) parity lives in `gpu/` and runs only on a CUDA device.
# (Grid constructors + CB/SB aliases come from runtests.jl.)

using Test: Test
using Random: Random
using KernelAbstractions: KernelAbstractions as KA
using FlowFieldSpectra: FlowFieldSpectra as FFS

Test.@testset "GPU Backend Parity via KernelAbstractions.CPU()" begin
    dev = CB.GPUBackend(KA.CPU())

    Random.seed!(42)
    L = 10.0
    ms = (16, 16)
    xs = ucg_axis(Float64, L, 16); ys = ucg_axis(Float64, L, 16)
    ug = ucg((L, L), ms)
    kx1, ky1 = 2π * 2 / L, 2π * 1 / L
    u = [cos(kx1 * x + ky1 * y) for x in xs, y in ys]
    ct_s, k_s = FFS.calculate_spectrum(ug, u, ms; transform = SB.DirectSumSpectralBackend(), execution = CB.SerialBackend())
    ct_g, k_g = FFS.calculate_spectrum(ug, u, ms; transform = SB.DirectSumSpectralBackend(), execution = dev)
    Test.@test isapprox(ct_s, ct_g; atol = 1e-12)
    Test.@test all(isapprox(k_s[d], k_g[d]; rtol = 1e-12) for d in 1:2)

    xv = vec([x for x in xs, y in ys])
    yv = vec([y for x in xs, y in ys])
    fv = vec(u)
    sc = scg((xv, yv), (L, L))
    cs_s, _ = FFS.calculate_spectrum(sc, fv, ms; transform = SB.DirectSumSpectralBackend(), execution = CB.SerialBackend())
    cs_g, _ = FFS.calculate_spectrum(sc, fv, ms; transform = SB.DirectSumSpectralBackend(), execution = dev)
    Test.@test isapprox(cs_s, cs_g; atol = 1e-12)

    # Batched (B > 1) Cartesian: exercises the kernels' inner batch loop on both grid kinds.
    ub = cat(u, 2 .* u, -u; dims = 3)                        # tensor field (16, 16, 3)
    cb_s, _ = FFS.calculate_spectrum(ug, ub, ms; transform = SB.DirectSumSpectralBackend(), execution = CB.SerialBackend())
    cb_g, _ = FFS.calculate_spectrum(ug, ub, ms; transform = SB.DirectSumSpectralBackend(), execution = dev)
    Test.@test isapprox(cb_s, cb_g; atol = 1e-12)
    fvb = cat(fv, 2 .* fv, -fv; dims = 2)                    # scattered field (256, 3)
    cbs_s, _ = FFS.calculate_spectrum(sc, fvb, ms; transform = SB.DirectSumSpectralBackend(), execution = CB.SerialBackend())
    cbs_g, _ = FFS.calculate_spectrum(sc, fvb, ms; transform = SB.DirectSumSpectralBackend(), execution = dev)
    Test.@test isapprox(cbs_s, cbs_g; atol = 1e-12)

    # D = 1 and D = 3 tensor-grid coverage (flat-index kernel decode).
    for D in (1, 3)
        Nd = D == 3 ? 6 : 12
        Ld = 2π
        gd = ucg(ntuple(_ -> Ld, D), ntuple(_ -> Nd, D))
        axs = ntuple(d -> ucg_axis(Float64, Ld, Nd), D)
        f = collect([sum(cos((d + 1) * pt[d]) for d in 1:D) for pt in Iterators.product(axs...)])
        msd = ntuple(_ -> Nd, D)
        cs, _ = FFS.calculate_spectrum(gd, f, msd; transform = SB.DirectSumSpectralBackend(), execution = CB.SerialBackend())
        cg, _ = FFS.calculate_spectrum(gd, f, msd; transform = SB.DirectSumSpectralBackend(), execution = dev)
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
    sgrid = sph_scat(θ, φ)
    csph_s, _ = FFS.calculate_spectrum(sgrid, fval, (Nθ, Nφ); transform = SB.DirectSumSpectralBackend(), execution = CB.SerialBackend())
    csph_g, _ = FFS.calculate_spectrum(sgrid, fval, (Nθ, Nφ); transform = SB.DirectSumSpectralBackend(), execution = dev)
    Test.@test isapprox(csph_s, csph_g; atol = 1e-10)
end
