# Cross-backend agreement on the sphere.
#
# `calculate_spectrum` returns the quadrature projection `C[l,m] = Σⱼ wⱼ fⱼ Y_lm(xⱼ)` with the grid's own
# weights totalling `4π`. Every spherical backend returns that same functional in the same basis, so a
# caller may pick one for speed and read the same coefficients: DirectSum states it directly, the GPU SHT
# runs the same sum on a device, and NUFSHT reaches it through its type-1.
#
# NUFSHT's type-1 is the adjoint `Aᴴf = Σⱼ fⱼ Y_lm(xⱼ)` and takes no weights, so the quadrature reaches it
# through the field. A grid whose per-node measures VARY is the discriminating case: an unweighted
# transform returns identical coefficients for any measures at all.
#
# `solve = true` is a different and legitimate problem — the least-squares `A c ≈ f`, recovering the
# coefficients a field was synthesized from — and consumes the raw field.

using Test: Test
using FlowFieldSpectra: FlowFieldSpectra as FFS
using FlowGeometries: FlowGeometries as FG
using SpectralBackends: SpectralBackends as SB
using ComputationalBackends: ComputationalBackends as CB
using FastSphericalHarmonics: FastSphericalHarmonics as FSH
using NUFSHT: NUFSHT
using FINUFFT: FINUFFT
using KernelAbstractions: KernelAbstractions as KA
using Random: Random

const SP_DS = SB.DirectSumSpectralBackend()
const SP_NU = SB.NUFSHTSpectralBackend()
const SP_SER = CB.SerialBackend()
sp_rel(a, b) = maximum(abs, a .- b) / max(maximum(abs, b), eps())

# Fibonacci-like nodes: a genuine point cloud with no iso-latitude structure.
function sp_nodes(N)
    ga = π * (3 - sqrt(5))
    λ = [mod(ga * i, 2π) for i in 1:N]
    z = [-1 + 2 * (i - 0.5) / N for i in 1:N]
    return λ, asin.(z), z
end

Test.@testset "NUFSHT forward is the quadrature projection" begin
    Random.seed!(4242)
    lmax = 4
    ms = (lmax + 1, 2 * lmax + 1)
    N = 300
    λ, φlat, z = sp_nodes(N)
    f = [1.0 + 0.5 * sin(λ[i]) + 0.3 * z[i] + 0.2 * cos(2 * λ[i]) * sqrt(1 - z[i]^2) for i in 1:N]

    # Equal-measure nodes: the weights are the constant 4π/N.
    ge = FG.Grids.UnstructuredGrid(FG.Geometry.SphericalGeometry(1.0), (λ, φlat), fill(4π / N, N))
    cd, ksd = FFS.calculate_spectrum(ge, f, ms; transform = SP_DS, execution = SP_SER)
    cn, ksn = FFS.calculate_spectrum(ge, f, ms; transform = SP_NU)
    Test.@test size(cn) == size(cd)
    Test.@test ksn == ksd
    Test.@test sp_rel(cn, cd) < 1e-9

    # Varying measures pin the quadrature: both backends must move away from the equal-measure answer,
    # and must agree on where they move to.
    w = [1.0 + 0.8 * sin(λ[i])^2 for i in 1:N]
    gv = FG.Grids.UnstructuredGrid(FG.Geometry.SphericalGeometry(1.0), (λ, φlat), w)
    cdv, _ = FFS.calculate_spectrum(gv, f, ms; transform = SP_DS, execution = SP_SER)
    cnv, _ = FFS.calculate_spectrum(gv, f, ms; transform = SP_NU)
    Test.@test sp_rel(cnv, cdv) < 1e-9
    Test.@test sp_rel(cdv, cd) > 1e-3          # the measures genuinely change the projection
    Test.@test sp_rel(cnv, cn) > 1e-3          # and NUFSHT tracks them

    # The weights the grid states, and the total the transform's convention fixes.
    ws = collect(FFS.Grids._sph_node_weights(gv, Float64, N, nothing))
    Test.@test sum(ws) ≈ 4π

    # Batched, and the device DirectSum still agrees with the serial projection.
    fb = hcat(f, reverse(f), 0.5 .* f)
    cnb, _ = FFS.calculate_spectrum(gv, fb, ms; transform = SP_NU)
    Test.@test size(cnb) == (ms[1], ms[2], 3)
    for b in 1:3
        c1, _ = FFS.calculate_spectrum(gv, fb[:, b], ms; transform = SP_NU)
        Test.@test sp_rel(cnb[:, :, b], c1) < 1e-9
    end
    cgv, _ = FFS.calculate_spectrum(gv, f, ms; transform = SP_DS, execution = CB.GPUBackend(KA.CPU()))
    Test.@test sp_rel(cgv, cdv) < 1e-12

    # A reusable plan gives the same coefficients as the one-shot.
    p = FFS.plan_spectrum(gv, Float64, ms; transform = SP_NU)
    buf = zeros(ComplexF64, ms[1], ms[2])
    FFS.calculate_spectrum!(buf, p, f)
    Test.@test sp_rel(buf, cnv) < 1e-12
end

Test.@testset "NUFSHT solve consumes the raw field" begin
    lmax = 3
    Nθ, Nφ = lmax + 1, 2 * lmax + 1
    N = 400
    λ, φlat, _ = sp_nodes(N)
    g = FG.Grids.UnstructuredGrid(FG.Geometry.SphericalGeometry(1.0), (λ, φlat), fill(4π / N, N))

    # A field synthesized from known coefficients; the solve returns them for any node measures.
    Ct = zeros(Nθ, Nφ)
    Ct[FSH.sph_mode(0, 0)] = 1.0
    Ct[FSH.sph_mode(2, 1)] = -0.4
    Ct[FSH.sph_mode(3, -2)] = 0.7
    plan = NUFSHT.make_plan(Float64, Float64(π) / 2 .- φlat, λ, lmax)
    fv = zeros(N)
    NUFSHT.nusht_type2!(fv, Ct, plan)

    cs, _ = FFS.calculate_spectrum(g, fv, (Nθ, Nφ); transform = SP_NU, solve = true,
        rtol = 1.0e-10, maxiter = 2000)
    for l in 0:lmax, m in -l:l
        Test.@test isapprox(real(cs[FFS.sph_mode_index(l, m)]), Ct[FSH.sph_mode(l, m)]; atol = 1.0e-6)
    end

    # The solve fits the field, so a varying measure leaves it where it was.
    gv = FG.Grids.UnstructuredGrid(FG.Geometry.SphericalGeometry(1.0), (λ, φlat),
        [1.0 + 0.8 * sin(λ[i])^2 for i in 1:N])
    csv, _ = FFS.calculate_spectrum(gv, fv, (Nθ, Nφ); transform = SP_NU, solve = true,
        rtol = 1.0e-10, maxiter = 2000)
    Test.@test sp_rel(csv, cs) < 1e-6
end

Test.@testset "DirectSum spherical plan reuses its setup" begin
    # A spherical direct sum rebuilds the grid's nodes, its latitude quadrature (a Gauss–Legendre root
    # solve), its ring table and the Legendre tables on every call — a minority of the time and about
    # three quarters of the allocations. The plan holds them, so a reused execution allocates almost
    # nothing and returns the same coefficients.
    Random.seed!(5)
    lmax = 6
    ms = (lmax + 1, 2 * lmax + 1)
    sph = FG.Geometry.SphericalGeometry(1.0)
    grids = (
        FG.Connectivity.structured_grid(Float64, FG.SphericalSampling.GaussLegendreSampling(), lmax + 1),
        FG.Grids.HEALPixGrid(sph, 2),
        FG.Grids.RingGrid(sph, FG.SphericalSampling.OctahedralGaussianSampling(2)),
        FG.Grids.CubedSphereGrid(sph, 4),
    )
    for g in grids                                     # tensor, ring, ring, and point-cloud layouts
        f = randn(size(g)...)
        cref, ksref = FFS.calculate_spectrum(g, f, ms; transform = SP_DS, execution = SP_SER)
        p = FFS.plan_spectrum(g, Float64, ms; transform = SP_DS, execution = SP_SER)
        buf = zeros(ComplexF64, ms[1], ms[2])
        Test.@test FFS.calculate_spectrum!(buf, p, f) == ksref
        Test.@test sp_rel(buf, cref) < 1e-13
        # Reuse leaves the plan intact, and a different field goes through the same one.
        for _ in 1:3
            FFS.calculate_spectrum!(buf, p, f)
            Test.@test sp_rel(buf, cref) < 1e-13
        end
        f2 = randn(size(g)...)
        c2, _ = FFS.calculate_spectrum(g, f2, ms; transform = SP_DS, execution = SP_SER)
        FFS.calculate_spectrum!(buf, p, f2)
        Test.@test sp_rel(buf, c2) < 1e-13
    end

    g = grids[1]
    fb = randn(size(g)..., 3)
    cb, _ = FFS.calculate_spectrum(g, fb, ms; transform = SP_DS, execution = SP_SER)
    pb = FFS.plan_spectrum(g, Float64, ms; transform = SP_DS, execution = SP_SER, batch = (3,))
    bufb = zeros(ComplexF64, ms[1], ms[2], 3)
    FFS.calculate_spectrum!(bufb, pb, fb)
    Test.@test sp_rel(bufb, cb) < 1e-13
    # A buffer for the wrong batch says so.
    Test.@test_throws DimensionMismatch FFS.calculate_spectrum!(zeros(ComplexF64, ms[1], ms[2]), pb, fb)

    # The setup is what the plan removes, so a reused execution stays well under the one-shot.
    f = randn(size(g)...)
    buf = zeros(ComplexF64, ms[1], ms[2])
    p = FFS.plan_spectrum(g, Float64, ms; transform = SP_DS, execution = SP_SER)
    FFS.calculate_spectrum!(buf, p, f)
    a_plan = @allocated FFS.calculate_spectrum!(buf, p, f)
    a_one = @allocated FFS.calculate_spectrum(g, f, ms; transform = SP_DS, execution = SP_SER)
    Test.@test a_plan < 0.2 * a_one

    # NUFSHT's `batch_chunk` splits the batch across executions, so the answer must not depend on it:
    # every setting is held to the whole-batch result, including a chunk that leaves a PARTIAL final
    # chunk whose unused slices are zero-filled.
    Random.seed!(31)
    lmaxb = 4
    msb = (lmaxb + 1, 2 * lmaxb + 1)
    Nb = 120
    gab = π * (3 - sqrt(5))
    λb = [mod(gab * i, 2π) for i in 1:Nb]
    φb = asin.([-1 + 2 * (i - 0.5) / Nb for i in 1:Nb])
    gb = FG.Grids.UnstructuredGrid(FG.Geometry.SphericalGeometry(1.0), (λb, φb), fill(4π / Nb, Nb))
    B = 6
    fb = randn(Nb, B)
    ref, _ = FFS.calculate_spectrum(gb, fb, msb; transform = SP_NU, batch_chunk = 0)
    for bc in (1, 2, 4, 6, 8)                       # 4 leaves a partial final chunk of 2
        cb, _ = FFS.calculate_spectrum(gb, fb, msb; transform = SP_NU, batch_chunk = bc)
        Test.@test size(cb) == size(ref)
        Test.@test maximum(abs, cb .- ref) < 1e-9
        pb = FFS.plan_spectrum(gb, Float64, msb; transform = SP_NU, execution = SP_SER,
            batch = (B,), batch_chunk = bc)
        buf = zeros(Float64, msb..., B)
        FFS.calculate_spectrum!(buf, pb, fb)
        Test.@test maximum(abs, buf .- ref) < 1e-9
    end
    # Each slice is independent of the others, so a chunk boundary cannot mix them.
    single, _ = FFS.calculate_spectrum(gb, fb[:, 3], msb; transform = SP_NU)
    Test.@test maximum(abs, ref[:, :, 3] .- single) < 1e-9

    # A Cartesian grid gets the Cartesian plan; `test_directsum_plan.jl` gates its values.
    ax = range(0.0, 2π; length = 9)[1:8]
    gc = FG.Grids.StructuredGrid(FG.Geometry.CartesianGeometry{Float64}(), ax, ax;
        periodic = (true, true), period = (2π, 2π))
    Test.@test FFS.plan_spectrum(gc, Float64, (8, 8); transform = SP_DS, execution = SP_SER) isa
        FFS.DirectSumCartesianPlan
end

Test.@testset "NUFSHT on a structured spherical grid" begin
    # `_calculate_spectrum_nufsht` accepts any spherical grid, so a structured one arrives with TWO
    # spatial dims. Its batch and its synthesis shape both come from the grid, and its weights are the
    # sampling's own latitude quadrature.
    lmax = 4
    Nθ, Nφ = lmax + 1, 2 * lmax + 1
    g = FG.Connectivity.structured_grid(Float64, FG.SphericalSampling.GaussLegendreSampling(), Nθ)
    Test.@test size(g) == (Nφ, Nθ)
    f = [1.0 + 0.3 * sin(2 * λ) * cos(φ) for λ in FG.Grids.coordinates(g, 1),
        φ in FG.Grids.coordinates(g, 2)]

    cn, _ = FFS.calculate_spectrum(g, f, (Nθ, Nφ); transform = SP_NU)
    Test.@test size(cn) == (Nθ, Nφ)                       # no phantom batch axis
    cd, _ = FFS.calculate_spectrum(g, f, (Nθ, Nφ); transform = SP_DS, execution = SP_SER)
    Test.@test sp_rel(cn, cd) < 1e-8

    fb = cat(f, 2 .* f; dims = 3)
    cnb, _ = FFS.calculate_spectrum(g, fb, (Nθ, Nφ); transform = SP_NU)
    Test.@test size(cnb) == (Nθ, Nφ, 2)
    Test.@test sp_rel(cnb[:, :, 1], cn) < 1e-9

    # Synthesis writes the grid's own shape, so a round trip compares against the forward's input.
    out = FFS.synthesize(g, cn, (Nθ, Nφ); transform = SP_NU)
    Test.@test size(out) == size(g)
end
