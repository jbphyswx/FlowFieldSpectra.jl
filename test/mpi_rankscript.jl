# MPI rank script — launched under `mpiexec -n 2` by test_mpi.jl. Every rank builds the same replicated
# grid/fields (seeded identically), computes with MPIBackend (point-partition + α_w-weighted
# MPI.Allreduce!, or batch-partition for FFT), and rank 0 compares to the serial reference, printing a
# marker the launcher greps for.

using MPI: MPI
MPI.Init()
using FlowFieldSpectra: FlowFieldSpectra as FFS
using FlowTransformBindings: FlowTransformBindings as FTB
using FFTW: FFTW
using FINUFFT: FINUFFT
using Random: Random
using ComputationalBackends: ComputationalBackends as CB
using SpectralBackends: SpectralBackends as SB
using FlowGeometries: FlowGeometries as FG

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

Random.seed!(123)
L = 2π
ms = (16, 16)
N = 200
xv = rand(N) .* L
yv = rand(N) .* L
f = rand(N, 2)
cart = FG.Geometry.CartesianGeometry{Float64}()
sc = FG.Grids.UnstructuredGrid(cart, (xv, yv), ones(N); periodic = (true, true), period = (L, L))
xs = range(0.0, L; length = 17)[1:16]
ys = range(0.0, L; length = 17)[1:16]
ug = FG.Grids.StructuredGrid(cart, xs, ys; periodic = (true, true), period = (L, L))
u = [cos(2x) + 0.5 * sin(3y) for x in xs, y in ys]
ub = cat(u, 2 .* u, 3 .* u, 4 .* u; dims = 3)

# The round-robin shares carry different mean measures, and a masked sphere holds NaN at its masked
# nodes.
scw = FG.Grids.UnstructuredGrid(cart, (xv, yv), [isodd(j) ? 1.0 : 3.0 for j in 1:N]; periodic = (true, true),
                                period = (L, L))
θ = rand(N) .* π; φ = rand(N) .* 2π
smask = trues(N); smask[1:3:N] .= false
fs = rand(N); fs[.!smask] .= NaN
sphm = FG.Grids.UnstructuredGrid(FG.Geometry.SphericalGeometry(1.0), (φ, π / 2 .- θ),
                                 [isodd(j) ? 1.0 : 2.0 for j in 1:N], smask)
# A bounded cloud, whose shares' own origins and Fourier lengths differ from the whole grid's.
scb = FG.Grids.UnstructuredGrid(cart, (xv, yv), ones(N))

serial(g, x, m, t; kw...) = FFS.calculate_spectrum(g, x, m; transform = t, execution = CB.SerialBackend(), kw...)[1]
mpi(g, x, m, t; kw...) = FFS.calculate_spectrum(g, x, m; transform = t, execution = CB.MPIBackend(), kw...)[1]
# The same transform through a plan divided among the ranks, executed twice.
function mpiplan(g, x, m, t; kw...)
    batch = ntuple(i -> size(x, ndims(g) + i), ndims(x) - ndims(g))
    p = FFS.plan_spectrum(g, eltype(x), m; transform = t, execution = CB.MPIBackend(), batch = batch, kw...)
    c = FFS.allocate_coefficients(p)
    FFS.calculate_spectrum!(c, p, x)
    FFS.calculate_spectrum!(c, p, x)
    FFS.close!(p)
    return c
end
synth(g, c, m, t, exec; kw...) = FFS.synthesize(g, c, m; transform = t, execution = exec, kw...)
DS, NU, FF = SB.DirectSumSpectralBackend(), FTB.FINUFFTBackend(), SB.FFTSpectralBackend()
cds = serial(scw, f, ms, DS)
cnu = serial(scw, f, ms, NU; eps = 1e-12)
cff = serial(ug, ub, ms, FF)
checks = [
    ("FINUFFT", isapprox(mpi(sc, f, ms, NU; eps = 1e-12), serial(sc, f, ms, NU; eps = 1e-12); rtol = 1e-9, atol = 1e-10)),
    ("FFT batch", isapprox(mpi(ug, ub, ms, FF), serial(ug, ub, ms, FF); atol = 1e-12)),
    ("unequal measure", isapprox(mpi(scw, f, ms, DS), serial(scw, f, ms, DS); rtol = 1e-10, atol = 1e-12)),
    ("masked sphere", isapprox(mpi(sphm, fs, (8, 15), DS), serial(sphm, fs, (8, 15), DS); atol = 1e-10)),
    ("bounded cloud", isapprox(mpi(scb, f, ms, DS), serial(scb, f, ms, DS); rtol = 1e-10, atol = 1e-12)),
    ("bounded cloud FINUFFT", isapprox(mpi(scb, f, ms, NU; eps = 1e-12), serial(scb, f, ms, NU; eps = 1e-12);
                                       rtol = 1e-9, atol = 1e-10)),
    ("plan direct sum", isapprox(mpiplan(scw, f, ms, DS), cds; rtol = 1e-10, atol = 1e-12)),
    ("plan FINUFFT", isapprox(mpiplan(scw, f, ms, NU; eps = 1e-12), cnu; rtol = 1e-9, atol = 1e-10)),
    ("plan FFT batch", isapprox(mpiplan(ug, ub, ms, FF), cff; atol = 1e-12)),
    ("plan masked sphere", isapprox(mpiplan(sphm, fs, (8, 15), DS), serial(sphm, fs, (8, 15), DS); atol = 1e-10)),
    ("synthesis direct sum", isapprox(synth(scw, cds, ms, DS, CB.MPIBackend()),
                                      synth(scw, cds, ms, DS, CB.SerialBackend()); atol = 1e-12)),
    ("synthesis FINUFFT", isapprox(synth(scw, cnu, ms, NU, CB.MPIBackend(); eps = 1e-12),
                                   synth(scw, cnu, ms, NU, CB.SerialBackend(); eps = 1e-12); atol = 1e-10)),
    ("synthesis FFT batch", isapprox(synth(ug, cff, ms, FF, CB.MPIBackend()), ub; atol = 1e-12)),
]
failed = [name for (name, ok) in checks if !ok]
rank == 0 && println(isempty(failed) ? "MPI_PARITY_OK np=$(MPI.Comm_size(comm))" : "MPI_PARITY_FAIL $(failed)")

MPI.Finalize()
