# MPI rank script — launched under `mpiexec -n 2` by test_mpi.jl. Every rank builds the SAME replicated
# grid/fields (seeded identically), computes with MPIBackend (point-partition + α_w-weighted
# MPI.Allreduce!, or batch-partition for FFT), and rank 0 compares to the serial reference, printing a
# marker the launcher greps for.

using MPI: MPI
MPI.Init()
using FlowFieldSpectra: FlowFieldSpectra as FFS
using FFTW: FFTW
using FINUFFT: FINUFFT
using Random: Random

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

Random.seed!(123)
L = 2π
ms = (16, 16)
N = 200
xv = rand(N) .* L
yv = rand(N) .* L
f = rand(N, 2)
sc = FFS.ScatteredCartesianGrid((xv, yv); domain_size = (L, L))
ug = FFS.UniformCartesianGrid(; domain = (L, L), n = ms)
xs, ys = ug.axes
u = [cos(2x) + 0.5 * sin(3y) for x in xs, y in ys]
ub = cat(u, 2 .* u, 3 .* u, 4 .* u; dims = 3)

cd, _ = FFS.calculate_spectrum(sc, f, ms; transform = FFS.DirectSumBackend(), execution = FFS.MPIBackend())
cn, _ = FFS.calculate_spectrum(sc, f, ms; transform = FFS.NUFFTBackend(), execution = FFS.MPIBackend(), eps = 1e-12)
cf, _ = FFS.calculate_spectrum(ug, ub, ms; transform = FFS.FFTBackend(), execution = FFS.MPIBackend())

if rank == 0
    dref, _ = FFS.calculate_spectrum(sc, f, ms; transform = FFS.DirectSumBackend(), execution = FFS.SerialBackend())
    nref, _ = FFS.calculate_spectrum(sc, f, ms; transform = FFS.NUFFTBackend(), execution = FFS.SerialBackend(), eps = 1e-12)
    fref, _ = FFS.calculate_spectrum(ug, ub, ms; transform = FFS.FFTBackend(), execution = FFS.SerialBackend())
    ok = isapprox(cd, dref; rtol = 1e-10, atol = 1e-12) &&
         isapprox(cn, nref; rtol = 1e-9, atol = 1e-10) &&
         isapprox(cf, fref; atol = 1e-12)
    println(ok ? "MPI_PARITY_OK np=$(MPI.Comm_size(comm))" : "MPI_PARITY_FAIL")
end

MPI.Finalize()
