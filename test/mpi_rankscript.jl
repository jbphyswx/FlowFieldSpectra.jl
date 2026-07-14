# MPI rank script — launched under `mpiexec -n 2` by test_mpi.jl. Every rank builds the SAME
# replicated grid/fields (seeded identically), computes the spectrum with MPIBackend (each rank
# does its round-robin point share, then MPI.Allreduce! sums the α_w-weighted partials), and rank 0
# compares against the serial reference, printing a marker the launcher greps for.

using MPI: MPI
MPI.Init()
using FlowFieldSpectra: FlowFieldSpectra as FFS
using Random: Random

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

Random.seed!(123)
L = 2π
ms = (16, 16)
dx = L / ms[1]
dy = L / ms[2]
xs = range(0.0, stop = L - dx, length = ms[1])
ys = range(0.0, stop = L - dy, length = ms[2])
xv = vec([x for x in xs, y in ys])
yv = vec([y for x in xs, y in ys])
u = @. cos(2xv) + 0.5 * sin(3yv)
v = @. sin(xv)
g = FFS.ScatteredCartesianGrid((xv, yv); domain_size = (L, L))

cd, _ = FFS.calculate_spectrum(g, (u, v), ms; transform = FFS.DirectSumBackend(), execution = FFS.MPIBackend())

if rank == 0
    cref, _ = FFS.calculate_spectrum(g, (u, v), ms; transform = FFS.DirectSumBackend(), execution = FFS.SerialBackend())
    ok = isapprox(cd, cref; rtol = 1e-10, atol = 1e-12)
    println(ok ? "MPI_PARITY_OK np=$(MPI.Comm_size(comm))" : "MPI_PARITY_FAIL")
end

MPI.Finalize()
