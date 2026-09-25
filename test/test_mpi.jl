# MPI execution parity: launches `mpiexec -n 2` running mpi_rankscript.jl and checks the marker rank 0
# prints.

using Test: Test
using MPI: MPI

Test.@testset "MPI spectrum parity (multi-rank)" begin
    script = joinpath(@__DIR__, "mpi_rankscript.jl")
    proj = Base.active_project()
    buf = IOBuffer()
    MPI.mpiexec() do exe
        run(pipeline(`$exe -n 2 $(Base.julia_cmd()) --project=$proj $script`; stdout = buf, stderr = buf))
    end
    out = String(take!(buf))
    Test.@test occursin("MPI_PARITY_OK", out)
    occursin("MPI_PARITY_OK", out) || println(out)
end
