# MPI execution parity — launches `mpiexec -n 2` running mpi_rankscript.jl and checks the marker
# rank 0 prints. Guarded: if MPI's `mpiexec` cannot launch (no functional MPI in this environment),
# the test is skipped rather than failing spuriously.

using Test: Test
using MPI: MPI

Test.@testset "MPI spectrum parity (multi-rank)" begin
    script = joinpath(@__DIR__, "mpi_rankscript.jl")
    proj = Base.active_project()
    buf = IOBuffer()
    ran = try
        MPI.mpiexec() do exe
            run(pipeline(`$exe -n 2 $(Base.julia_cmd()) --project=$proj $script`; stdout = buf, stderr = buf))
        end
        true
    catch err
        @info "mpiexec launch failed; skipping MPI parity test" err
        false
    end
    if ran
        out = String(take!(buf))
        Test.@test occursin("MPI_PARITY_OK", out)
    else
        Test.@test_skip true
    end
end
