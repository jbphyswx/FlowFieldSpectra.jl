# Two orthogonal backend axes: transform × execution.
#
# `calculate_spectrum(grid, field, ms; transform=…, execution=…)` takes an independent TRANSFORM
# backend (the spectral math) and EXECUTION backend (where/how it runs). The execution axis is
# result-invariant: one FFT transform under Serial, Threaded, and GPUBackend(KA.CPU()) gives identical
# spectra.
#
#   julia --project=examples examples/execution_backends.jl

using FlowFieldSpectra: FlowFieldSpectra as FFS
using FFTW: FFTW                                   # FFT transform + the device-generic GPU FFT (KA.CPU→FFTW)
using KernelAbstractions: KernelAbstractions as KA # GPUBackend (KA.CPU() here; CUDABackend() on a GPU)
using OhMyThreads: OhMyThreads                     # ThreadedBackend
using CairoMakie: CairoMakie as Mke
using Random: Random
using ComputationalBackends: ComputationalBackends as CB
using SpectralBackends: SpectralBackends as SB
using FlowGeometries: FlowGeometries as FG

function run_execution_backends_example()
    println("--- Two-axis backends: one transform, several execution backends ---")

    L = 2π
    N = 96
    xs = range(0.0, L; length = N + 1)[1:N]
    grid = FG.Grids.StructuredGrid(FG.Geometry.CartesianGeometry{Float64}(), xs, xs;
        periodic = (true, true), period = (L, L))
    ms = (N, N)

    # A broadband field with a k^-5/3 cascade — built and kept as a plain (N, N) tensor.
    Random.seed!(5)
    freq = [0:(N ÷ 2 - 1); -(N ÷ 2):-1] .* (2π / L)
    f̂ = FFTW.fft(randn(N, N))
    for j in 1:N, i in 1:N
        k = hypot(freq[i], freq[j])
        f̂[i, j] *= k == 0 ? 0.0 : k^(-(5 / 3 + 1) / 2)
    end
    f = real(FFTW.ifft(f̂))                          # (N, N) — no flattening

    execs = [
        ("Serial", CB.SerialBackend()),
        ("Threaded", CB.ThreadedBackend()),
        ("GPU (KA.CPU())", CB.GPUBackend(KA.CPU())),
    ]
    results = [FFS.calculate_spectrum(grid, f, ms; transform = SB.FFTSpectralBackend(), execution = e) for (_, e) in execs]
    ks = results[1][2]
    cref = results[1][1]

    println("Max |coefficient difference| vs. Serial (execution is result-invariant):")
    for (i, (name, _)) in enumerate(execs)
        println("  $(rpad(name, 16)) $(maximum(abs.(results[i][1] .- cref)))")
    end

    # ── real GPU / Distributed / MPI (uncomment on the appropriate machine) ──
    # using CUDA; dev = CB.GPUBackend(CUDA.CUDABackend())
    #   FFS.calculate_spectrum(grid, f, ms; transform=SB.FFTSpectralBackend(),  execution=dev)   # CUFFT
    #   scat = FG.Grids.UnstructuredGrid(FG.Geometry.CartesianGeometry{Float64}(), (xv, yv), ones(length(xv)); periodic=(true,true), period=(L,L))
    #   FFS.calculate_spectrum(scat, fscat, ms; transform=FFS.FINUFFTBackend(), execution=dev)  # cuFINUFFT
    # using Distributed; addprocs(4); @everywhere using FlowFieldSpectra, FINUFFT
    #   FFS.calculate_spectrum(scat, fscat, ms; transform=FFS.FINUFFTBackend(), execution=CB.DistributedBackend())
    # using MPI; MPI.Init()
    #   FFS.calculate_spectrum(scat, fscat, ms; transform=SB.DirectSumSpectralBackend(), execution=CB.MPIBackend())

    specs = [FFS.isotropic_spectrum(ks, r[1]; num_bins = 40) for r in results]
    kb1 = specs[1][1]
    rng = 2:findlast(<=(0.6 * maximum(kb1)), kb1)

    fig = Mke.Figure(size = (1360, 470), fontsize = 15)
    Mke.Label(fig[0, 1:3], "One FFT transform × three execution backends — identical results",
        fontsize = 19, font = :bold)
    ax1 = Mke.Axis(fig[1, 1]; title = "Isotropic E(k) (curves coincide)", xlabel = "k", ylabel = "E(k)",
        xscale = log10, yscale = log10)
    styles = [(:solid, 5), (:dash, 3), (:dot, 2)]
    for (i, (name, _)) in enumerate(execs)
        kb, Ek = specs[i]
        Mke.lines!(ax1, kb[rng], Ek[rng] .+ 1e-30; linestyle = styles[i][1], linewidth = styles[i][2], label = name)
    end
    Mke.axislegend(ax1; position = :lb)
    ax2 = Mke.Axis(fig[1, 2]; title = "2D spectral energy  log₁₀|C(kₓ, k_y)|²", xlabel = "kₓ", ylabel = "k_y",
        aspect = Mke.DataAspect())
    hm = Mke.heatmap!(ax2, ks[1], ks[2], log10.(abs2.(cref) .+ 1e-30); colormap = :viridis)
    Mke.Colorbar(fig[1, 3], hm)

    outpath = joinpath(@__DIR__, "execution_backends.png")
    Mke.save(outpath, fig)
    println("Saved figure: $outpath")
    println("Example run successfully!")
    return nothing
end

run_execution_backends_example()
