# Two orthogonal backend axes: transform × execution.
#
# `calculate_spectrum(grid, fields, ms; transform=…, execution=…)` takes an independent
# TRANSFORM backend (the spectral math) and EXECUTION backend (where/how it runs). This example
# shows that the execution axis is *result-invariant*: one FFT transform run under Serial, Threaded,
# and GPUBackend(KA.CPU()) gives identical spectra — and documents the GPU/Distributed/MPI paths.
#
#   julia --project=examples examples/execution_backends.jl

using FlowFieldSpectra: FlowFieldSpectra as FFS
using FFTW: FFTW                                   # activates FFTBackend + the device-generic GPU FFT
using KernelAbstractions: KernelAbstractions as KA # activates GPUBackend (KA.CPU() here; CUDABackend() on a GPU)
using OhMyThreads: OhMyThreads                     # activates ThreadedBackend
using CairoMakie: CairoMakie as Mke
using Random: Random

function run_execution_backends_example()
    println("--- Two-axis backends: one transform, several execution backends ---")

    L = 2π
    N = 96
    dx = L / N
    xs = range(0.0, stop = L - dx, length = N)
    xv = vec([x for x in xs, y in xs])
    yv = vec([y for x in xs, y in xs])

    # A broadband field with a k^-5/3 cascade.
    Random.seed!(5)
    freq = [0:(N ÷ 2 - 1); -(N ÷ 2):-1] .* (2π / L)
    f̂ = FFTW.fft(randn(N, N))
    for j in 1:N, i in 1:N
        k = hypot(freq[i], freq[j])
        f̂[i, j] *= k == 0 ? 0.0 : k^(-(5 / 3 + 1) / 2)
    end
    f = vec(real(FFTW.ifft(f̂)))

    grid = FFS.UniformCartesianGrid((xv, yv); domain_size = (L, L))
    ms = (N, N)

    # Same FFT transform, three execution backends. The execution axis never changes the result:
    #   • SerialBackend()        — single-threaded CPU
    #   • ThreadedBackend()      — FFTW internal threads (OhMyThreads for the direct-sum outer loop)
    #   • GPUBackend(KA.CPU())   — the device-generic GPU FFT staged to a KA device; on KA.CPU() it
    #                              uses FFTW, on a real GPU `GPUBackend(CUDABackend())` uses CUFFT.
    execs = [
        ("Serial", FFS.SerialBackend()),
        ("Threaded", FFS.ThreadedBackend()),
        ("GPU (KA.CPU())", FFS.GPUBackend(KA.CPU())),
    ]
    results = [FFS.calculate_spectrum(grid, (f,), ms; transform = FFS.FFTBackend(), execution = e)
               for (_, e) in execs]
    ks = results[1][2]
    cref = results[1][1]

    println("Max |coefficient difference| vs. Serial (execution is result-invariant):")
    for (i, (name, _)) in enumerate(execs)
        println("  $(rpad(name, 16)) $(maximum(abs.(results[i][1] .- cref)))")
    end

    # ── GPU on a real CUDA device (uncomment on a machine with an NVIDIA GPU + `using CUDA`) ──
    # using CUDA
    # dev = FFS.GPUBackend(CUDA.CUDABackend())
    # c_cufft, _   = FFS.calculate_spectrum(grid, (f,), ms; transform = FFS.FFTBackend(),   execution = dev)   # CUFFT
    # sgrid = FFS.ScatteredCartesianGrid((xv, yv); domain_size = (L, L))
    # c_cufinu, _  = FFS.calculate_spectrum(sgrid, (f,), ms; transform = FFS.NUFFTBackend(), execution = dev)  # cuFINUFFT
    #
    # ── Distributed across worker processes (uncomment; needs `addprocs`) ──
    # using Distributed; addprocs(4); @everywhere using FlowFieldSpectra, FINUFFT
    # sgrid = FFS.ScatteredCartesianGrid((xv, yv); domain_size = (L, L))
    # c_dist, _ = FFS.calculate_spectrum(sgrid, (f,), ms; transform = FFS.NUFFTBackend(),
    #                                    execution = FFS.DistributedBackend(FFS.SerialBackend()))
    #
    # ── MPI across ranks (uncomment; run under `mpiexec -n N julia --project=examples this.jl`) ──
    # using MPI; MPI.Init()
    # c_mpi, _ = FFS.calculate_spectrum(sgrid, (f,), ms; transform = FFS.DirectSumBackend(),
    #                                   execution = FFS.MPIBackend(FFS.SerialBackend()))   # multi-GPU: MPIBackend(GPUBackend(dev))

    # Figure: overlaid isotropic spectra (they coincide) + the residual vs. Serial.
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
    ax2 = Mke.Axis(fig[1, 2]; title = "2D spectral energy  log₁₀|C(kₓ, k_y)|²", xlabel = "kₓ",
        ylabel = "k_y", aspect = Mke.DataAspect())
    hm = Mke.heatmap!(ax2, ks[1], ks[2], log10.(abs2.(cref[:, :, 1]) .+ 1e-30); colormap = :viridis)
    Mke.Colorbar(fig[1, 3], hm)

    outpath = joinpath(@__DIR__, "execution_backends.png")
    Mke.save(outpath, fig)
    println("Saved figure: $outpath")
    println("Example run successfully!")
    return nothing
end

run_execution_backends_example()
