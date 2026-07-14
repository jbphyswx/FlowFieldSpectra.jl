# Real-device GPU parity for FlowFieldSpectra.jl.
#
#   julia --project=gpu gpu/gpu_parity.jl
#
# Requires a functional NVIDIA CUDA device. This is the ONLY place the CUDA-only fast transforms
# (CUFFT for uniform grids, cuFINUFFT for scattered grids) are validated numerically — CI has no
# GPU and never runs this. It checks each fast GPU transform against its CPU reference, plus the
# portable KA direct-sum kernels running on the real CUDA device vs the serial CPU direct sum.
#
# If no functional CUDA device is present the script prints a notice and returns without failing,
# so it is safe to invoke unconditionally.

using CUDA: CUDA
using FFTW: FFTW
using FINUFFT: FINUFFT
using AbstractFFTs: AbstractFFTs
using KernelAbstractions: KernelAbstractions as KA
using FlowFieldSpectra: FlowFieldSpectra as FFS
using Random: Random
using Test: Test

function main()
    println("CUDA.functional(): ", CUDA.functional())
    if !CUDA.functional()
        println("No functional CUDA device — skipping GPU parity (expected on CI / CPU-only machines).")
        return nothing
    end
    println("Device: ", CUDA.name(CUDA.device()))
    dev = FFS.GPUBackend(CUDA.CUDABackend())

    Random.seed!(42)
    L = 2π
    ms = (16, 16)
    dx = L / ms[1]
    dy = L / ms[2]
    xs = range(0.0, stop = L - dx, length = ms[1])
    ys = range(0.0, stop = L - dy, length = ms[2])
    xv = vec([x for x in xs, y in ys])
    yv = vec([y for x in xs, y in ys])
    u = @. cos(2xv) + 0.5 * sin(3yv)
    ug = FFS.UniformCartesianGrid((xv, yv); domain_size = (L, L))

    Test.@testset "GPU fast-transform parity vs CPU (real CUDA device)" begin
        # CUFFT (uniform grid) — fast GPU FFT vs CPU FFTW.
        c_fftw, _ = FFS.calculate_spectrum(ug, (u,), ms; transform = FFS.FFTBackend(), execution = FFS.SerialBackend())
        c_cufft, _ = FFS.calculate_spectrum(ug, (u,), ms; transform = FFS.FFTBackend(), execution = dev)
        Test.@test isapprox(Array(c_cufft), c_fftw; atol = 1e-10)

        # cuFINUFFT (scattered grid) — fast GPU NUFFT vs CPU FINUFFT.
        Random.seed!(7)
        M = 400
        px = rand(M) .* L
        py = rand(M) .* L
        fs = @. cos(px) + sin(2py)
        sg = FFS.ScatteredCartesianGrid((px, py); domain_size = (L, L))
        c_finufft, _ = FFS.calculate_spectrum(sg, (fs,), ms; transform = FFS.NUFFTBackend(), execution = FFS.SerialBackend(), eps = 1e-12)
        c_cuf, _ = FFS.calculate_spectrum(sg, (fs,), ms; transform = FFS.NUFFTBackend(), execution = dev, eps = 1e-12)
        Test.@test isapprox(Array(c_cuf), c_finufft; atol = 1e-8)

        # KA direct sum on the real CUDA device vs the serial CPU direct sum.
        c_dir, _ = FFS.calculate_spectrum(ug, (u,), ms; transform = FFS.DirectSumBackend(), execution = FFS.SerialBackend())
        c_kacuda, _ = FFS.calculate_spectrum(ug, (u,), ms; transform = FFS.DirectSumBackend(), execution = dev)
        Test.@test isapprox(Array(c_kacuda), c_dir; atol = 1e-10)

        # Spherical direct sum on the real CUDA device vs serial (no fast GPU SHT exists).
        Random.seed!(11)
        θ = rand(120) .* π
        φ = rand(120) .* 2π
        fθ = rand(120)
        sph = FFS.ScatteredSphericalGrid(θ, φ)
        cs_cpu, _ = FFS.calculate_spectrum(sph, (fθ,), (6, 11); transform = FFS.DirectSumBackend(), execution = FFS.SerialBackend())
        cs_gpu, _ = FFS.calculate_spectrum(sph, (fθ,), (6, 11); transform = FFS.DirectSumBackend(), execution = dev)
        Test.@test isapprox(Array(cs_gpu), cs_cpu; atol = 1e-9)
    end
    return nothing
end

main()
