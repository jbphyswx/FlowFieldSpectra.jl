module FlowFieldSpectraNUFSHTKernelAbstractionsExt

using NUFSHT: NUFSHT
using KernelAbstractions: KernelAbstractions as KA
using FlowFieldSpectra: FlowFieldSpectra as FFS
using ComputationalBackends: ComputationalBackends
using SpectralBackends: SpectralBackends as SB
using FlowGeometries: FlowGeometries

# =============================================================================
# GPU NUFSHT (scattered sphere): place the nodes + field on the execution backend (`KA.allocate`), then
# run NUFSHT's device-generic transform. NUFSHT.make_plan builds a device-RESIDENT plan from device
# nodes — cuFINUFFT on CUDA, FFTW/FINUFFT on KA.CPU() — so this is verifiable on KA.CPU and fast on a
# real GPU. Real coefficients map into FFS's complex `(Nθ, Nφ, batch…)` (FSH sph_mode layout). The
# small, M-independent FastTransforms sph2fourier step is a host bounce inside NUFSHT (its ceiling).
# The reusable plan holds the device plan + device field/coeff buffers + host staging, so a fixed
# point set pays the (device) planning cost once. Points `(θ, φ)` come from the FG convention bridge.
# =============================================================================

"""
    NUSHTSphericalGPUPlan{T}

Device-resident reusable NUFSHT plan (GPU execution): the device NUFSHT plan, the device field/coeff
buffers, the host staging buffer for the layout remap, and — for `solve=true` — the device CG
workspace. Built once for a fixed point set + batch shape; reuse via `calculate_spectrum!`.
"""
struct NUSHTSphericalGPUPlan{T, NB, P, FD, CD, HB, WS, KS} <: FFS.AbstractSpectralPlan
    plan::P            # device-resident NUFSHT plan (fixed nodes)
    fd::FD             # device (N, B) field buffer (host field copied in per call)
    Cd::CD             # device (Nθ, Nφ, B) real coeff buffer
    Cr_host::HB        # host (Nθ, Nφ, B) staging for the small layout remap
    ws::WS             # device LSMRWorkspace for solve=true; nothing otherwise
    lmax::Int
    Nθ::Int
    Nφ::Int
    batch::NTuple{NB, Int}
    B::Int
    solve::Bool
    maxiter::Int
    rtol::T
    ks::KS
end

# Custom show: the wrapped NUFSHT plan holds FINUFFT plans, whose default printing can segfault.
Base.show(io::IO, p::NUSHTSphericalGPUPlan{T}) where {T} =
    print(io, "NUSHTSphericalGPUPlan{", T, "}(lmax=", p.lmax, ", B=", p.B, p.solve ? ", solve" : "", ")")

function _nusht_gpu_plan(::Type{FT}, backend, g, ms::Tuple, batch::NTuple{NB, Int};
        tol::Real, solve::Bool, maxiter::Int, rtol::Real, nufft) where {FT, NB}
    lmax = ms[1] - 1
    Nθ = lmax + 1
    Nφ = 2 * lmax + 1
    θ, φ = FFS.Grids._sph_points(g)
    N = length(θ)
    B = prod(batch; init = 1)
    θd = KA.allocate(backend, FT, N); copyto!(θd, FT.(θ))
    φd = KA.allocate(backend, FT, N); copyto!(φd, FT.(φ))
    plan = NUFSHT.make_plan(FT, θd, φd, lmax; tol = tol, ntrans = B, nufft = nufft)   # device-resident
    fd = KA.allocate(backend, FT, N, B)
    Cd = KA.zeros(backend, FT, Nθ, Nφ, B)
    Cr_host = zeros(FT, Nθ, Nφ, B)
    ws = solve ? NUFSHT.LSMRWorkspace(plan) : nothing
    ks = (0:lmax, -lmax:lmax)
    return NUSHTSphericalGPUPlan{FT, NB, typeof(plan), typeof(fd), typeof(Cd), typeof(Cr_host), typeof(ws), typeof(ks)}(
        plan, fd, Cd, Cr_host, ws, lmax, Nθ, Nφ, batch, B, solve, maxiter, FT(rtol), ks)
end

function FFS.plan_spectrum(::SB.AbstractNUFSHTSpectralBackend, exec::ComputationalBackends.GPUBackend{<:KA.Backend},
        g::FlowGeometries.Grids.AbstractGrid{<:FlowGeometries.Geometry.AbstractSphericalGeometry},
        ::Type{T}, ms::Tuple; batch::Tuple = (), tol::Real = 1.0e-8, solve::Bool = false,
        maxiter::Int = 500, rtol::Real = 1.0e-6, nufft::SB.AbstractSpectralBackend = SB.AutoSpectralBackend(),
        kwargs...) where {T}
    FT = real(float(T))
    return _nusht_gpu_plan(FT, exec.backend, g, ms, NTuple{length(batch), Int}(batch);
        tol = tol, solve = solve, maxiter = maxiter, rtol = rtol, nufft = nufft)
end

function FFS.calculate_spectrum!(coeffs::AbstractArray{Complex{T}}, plan::NUSHTSphericalGPUPlan{T}, field) where {T}
    copyto!(plan.fd, field)                                    # host field → device buffer
    if plan.solve
        NUFSHT.nusht_solve!(plan.Cd, plan.fd, plan.plan; ws = plan.ws, maxiter = plan.maxiter, rtol = plan.rtol)
    else
        NUFSHT.nusht_type1!(plan.Cd, plan.fd, plan.plan)
    end
    copyto!(plan.Cr_host, plan.Cd)                             # device → host for the small layout remap
    lmax = plan.lmax
    Cr = reshape(plan.Cr_host, plan.Nθ, plan.Nφ, plan.B)
    Cc = reshape(coeffs, plan.Nθ, plan.Nφ, plan.B)
    @inbounds for b in 1:plan.B
        for l in 0:lmax
            for m in -l:l
                Cc[FFS.sph_mode_index(l, m), b] = Cr[NUFSHT.FastSphericalHarmonics.sph_mode(l, m), b]
            end
        end
    end
    return plan.ks
end

# One-shot GPU: build a device plan for this field's batch shape, then execute it once.
# More specific than the NUFSHT-only `AbstractGPUBackend` stub, so it wins when KernelAbstractions is
# loaded.
function FFS._calculate_spectrum_nufsht(exec::ComputationalBackends.GPUBackend{<:KA.Backend},
        g::FlowGeometries.Grids.AbstractGrid{<:FlowGeometries.Geometry.AbstractSphericalGeometry},
        field::AbstractArray, ms::Tuple;
        tol::Real = 1.0e-8, solve::Bool = false, maxiter::Int = 500, rtol::Real = 1.0e-6,
        nufft::SB.AbstractSpectralBackend = SB.AutoSpectralBackend(), kwargs...)
    FT = real(float(eltype(field)))
    batch = ntuple(i -> size(field, 1 + i), ndims(field) - 1)   # scattered sphere: 1 spatial dim
    plan = _nusht_gpu_plan(FT, exec.backend, g, ms, batch; tol = tol, solve = solve, maxiter = maxiter, rtol = rtol, nufft = nufft)
    coeffs = zeros(Complex{FT}, plan.Nθ, plan.Nφ, batch...)
    FFS.calculate_spectrum!(coeffs, plan, field)
    return coeffs, plan.ks
end

end # module FlowFieldSpectraNUFSHTKernelAbstractionsExt
