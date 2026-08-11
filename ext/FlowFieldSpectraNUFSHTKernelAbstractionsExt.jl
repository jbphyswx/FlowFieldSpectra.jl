module FlowFieldSpectraNUFSHTKernelAbstractionsExt

using NUFSHT: NUFSHT
using KernelAbstractions: KernelAbstractions as KA
using FlowFieldSpectra: FlowFieldSpectra as FFS
using ComputationalBackends: ComputationalBackends
using FlowGeometries: FlowGeometries

# =============================================================================
# GPU NUFSHT (scattered sphere): place the nodes + field on the execution backend (`KA.allocate`), then
# run NUFSHT's device-generic transform. NUFSHT.make_plan builds a device-RESIDENT plan from device
# nodes — cuFINUFFT on CUDA, FFTW/FINUFFT on KA.CPU() — so this is verifiable on KA.CPU and fast on a
# real GPU. Real coefficients map into FFS's complex `(Nθ, Nφ, batch…)` (FSH sph_mode layout). The
# small, M-independent FastTransforms sph2fourier step is a host bounce inside NUFSHT (its ceiling).
# More specific than the NUFSHT-only `AbstractGPUBackend` stub, so it wins when KernelAbstractions is
# loaded. Points `(θ, φ)` come from the FlowGeometries adapter's convention bridge.
# =============================================================================

function FFS._calculate_spectrum_nufsht(exec::ComputationalBackends.GPUBackend{<:KA.Backend},
        g::FlowGeometries.Grids.AbstractGrid{<:FlowGeometries.Geometry.AbstractSphericalGeometry},
        field::AbstractArray, ms::Tuple;
        tol::Real = 1.0e-8, solve::Bool = false, maxiter::Int = 500, rtol::Real = 1.0e-6, kwargs...)
    backend = exec.backend
    FT = real(float(eltype(field)))
    lmax = ms[1] - 1
    Nθ = lmax + 1
    Nφ = 2 * lmax + 1
    θ, φ = FFS.Grids._sph_points(g)
    N = length(θ)
    batch = ntuple(i -> size(field, 1 + i), ndims(field) - 1)   # scattered sphere: 1 spatial dim
    B = prod(batch; init = 1)
    # Nodes + field on the execution backend ⇒ NUFSHT.make_plan yields a device-resident plan.
    θd = KA.allocate(backend, FT, N); copyto!(θd, FT.(θ))
    φd = KA.allocate(backend, FT, N); copyto!(φd, FT.(φ))
    fd = KA.allocate(backend, FT, N, B); copyto!(fd, field)     # (N, ∏batch), linear
    plan = NUFSHT.make_plan(FT, θd, φd, lmax; tol = tol, ntrans = B)
    Cd = KA.zeros(backend, FT, Nθ, Nφ, B)                       # real coeffs, device-resident
    if solve
        NUFSHT.nusht_solve!(Cd, fd, plan; maxiter = maxiter, rtol = rtol)
    else
        NUFSHT.nusht_type1!(Cd, fd, plan)
    end
    Cr = Array(Cd)                                              # → host for the small (Nθ·Nφ) layout remap
    coeffs = zeros(Complex{FT}, Nθ, Nφ, B)
    @inbounds for b in 1:B
        for l in 0:lmax
            for m in -l:l
                coeffs[FFS.sph_mode_index(l, m), b] = Cr[NUFSHT.FastSphericalHarmonics.sph_mode(l, m), b]
            end
        end
    end
    return reshape(coeffs, Nθ, Nφ, batch...), (0:lmax, -lmax:lmax)
end

end # module FlowFieldSpectraNUFSHTKernelAbstractionsExt
