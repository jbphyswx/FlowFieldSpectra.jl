module FlowFieldSpectraNUFSHTExt

using NUFSHT: NUFSHT
using FlowFieldSpectra: FlowFieldSpectra as FFS

# Non-uniform Spherical Harmonic Transform (scattered sphere) via NUFSHT, tensor-native + batched. The
# field `(N, batch…)` and its `ntrans = ∏batch` co-located slices transform in ONE guru-plan call;
# NUFSHT's real coefficients (`Nθ·Nφ·B`, FastSphericalHarmonics `sph_mode` layout) map into FFS's
# complex `(Nθ, Nφ, batch…)`. The plan (fixed nodes) is built once. `solve=true` runs the CG inverse.

# NUFSHT FINUFFT thread count: 0 uses all cores (its default); 1 is serial.
_nthreads(::FFS.ThreadedBackend) = 0
_nthreads(::FFS.AbstractExecutionBackend) = 1

function FFS._calculate_spectrum_nufsht(exec::Union{FFS.SerialBackend, FFS.ThreadedBackend},
        g::FFS.AbstractSphericalGrid, field::AbstractArray, ms::Tuple;
        tol::Real = 1.0e-8, solve::Bool = false, maxiter::Int = 500, rtol::Real = 1.0e-6, kwargs...)
    FT = real(float(eltype(field)))
    lmax = ms[1] - 1
    Nθ = lmax + 1
    Nφ = 2 * lmax + 1
    θ = g.coords[1]
    φ = g.coords[2]
    batch = ntuple(i -> size(field, 1 + i), ndims(field) - 1)
    B = prod(batch; init = 1)
    plan = NUFSHT.make_plan(θ, φ, lmax; tol = tol, T = FT, ntrans = B, nthreads = _nthreads(exec))
    C_real = zeros(FT, Nθ, Nφ, B)                       # NUFSHT real coeffs (FSH sph_mode layout)
    if solve
        NUFSHT.nusht_solve!(C_real, field, plan; maxiter = maxiter, rtol = rtol)
    else
        NUFSHT.nusht_type1!(C_real, field, plan)
    end
    coeffs = zeros(Complex{FT}, Nθ, Nφ, B)
    Cr = reshape(C_real, Nθ, Nφ, B)
    Cc = reshape(coeffs, Nθ, Nφ, B)
    @inbounds for b in 1:B
        for l in 0:lmax
            for m in -l:l
                Cc[FFS.sph_mode_index(l, m), b] = Cr[NUFSHT.FastSphericalHarmonics.sph_mode(l, m), b]
            end
        end
    end
    return reshape(coeffs, Nθ, Nφ, batch...), (0:lmax, -lmax:lmax)
end

# NUFSHT.jl has a device (cuFINUFFT) path selected by a CuArray node set; wiring GPUBackend to it is a
# CUDA-only follow-up. Until then, a clear error (never a silent CPU fallback).
function FFS._calculate_spectrum_nufsht(::FFS.GPUBackend, g::FFS.AbstractSphericalGrid,
        field::AbstractArray, ms::Tuple; kwargs...)
    throw(ArgumentError(
        "GPU NUFSHT is not wired in FlowFieldSpectra yet. Use a CPU execution backend for " *
        "NUFSHTBackend, or transform=DirectSumBackend() with a GPUBackend for an on-device spherical " *
        "direct sum.",
    ))
end

end # module FlowFieldSpectraNUFSHTExt
