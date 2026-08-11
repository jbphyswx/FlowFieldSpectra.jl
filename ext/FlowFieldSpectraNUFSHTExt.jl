module FlowFieldSpectraNUFSHTExt

using NUFSHT: NUFSHT
using FlowFieldSpectra: FlowFieldSpectra as FFS
using ComputationalBackends: ComputationalBackends
using FlowGeometries: FlowGeometries

# Non-uniform Spherical Harmonic Transform via NUFSHT, tensor-native + batched. The field `(N, batch…)`
# and its `ntrans = ∏batch` co-located slices transform in ONE guru-plan call; NUFSHT's real
# coefficients (`Nθ·Nφ·B`, FastSphericalHarmonics `sph_mode` layout) map into FFS's complex
# `(Nθ, Nφ, batch…)`. The plan (fixed nodes) is built once. `solve=true` runs the CG inverse. Points
# `(θ, φ)` = (colatitude, longitude) come from the FlowGeometries adapter's convention bridge (a
# structured grid is expanded to per-point form in the same column-major order as its field).

# NUFSHT FINUFFT thread count: 0 uses all cores (its default); 1 is serial.
_nthreads(::ComputationalBackends.AbstractThreadedBackend) = 0
_nthreads(::ComputationalBackends.AbstractExecutionBackend) = 1

function FFS._calculate_spectrum_nufsht(exec::Union{ComputationalBackends.AbstractSerialBackend, ComputationalBackends.AbstractThreadedBackend},
        g::FlowGeometries.Grids.AbstractGrid{<:FlowGeometries.Geometry.AbstractSphericalGeometry}, field::AbstractArray, ms::Tuple;
        tol::Real = 1.0e-8, solve::Bool = false, maxiter::Int = 500, rtol::Real = 1.0e-6, kwargs...)
    FT = real(float(eltype(field)))
    lmax = ms[1] - 1
    Nθ = lmax + 1
    Nφ = 2 * lmax + 1
    θ, φ = FFS.Grids._sph_points(g)
    batch = ntuple(i -> size(field, 1 + i), ndims(field) - 1)
    B = prod(batch; init = 1)
    plan = NUFSHT.make_plan(FT, θ, φ, lmax; tol = tol, ntrans = B, nthreads = _nthreads(exec))
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

# GPU NUFSHT (device-resident NUFSHT plan via cuFINUFFT) is provided by the NUFSHT × KernelAbstractions
# extension, which allocates the node/field arrays on the execution backend so NUFSHT.make_plan builds a
# device plan. This less-specific stub fires only when KernelAbstractions is not loaded.
function FFS._calculate_spectrum_nufsht(::ComputationalBackends.AbstractGPUBackend,
        g::FlowGeometries.Grids.AbstractGrid{<:FlowGeometries.Geometry.AbstractSphericalGeometry},
        field::AbstractArray, ms::Tuple; kwargs...)
    throw(ArgumentError("GPU NUFSHT requires `using KernelAbstractions` (to place the transform on the execution backend)."))
end

end # module FlowFieldSpectraNUFSHTExt
