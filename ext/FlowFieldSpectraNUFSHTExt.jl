module FlowFieldSpectraNUFSHTExt

using NUFSHT: NUFSHT
using FlowFieldSpectra: FlowFieldSpectra as FFS
using ComputationalBackends: ComputationalBackends
using SpectralBackends: SpectralBackends as SB
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

"""
    NUSHTSphericalPlan{T}

Reusable scattered-spherical NUFSHT plan: the fixed-node NUFSHT plan (point preset + FINUFFT setup),
the reused real-coefficient buffer, and — for `solve=true` — the LSMR solve workspace, all built once for a
fixed point set and batch shape. Reuse across many fields via `calculate_spectrum!`.
"""
struct NUSHTSphericalPlan{T, NB, P, CR, WS, KS} <: FFS.AbstractSpectralPlan
    plan::P                       # NUFSHT.NUSHTplan (fixed nodes + FINUFFT setup)
    C_real::CR                    # (Nθ, Nφ, B) real NUFSHT coeff buffer (FSH sph_mode layout), reused
    ws::WS                        # LSMRWorkspace for solve=true (built once); nothing otherwise
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
Base.show(io::IO, p::NUSHTSphericalPlan{T}) where {T} =
    print(io, "NUSHTSphericalPlan{", T, "}(lmax=", p.lmax, ", B=", p.B, p.solve ? ", solve" : "", ")")

function _nusht_plan(::Type{FT}, g, ms::Tuple, batch::NTuple{NB, Int}, exec;
        tol::Real, solve::Bool, maxiter::Int, rtol::Real, nufft) where {FT, NB}
    lmax = ms[1] - 1
    Nθ = lmax + 1
    Nφ = 2 * lmax + 1
    θ, φ = FFS.Grids._sph_points(g)
    B = prod(batch; init = 1)
    plan = NUFSHT.make_plan(FT, θ, φ, lmax; tol = tol, ntrans = B, nthreads = _nthreads(exec), nufft = nufft)
    C_real = zeros(FT, Nθ, Nφ, B)
    ws = solve ? NUFSHT.LSMRWorkspace(plan) : nothing
    ks = (0:lmax, -lmax:lmax)
    return NUSHTSphericalPlan{FT, NB, typeof(plan), typeof(C_real), typeof(ws), typeof(ks)}(
        plan, C_real, ws, lmax, Nθ, Nφ, batch, B, solve, maxiter, FT(rtol), ks)
end

"""
    plan_spectrum(NUFSHTSpectralBackend(), execution, grid, T, ms; batch=(), tol, solve, maxiter, rtol, nufft)

Reusable [`FFS.AbstractSpectralPlan`](@ref) for the scattered-spherical NUFSHT on a fixed point set.
Presets the points / NUFSHT plan / CG setup once; execute across many fields with
`calculate_spectrum!(coeffs, plan, field)`. `nufft` selects NUFSHT's internal NUFFT engine (a
`SpectralBackends` marker; default `AutoSpectralBackend()`); pass `SpectralBackends.NonuniformFFTsBackend()`
for the real-data half-spectrum fast path on a real field.
"""
function FFS.plan_spectrum(::SB.AbstractNUFSHTSpectralBackend,
        exec::Union{ComputationalBackends.AbstractSerialBackend, ComputationalBackends.AbstractThreadedBackend},
        g::FlowGeometries.Grids.AbstractGrid{<:FlowGeometries.Geometry.AbstractSphericalGeometry},
        ::Type{T}, ms::Tuple; batch::Tuple = (), tol::Real = 1.0e-8, solve::Bool = false,
        maxiter::Int = 500, rtol::Real = 1.0e-6, nufft::SB.AbstractSpectralBackend = SB.AutoSpectralBackend(),
        kwargs...) where {T}
    FT = real(float(T))
    return _nusht_plan(FT, g, ms, NTuple{length(batch), Int}(batch), exec;
        tol = tol, solve = solve, maxiter = maxiter, rtol = rtol, nufft = nufft)
end

"""
    calculate_spectrum!(coeffs, plan::NUSHTSphericalPlan, field) -> ks

Fill preallocated `coeffs` `(Nθ, Nφ, batch…)` with the scattered-spherical spectrum of `field`
`(N, batch…)`, reusing `plan`'s nodes / NUFSHT plan / LSMR solve workspace (no re-planning).
"""
function FFS.calculate_spectrum!(coeffs::AbstractArray{Complex{T}}, plan::NUSHTSphericalPlan{T}, field) where {T}
    if plan.solve
        NUFSHT.nusht_solve!(plan.C_real, field, plan.plan; ws = plan.ws, maxiter = plan.maxiter, rtol = plan.rtol)
    else
        NUFSHT.nusht_type1!(plan.C_real, field, plan.plan)
    end
    # Remap NUFSHT's FastSphericalHarmonics `sph_mode` layout → FFS's `sph_mode_index`, per batch slice.
    # Both index functions return a `CartesianIndex{2}` into the (Nθ, Nφ) block, so index the reshaped
    # (Nθ, Nφ, B) views with `[ci, b]`.
    lmax = plan.lmax
    Cr = reshape(plan.C_real, plan.Nθ, plan.Nφ, plan.B)
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

# One-shot: build a plan for this field's batch shape, then execute it once.
function FFS._calculate_spectrum_nufsht(exec::Union{ComputationalBackends.AbstractSerialBackend, ComputationalBackends.AbstractThreadedBackend},
        g::FlowGeometries.Grids.AbstractGrid{<:FlowGeometries.Geometry.AbstractSphericalGeometry}, field::AbstractArray, ms::Tuple;
        tol::Real = 1.0e-8, solve::Bool = false, maxiter::Int = 500, rtol::Real = 1.0e-6,
        nufft::SB.AbstractSpectralBackend = SB.AutoSpectralBackend(), kwargs...)
    FT = real(float(eltype(field)))
    batch = ntuple(i -> size(field, 1 + i), ndims(field) - 1)
    plan = _nusht_plan(FT, g, ms, batch, exec; tol = tol, solve = solve, maxiter = maxiter, rtol = rtol, nufft = nufft)
    coeffs = zeros(Complex{FT}, plan.Nθ, plan.Nφ, batch...)
    FFS.calculate_spectrum!(coeffs, plan, field)
    return coeffs, plan.ks
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
