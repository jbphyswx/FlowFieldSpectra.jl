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
struct NUSHTSphericalGPUPlan{T, CT, NB, P, FD, FH, CD, HB, WS, QW, KS} <: FFS.AbstractSpectralPlan
    plan::P            # device-resident NUFSHT plan (fixed nodes)
    fd::FD             # device (N, B) field buffer (host field copied in per call)
    fh::FH             # host (N, B) real staging for one component of a complex field
    Cd::CD             # device (Nθ, Nφ, B) real coeff buffer
    Cr_host::HB        # host (Nθ, Nφ, B) staging for the small layout remap
    ws::WS             # device LSMRWorkspace for solve=true; nothing otherwise
    qwd::QW            # device (N, 1) per-node quadrature weights, Σw = 4π; nothing for solve=true
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

function _nusht_gpu_plan(::Type{CT}, ::Type{FT}, backend, g, ms::Tuple, batch::NTuple{NB, Int};
        tol::Real, solve::Bool, maxiter::Int, rtol::Real, nufft,
        sampling = nothing, weights = nothing) where {CT, FT, NB}
    lmax = ms[1] - 1
    Nθ = lmax + 1
    Nφ = 2 * lmax + 1
    θ, φ, w = FFS.DirectSum._sph_point_data(g, FT; sampling = sampling, weights = weights)
    N = length(θ)
    B = prod(batch; init = 1)
    θd = KA.allocate(backend, FT, N); copyto!(θd, FT.(θ))
    φd = KA.allocate(backend, FT, N); copyto!(φd, FT.(φ))
    plan = NUFSHT.make_plan(FT, θd, φd, lmax; tol = tol, ntrans = B, nufft = nufft)   # device-resident
    fd = KA.allocate(backend, FT, N, B)
    fh = zeros(FT, N, B)
    Cd = KA.zeros(backend, FT, Nθ, Nφ, B)
    Cr_host = zeros(FT, Nθ, Nφ, B)
    ws = solve ? NUFSHT.LSMRWorkspace(plan) : nothing
    # `nusht_type1!` is the unweighted adjoint, so the quadrature reaches it through the field. Shaped
    # `(N, 1)` so one broadcast weights every transform in the batch. The solve reads the raw field.
    qwd = if solve
        nothing
    else
        d = KA.allocate(backend, FT, N, 1)
        copyto!(d, reshape(FT.(collect(w)), N, 1))
        d
    end
    ks = (0:lmax, -lmax:lmax)
    return NUSHTSphericalGPUPlan{FT, CT, NB, typeof(plan), typeof(fd), typeof(fh), typeof(Cd), typeof(Cr_host), typeof(ws), typeof(qwd), typeof(ks)}(
        plan, fd, fh, Cd, Cr_host, ws, qwd, lmax, Nθ, Nφ, batch, B, solve, maxiter, FT(rtol), ks)
end

FFS.Plans.coefficient_size(p::NUSHTSphericalGPUPlan) = (p.Nθ, p.Nφ, p.batch...)
FFS.Plans.coefficient_type(::NUSHTSphericalGPUPlan{T, CT}) where {T, CT} = CT
FFS.Plans.wavenumbers(p::NUSHTSphericalGPUPlan) = p.ks

function FFS.plan_spectrum(::SB.AbstractNUFSHTSpectralBackend, exec::ComputationalBackends.GPUBackend{<:KA.Backend},
        g::FlowGeometries.Grids.AbstractGrid{<:FFS.Grids.SphericalHarmonicGeometry},
        ::Type{T}, ms::Tuple; batch::Tuple = (), tol::Real = 1.0e-8, solve::Bool = false,
        maxiter::Int = 500, rtol::Real = 1.0e-6, nufft::SB.AbstractSpectralBackend = SB.AutoSpectralBackend(),
        sampling = nothing, weights = nothing, kwargs...) where {T}
    FT = real(float(T))
    return _nusht_gpu_plan(FFS.sph_coeff_type(T, FT), FT, exec.backend, g, ms,
        NTuple{length(batch), Int}(batch);
        tol = tol, solve = solve, maxiter = maxiter, rtol = rtol, nufft = nufft,
        sampling = sampling, weights = weights)
end

function FFS.calculate_spectrum!(coeffs::AbstractArray{<:Number}, plan::NUSHTSphericalGPUPlan{T}, field) where {T}
    fill!(coeffs, zero(eltype(coeffs)))
    _nusht_gpu_pass!(coeffs, plan, field, real, one(T))
    eltype(field) <: Real || _nusht_gpu_pass!(coeffs, plan, field, imag, im)
    return plan.ks
end

# The host array staged to the device for one component: a real field crosses directly, and one component
# of a complex field is written into the plan's host buffer first.
@inline _gpu_component(::AbstractMatrix, field::AbstractArray{<:Real}, part) = field
@inline function _gpu_component(fh::AbstractMatrix{FT}, field::AbstractArray{<:Complex}, part) where {FT}
    N = size(fh, 1)
    @inbounds for b in axes(fh, 2)
        off = (b - 1) * N
        for j in 1:N
            fh[j, b] = FT(part(field[off + j]))
        end
    end
    return fh
end

# One real field component through the device NUFSHT, added into `coeffs` at weight `scale`. NUFSHT's
# transform is real and linear, so a real field runs this once with `real` (the identity on it) at weight
# 1, and a complex field runs it again with `imag` at weight `im`.
function _nusht_gpu_pass!(coeffs, plan::NUSHTSphericalGPUPlan{T}, field, part, scale) where {T}
    copyto!(plan.fd, _gpu_component(plan.fh, field, part))     # host field → device buffer
    if plan.solve
        NUFSHT.nusht_solve!(plan.Cd, plan.fd, plan.plan; ws = plan.ws, maxiter = plan.maxiter, rtol = plan.rtol)
    else
        # Weight the field into the quadrature the projection integrates. `ifelse` selects a value, so a
        # zero-weight node yields zero even where the field holds `NaN`.
        qwd = plan.qwd
        plan.fd .= ifelse.(iszero.(qwd), zero(T), plan.fd .* qwd)
        NUFSHT.nusht_type1!(plan.Cd, plan.fd, plan.plan)
    end
    copyto!(plan.Cr_host, plan.Cd)                             # device → host for the small layout remap
    lmax = plan.lmax
    Cr = reshape(plan.Cr_host, plan.Nθ, plan.Nφ, plan.B)
    Cc = reshape(coeffs, plan.Nθ, plan.Nφ, plan.B)
    @inbounds for b in 1:plan.B
        for l in 0:lmax
            for m in -l:l
                Cc[FFS.sph_mode_index(l, m), b] += scale * Cr[NUFSHT.FastSphericalHarmonics.sph_mode(l, m), b]
            end
        end
    end
    return coeffs
end

# One-shot GPU: build a device plan for this field's batch shape, then execute it once.
# More specific than the NUFSHT-only `AbstractGPUBackend` stub, so it wins when KernelAbstractions is
# loaded.
function FFS._calculate_spectrum_nufsht(exec::ComputationalBackends.GPUBackend{<:KA.Backend},
        g::FlowGeometries.Grids.AbstractGrid{<:FFS.Grids.SphericalHarmonicGeometry},
        field::AbstractArray, ms::Tuple;
        tol::Real = 1.0e-8, solve::Bool = false, maxiter::Int = 500, rtol::Real = 1.0e-6,
        nufft::SB.AbstractSpectralBackend = SB.AutoSpectralBackend(),
        sampling = nothing, weights = nothing, kwargs...)
    FT = real(float(eltype(field)))
    batch = FFS.Grids.field_batch_shape(g, field)
    plan = _nusht_gpu_plan(FFS.sph_coeff_type(eltype(field), FT), FT, exec.backend, g, ms, batch;
        tol = tol, solve = solve, maxiter = maxiter, rtol = rtol, nufft = nufft,
        sampling = sampling, weights = weights)
    coeffs = FFS.allocate_coefficients(plan)
    FFS.calculate_spectrum!(coeffs, plan, field)
    return coeffs, plan.ks
end

# =============================================================================
# Inverse (synthesis) on a device backend via `nusht_type2!`, the adjoint pair of the `nusht_type1!` the
# forward runs. Nodes are placed on the execution backend, so `NUFSHT.make_plan` builds a device-resident
# plan. Coefficients arrive in `sph_mode_index` layout and are remapped on the host into NUFSHT's
# FastSphericalHarmonics `sph_mode` real buffer, which is then staged to the device; the remap is over
# the mode set, whose size is independent of the node count. NUFSHT's coefficients are real, so a complex
# coefficient array is evaluated one real component at a time, which the transform's linearity permits.
# =============================================================================

# Reusable device synthesis: the device-resident NUFSHT plan (nodes preset once) plus the host staging
# for the mode remap, whose size is independent of the node count.
struct NUSHTSynthesisGPUPlan{FT, R, NB, P, CD, FD, HB, FH, SP} <: FFS.AbstractSynthesisPlan
    plan::P
    Cd::CD
    fd::FD
    Cr_host::HB
    f_host::FH
    lmax::Int
    Nθ::Int
    Nφ::Int
    N::Int
    spatial::SP
    batch::NTuple{NB, Int}
    B::Int
end

# A default show of a struct holding FINUFFT plans can segfault.
Base.show(io::IO, p::NUSHTSynthesisGPUPlan{FT, R}) where {FT, R} =
    print(io, "NUSHTSynthesisGPUPlan{", FT, "}(lmax=", p.lmax, ", ", R ? "real" : "complex", ")")

FFS.Plans.field_size(p::NUSHTSynthesisGPUPlan) = (p.spatial..., p.batch...)
FFS.Plans.field_type(::NUSHTSynthesisGPUPlan{FT, R}) where {FT, R} = R ? FT : Complex{FT}

function FFS.Plans.plan_synthesis(::SB.AbstractNUFSHTSpectralBackend,
        exec::ComputationalBackends.GPUBackend{<:KA.Backend},
        g::FlowGeometries.Grids.AbstractGrid{<:FFS.Grids.SphericalHarmonicGeometry},
        ::Type{T}, ms::NTuple{2, Int}; batch::Tuple = (), tol::Real = 1.0e-8,
        nufft::SB.AbstractSpectralBackend = SB.AutoSpectralBackend(), kwargs...) where {T}
    FT = real(float(T))
    backend = exec.backend
    lmax = ms[1] - 1
    Nθ = lmax + 1
    Nφ = 2 * lmax + 1
    θ, φ = FFS.Grids._sph_points(g)
    N = length(θ)
    bt = NTuple{length(batch), Int}(batch)
    B = prod(bt; init = 1)
    θd = KA.allocate(backend, FT, N); copyto!(θd, FT.(θ))
    φd = KA.allocate(backend, FT, N); copyto!(φd, FT.(φ))
    plan = NUFSHT.make_plan(FT, θd, φd, lmax; tol = tol, ntrans = B, nufft = nufft)
    sp = size(g)
    return NUSHTSynthesisGPUPlan{FT, T <: Real, length(bt), typeof(plan), typeof(KA.zeros(backend, FT, 1, 1, 1)),
            typeof(KA.allocate(backend, FT, 1, 1)), Array{FT, 3}, Matrix{FT}, typeof(sp)}(
        plan, KA.zeros(backend, FT, Nθ, Nφ, B), KA.allocate(backend, FT, N, B),
        zeros(FT, Nθ, Nφ, B), zeros(FT, N, B), lmax, Nθ, Nφ, N, sp, bt, B)
end

function FFS.Plans.synthesize!(out::AbstractArray, plan::NUSHTSynthesisGPUPlan{FT, R},
        coeffs::AbstractArray; ks = nothing) where {FT, R}
    size(out) == FFS.Plans.field_size(plan) || throw(DimensionMismatch(
        "out is $(size(out)); this plan writes $(FFS.Plans.field_size(plan))"))
    size(coeffs)[1:2] == (plan.Nθ, plan.Nφ) || throw(DimensionMismatch(
        "spherical coefficients must be (Nθ, Nφ) = ($(plan.Nθ), $(plan.Nφ)) on the spectral dims; " *
        "got $(size(coeffs)[1:2])"))
    lmax = plan.lmax
    B = plan.B
    Cc = reshape(coeffs, plan.Nθ, plan.Nφ, B)
    O = reshape(out, plan.N, B)
    ET = R ? FT : Complex{FT}
    fill!(O, zero(ET))
    ncomp = R ? 1 : 2
    @inbounds for comp in 1:ncomp
        fill!(plan.Cr_host, zero(FT))
        for b in 1:B, l in 0:lmax, m in -l:l
            z = Cc[FFS.sph_mode_index(l, m), b]
            plan.Cr_host[NUFSHT.FastSphericalHarmonics.sph_mode(l, m), b] = comp == 1 ? real(z) : imag(z)
        end
        copyto!(plan.Cd, plan.Cr_host)
        NUFSHT.nusht_type2!(plan.fd, plan.Cd, plan.plan)
        copyto!(plan.f_host, plan.fd)
        for b in 1:B, j in 1:plan.N
            O[j, b] += comp == 1 ? ET(plan.f_host[j, b]) : ET(im * plan.f_host[j, b])
        end
    end
    return out
end

function FFS._synthesize(::SB.AbstractNUFSHTSpectralBackend,
        exec::ComputationalBackends.GPUBackend{<:KA.Backend},
        g::FlowGeometries.Grids.AbstractGrid{<:FFS.Grids.SphericalHarmonicGeometry},
        coeffs::AbstractArray, ms::NTuple{2, Int}; real_output::Bool = true, iflag::Int = 1,
        ks = nothing, tol::Real = 1.0e-8,
        nufft::SB.AbstractSpectralBackend = SB.AutoSpectralBackend(), kwargs...)
    FT = real(float(eltype(coeffs)))
    backend = exec.backend
    lmax = ms[1] - 1
    Nθ = lmax + 1
    Nφ = 2 * lmax + 1
    size(coeffs)[1:2] == (Nθ, Nφ) || throw(DimensionMismatch(
        "spherical coefficients must be (Nθ, Nφ) = ($Nθ, $Nφ) on the spectral dims; got $(size(coeffs)[1:2])."))
    θ, φ = FFS.Grids._sph_points(g)
    N = length(θ)
    batch = ntuple(i -> size(coeffs, 2 + i), ndims(coeffs) - 2)
    B = prod(batch; init = 1)
    θd = KA.allocate(backend, FT, N); copyto!(θd, FT.(θ))
    φd = KA.allocate(backend, FT, N); copyto!(φd, FT.(φ))
    plan = NUFSHT.make_plan(FT, θd, φd, lmax; tol = tol, ntrans = B, nufft = nufft)
    Cr_host = zeros(FT, Nθ, Nφ, B)
    Cd = KA.zeros(backend, FT, Nθ, Nφ, B)
    fd = KA.allocate(backend, FT, N, B)
    f_host = zeros(FT, N, B)
    Cc = reshape(coeffs, Nθ, Nφ, B)
    # Synthesis writes the grid's own spatial shape, matching what the forward consumed; a node cloud's
    # `size` is `(N,)`, so the two coincide there.
    sp = size(g)
    # Branch on the output element type, so `_synth_nusht_gpu!` specializes on a concrete array.
    return real_output ?
        _synth_nusht_gpu!(zeros(FT, sp..., batch...), Cc, Cr_host, Cd, fd, f_host, plan, lmax, N, B, 1) :
        _synth_nusht_gpu!(zeros(Complex{FT}, sp..., batch...), Cc, Cr_host, Cd, fd, f_host, plan, lmax, N, B, 2)
end

# `ET` fixes the destination type; `ncomp` is 1 for a real field and 2 for a complex one, whose real and
# imaginary coefficient parts are evaluated in turn and recombined.
function _synth_nusht_gpu!(out::AbstractArray{ET}, Cc, Cr_host, Cd, fd, f_host, plan, lmax::Int,
        N::Int, B::Int, ncomp::Int) where {ET}
    O = reshape(out, N, B)
    @inbounds for comp in 1:ncomp
        fill!(Cr_host, zero(eltype(Cr_host)))
        for b in 1:B, l in 0:lmax, m in -l:l
            z = Cc[FFS.sph_mode_index(l, m), b]
            Cr_host[NUFSHT.FastSphericalHarmonics.sph_mode(l, m), b] = comp == 1 ? real(z) : imag(z)
        end
        copyto!(Cd, Cr_host)                                   # host → device
        NUFSHT.nusht_type2!(fd, Cd, plan)
        copyto!(f_host, fd)                                    # device → host
        for b in 1:B, j in 1:N
            O[j, b] += comp == 1 ? ET(f_host[j, b]) : ET(im * f_host[j, b])
        end
    end
    return out
end

end # module FlowFieldSpectraNUFSHTKernelAbstractionsExt
