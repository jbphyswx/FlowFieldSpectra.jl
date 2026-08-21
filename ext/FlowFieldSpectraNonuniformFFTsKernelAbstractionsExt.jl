module FlowFieldSpectraNonuniformFFTsKernelAbstractionsExt

# `@index`/`@Const` are imported UNQUALIFIED: the `@kernel` macro only recognizes them by their bare
# names inside a kernel body. Everything else stays module-qualified.
using KernelAbstractions: KernelAbstractions as KA, @index, @Const
using NonuniformFFTs: NonuniformFFTs
using FlowFieldSpectra: FlowFieldSpectra as FFS
using ComputationalBackends: ComputationalBackends
using SpectralBackends: SpectralBackends
using FlowGeometries: FlowGeometries

# =============================================================================
# Device-generic NonuniformFFTs (type-1) for `GPUBackend`: thread the execution backend into
# `PlanNUFFT(…; backend=…)`, stage the points/phase/tables to that backend, and reconstruct FFS's full
# centered spectrum with broadcasts (complex path) / a KA kernel (real path). One code path serves any
# KA backend (`KA.CPU()`, JLArrays' `JLBackend`, CUDA, ROCm, …); it never scalar-indexes a device array
# on the host. Same math/convention as the CPU ext (`Σⱼ vⱼ e^{-i k·xⱼ}`, centered modes, ×1/M). The CPU
# (Serial/Threaded) fast paths live in the base ext; `GPUBackend` dispatch here is strictly more specific.
# =============================================================================

_default_eps(::Type{Float32}) = 1.0f-6
_default_eps(::Type{T}) where {T <: Real} = 1.0e-8
_default_eps(::Type{Complex{T}}) where {T} = _default_eps(T)

function _plan_accuracy(eps::Real, ms::NTuple)
    m = clamp(ceil(Int, -log10(eps)) + 1, 1, 16)
    σ = float(max(2, cld(2 * m, minimum(ms))))
    return NonuniformFFTs.HalfSupport(m), σ
end

# Scaled points (→ [0, 2π)), centered offset-correction phase (× 1/M; built for iflag=+1), physical
# wavenumbers — all on the host; the device buffers are staged from these once at plan construction.
function _scaled_phase_ks(::Type{Tr}, coords::Tuple, ms::NTuple{D, Int}, Ls::NTuple{D}, iflag::Int) where {Tr <: Real, D}
    M = length(coords[1])
    for d in 1:D
        length(coords[d]) == M || throw(DimensionMismatch("coordinate $d length mismatch"))
    end
    offsets = ntuple(d -> Tr(minimum(coords[d])), D)
    ranges = ntuple(d -> (r = Tr(Ls[d]); r == 0 ? one(Tr) : r), D)
    scaled = ntuple(d -> Tr(2π) .* (Tr.(coords[d]) .- offsets[d]) ./ ranges[d], D)
    k_ints = ntuple(d -> collect(-(ms[d] ÷ 2):((ms[d] - 1) ÷ 2)), D)
    inv_M = one(Tr) / M
    phase = Array{Complex{Tr}, D}(undef, ms...)
    @inbounds for I in CartesianIndices(ms)
        p = one(Complex{Tr})
        for d in 1:D
            p *= cis(-iflag * k_ints[d][I[d]] * (offsets[d] * Tr(2π) / ranges[d]))
        end
        phase[I] = p * inv_M
    end
    ks_phys = FFS.Grids.physical_wavenumbers(ranges, ms, Tr)
    return scaled, phase, ks_phys, M
end

function _mirror_index(N::Int)
    half = N ÷ 2
    hi = (N - 1) ÷ 2
    mir = Vector{Int}(undef, N)
    @inbounds for i in 1:N
        m = -(i - 1 - half)
        m > hi && (m -= N)
        mir[i] = m + half + 1
    end
    return mir
end

@inline _ovs_index(f::Int, Ñ::Int) = f >= 0 ? f + 1 : Ñ + f + 1
@inline function _gk_index(f::Int, N::Int)
    half = N ÷ 2
    f > (N - 1) ÷ 2 && (f -= N)
    return f + half + 1
end
@inline function _has_even_nyquist(I, ms::NTuple{D, Int}, half, ::Val{D}) where {D}
    any(ntuple(d -> d >= 2 && iseven(ms[d]) && (I[d] - 1 - half[d]) == -half[d], Val(D)))
end

# Stage a host array/tuple onto the KA backend.
_to_dev(backend, ::Type{S}, v) where {S} = (d = KA.allocate(backend, S, size(v)...); copyto!(d, S.(v)); d)

# ---- device plans (buffers + tables live on the execution backend) ----

struct NUFFTNonuniformComplexGPUPlan{T, D, P, CJ, FK, OB, HB, PH, KS} <: FFS.AbstractSpectralPlan
    plan::P
    cj::CJ                           # device (M,) complex strengths
    fk::FK                           # device (ms…) complex modes
    out::OB                          # device (ms…) complex output (fk × phase)
    out_host::HB                     # host (ms…) staging for the copy back to `coeffs`
    ms::NTuple{D, Int}
    M::Int
    iflag::Int
    phase::PH                        # device (ms…) offset phase × 1/M
    ks_phys::KS
end

struct NUFFTNonuniformRealGPUPlan{T, D, P, CJ, FK, OB, HB, PH, MIR, US, GK, KS} <: FFS.AbstractSpectralPlan
    plan::P                          # device PlanNUFFT{T<:Real}, half-spectrum
    cj::CJ                           # device (M,) REAL strengths
    fk_half::FK                      # device (N₁÷2+1, N₂…) complex half-spectrum
    out::OB                          # device (ms…) complex output
    out_host::HB                     # host (ms…) staging
    ms::NTuple{D, Int}
    M::Int
    iflag::Int
    phase::PH                        # device (ms…) phase × 1/M (built for iflag=+1)
    mir::MIR                         # NTuple{D, device Vector{Int}} per-axis mirror indices
    us_ovs::US                       # ref to plan.data.ûs[1] (device oversampled spectrum; refilled per exec)
    novs::NTuple{D, Int}
    gk::GK                           # NTuple{D, device Vector} kernel Fourier coefficients
    normfactor::T
    ks_phys::KS
end

Base.show(io::IO, ::NUFFTNonuniformComplexGPUPlan{T, D}) where {T, D} = print(io, "NUFFTNonuniformComplexGPUPlan{$T, $D}(NonuniformFFTs, device)")
Base.show(io::IO, ::NUFFTNonuniformRealGPUPlan{T, D}) where {T, D} = print(io, "NUFFTNonuniformRealGPUPlan{$T, $D}(NonuniformFFTs, device)")

function _nu_gpu_plan(::Type{Complex{Tr}}, backend, coords::Tuple, ms::NTuple{D, Int}, Ls::NTuple{D},
        iflag::Int, eps::Real) where {Tr <: Real, D}
    scaled_h, phase_h, ks_phys, M = _scaled_phase_ks(Tr, coords, ms, Ls, iflag)
    hs, σ = _plan_accuracy(eps, ms)
    plan = NonuniformFFTs.PlanNUFFT(Complex{Tr}, ms; ntransforms = Val(1), fftshift = true, m = hs, σ = σ, backend = backend)
    NonuniformFFTs.set_points!(plan, ntuple(d -> _to_dev(backend, Tr, scaled_h[d]), D))
    cj = KA.allocate(backend, Complex{Tr}, M)
    fk = KA.allocate(backend, Complex{Tr}, ms...)
    out = KA.allocate(backend, Complex{Tr}, ms...)
    out_host = Array{Complex{Tr}, D}(undef, ms...)
    phase = _to_dev(backend, Complex{Tr}, phase_h)
    return NUFFTNonuniformComplexGPUPlan{Tr, D, typeof(plan), typeof(cj), typeof(fk), typeof(out), typeof(out_host), typeof(phase), typeof(ks_phys)}(
        plan, cj, fk, out, out_host, ms, M, iflag, phase, ks_phys)
end

function _nu_gpu_plan(::Type{Tr}, backend, coords::Tuple, ms::NTuple{D, Int}, Ls::NTuple{D},
        iflag::Int, eps::Real) where {Tr <: Real, D}
    scaled_h, phase_h, ks_phys, M = _scaled_phase_ks(Tr, coords, ms, Ls, 1)   # sign via final conj
    hs, σ = _plan_accuracy(eps, ms)
    plan = NonuniformFFTs.PlanNUFFT(Tr, ms; ntransforms = Val(1), fftshift = true, m = hs, σ = σ, backend = backend)
    NonuniformFFTs.set_points!(plan, ntuple(d -> _to_dev(backend, Tr, scaled_h[d]), D))
    cj = KA.allocate(backend, Tr, M)
    fk_half = KA.allocate(backend, Complex{Tr}, size(plan)...)
    out = KA.allocate(backend, Complex{Tr}, ms...)
    out_host = Array{Complex{Tr}, D}(undef, ms...)
    phase = _to_dev(backend, Complex{Tr}, phase_h)
    mir = ntuple(d -> _to_dev(backend, Int, _mirror_index(ms[d])), D)
    us_ovs = plan.data.ûs[1]                                     # device oversampled spectrum
    novs = ntuple(d -> d == 1 ? 2 * (size(us_ovs, 1) - 1) : size(us_ovs, d), D)
    gk = ntuple(d -> _to_dev(backend, Tr, collect(Tr, NonuniformFFTs.fourier_coefficients(plan.kernels[d]))), D)
    normfactor = prod(2 * Tr(π) / novs[d] for d in 1:D)
    return NUFFTNonuniformRealGPUPlan{Tr, D, typeof(plan), typeof(cj), typeof(fk_half), typeof(out), typeof(out_host), typeof(phase), typeof(mir), typeof(us_ovs), typeof(gk), typeof(ks_phys)}(
        plan, cj, fk_half, out, out_host, ms, M, iflag, phase, mir, us_ovs, novs, gk, normfactor, ks_phys)
end

# ---- complex path (broadcasts) ----

function FFS.calculate_spectrum!(coeffs::AbstractArray{Complex{T}}, plan::NUFFTNonuniformComplexGPUPlan{T, D},
        field) where {T, D}
    M = plan.M
    Pm = prod(plan.ms)
    ntrans = length(field) ÷ M
    neg = plan.iflag < 0
    @inbounds for t in 1:ntrans
        foff = (t - 1) * M
        copyto!(plan.cj, view(field, (foff + 1):(foff + M)))     # host/device field → device
        neg && (plan.cj .= conj.(plan.cj))
        NonuniformFFTs.exec_type1!(plan.fk, plan.plan, plan.cj)
        if neg
            plan.out .= conj.(plan.fk) .* plan.phase
        else
            plan.out .= plan.fk .* plan.phase
        end
        copyto!(plan.out_host, plan.out)                          # device → host
        coff = (t - 1) * Pm
        copyto!(view(coeffs, (coff + 1):(coff + Pm)), vec(plan.out_host))
    end
    return plan.ks_phys
end

# ---- real path (reconstruction KA kernel) ----

# Full-spectrum value at output index `I` from the r2c half-spectrum + oversampled interior modes.
@inline function _recon_value(I, fk_half, us_ovs, phase, mir, gk, ms::NTuple{D, Int}, half::NTuple{D, Int},
        novs::NTuple{D, Int}, normfactor) where {D}
    k1 = I[1] - 1 - half[1]
    if k1 >= 0
        src = CartesianIndex(ntuple(d -> d == 1 ? k1 + 1 : Int(I[d]), Val(D)))
        return fk_half[src] * phase[I]
    elseif !_has_even_nyquist(I, ms, half, Val(D))
        mI = CartesianIndex(ntuple(d -> d == 1 ? -k1 + 1 : mir[d][I[d]], Val(D)))
        return conj(fk_half[mI]) * phase[I]
    else
        negk = ntuple(d -> -(I[d] - 1 - half[d]), Val(D))
        ovsI = CartesianIndex(ntuple(d -> _ovs_index(negk[d], novs[d]), Val(D)))
        β = normfactor / gk[1][negk[1] + 1]
        for d in 2:D
            β = β / gk[d][_gk_index(negk[d], ms[d])]
        end
        return conj(β * us_ovs[ovsI]) * phase[I]
    end
end

KA.@kernel function _recon_kernel!(out, @Const(fk_half), @Const(us_ovs), @Const(phase), @Const(mir), @Const(gk),
        ms, half, novs, normfactor, neg::Bool)
    I = @index(Global, Cartesian)
    v = _recon_value(I, fk_half, us_ovs, phase, mir, gk, ms, half, novs, normfactor)
    @inbounds out[I] = neg ? conj(v) : v
end

function FFS.calculate_spectrum!(coeffs::AbstractArray{Complex{T}}, plan::NUFFTNonuniformRealGPUPlan{T, D},
        field) where {T, D}
    M = plan.M
    ms = plan.ms
    Pm = prod(ms)
    ntrans = length(field) ÷ M
    half = ntuple(d -> ms[d] ÷ 2, Val(D))
    backend = KA.get_backend(plan.out)
    kernel! = _recon_kernel!(backend)
    neg = plan.iflag < 0
    @inbounds for t in 1:ntrans
        foff = (t - 1) * M
        copyto!(plan.cj, view(field, (foff + 1):(foff + M)))     # real strengths → device
        NonuniformFFTs.exec_type1!(plan.fk_half, plan.plan, plan.cj)   # fills fk_half AND plan.data.ûs
        kernel!(plan.out, plan.fk_half, plan.us_ovs, plan.phase, plan.mir, plan.gk, ms, half, plan.novs, plan.normfactor, neg; ndrange = ms)
        KA.synchronize(backend)
        copyto!(plan.out_host, plan.out)                          # device → host
        coff = (t - 1) * Pm
        copyto!(view(coeffs, (coff + 1):(coff + Pm)), vec(plan.out_host))
    end
    return plan.ks_phys
end

# ---- plan_spectrum + one-shot entries (GPUBackend); real/complex dispatched on the plan eltype ----

function FFS.plan_spectrum(::FFS.NonuniformFFTsBackend, exec::ComputationalBackends.GPUBackend{<:KA.Backend},
        g::FlowGeometries.Grids.AbstractUnstructuredGrid{<:FlowGeometries.Geometry.AbstractCartesianGeometry}, ::Type{T}, ms::NTuple{D, Int};
        batch::Tuple = (), iflag::Int = 1, eps::Real = _default_eps(T)) where {T, D}
    coords = FlowGeometries.Grids.coordinates(g)
    Ls = ntuple(d -> real(float(T))(FlowGeometries.Grids.period(g, d)), D)
    return _nu_gpu_plan(T, exec.backend, coords, ms, Ls, iflag, eps)
end

function FFS._calculate_spectrum_gpu_nufft(::FFS.NonuniformFFTsBackend, exec::ComputationalBackends.GPUBackend{<:KA.Backend},
        g::FlowGeometries.Grids.AbstractUnstructuredGrid{<:FlowGeometries.Geometry.AbstractCartesianGeometry},
        field::AbstractArray, ms::Tuple; iflag::Int = 1, eps::Union{Nothing, Real} = nothing, kwargs...)
    D = length(ms)
    coords = FlowGeometries.Grids.coordinates(g)
    E = float(eltype(field))
    Tr = real(E)
    Ls = ntuple(d -> Tr(FlowGeometries.Grids.period(g, d)), D)
    batch = ntuple(i -> size(field, 1 + i), ndims(field) - 1)
    epsv = eps === nothing ? _default_eps(E) : eps
    plan = _nu_gpu_plan(E, exec.backend, coords, NTuple{D, Int}(ms), Ls, iflag, epsv)
    coeffs = zeros(Complex{Tr}, ms..., batch...)
    ks = FFS.calculate_spectrum!(coeffs, plan, field)
    return coeffs, ks
end

# Nonuniform tensor-product (structured) Cartesian grid: materialize the point cloud, reuse the scattered path.
function FFS._calculate_spectrum_gpu_nufft(::FFS.NonuniformFFTsBackend, exec::ComputationalBackends.GPUBackend{<:KA.Backend},
        g::FlowGeometries.Grids.AbstractStructuredGrid{<:FlowGeometries.Geometry.AbstractCartesianGeometry},
        field::AbstractArray, ms::Tuple; iflag::Int = 1, eps::Union{Nothing, Real} = nothing, kwargs...)
    D = ndims(g)
    E = float(eltype(field))
    Tr = real(E)
    axes_d = ntuple(d -> Tr.(FlowGeometries.Grids.coordinates(g, d)), D)
    Ls = ntuple(d -> Tr(FlowGeometries.Grids.period(g, d)), D)
    Ns = size(g)
    npts = prod(Ns)
    CIg = CartesianIndices(Ns)
    coords = ntuple(d -> [axes_d[d][CIg[p][d]] for p in 1:npts], D)
    batch = ntuple(i -> size(field, D + i), ndims(field) - D)
    fieldflat = reshape(field, npts, batch...)
    epsv = eps === nothing ? _default_eps(E) : eps
    plan = _nu_gpu_plan(E, exec.backend, coords, NTuple{D, Int}(ms), Ls, iflag, epsv)
    coeffs = zeros(Complex{Tr}, ms..., batch...)
    ks = FFS.calculate_spectrum!(coeffs, plan, fieldflat)
    return coeffs, ks
end

end # module FlowFieldSpectraNonuniformFFTsKernelAbstractionsExt
