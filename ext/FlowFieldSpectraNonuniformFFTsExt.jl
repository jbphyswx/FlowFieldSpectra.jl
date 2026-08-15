module FlowFieldSpectraNonuniformFFTsExt

using NonuniformFFTs: NonuniformFFTs
using FlowFieldSpectra: FlowFieldSpectra as FFS
using ComputationalBackends: ComputationalBackends
using SpectralBackends: SpectralBackends
using FlowGeometries: FlowGeometries

# =============================================================================
# NUFFT (type-1) on the tensor-native model, via NonuniformFFTs.jl — a portable provider for
# `NUFFTSpectralBackend`, matching the FINUFFT ext's convention exactly (`Σⱼ vⱼ exp(-i k·xⱼ)`, centered
# modes, ×1/M, offset-correction phase; FFS `iflag = 1` ≡ FINUFFT `isign = -1`). Load ONE NUFFT provider
# — both exts define the same `_calculate_spectrum_nufft`/`plan_spectrum` methods (last-one-wins).
#
# Fast path by input type:
#   • COMPLEX field → a `Complex{T}` plan returning the full `(ms…)` spectrum directly.
#   • REAL field → a real-input plan (`PlanNUFFT(T<:Real, …)`): a real-to-complex FFT (≈2× faster, half
#     the memory — no real→complex widen). It returns the non-redundant half (`rfftfreq` on axis 1 =
#     modes `0…N₁/2`; centered `fftfreq` on the rest). We build FFS's full centered array from it:
#       – axis-1 `k₁≥0`: taken directly; `k₁<0`, non-Nyquist: Hermitian mirror `C[k]=conj(C[-k])`;
#       – an EVEN axis d≥2 at its Nyquist `k_d=-N_d/2` (with `k₁<0`) needs `C[-k₁,+N_d/2]`, and `+N_d/2`
#         is not an OUTPUT mode. But it is an INTERIOR mode of the oversampled spectrum the transform
#         already computed, so we read it straight from `plan.data.ûs` and apply the SAME deconvolution
#         `copy_deconvolve_to_non_oversampled!` uses (`normfactor / ∏ĝₖ`) — no extra transform.
#     (This reaches into NonuniformFFTs internals — `data.ûs`, kernel `fourier_coefficients`, the
#     oversampling factor — validated against the complex plan; FFS pins NonuniformFFTs to 0.9.x, so a
#     breaking internal change would land as 0.10.) Since the field is real, `iflag=-1` is the
#     elementwise conjugate of the `iflag=+1` result.
# A scattered grid maps directly; a nonuniform tensor-product (structured) grid materializes its point
# cloud and reuses the scattered path. Points are FIXED by the grid, so plans + `set_points!` run ONCE.
# NonuniformFFTs' `exec_type1!` allocates small internal buffers each call (not 0-alloc, unlike FINUFFT);
# the wrapper itself adds none.
# =============================================================================

_default_eps(::Type{Float32}) = 1.0f-6
_default_eps(::Type{T}) where {T <: Real} = 1.0e-8
_default_eps(::Type{Complex{T}}) where {T} = _default_eps(T)

# Half-support + oversampling for a target tolerance. NonuniformFFTs needs `σ·N ≥ 2m` per axis, so for a
# grid too small at the default σ = 2 we RAISE σ (honoring eps) rather than shrink m (a silent downgrade).
function _plan_accuracy(eps::Real, ms::NTuple)
    m = clamp(ceil(Int, -log10(eps)) + 1, 1, 16)
    σ = float(max(2, cld(2 * m, minimum(ms))))
    return NonuniformFFTs.HalfSupport(m), σ
end

# Scaled points (→ [0, 2π)), centered offset-correction phase (× 1/M; built for `iflag = +1`, the real
# path applies the sign by a final conjugate), and physical wavenumbers.
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

# `mir[i]` = index of mode `-ks[i]` in the centered axis of length N (even-N Nyquist wraps to itself; a
# Nyquist column is instead filled by the oversampled-spectrum extraction, so that entry is unused).
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

# ---- immutable plans (no C resource → no finalizer) ----

struct NUFFTNonuniformComplexPlan{T, D, P, CJ, FK, PH, KS} <: FFS.AbstractSpectralPlan
    plan::P
    cj::CJ
    fk::FK
    ms::NTuple{D, Int}
    M::Int
    iflag::Int
    phase::PH
    ks_phys::KS
end

struct NUFFTNonuniformRealPlan{T, D, P, CJ, FK, PH, MIR, US, GK, KS} <: FFS.AbstractSpectralPlan
    plan::P                          # NonuniformFFTs.PlanNUFFT{T<:Real}, half-spectrum
    cj::CJ                           # (M,) reused REAL strengths (no widen)
    fk_half::FK                      # (N₁÷2+1, N₂…) reused complex half-spectrum
    ms::NTuple{D, Int}
    M::Int
    iflag::Int
    phase::PH                        # (ms…) offset phase × 1/M  (built for iflag=+1)
    mir::MIR                         # NTuple{D,Vector{Int}} per-axis mirror indices
    us_ovs::US                       # ref to plan.data.ûs[1] (oversampled spectrum; refilled each exec)
    novs::NTuple{D, Int}             # oversampled sizes Ñ (axis 1 is the full Ñ₁, not the rfft length)
    gk::GK                           # NTuple{D,Vector} kernel Fourier coefficients (deconvolution)
    normfactor::T                    # ∏ 2π/Ñ_d, in the plan's real precision T
    ks_phys::KS
end

Base.show(io::IO, ::NUFFTNonuniformComplexPlan{T, D}) where {T, D} = print(io, "NUFFTNonuniformComplexPlan{$T, $D}(NonuniformFFTs)")
Base.show(io::IO, ::NUFFTNonuniformRealPlan{T, D}) where {T, D} = print(io, "NUFFTNonuniformRealPlan{$T, $D}(NonuniformFFTs)")

function _nu_plan(::Type{Complex{Tr}}, coords::Tuple, ms::NTuple{D, Int}, Ls::NTuple{D},
        iflag::Int, eps::Real) where {Tr <: Real, D}
    scaled, phase, ks_phys, M = _scaled_phase_ks(Tr, coords, ms, Ls, iflag)
    hs, σ = _plan_accuracy(eps, ms)
    plan = NonuniformFFTs.PlanNUFFT(Complex{Tr}, ms; ntransforms = Val(1), fftshift = true, m = hs, σ = σ)
    NonuniformFFTs.set_points!(plan, scaled)
    cj = Vector{Complex{Tr}}(undef, M)
    fk = Array{Complex{Tr}, D}(undef, ms...)
    return NUFFTNonuniformComplexPlan{Tr, D, typeof(plan), typeof(cj), typeof(fk), typeof(phase), typeof(ks_phys)}(
        plan, cj, fk, ms, M, iflag, phase, ks_phys,
    )
end

function _nu_plan(::Type{Tr}, coords::Tuple, ms::NTuple{D, Int}, Ls::NTuple{D},
        iflag::Int, eps::Real) where {Tr <: Real, D}
    scaled, phase, ks_phys, M = _scaled_phase_ks(Tr, coords, ms, Ls, 1)   # sign applied via final conj
    hs, σ = _plan_accuracy(eps, ms)
    plan = NonuniformFFTs.PlanNUFFT(Tr, ms; ntransforms = Val(1), fftshift = true, m = hs, σ = σ)
    NonuniformFFTs.set_points!(plan, scaled)
    cj = Vector{Tr}(undef, M)
    fk_half = Array{Complex{Tr}, D}(undef, size(plan)...)
    mir = ntuple(d -> _mirror_index(ms[d]), D)
    us_ovs = plan.data.ûs[1]                                        # oversampled spectrum (refilled each exec)
    novs = ntuple(d -> d == 1 ? 2 * (size(us_ovs, 1) - 1) : size(us_ovs, d), D)   # Ñ (axis 1 is rfft → full)
    gk = ntuple(d -> NonuniformFFTs.fourier_coefficients(plan.kernels[d]), D)
    normfactor = prod(2 * Tr(π) / novs[d] for d in 1:D)
    return NUFFTNonuniformRealPlan{Tr, D, typeof(plan), typeof(cj), typeof(fk_half), typeof(phase), typeof(mir), typeof(us_ovs), typeof(gk), typeof(ks_phys)}(
        plan, cj, fk_half, ms, M, iflag, phase, mir, us_ovs, novs, gk, normfactor, ks_phys,
    )
end

# ---- complex path ----

function FFS.calculate_spectrum!(coeffs::AbstractArray{Complex{T}}, plan::NUFFTNonuniformComplexPlan{T, D},
        field) where {T, D}
    M = plan.M
    Pm = prod(plan.ms)
    ntrans = length(field) ÷ M
    neg = plan.iflag < 0
    @inbounds for t in 1:ntrans
        foff = (t - 1) * M
        if neg
            for i in 1:M
                plan.cj[i] = conj(Complex{T}(field[foff + i]))
            end
        else
            for i in 1:M
                plan.cj[i] = Complex{T}(field[foff + i])
            end
        end
        NonuniformFFTs.exec_type1!(plan.fk, plan.plan, plan.cj)
        coff = (t - 1) * Pm
        if neg
            for i in 1:Pm
                coeffs[coff + i] = conj(plan.fk[i]) * plan.phase[i]
            end
        else
            for i in 1:Pm
                coeffs[coff + i] = plan.fk[i] * plan.phase[i]
            end
        end
    end
    return plan.ks_phys
end

# ---- real path ----

# Oversampled `fftfreq(Ñ)` index (1-based) of integer frequency `f`; axis 1 is the r2c half (`f ≥ 0`).
@inline _ovs_index(f::Int, Ñ::Int) = f >= 0 ? f + 1 : Ñ + f + 1
# Kernel-Fourier-coefficient index for frequency `f` on a centered axis of length N (ĝₖ is even, so the
# out-of-range `+N/2` maps to `-N/2`).
@inline function _gk_index(f::Int, N::Int)
    half = N ÷ 2
    f > (N - 1) ÷ 2 && (f -= N)
    return f + half + 1
end

function FFS.calculate_spectrum!(coeffs::AbstractArray{Complex{T}}, plan::NUFFTNonuniformRealPlan{T, D},
        field) where {T, D}
    M = plan.M
    ms = plan.ms
    ntrans = length(field) ÷ M
    half = ntuple(d -> ms[d] ÷ 2, Val(D))
    coeffsB = reshape(coeffs, ms..., ntrans)
    CI = CartesianIndices(ms)
    colon = ntuple(_ -> Colon(), Val(D))
    @inbounds for t in 1:ntrans
        cslab = view(coeffsB, colon..., t)
        foff = (t - 1) * M
        for i in 1:M
            plan.cj[i] = field[foff + i]                              # real strengths — no widen
        end
        NonuniformFFTs.exec_type1!(plan.fk_half, plan.plan, plan.cj)  # fills fk_half AND plan.data.ûs
        for I in CI
            k1 = I[1] - 1 - half[1]
            if k1 >= 0
                cslab[I] = plan.fk_half[CartesianIndex(ntuple(d -> d == 1 ? k1 + 1 : Int(I[d]), Val(D)))] * plan.phase[I]
            elseif !_has_even_nyquist(I, ms, half, Val(D))
                mI = CartesianIndex(ntuple(d -> d == 1 ? -k1 + 1 : plan.mir[d][I[d]], Val(D)))
                cslab[I] = conj(plan.fk_half[mI]) * plan.phase[I]
            else
                # C[k1<0, …] = conj(C[-k]); read C[-k]'s oversampled coefficient + deconvolve.
                negk = ntuple(d -> -(I[d] - 1 - half[d]), Val(D))     # -k
                ovsI = CartesianIndex(ntuple(d -> _ovs_index(negk[d], plan.novs[d]), Val(D)))
                β = plan.normfactor / plan.gk[1][negk[1] + 1]         # axis 1: rfftfreq index (negk[1] = -k1 ≥ 1)
                for d in 2:D
                    β /= plan.gk[d][_gk_index(negk[d], ms[d])]        # axes ≥ 2: centered
                end
                cslab[I] = conj(β * plan.us_ovs[ovsI]) * plan.phase[I]
            end
        end
        plan.iflag < 0 && (cslab .= conj.(cslab))                     # real field: iflag=-1 ≡ conjugate
    end
    return plan.ks_phys
end

# Does output index `I` sit on an even non-reduced axis's Nyquist (`k_d = -N_d/2`)?
@inline function _has_even_nyquist(I, ms::NTuple{D, Int}, half, ::Val{D}) where {D}
    any(ntuple(d -> d >= 2 && iseven(ms[d]) && (I[d] - 1 - half[d]) == -half[d], Val(D)))
end

# ---- plan_spectrum + one-shot entries; dispatch the real/complex path on the field eltype ----

function FFS.plan_spectrum(::FFS.NonuniformFFTsBackend, ::ComputationalBackends.AbstractExecutionBackend,
        g::FlowGeometries.Grids.AbstractUnstructuredGrid{<:FlowGeometries.Geometry.AbstractCartesianGeometry}, ::Type{T}, ms::NTuple{D, Int};
        batch::Tuple = (), iflag::Int = 1, eps::Real = _default_eps(T)) where {T, D}
    coords = FlowGeometries.Grids.coordinates(g)
    Ls = ntuple(d -> real(float(T))(FlowGeometries.Grids.period(g, d)), D)
    return _nu_plan(T, coords, ms, Ls, iflag, eps)
end

function FFS._calculate_spectrum_nufft(::FFS.NonuniformFFTsBackend, ::ComputationalBackends.AbstractExecutionBackend,
        g::FlowGeometries.Grids.AbstractUnstructuredGrid{<:FlowGeometries.Geometry.AbstractCartesianGeometry},
        field::AbstractArray, ms::Tuple; iflag::Int = 1, eps::Union{Nothing, Real} = nothing, kwargs...)
    D = length(ms)
    coords = FlowGeometries.Grids.coordinates(g)
    E = float(eltype(field))
    Tr = real(E)
    Ls = ntuple(d -> Tr(FlowGeometries.Grids.period(g, d)), D)
    batch = ntuple(i -> size(field, 1 + i), ndims(field) - 1)
    epsv = eps === nothing ? _default_eps(E) : eps
    plan = _nu_plan(E, coords, NTuple{D, Int}(ms), Ls, iflag, epsv)
    coeffs = zeros(Complex{Tr}, ms..., batch...)
    ks = FFS.calculate_spectrum!(coeffs, plan, field)
    return coeffs, ks
end

# Nonuniform tensor-product (structured) Cartesian grid: materialize the tensor-product point cloud
# (column-major, axis 1 fastest — matching the field layout) and reuse the scattered path.
function FFS._calculate_spectrum_nufft(::FFS.NonuniformFFTsBackend, ::ComputationalBackends.AbstractExecutionBackend,
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
    plan = _nu_plan(E, coords, NTuple{D, Int}(ms), Ls, iflag, epsv)
    coeffs = zeros(Complex{Tr}, ms..., batch...)
    ks = FFS.calculate_spectrum!(coeffs, plan, fieldflat)
    return coeffs, ks
end

end # module FlowFieldSpectraNonuniformFFTsExt
