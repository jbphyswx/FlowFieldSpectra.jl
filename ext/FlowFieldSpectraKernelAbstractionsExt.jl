module FlowFieldSpectraKernelAbstractionsExt

# `@index` and `@Const` are imported by bare name: the `@kernel` macro recognizes these DSL keywords
# only unqualified inside a kernel body, and `KA.@index` expands to the wrong `__index_*` call.
using KernelAbstractions: KernelAbstractions as KA, @index, @Const
using FlowFieldSpectra: FlowFieldSpectra as FFS
using FlowTransformBindings: FlowTransformBindings as FTB
using ComputationalBackends: ComputationalBackends
using SpectralBackends: SpectralBackends
using FlowGeometries: FlowGeometries

# =============================================================================
# GPUBackend execution of the DirectSum transform and its synthesis on any KernelAbstractions device
# (`KA.CPU()`, `CUDABackend()`, …). The forward runs one thread per spectral mode, which owns that mode's
# coefficient rows; the synthesis runs one thread per point, which owns that point's values. Neither needs
# atomics. A tensor grid's point coordinates decode from its 1-D axes on device (`axes[d][decode(p)[d]]`).
# The batch is the inner loop. The one-shots build a plan and execute it once.
# =============================================================================

# Is `a` resident on `backend`? Only a strided array can be, and `KA.get_backend` answers for a host
# `Array` and for each vendor's device array. A lazy wrapper — the reshaped, scaled `SeparableMeasure`
# a grid's quadrature weights come back as — is resident nowhere and is materialized by `_host_dense`.
_resident(::Any, ::KA.Backend) = false
_resident(a::DenseArray, backend::KA.Backend) = KA.get_backend(a) === backend

# A dense array of element type `FT`, ready for `copyto!` onto a device array. A device array converts in
# place (staying on its own backend); a lazy wrapper materializes in one pass.
_host_dense(::Type{FT}, v::DenseArray) where {FT} = eltype(v) === FT ? v : FT.(v)
_host_dense(::Type{FT}, v) where {FT} = collect(FT, v)

# Stage a vector as a length-N device array of element type FT (identity if already resident).
function _dev_vec(backend::KA.Backend, v, ::Type{FT}) where {FT}
    _resident(v, backend) && eltype(v) === FT && return v
    d = KA.allocate(backend, FT, length(v))
    copyto!(d, _host_dense(FT, v))
    return d
end

# A field `(spatial…, batch…)`, host or device, into the device `(N, B)` buffer `fld`, weighted by the
# per-node quadrature factor `qw` (`nothing` for a constant measure).
function _stage_field!(fld, field, qw)
    copyto!(fld, eltype(field) === eltype(fld) ? field : eltype(fld).(field))
    qw === nothing || (fld .*= qw)
    return fld
end

# =============================================================================
# The device side of `NUFFT.jl`: a plan's arrays on the execution backend, and its index gathers.
# =============================================================================

FFS._alloc(exec::ComputationalBackends.GPUBackend{<:KA.Backend}, ::Type{T}, dims::Integer...) where {T} =
    KA.allocate(exec.backend, T, dims...)

# One thread per entry of every transform: entry `i` of transform `c` reads `src` at `soff + c·sstride +
# idx[i]` and writes `dst` at `doff + c·n + i`. `w` is an array of per-entry weights or one number for
# every entry (`FFS._weight`).
KA.@kernel function _gather_scaled_kernel!(dst, doff::Int, @Const(src), soff::Int, sstride::Int, @Const(idx),
        w, n::Int, csrc::Bool, neg::Bool)
    k = @index(Global)
    i = (k - 1) % n + 1
    c = (k - 1) ÷ n
    @inbounds begin
        s = src[soff + c * sstride + idx[i]]
        v = (csrc ? conj(s) : s) * FFS._weight(w, i)
        dst[doff + c * n + i] = neg ? conj(v) : v
    end
end

function FFS._gather_scaled!(dst::AbstractArray, doff::Int, src::AbstractArray, soff::Int, sstride::Int,
        idx::AbstractArray{Int}, w::Union{AbstractArray, Number}, n::Int, ntr::Int, csrc::Bool, neg::Bool)
    n * ntr == 0 && return dst
    backend = KA.get_backend(src)
    _gather_scaled_kernel!(backend)(dst, doff, src, soff, sstride, idx, w, n, csrc, neg; ndrange = n * ntr)
    KA.synchronize(backend)
    return dst
end

# One thread per coefficient of every transform: coefficient `i` of transform `c` is `scale` times entry
# `idx[i]` of that transform's spectrum, its conjugate at `-idx[i]` where negative, zero at `0`.
KA.@kernel function _gather_packed_kernel!(dst, doff::Int, @Const(src), soff::Int, sstride::Int, @Const(idx),
        scale, n::Int, neg::Bool)
    k = @index(Global)
    i = (k - 1) % n + 1
    c = (k - 1) ÷ n
    @inbounds begin
        j = idx[i]
        so = soff + c * sstride
        v = j > 0 ? src[so + j] * scale : j < 0 ? conj(src[so - j]) * scale : zero(eltype(dst))
        dst[doff + c * n + i] = neg ? conj(v) : v
    end
end

function FFS._gather_packed!(dst::AbstractArray, doff::Int, src::AbstractArray, soff::Int, sstride::Int,
        idx::AbstractArray{Int}, scale::Real, n::Int, ntr::Int, neg::Bool)
    n * ntr == 0 && return dst
    backend = KA.get_backend(src)
    _gather_packed_kernel!(backend)(dst, doff, src, soff, sstride, idx, scale, n, neg; ndrange = n * ntr)
    KA.synchronize(backend)
    return dst
end

# One thread per listed coefficient of every transform: coefficient `pos[i]` of transform `c`, `n` apart
# from `doff`, is entry `idx[i]` of `srcs[c]` times `fac[i]`.
KA.@kernel function _scatter_each_kernel!(dst, doff::Int, n::Int, srcs, @Const(pos), @Const(idx), @Const(fac),
        S::Int, neg::Bool)
    k = @index(Global)
    i = (k - 1) % S + 1
    c = (k - 1) ÷ S
    @inbounds begin
        v = srcs[c + 1][idx[i]] * fac[i]
        dst[doff + c * n + pos[i]] = neg ? conj(v) : v
    end
end

function FFS._scatter_each!(dst::AbstractArray, doff::Int, n::Int, srcs::Tuple, ntr::Int,
        pos::AbstractArray{Int}, idx::AbstractArray{Int}, fac::AbstractArray, neg::Bool)
    S = length(pos)
    S * ntr == 0 && return dst
    backend = KA.get_backend(dst)
    _scatter_each_kernel!(backend)(dst, doff, n, srcs, pos, idx, fac, S, neg; ndrange = S * ntr)
    KA.synchronize(backend)
    return dst
end

# Column-major decode of a 1-based linear index `p` into a size-`s` grid (pure ⇒ GPU-safe).
@inline function _ka_decode(p::Int, s::NTuple{D, Int}) where {D}
    idx = p - 1
    return ntuple(Val(D)) do d
        stride = 1
        for i in 1:(d - 1)
            stride *= s[i]
        end
        (idx ÷ stride) % s[d] + 1
    end
end

# =============================================================================
# Cartesian direct sum
# =============================================================================

# The positions `x − x₀` a kernel reads: one vector per axis on a tensor grid, and one value per point per
# direction otherwise, a curvilinear grid's coordinate arrays flattened.
_stage_positions(backend, ::FFS.DirectSum.TensorCartesian, xs::Tuple, ::Type{FT}) where {FT} =
    map(x -> _dev_vec(backend, collect(x), FT), xs)
_stage_positions(backend, ::FFS.DirectSum.CloudCartesian, xs::Tuple, ::Type{FT}) where {FT} =
    map(x -> _dev_vec(backend, vec(collect(x)), FT), xs)

# `coeffs[k, b] += Σ_p field[p, b] e^{-iflag·i·k·x_p}` over the modes `live` (linear indices into the mode
# set `pms`), wavenumber axes `ksd`.
_run_cartesian!(coeffs, backend, ::FFS.DirectSum.TensorCartesian, xsd, field, ksd, ss, pms, live, N, L, B,
    iflag) = _cart_tensor_kernel!(backend)(coeffs, field, xsd, ksd, ss, pms, live, N, L, B, length(pms), iflag;
    ndrange = L)
_run_cartesian!(coeffs, backend, ::FFS.DirectSum.CloudCartesian, xsd, field, ksd, ss, pms, live, N, L, B,
    iflag) = _cart_scattered_kernel!(backend)(coeffs, field, xsd, ksd, pms, live, N, L, B, length(pms), iflag;
    ndrange = L)

KA.@kernel function _cart_tensor_kernel!(coeffs, @Const(field), @Const(axesd), @Const(ksd),
        @Const(ss), @Const(pms), @Const(live), N::Int, L::Int, B::Int, D::Int, iflag::Int)
    t = @index(Global)
    if t <= L
        mi = live[t]
        I = _ka_decode(mi, pms)
        FT = eltype(axesd[1])
        @inbounds for p in 1:N
            P = _ka_decode(p, ss)
            phi = zero(FT)
            for d in 1:D
                phi += ksd[d][I[d]] * axesd[d][P[d]]
            end
            W = cis(-iflag * phi)
            for b in 1:B
                coeffs[mi, b] += field[p, b] * W
            end
        end
    end
end

KA.@kernel function _cart_scattered_kernel!(coeffs, @Const(field), @Const(coordsd), @Const(ksd),
        @Const(pms), @Const(live), N::Int, L::Int, B::Int, D::Int, iflag::Int)
    t = @index(Global)
    if t <= L
        mi = live[t]
        I = _ka_decode(mi, pms)
        FT = eltype(coordsd[1])
        @inbounds for p in 1:N
            phi = zero(FT)
            for d in 1:D
                phi += ksd[d][I[d]] * coordsd[d][p]
            end
            W = cis(-iflag * phi)
            for b in 1:B
                coeffs[mi, b] += field[p, b] * W
            end
        end
    end
end

"""
    DirectSumGPUCartesianPlan{FT,R}

Reusable Cartesian direct sum on a KernelAbstractions device: the positions `x − x₀` and the wavenumber
axes of the coefficient box staged once, the modes it computes (a real field's box holds zero at the rest,
`FlowTransformBindings.outside_modes`), the device field and coefficient buffers, and the per-node
quadrature factor. `R` marks a real field. Execute with `calculate_spectrum!(coeffs, plan, field)`,
`field` and `coeffs` on the host or the device.
"""
struct DirectSumGPUCartesianPlan{FT, R, NB, BK, L, X, K, LV, FB, CB, QW, KS, SS, PS} <: FFS.AbstractSpectralPlan
    backend::BK
    layout::L                     # TensorCartesian / CloudCartesian
    xs::X                         # device positions
    ks::K                         # device wavenumber axes
    live::LV                      # device linear indices of the computed modes
    fld::FB                       # device (Npts, B) field
    cd::CB                        # device (M, B) coefficients
    qw::QW                        # device (Npts,) quadrature factor, `nothing` where the measure is constant
    ss::SS                        # the grid's shape
    pms::PS                       # packed coefficient box
    Npts::Int
    M::Int
    B::Int
    iflag::Int
    batch::NTuple{NB, Int}
    ks_phys::KS                   # host wavenumber axes
end

Base.show(io::IO, p::DirectSumGPUCartesianPlan{FT, R}) where {FT, R} =
    print(io, "DirectSumGPUCartesianPlan{", FT, "}(", R ? "real" : "complex", ", ",
        nameof(typeof(p.layout)), ", B=", p.B, ")")

FFS.Plans.coefficient_size(p::DirectSumGPUCartesianPlan) = (p.pms..., p.batch...)
FFS.Plans.coefficient_type(::DirectSumGPUCartesianPlan{FT}) where {FT} = Complex{FT}
FFS.Plans.wavenumbers(p::DirectSumGPUCartesianPlan) = p.ks_phys

function FFS.plan_spectrum(::SpectralBackends.AbstractDirectSumSpectralBackend,
        exec::ComputationalBackends.GPUBackend{<:KA.Backend},
        g::FlowGeometries.Grids.AbstractGrid{<:FlowGeometries.Geometry.AbstractCartesianGeometry},
        ::Type{T}, ms::NTuple{D, Int}; batch::Tuple = (), iflag::Int = 1, box = nothing,
        kwargs...) where {T, D}
    FT = real(float(T))
    R = T <: Real
    backend = exec.backend
    layout = FFS.DirectSum._cart_layout(g)
    ss = size(g)
    Npts = prod(ss)
    bt = NTuple{length(batch), Int}(batch)
    B = prod(bt; init = 1)
    xs, ks_phys = FFS.DirectSum.phase_geometry(g, FT, ms, Val(R), box)
    pms = map(length, ks_phys)
    M = prod(pms)
    outside = R ? FTB.outside_modes(ms, FFS.Grids.mode_box(g, ms, Val(true))) : Int[]
    live = _dev_vec(backend, setdiff(1:M, outside), Int)
    xsd = _stage_positions(backend, layout, xs, FT)
    ksd = map(k -> _dev_vec(backend, collect(k), FT), ks_phys)
    fld = KA.allocate(backend, R ? FT : Complex{FT}, Npts, B)
    cd = KA.allocate(backend, Complex{FT}, M, B)
    qwh = FFS.Grids.quadrature_scale(g, FT, Npts)
    qw = qwh === nothing ? nothing : _dev_vec(backend, qwh, FT)
    return DirectSumGPUCartesianPlan{FT, R, length(bt), typeof(backend), typeof(layout), typeof(xsd),
            typeof(ksd), typeof(live), typeof(fld), typeof(cd), typeof(qw), typeof(ks_phys), typeof(ss),
            typeof(pms)}(
        backend, layout, xsd, ksd, live, fld, cd, qw, ss, pms, Npts, M, B, iflag, bt, ks_phys)
end

"""
    calculate_spectrum!(coeffs, plan::DirectSumGPUCartesianPlan, field) -> ks

Fill `coeffs` (the packed half of the coefficient box for a real field, the full native spectrum for a
complex one) with the direct sum of `field` `(spatial…, batch…)` on the plan's device, and return the
wavenumber axes.
"""
function FFS.calculate_spectrum!(coeffs::AbstractArray{Complex{FT}}, p::DirectSumGPUCartesianPlan{FT, R},
        field) where {FT, R}
    FFS.Plans._check_coefficients(coeffs, p)
    length(field) == p.Npts * p.B || throw(DimensionMismatch(
        "field holds $(length(field)) values; this plan transforms $(p.B) field(s) of $(p.Npts) points — " *
        "pass the matching `batch=` to plan_spectrum"))
    (R && eltype(field) <: Complex) && throw(ArgumentError(
        "this plan was built for a real field; build one for $(eltype(field)) to transform a complex field"))
    _stage_field!(p.fld, field, p.qw)
    fill!(p.cd, zero(eltype(p.cd)))
    L = length(p.live)
    if L > 0
        _run_cartesian!(p.cd, p.backend, p.layout, p.xs, p.fld, p.ks, p.ss, p.pms, p.live, p.Npts, L, p.B,
            p.iflag)
        KA.synchronize(p.backend)
        p.cd ./= p.Npts
    end
    copyto!(coeffs, p.cd)
    return p.ks_phys
end

function FFS._gpu_directsum_cartesian(exec::ComputationalBackends.GPUBackend{<:KA.Backend},
        g::FlowGeometries.Grids.AbstractGrid{<:FlowGeometries.Geometry.AbstractCartesianGeometry},
        field::AbstractArray, ms::NTuple{D, Int}, iflag::Int, box = nothing) where {D}
    FT = eltype(g)
    p = FFS.plan_spectrum(SpectralBackends.DirectSumSpectralBackend(), exec, g,
        eltype(field) <: Real ? FT : Complex{FT}, ms; batch = FFS.Grids.field_batch_shape(g, field),
        iflag = iflag, box = box)
    coeffs = FFS._allocate_coefficients_like(field, p)
    return coeffs, FFS.calculate_spectrum!(coeffs, p, field)
end

# =============================================================================
# Cartesian direct-sum synthesis: `f_p = Σ_k C_k e^{+iflag·i·k·x_p}` over the native spectrum. A real field's
# packed half is written on the host as the series of `Grids.series_modes` whose real part is its series
# over the coefficient box (`DirectSum.native_coefficients!`), and the device sum keeps the real part of
# each term.
# =============================================================================

@inline _part(::Type{<:Real}, z) = real(z)
@inline _part(::Type{<:Complex}, z) = z

_run_synthesis!(out, backend, ::FFS.DirectSum.TensorCartesian, xsd, C, ksd, ss, ms, N, M, B, iflag) =
    _cart_synth_tensor_kernel!(backend)(out, C, xsd, ksd, ss, ms, N, M, B, length(ms), iflag; ndrange = N)
_run_synthesis!(out, backend, ::FFS.DirectSum.CloudCartesian, xsd, C, ksd, ss, ms, N, M, B, iflag) =
    _cart_synth_scattered_kernel!(backend)(out, C, xsd, ksd, ms, N, M, B, length(ms), iflag; ndrange = N)

KA.@kernel function _cart_synth_tensor_kernel!(out, @Const(C), @Const(axesd), @Const(ksd), @Const(ss),
        @Const(ms), N::Int, M::Int, B::Int, D::Int, iflag::Int)
    p = @index(Global)
    if p <= N
        P = _ka_decode(p, ss)
        FT = eltype(axesd[1])
        @inbounds for mi in 1:M
            I = _ka_decode(mi, ms)
            phi = zero(FT)
            for d in 1:D
                phi += ksd[d][I[d]] * axesd[d][P[d]]
            end
            W = cis(iflag * phi)
            for b in 1:B
                out[p, b] += _part(eltype(out), C[mi, b] * W)
            end
        end
    end
end

KA.@kernel function _cart_synth_scattered_kernel!(out, @Const(C), @Const(coordsd), @Const(ksd), @Const(ms),
        N::Int, M::Int, B::Int, D::Int, iflag::Int)
    p = @index(Global)
    if p <= N
        FT = eltype(coordsd[1])
        @inbounds for mi in 1:M
            I = _ka_decode(mi, ms)
            phi = zero(FT)
            for d in 1:D
                phi += ksd[d][I[d]] * coordsd[d][p]
            end
            W = cis(iflag * phi)
            for b in 1:B
                out[p, b] += _part(eltype(out), C[mi, b] * W)
            end
        end
    end
end

"""
    DirectSumGPUSynthesisPlan{FT,R}

Reusable Cartesian direct-sum synthesis on a KernelAbstractions device: the positions `x − x₀` and the
wavenumber axes of the series staged once, the device spectrum and field buffers, and for a real field the
host buffers its packed half is written through. `R` marks a real field. Execute with
`synthesize!(out, plan, coeffs)`.
"""
struct DirectSumGPUSynthesisPlan{FT, R, NB, BK, L, X, K, CB, OB, HB, FB, SS, MS} <: FFS.AbstractSynthesisPlan
    backend::BK
    layout::L                     # TensorCartesian / CloudCartesian
    xs::X                         # device positions
    ks::K                         # device wavenumber axes of the series
    cd::CB                        # device (M, B) series coefficients
    out::OB                       # device (Npts, B) field
    hc::HB                        # host packed half, `nothing` for a complex field
    full::FB                      # host series coefficients it is written into, `nothing` for a complex field
    ss::SS                        # the grid's shape, which the field takes
    ms::MS
    box::MS                       # coefficient box
    ns::MS                        # mode counts of the series
    Npts::Int
    M::Int
    B::Int
    iflag::Int
    batch::NTuple{NB, Int}
end

Base.show(io::IO, p::DirectSumGPUSynthesisPlan{FT, R}) where {FT, R} =
    print(io, "DirectSumGPUSynthesisPlan{", FT, "}(", R ? "real" : "complex", ", ms=", p.ms, ")")

FFS.Plans.field_size(p::DirectSumGPUSynthesisPlan) = (p.ss..., p.batch...)
FFS.Plans.field_type(::DirectSumGPUSynthesisPlan{FT, R}) where {FT, R} = R ? FT : Complex{FT}

function FFS.Plans.plan_synthesis(::SpectralBackends.AbstractDirectSumSpectralBackend,
        exec::ComputationalBackends.GPUBackend{<:KA.Backend},
        g::FlowGeometries.Grids.AbstractGrid{<:FlowGeometries.Geometry.AbstractCartesianGeometry},
        ::Type{T}, ms::NTuple{D, Int}; batch::Tuple = (), iflag::Int = 1, box = nothing,
        kwargs...) where {T, D}
    FT = real(float(eltype(g)))
    R = T <: Real
    backend = exec.backend
    layout = FFS.DirectSum._cart_layout(g)
    ss = size(g)
    Npts = prod(ss)
    bt = NTuple{length(batch), Int}(batch)
    B = prod(bt; init = 1)
    offs, ranges = FFS.Grids.axis_geometry(FT, g, D, box)
    xsd = _stage_positions(backend, layout, FFS.Grids.relative_coordinates(g, offs), FT)
    mb = FFS.Grids.mode_box(g, ms, Val(R))
    ns = FFS.Grids.series_modes(g, ms, Val(R))
    ksd = map(k -> _dev_vec(backend, collect(k), FT), FFS.Grids.physical_wavenumbers(ranges, ns, Val(false)))
    M = prod(ns)
    cd = KA.allocate(backend, Complex{FT}, M, B)
    out = KA.allocate(backend, R ? FT : Complex{FT}, Npts, B)
    hc = R ? Array{Complex{FT}}(undef, FTB.packed_size(mb, Val(true))..., B) : nothing
    full = R ? Array{Complex{FT}}(undef, ns..., B) : nothing
    return DirectSumGPUSynthesisPlan{FT, R, length(bt), typeof(backend), typeof(layout), typeof(xsd), typeof(ksd),
            typeof(cd), typeof(out), typeof(hc), typeof(full), typeof(ss), typeof(ms)}(
        backend, layout, xsd, ksd, cd, out, hc, full, ss, ms, mb, ns, Npts, M, B, iflag, bt)
end

"""
    synthesize!(out, plan::DirectSumGPUSynthesisPlan, coeffs) -> out

Evaluate the series `coeffs` (the packed half of the coefficient box for a real field, the full native
spectrum for a complex one) at the grid's points on the plan's device, into `out` on the host or the
device.
"""
function FFS.Plans.synthesize!(out::AbstractArray, p::DirectSumGPUSynthesisPlan{FT, R},
        coeffs::AbstractArray) where {FT, R}
    size(out) == FFS.Plans.field_size(p) || throw(DimensionMismatch(
        "out is $(size(out)); this plan writes $(FFS.Plans.field_size(p))"))
    want = (FTB.packed_size(p.box, Val(R))..., p.batch...)
    size(coeffs) == want || throw(DimensionMismatch("coeffs is $(size(coeffs)); this plan reads $want"))
    if R
        copyto!(p.hc, coeffs)
        FFS.DirectSum.native_coefficients!(p.full, p.hc, p.ms, p.box)
        copyto!(p.cd, p.full)
    else
        copyto!(p.cd, coeffs)
    end
    fill!(p.out, zero(eltype(p.out)))
    _run_synthesis!(p.out, p.backend, p.layout, p.xs, p.cd, p.ks, p.ss, p.ns, p.Npts, p.M, p.B, p.iflag)
    KA.synchronize(p.backend)
    copyto!(out, p.out)
    return out
end

# =============================================================================
# Normalized associated Legendre values on the device: the packed table
# `P[legendre_index(ℓ, m, lmax), j] = P̄_ℓ^m(cos θ_j)`, one thread per (order, node), each running
# `SphericalKernels.legendre_order!` on the tables `legendre_tables` builds, staged once, from the starts
# `legendre_start` forms on the host in `Float64`. The spherical kernels read values from it and evaluate
# none.
# =============================================================================

function FFS.SphericalKernels.device_legendre_table(backend::KA.Backend, ::Type{FT}, lmax::Int,
        θ::AbstractVector) where {FT}
    SK = FFS.SphericalKernels
    t = SK.legendre_tables(FT, lmax)
    n = length(θ)
    P = KA.allocate(backend, FT, SK.legendre_count(lmax), n)
    n == 0 && return P
    θh = FT.(collect(θ))
    Y = Matrix{FT}(undef, lmax + 1, n); K = Matrix{Int32}(undef, lmax + 1, n)
    pos = Vector{Bool}(undef, n); u = Vector{FT}(undef, n)
    for j in 1:n
        x = cos(θh[j]); s = sin(θh[j])
        pos[j] = x >= 0; u[j] = s^2 / (1 + abs(x))
        ls = log2(abs(eltype(t.lD)(s)))
        for m in 0:lmax
            Y[m + 1, j], K[m + 1, j] = SK.legendre_start(FT, t.lD[m + 1], m, ls)
        end
    end
    dev(a) = copyto!(KA.allocate(backend, eltype(a), size(a)...), a)
    _legendre_table_kernel!(backend)(P, dev(Y), dev(K), dev(pos), dev(u), dev(t.α), dev(t.c), dev(t.γ),
        dev(t.nrm), lmax, n; ndrange = (lmax + 1) * n)
    KA.synchronize(backend)
    return P
end

KA.@kernel function _legendre_table_kernel!(P, @Const(Y), @Const(K), @Const(pos), @Const(u), @Const(α),
        @Const(c), @Const(γ), @Const(nrm), lmax::Int, n::Int)
    idx = @index(Global)
    if idx <= (lmax + 1) * n
        m = (idx - 1) % (lmax + 1)
        j = (idx - 1) ÷ (lmax + 1) + 1
        r0 = FFS.SphericalKernels.legendre_index(m, m, lmax) - m
        @inbounds FFS.SphericalKernels.legendre_order!(P, ℓ -> CartesianIndex(r0 + ℓ, j), α, c, γ, nrm, m,
            lmax, Y[m + 1, j], K[m + 1, j], pos[j], u[j])
    end
end

# =============================================================================
# Spherical direct sum and synthesis — the projection runs one thread per (row, col) coefficient slot (owns
# it across batch, no atomics), the synthesis one thread per node, both reading the Legendre values and
# each node's longitude factor from tables. The tables hold one column per node, so the nodes are taken in
# chunks whose tables stay within `_SPH_TABLE_ENTRIES` entries each.
# =============================================================================

const _SPH_TABLE_ENTRIES = 1 << 24

_sph_chunk(lmax::Int, N::Int) =
    clamp(_SPH_TABLE_ENTRIES ÷ max(FFS.SphericalKernels.legendre_count(lmax), 2 * lmax + 1), 1, max(N, 1))

# `Y[col, p] = w_p · s(m) · (m ≥ 0 ? cos(mφ_p) : sin(|m|φ_p))`, the order-`m` factor of the real
# spherical harmonic (FastSphericalHarmonics convention) times the node's quadrature weight, or times one
# for `w = nothing`.
function _weighted_trig(::Type{FT}, φ, w, lmax::Int) where {FT}
    Y = Matrix{FT}(undef, 2 * lmax + 1, length(φ))
    r2 = sqrt(FT(2))
    @inbounds for p in eachindex(φ), col in 1:(2 * lmax + 1)
        m = col == 1 ? 0 : (iseven(col) ? -(col ÷ 2) : (col - 1) ÷ 2)
        am = abs(m)
        sfac = m == 0 ? one(FT) : (isodd(am) ? -r2 : r2)
        Y[col, p] = _node_weight(FT, w, p) * sfac * (m >= 0 ? cos(m * FT(φ[p])) : sin(am * FT(φ[p])))
    end
    return Y
end

@inline _node_weight(::Type{FT}, ::Nothing, p::Int) where {FT} = one(FT)
@inline _node_weight(::Type{FT}, w, p::Int) where {FT} = FT(w[p])

# The tables of the nodes `pts`: the Legendre values on the device, the longitude factors from the host.
function _node_tables(backend, ::Type{FT}, θ, φ, w, pts, lmax::Int) where {FT}
    P = FFS.SphericalKernels.device_legendre_table(backend, FT, lmax, view(θ, pts))
    Y = KA.allocate(backend, FT, 2 * lmax + 1, length(pts))
    copyto!(Y, _weighted_trig(FT, view(φ, pts), w === nothing ? nothing : view(w, pts), lmax))
    return P, Y
end

# Points `p0 + 1 … p0 + n`, whose table columns are `1 … n`.
KA.@kernel function _spherical_kernel!(coeffs, @Const(P), @Const(Y), @Const(field), @Const(w),
        lmax::Int, p0::Int, n::Int, B::Int)
    idx = @index(Global)
    Nθ = lmax + 1
    Nφ = 2 * lmax + 1
    if idx <= Nθ * Nφ
        col = (idx - 1) ÷ Nθ + 1
        row = (idx - 1) % Nθ + 1
        m = col == 1 ? 0 : (iseven(col) ? -(col ÷ 2) : (col - 1) ÷ 2)
        abs_m = abs(m)
        l = row - 1 + abs_m
        if l <= lmax
            CT = eltype(coeffs)
            r = FFS.SphericalKernels.legendre_index(l, abs_m, lmax)
            @inbounds for q in 1:n
                p = p0 + q
                # A zero-weight node is skipped, so a masked cell's datum (commonly NaN) never enters.
                iszero(w[p]) && continue
                gw = P[r, q] * Y[col, q]
                for b in 1:B
                    coeffs[row, col, b] += CT(field[p, b] * gw)
                end
            end
        end
    end
end

# The series at nodes `p0 + 1 … p0 + n`, whose table columns are `1 … n`.
KA.@kernel function _spherical_synth_kernel!(out, @Const(P), @Const(Y), @Const(C), lmax::Int, p0::Int, n::Int,
        B::Int)
    q = @index(Global)
    if q <= n
        @inbounds for b in 1:B
            acc = zero(eltype(out))
            for l in 0:lmax
                for m in (-l):l
                    am = abs(m)
                    col = m == 0 ? 1 : (m < 0 ? 2 * am : 2 * m + 1)
                    r = FFS.SphericalKernels.legendre_index(l, am, lmax)
                    acc += C[l - am + 1, col, b] * (P[r, q] * Y[col, q])
                end
            end
            out[p0 + q, b] = acc
        end
    end
end

"""
    DirectSumGPUSphericalPlan{FT,CT}

Reusable spherical direct sum on a KernelAbstractions device: each node's colatitude, longitude and
quadrature weight, read from the grid once, with the device weights, field and coefficient buffers. Each
execution builds the Legendre and longitude tables of one chunk of nodes at a time. `CT` is the coefficient
element type, real for a real field. Execute with `calculate_spectrum!(coeffs, plan, field)`.
"""
struct DirectSumGPUSphericalPlan{FT, CT, NB, BK, V, W, FB, CB, KS} <: FFS.AbstractSpectralPlan
    backend::BK
    θ::V                          # host colatitude per node
    φ::V                          # host longitude per node
    w::V                          # host quadrature weight per node, an inactive node at zero
    wd::W                         # device weights
    fld::FB                       # device (N, B) field
    cd::CB                        # device (lmax + 1, 2lmax + 1, B) coefficients
    lmax::Int
    N::Int
    chunk::Int
    batch::NTuple{NB, Int}
    B::Int
    ks::KS
end

Base.show(io::IO, p::DirectSumGPUSphericalPlan{FT, CT}) where {FT, CT} =
    print(io, "DirectSumGPUSphericalPlan{", FT, "}(lmax=", p.lmax, ", ", CT <: Real ? "real" : "complex",
        ", B=", p.B, ")")

FFS.Plans.coefficient_size(p::DirectSumGPUSphericalPlan) = (p.lmax + 1, 2 * p.lmax + 1, p.batch...)
FFS.Plans.coefficient_type(::DirectSumGPUSphericalPlan{FT, CT}) where {FT, CT} = CT
FFS.Plans.wavenumbers(p::DirectSumGPUSphericalPlan) = p.ks

function FFS.plan_spectrum(::SpectralBackends.AbstractDirectSumSpectralBackend,
        exec::ComputationalBackends.GPUBackend{<:KA.Backend},
        g::FlowGeometries.Grids.AbstractGrid{<:FFS.Grids.SphericalHarmonicGeometry},
        ::Type{T}, ms::Tuple; batch::Tuple = (), sampling = nothing, weights = nothing,
        inactive::FFS.AbstractInactiveCells = FFS.ZeroFill(), kwargs...) where {T}
    FT = real(float(T))
    lmax = ms[1] - 1
    lmax >= 0 || throw(ArgumentError("ms[1] must be ≥ 1 (lmax = ms[1] - 1); got $(ms[1])"))
    backend = exec.backend
    θ, φ, wpt = FFS.DirectSum._sph_point_data(g, FT; sampling = sampling,
        weights = FFS.Grids._inactive_weights(inactive, g, FT; sampling, weights))
    w = collect(FT, wpt)
    N = length(θ)
    bt = NTuple{length(batch), Int}(batch)
    B = prod(bt; init = 1)
    CT = FFS.sph_coeff_type(T, FT)
    wd = _dev_vec(backend, w, FT)
    fld = KA.allocate(backend, CT <: Real ? FT : Complex{FT}, N, B)
    cd = KA.allocate(backend, CT, lmax + 1, 2 * lmax + 1, B)
    ks = (0:lmax, -lmax:lmax)
    return DirectSumGPUSphericalPlan{FT, CT, length(bt), typeof(backend), Vector{FT}, typeof(wd), typeof(fld),
            typeof(cd), typeof(ks)}(backend, θ, φ, w, wd, fld, cd, lmax, N, _sph_chunk(lmax, N), bt, B, ks)
end

"""
    calculate_spectrum!(coeffs, plan::DirectSumGPUSphericalPlan, field) -> ks

Fill `coeffs` `(lmax + 1, 2lmax + 1, batch…)` with the quadrature projection of `field` `(spatial…,
batch…)` onto the real spherical harmonics, on the plan's device.
"""
function FFS.calculate_spectrum!(coeffs::AbstractArray{<:Number}, p::DirectSumGPUSphericalPlan{FT, CT},
        field) where {FT, CT}
    FFS.Plans._check_coefficients(coeffs, p)
    length(field) == p.N * p.B || throw(DimensionMismatch(
        "field holds $(length(field)) values; this plan transforms $(p.B) field(s) of $(p.N) points — " *
        "pass the matching `batch=` to plan_spectrum"))
    (CT <: Real && eltype(field) <: Complex) && throw(ArgumentError(
        "this plan was built for a real field, whose coefficients are real; build one for " *
        "$(eltype(field)) to transform a complex field"))
    _stage_field!(p.fld, field, nothing)
    fill!(p.cd, zero(CT))
    for lo in 1:p.chunk:p.N
        pts = lo:min(p.N, lo + p.chunk - 1)
        P, Y = _node_tables(p.backend, FT, p.θ, p.φ, p.w, pts, p.lmax)
        _spherical_kernel!(p.backend)(p.cd, P, Y, p.fld, p.wd, p.lmax, lo - 1, length(pts), p.B;
            ndrange = (p.lmax + 1) * (2 * p.lmax + 1))
        KA.synchronize(p.backend)
    end
    copyto!(coeffs, p.cd)
    return p.ks
end

function FFS._gpu_directsum_spherical(exec::ComputationalBackends.GPUBackend{<:KA.Backend},
        g::FlowGeometries.Grids.AbstractGrid{<:FFS.Grids.SphericalHarmonicGeometry},
        field::AbstractArray, lmax::Int; sampling = nothing, weights = nothing,
        inactive::FFS.AbstractInactiveCells = FFS.ZeroFill())
    FT = eltype(g)
    p = FFS.plan_spectrum(SpectralBackends.DirectSumSpectralBackend(), exec, g,
        eltype(field) <: Real ? FT : Complex{FT}, (lmax + 1, 2 * lmax + 1);
        batch = FFS.Grids.field_batch_shape(g, field), sampling, weights, inactive)
    coeffs = FFS._allocate_coefficients_like(field, p)
    return coeffs, FFS.calculate_spectrum!(coeffs, p, field)
end

"""
    DirectSumGPUSphericalSynthesisPlan{FT,OT}

Reusable spherical direct-sum synthesis on a KernelAbstractions device: each node's colatitude and
longitude, read from the grid once, with the device coefficient and field buffers. `OT` is the field's
element type. Execute with `synthesize!(out, plan, coeffs)`.
"""
struct DirectSumGPUSphericalSynthesisPlan{FT, OT, NB, BK, V, CB, OB, SP <: Tuple} <: FFS.AbstractSynthesisPlan
    backend::BK
    θ::V                          # host colatitude per node
    φ::V                          # host longitude per node
    cd::CB                        # device (lmax + 1, 2lmax + 1, B) coefficients of element type `OT`
    out::OB                       # device (N, B) field
    lmax::Int
    N::Int
    chunk::Int
    spatial::SP
    batch::NTuple{NB, Int}
    B::Int
end

Base.show(io::IO, p::DirectSumGPUSphericalSynthesisPlan{FT, OT}) where {FT, OT} =
    print(io, "DirectSumGPUSphericalSynthesisPlan{", FT, "}(lmax=", p.lmax, ", ",
        OT <: Real ? "real" : "complex", ", B=", p.B, ")")

FFS.Plans.field_size(p::DirectSumGPUSphericalSynthesisPlan) = (p.spatial..., p.batch...)
FFS.Plans.field_type(::DirectSumGPUSphericalSynthesisPlan{FT, OT}) where {FT, OT} = OT

function FFS.Plans.plan_synthesis(::SpectralBackends.AbstractDirectSumSpectralBackend,
        exec::ComputationalBackends.GPUBackend{<:KA.Backend},
        g::FlowGeometries.Grids.AbstractGrid{<:FFS.Grids.SphericalHarmonicGeometry},
        ::Type{T}, ms::NTuple{2, Int}; batch::Tuple = (), kwargs...) where {T}
    FT = real(float(eltype(g)))
    OT = T <: Real ? FT : Complex{FT}
    lmax = ms[1] - 1
    lmax >= 0 || throw(ArgumentError("ms[1] must be ≥ 1 (lmax = ms[1] - 1); got $(ms[1])"))
    backend = exec.backend
    θraw, φraw = FFS.Grids._sph_points(g)
    θ = collect(FT, θraw)
    φ = collect(FT, φraw)
    N = length(θ)
    bt = NTuple{length(batch), Int}(batch)
    B = prod(bt; init = 1)
    cd = KA.allocate(backend, OT, lmax + 1, 2 * lmax + 1, B)
    out = KA.allocate(backend, OT, N, B)
    sp = size(g)
    return DirectSumGPUSphericalSynthesisPlan{FT, OT, length(bt), typeof(backend), Vector{FT}, typeof(cd),
            typeof(out), typeof(sp)}(backend, θ, φ, cd, out, lmax, N, _sph_chunk(lmax, N), sp, bt, B)
end

"""
    synthesize!(out, plan::DirectSumGPUSphericalSynthesisPlan, coeffs) -> out

Evaluate the real spherical-harmonic series `coeffs` `(lmax + 1, 2lmax + 1, batch…)` at the grid's nodes on
the plan's device, into `out` on the host or the device. A real-field plan reads the real part of complex
coefficients.
"""
function FFS.Plans.synthesize!(out::AbstractArray, p::DirectSumGPUSphericalSynthesisPlan{FT, OT},
        coeffs::AbstractArray) where {FT, OT}
    size(out) == FFS.Plans.field_size(p) || throw(DimensionMismatch(
        "out is $(size(out)); this plan writes $(FFS.Plans.field_size(p))"))
    want = (p.lmax + 1, 2 * p.lmax + 1, p.batch...)
    size(coeffs) == want || throw(DimensionMismatch("coeffs is $(size(coeffs)); this plan reads $want"))
    copyto!(p.cd, (OT <: Real && eltype(coeffs) <: Complex) ? real.(coeffs) : coeffs)
    for lo in 1:p.chunk:p.N
        pts = lo:min(p.N, lo + p.chunk - 1)
        P, Y = _node_tables(p.backend, FT, p.θ, p.φ, nothing, pts, p.lmax)
        _spherical_synth_kernel!(p.backend)(p.out, P, Y, p.cd, p.lmax, lo - 1, length(pts), p.B;
            ndrange = length(pts))
        KA.synchronize(p.backend)
    end
    copyto!(out, p.out)
    return out
end

end # module FlowFieldSpectraKernelAbstractionsExt
