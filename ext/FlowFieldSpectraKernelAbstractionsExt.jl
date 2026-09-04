module FlowFieldSpectraKernelAbstractionsExt

# `@index` and `@Const` are imported UNQUALIFIED on purpose: they are `@kernel` DSL keywords that the
# KernelAbstractions `@kernel` macro only recognizes by their bare names inside a kernel body (a
# qualified `KA.@index` is not rewritten and expands to the wrong `__index_*` call). Everything else
# stays module-qualified via `KA.`.
using KernelAbstractions: KernelAbstractions as KA, @index, @Const
using FlowFieldSpectra: FlowFieldSpectra as FFS
using ComputationalBackends: ComputationalBackends
using FlowGeometries: FlowGeometries

# =============================================================================
# GPUBackend execution of the DirectSum transform on ANY KernelAbstractions device (incl. `KA.CPU()`
# for CI parity, `CUDABackend()`, …). One thread per spectral mode, so each thread exclusively owns its
# coefficient rows — no atomics. Tensor grids decode a spatial point's coordinate from the 1-D axes on
# device (`axes[d][decode(p)[d]]`) — NO ∏N_d coordinate blob is staged. The batch is the inner loop.
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

# Stage a field `(spatial…, batch…)` as a device `(Npts, B)` array; `copyto!` converts and transfers in
# one step (`reshape` is a view, so no `(Npts, B)` host temp), and only widens when the type differs.
function _dev_field(backend::KA.Backend, field, Npts::Int, B::Int, ::Type{FT}) where {FT}
    d = KA.allocate(backend, FT, Npts, B)
    copyto!(d, reshape(eltype(field) === FT ? field : FT.(field), Npts, B))
    return d
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

function FFS._gpu_directsum_cartesian(exec::ComputationalBackends.AbstractGPUBackend,
        g::FlowGeometries.Grids.AbstractGrid{<:FlowGeometries.Geometry.AbstractCartesianGeometry},
        field::AbstractArray, ms::NTuple{D, Int}, iflag::Int) where {D}
    FT = eltype(g)
    backend = exec.backend
    ss = size(g)
    Npts = prod(ss)
    R = eltype(field) <: Real
    ks_cpu = FFS.Grids.physical_wavenumbers(g, ms, Val(R))
    pms = FFS.Packing.packed_size(ms, Val(R))
    M = prod(pms)
    B = length(field) ÷ Npts
    ksd = ntuple(d -> _dev_vec(backend, collect(ks_cpu[d]), FT), D)
    ET = eltype(field) <: Real ? FT : Complex{FT}          # stage the field in its own float type
    fieldd = _dev_field(backend, field, Npts, B, ET)
    coeffs_dev = KA.zeros(backend, Complex{FT}, M, B)
    _launch_cartesian!(coeffs_dev, backend, g, fieldd, ksd, ss, pms, Npts, M, B, iflag)
    KA.synchronize(backend)
    coeffs_dev ./= Npts
    tw = _gpu_nyquist_twin(backend, g, field, fieldd, ms, iflag, ss, Npts, B, FT)
    ks_out = tw === nothing ? ks_cpu : (FFS.Packing.with_twin(ks_cpu[1], tw), Base.tail(ks_cpu)...)
    return reshape(FFS._to_host(coeffs_dev), pms..., FFS.Grids.field_batch_shape(g, field)...), ks_out
end

# The twin the packed half needs where index negation misses `+N_d/2`, evaluated on device by rerunning
# the forward kernel over each masked mode set: the mask shape replaces `pms` and the masked wavenumbers
# replace `ks`. `nothing` when negation already reaches every partner.
function _gpu_nyquist_twin(backend, g, field, fieldd, ms::NTuple{D, Int}, iflag::Int, ss, Npts, B,
        ::Type{FT}) where {FT, D}
    (D >= 2 && eltype(field) <: Real && !FlowGeometries.Grids.isuniform(g)) || return nothing
    ks_cpu = FFS.Grids.physical_wavenumbers(g, ms, Val(true))
    pms = FFS.Packing.packed_size(ms, Val(true))
    batch = FFS.Grids.field_batch_shape(g, field)
    slices = ntuple(FFS.Packing.n_twin_slices(Val(D))) do mask
        sl = FFS.DirectSum._twin_shape(ms, pms, mask)
        S = prod(sl)
        out = KA.zeros(backend, Complex{FT}, S, B)
        if S > 0
            kaxd = ntuple(d -> _dev_vec(backend, FFS.DirectSum._twin_kaxis(FT, ks_cpu, ms, mask, d, sl[d]), FT), D)
            _launch_cartesian!(out, backend, g, fieldd, kaxd, ss, sl, Npts, S, B, iflag)
            KA.synchronize(backend)
            out ./= Npts
        end
        return reshape(FFS._to_host(out), sl..., batch...)
    end
    return FFS.Packing.NyquistTwin(slices)
end

# Structured (tensor-product) grid: stage the D axes and decode point coordinates on device. `pms` is the
# packed mode count per axis (axis 1 halved for a real field).
function _launch_cartesian!(coeffs_dev, backend, g::FlowGeometries.Grids.AbstractStructuredGrid{<:FlowGeometries.Geometry.AbstractCartesianGeometry},
        fieldd, ksd, ss::NTuple{D, Int}, pms::NTuple{D, Int}, Npts, M, B, iflag) where {D}
    FT = real(eltype(coeffs_dev))
    axesd = ntuple(d -> _dev_vec(backend, collect(FlowGeometries.Grids.coordinates(g, d)), FT), D)
    kernel! = _cart_tensor_kernel!(backend)
    kernel!(coeffs_dev, fieldd, axesd, ksd, ss, pms, Npts, M, B, D, iflag; ndrange = M)
    return coeffs_dev
end

# Pointwise grid: stage the D per-point coordinate arrays. The ambient dimension D is `length(pms)`; `ss`
# is the grid's own spatial shape and is unused here. A curvilinear grid holds one coordinate value per
# cell in an N-D array, so `vec` gives the point list the kernel indexes.
function _launch_cartesian!(coeffs_dev, backend, g::FFS.Grids.PointwiseCartesian,
        fieldd, ksd, ss, pms::NTuple{D, Int}, Npts, M, B, iflag) where {D}
    FT = real(eltype(coeffs_dev))
    coordsd = ntuple(d -> _dev_vec(backend, vec(collect(FlowGeometries.Grids.coordinates(g, d))), FT), D)
    kernel! = _cart_scattered_kernel!(backend)
    kernel!(coeffs_dev, fieldd, coordsd, ksd, pms, Npts, M, B, D, iflag; ndrange = M)
    return coeffs_dev
end

KA.@kernel function _cart_tensor_kernel!(coeffs, @Const(field), @Const(axesd), @Const(ksd),
        @Const(ss), @Const(pms), N::Int, M::Int, B::Int, D::Int, iflag::Int)
    mi = @index(Global)
    if mi <= M
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
        @Const(pms), N::Int, M::Int, B::Int, D::Int, iflag::Int)
    mi = @index(Global)
    if mi <= M
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

# =============================================================================
# Spherical direct sum — one thread per (row, col) coefficient slot (owns it across batch, no atomics)
# =============================================================================

function FFS._gpu_directsum_spherical(exec::ComputationalBackends.AbstractGPUBackend,
        g::FlowGeometries.Grids.AbstractGrid{<:FFS.Grids.SphericalHarmonicGeometry},
        field::AbstractArray, lmax::Int; sampling = nothing, weights = nothing)
    FT = eltype(g)
    backend = exec.backend
    θpt, φpt, wpt = FFS.DirectSum._sph_point_data(g, FT; sampling = sampling, weights = weights)
    N = length(θpt)
    Nθc = lmax + 1
    Nφc = 2 * lmax + 1
    B = length(field) ÷ N
    θd = _dev_vec(backend, θpt, FT)
    φd = _dev_vec(backend, φpt, FT)
    wd = _dev_vec(backend, wpt, FT)
    ET = eltype(field) <: Real ? FT : Complex{FT}          # stage the field in its own float type
    fieldd = _dev_field(backend, field, N, B, ET)
    # The kernel accumulates in `eltype(coeffs)`, so a real field's coefficients stay real on device.
    coeffs_dev = KA.zeros(backend, FFS.sph_coeff_type(eltype(field), FT), Nθc, Nφc, B)
    kernel! = _spherical_kernel!(backend)
    kernel!(coeffs_dev, θd, φd, fieldd, wd, lmax, N, B; ndrange = Nθc * Nφc)
    KA.synchronize(backend)
    batch = FFS.Grids.field_batch_shape(g, field)
    return reshape(FFS._to_host(coeffs_dev), Nθc, Nφc, batch...), (0:lmax, -lmax:lmax)
end

KA.@kernel function _spherical_kernel!(coeffs, @Const(θ), @Const(φ), @Const(field), @Const(w),
        lmax::Int, N::Int, B::Int)
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
            FT = eltype(θ)
            CT = eltype(coeffs)
            sfac = m == 0 ? one(FT) : (isodd(abs_m) ? -sqrt(FT(2)) : sqrt(FT(2)))
            @inbounds for p in 1:N
                wp = w[p]
                # A node of zero weight contributes nothing, and skipping it keeps a masked cell's value
                # out of the sum entirely: masked data is commonly NaN, and `0 * NaN` is NaN.
                iszero(wp) && continue
                xj = cos(θ[p])
                sj = sin(θ[p])
                P = _ka_normalized_legendre(l, abs_m, xj, sj)
                Ylm = sfac * P * (m >= 0 ? cos(m * φ[p]) : sin(abs_m * φ[p]))   # real SH (FSH convention)
                gw = Ylm * wp
                for b in 1:B
                    coeffs[row, col, b] += CT(field[p, b] * gw)
                end
            end
        end
    end
end

@inline function _ka_normalized_legendre(l::Int, m::Int, x::FT, s::FT)::FT where {FT}
    m > l && return zero(FT)
    P_mm = one(FT) / sqrt(FT(4π))
    for mm in 1:m
        P_mm *= -sqrt(FT(2mm + 1) / (2mm)) * s
    end
    l == m && return P_mm
    P_lm = x * sqrt(FT(2m + 3)) * P_mm
    P_lminus1_m = P_mm
    l == m + 1 && return P_lm
    for ll in (m+2):l
        coeff1 = sqrt(FT(4ll^2 - 1) / (ll^2 - m^2))
        coeff2 = sqrt(FT(2ll + 1) * ((ll - 1)^2 - m^2) / ((2ll - 3) * (ll^2 - m^2)))
        P_lminus1_m, P_lm = P_lm, x * coeff1 * P_lm - coeff2 * P_lminus1_m
    end
    return P_lm
end

end # module FlowFieldSpectraKernelAbstractionsExt
