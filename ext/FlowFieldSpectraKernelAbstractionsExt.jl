module FlowFieldSpectraKernelAbstractionsExt

# `@index` and `@Const` are imported UNQUALIFIED on purpose: they are `@kernel` DSL keywords that the
# KernelAbstractions `@kernel` macro only recognizes by their bare names inside a kernel body (a
# qualified `KA.@index` is not rewritten and expands to the wrong `__index_*` call). Everything else
# stays module-qualified via `KA.`.
using KernelAbstractions: KernelAbstractions as KA, @index, @Const
using FlowFieldSpectra: FlowFieldSpectra as FFS

# =============================================================================
# GPUBackend execution of the DirectSum transform on ANY KernelAbstractions device (incl. `KA.CPU()`
# for CI parity, `CUDABackend()`, …). One thread per spectral mode, so each thread exclusively owns its
# coefficient rows — no atomics. Tensor grids decode a spatial point's coordinate from the 1-D axes on
# device (`axes[d][decode(p)[d]]`) — NO ∏N_d coordinate blob is staged. The batch is the inner loop.
# =============================================================================

# Device→host copy (device-generic; `copyto!` exists for every KA/GPUArrays backend).
function _to_host(dev::AbstractArray{T}) where {T}
    host = Array{T}(undef, size(dev)...)
    copyto!(host, dev)
    return host
end

_on_backend(a, backend::KA.Backend) = (try KA.get_backend(a) == backend catch; false end)

# Stage a host vector as a length-N device array of element type FT (identity if already resident).
function _dev_vec(backend::KA.Backend, v, ::Type{FT}) where {FT}
    _on_backend(v, backend) && eltype(v) === FT && return v
    d = KA.allocate(backend, FT, length(v))
    copyto!(d, collect(FT.(v)))
    return d
end

# Stage a field `(spatial…, batch…)` as a device `(Npts, B)` array.
function _dev_field(backend::KA.Backend, field, Npts::Int, B::Int, ::Type{FT}) where {FT}
    d = KA.allocate(backend, FT, Npts, B)
    copyto!(d, reshape(collect(FT.(field)), Npts, B))
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

function FFS._gpu_directsum_cartesian(exec::FFS.GPUBackend, g::FFS.AbstractCartesianGrid{FT, D},
        field::AbstractArray, ms::NTuple{D, Int}, iflag::Int) where {FT, D}
    backend = exec.backend
    ss = FFS.spatial_size(g)
    Npts = prod(ss)
    M = prod(ms)
    B = length(field) ÷ Npts
    ks_cpu = FFS.Grids.physical_wavenumbers(g.domain_size, ms, FT)
    ksd = ntuple(d -> _dev_vec(backend, collect(ks_cpu[d]), FT), D)
    fieldd = _dev_field(backend, field, Npts, B, FT)
    coeffs_dev = KA.zeros(backend, Complex{FT}, M, B)
    _launch_cartesian!(coeffs_dev, backend, g, fieldd, ksd, ss, ms, Npts, M, B, iflag)
    KA.synchronize(backend)
    coeffs_dev ./= Npts
    return reshape(_to_host(coeffs_dev), ms..., _batch_shape(g, field)...), ks_cpu
end

_batch_shape(g::FFS.AbstractCartesianGrid, field) =
    ntuple(i -> size(field, FFS.ndims_spatial(g) + i), ndims(field) - FFS.ndims_spatial(g))

# Tensor grid: stage the D axes and decode point coordinates on device.
function _launch_cartesian!(coeffs_dev, backend, g::Union{FFS.UniformCartesianGrid, FFS.NonuniformCartesianGrid},
        fieldd, ksd, ss::NTuple{D, Int}, ms::NTuple{D, Int}, Npts, M, B, iflag) where {D}
    FT = real(eltype(coeffs_dev))
    axesd = ntuple(d -> _dev_vec(backend, collect(g.axes[d]), FT), D)
    kernel! = _cart_tensor_kernel!(backend)
    kernel!(coeffs_dev, fieldd, axesd, ksd, ss, ms, Npts, M, B, D, iflag; ndrange = M)
    return coeffs_dev
end

# Scattered grid: stage the D per-point coordinate vectors. `ss` (= (N,)) is unused here — the
# ambient dimension D comes from `ms`, not from `spatial_size` (which is the single point axis).
function _launch_cartesian!(coeffs_dev, backend, g::FFS.ScatteredCartesianGrid,
        fieldd, ksd, ss, ms::NTuple{D, Int}, Npts, M, B, iflag) where {D}
    FT = real(eltype(coeffs_dev))
    coordsd = ntuple(d -> _dev_vec(backend, collect(g.coords[d]), FT), D)
    kernel! = _cart_scattered_kernel!(backend)
    kernel!(coeffs_dev, fieldd, coordsd, ksd, ms, Npts, M, B, D, iflag; ndrange = M)
    return coeffs_dev
end

KA.@kernel function _cart_tensor_kernel!(coeffs, @Const(field), @Const(axesd), @Const(ksd),
        @Const(ss), @Const(ms), N::Int, M::Int, B::Int, D::Int, iflag::Int)
    mi = @index(Global)
    if mi <= M
        I = _ka_decode(mi, ms)
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
        @Const(ms), N::Int, M::Int, B::Int, D::Int, iflag::Int)
    mi = @index(Global)
    if mi <= M
        I = _ka_decode(mi, ms)
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

function FFS._gpu_directsum_spherical(exec::FFS.GPUBackend, g::FFS.AbstractSphericalGrid{FT},
        field::AbstractArray, lmax::Int) where {FT}
    backend = exec.backend
    θpt, φpt, wpt = FFS.DirectSum._sph_point_data(g)
    N = length(θpt)
    Nθc = lmax + 1
    Nφc = 2 * lmax + 1
    B = length(field) ÷ N
    θd = _dev_vec(backend, θpt, FT)
    φd = _dev_vec(backend, φpt, FT)
    wd = _dev_vec(backend, wpt, FT)
    fieldd = _dev_field(backend, field, N, B, FT)
    coeffs_dev = KA.zeros(backend, Complex{FT}, Nθc, Nφc, B)
    kernel! = _spherical_kernel!(backend)
    kernel!(coeffs_dev, θd, φd, fieldd, wd, lmax, N, B; ndrange = Nθc * Nφc)
    KA.synchronize(backend)
    batch = ntuple(i -> size(field, 1 + i), ndims(field) - 1)
    return reshape(_to_host(coeffs_dev), Nθc, Nφc, batch...), (0:lmax, -lmax:lmax)
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
            factor = (m < 0 && isodd(abs_m)) ? -one(FT) : one(FT)
            @inbounds for p in 1:N
                xj = cos(θ[p])
                sj = sin(θ[p])
                P_l_m = _ka_normalized_legendre(l, abs_m, xj, sj)
                Ylm = factor * P_l_m * cis(m * φ[p])
                gw = conj(Ylm) * w[p]
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
