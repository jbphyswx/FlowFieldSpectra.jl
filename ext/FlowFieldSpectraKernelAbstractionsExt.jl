module FlowFieldSpectraKernelAbstractionsExt

using KernelAbstractions: KernelAbstractions as KA, @index, @atomic, @Const
using FlowFieldSpectra: FlowFieldSpectra as FFS, GPUBackend

# =============================================================================
# Helper Utility
# =============================================================================

function _array_on_backend(a, backend::KA.Backend)
    return try
        KA.get_backend(a) == backend
    catch
        false
    end
end

# =============================================================================
# GPU DirectSum Dispatch
# =============================================================================

function FFS._calculate_spectrum_gpu(
    gpu_backend::GPUBackend,
    coords_vecs::Tuple,
    fields_vecs::Tuple,
    ms::Tuple;
    kwargs...
)
    backend = gpu_backend.backend
    D = length(coords_vecs)
    NU = length(fields_vecs)
    FT = eltype(coords_vecs[1])

    # Determine coordinate type (Spherical or Cartesian)
    if D == 2 && all(extrema(coords_vecs[1]) .<= (π + 1e-3)) && all(extrema(coords_vecs[1]) .>= -1e-5) &&
       all(extrema(coords_vecs[2]) .<= (2π + 1e-3)) && all(extrema(coords_vecs[2]) .>= -1e-5) &&
       (ms[2] == 2 * ms[1] - 1)
        # Spherical
        lmax = ms[1] - 1
        Nθ = lmax + 1
        Nφ = 2 * lmax + 1
        coeffs_dev = KA.zeros(backend, Complex{FT}, Nθ, Nφ, NU)
        ks = FFS._calculate_spectrum_gpu!(coeffs_dev, gpu_backend, coords_vecs, fields_vecs, ms; kwargs...)
        coeffs = Array(coeffs_dev)
        return (coeffs, ks)
    else
        # Cartesian
        coeffs_dev = KA.zeros(backend, Complex{FT}, ms..., NU)
        ks = FFS._calculate_spectrum_gpu!(coeffs_dev, gpu_backend, coords_vecs, fields_vecs, ms; kwargs...)
        coeffs = Array(coeffs_dev)
        return (coeffs, ks)
    end
end

function FFS._calculate_spectrum_gpu!(
    coeffs::AbstractArray{Complex{FT}},
    gpu_backend::GPUBackend,
    coords_vecs::Tuple,
    fields_vecs::Tuple,
    ms::Tuple;
    iflag::Int = 1,
    domain_size::Union{Nothing, Tuple} = nothing,
    weights::Union{Nothing, AbstractVector} = nothing,
) where {FT}
    backend = gpu_backend.backend
    D = length(coords_vecs)
    NU = length(fields_vecs)
    N = length(coords_vecs[1])

    for d in 1:D
        length(coords_vecs[d]) == N || throw(DimensionMismatch("Coordinates length mismatch"))
    end
    for u_idx in 1:NU
        length(fields_vecs[u_idx]) == N || throw(DimensionMismatch("Field length mismatch"))
    end

    # Stage inputs to device if they are not already on the backend
    is_gpu = _array_on_backend(coords_vecs[1], backend)

    coords_dev = is_gpu ? coords_vecs : ntuple(d -> begin
        dev_arr = KA.allocate(backend, FT, N)
        copyto!(dev_arr, collect(coords_vecs[d]))
        dev_arr
    end, D)

    fields_dev = is_gpu ? fields_vecs : ntuple(u -> begin
        dev_arr = KA.allocate(backend, FT, N)
        copyto!(dev_arr, collect(fields_vecs[u]))
        dev_arr
    end, NU)

    if D == 2 && all(extrema(coords_vecs[1]) .<= (π + 1e-3)) && all(extrema(coords_vecs[1]) .>= -1e-5) &&
       all(extrema(coords_vecs[2]) .<= (2π + 1e-3)) && all(extrema(coords_vecs[2]) .>= -1e-5) &&
       (ms[2] == 2 * ms[1] - 1)
        # Spherical coordinate path
        lmax = ms[1] - 1
        
        weights_dev = if weights === nothing
            w_host = fill(FT(4π) / N, N)
            w_dev = KA.allocate(backend, FT, N)
            copyto!(w_dev, w_host)
            w_dev
        else
            is_gpu ? weights : begin
                w_dev = KA.allocate(backend, FT, N)
                copyto!(w_dev, collect(weights))
                w_dev
            end
        end

        return _calculate_spectrum_spherical_gpu!(coeffs, backend, coords_dev, fields_dev, lmax, weights_dev)
    else
        # Cartesian coordinate path
        return _calculate_spectrum_cartesian_gpu!(coeffs, backend, coords_dev, fields_dev, ms, iflag, domain_size)
    end
end

# =============================================================================
# GPU Cartesian Direct Sum
# =============================================================================

function _calculate_spectrum_cartesian_gpu!(
    coeffs::AbstractArray{Complex{FT}},
    backend::KA.Backend,
    coords_dev::Tuple,
    fields_dev::Tuple,
    ms::Tuple,
    iflag::Int,
    domain_size::Union{Nothing, Tuple} = nothing,
) where {FT}
    D = length(coords_dev)
    NU = length(fields_dev)
    N = length(coords_dev[1])

    # Zero out coefficients
    fill!(coeffs, zero(Complex{FT}))

    # Coordinate ranges for physical wavenumbers (compute on CPU)
    ranges = ntuple(Val(D)) do d
        if domain_size !== nothing
            return domain_size[d]
        else
            c_host = coords_dev[d] isa Array ? coords_dev[d] : Array(coords_dev[d])
            min_x, max_x = extrema(c_host)
            return max_x - min_x
        end
    end

    # Generate physical wavenumbers on CPU
    ks_phys_cpu = ntuple(
        d ->
            range(FT(-ms[d] ÷ 2), stop = FT((ms[d] - 1) ÷ 2), length = ms[d]) .*
            (FT(2π) / (ranges[d] == 0 ? one(FT) : ranges[d])),
        Val(D),
    )

    # Allocate and copy physical wavenumbers to device
    ks_dev = ntuple(d -> begin
        v = KA.allocate(backend, FT, ms[d])
        copyto!(v, collect(ks_phys_cpu[d]))
        v
    end, D)

    # Launch Cartesian direct sum kernel
    kernel! = _cartesian_direct_sum_kernel!(backend)
    kernel!(coeffs, coords_dev, fields_dev, ks_dev, ms, N, NU, D, iflag; ndrange = prod(ms))
    KA.synchronize(backend)

    # Scale by 1/N
    coeffs ./= N

    return ks_phys_cpu
end

KA.@kernel function _cartesian_direct_sum_kernel!(
    coeffs,
    @Const(coords),
    @Const(fields),
    @Const(ks),
    @Const(ms),
    N::Int,
    NU::Int,
    D::Int,
    iflag::Int,
)
    # Get flat mode index
    mode_idx = @index(Global)
    total_modes = prod(ms)

    if mode_idx <= total_modes
        # Convert flat index to Cartesian index
        I = _ka_flat_to_cartesian(mode_idx, ms, D)

        # Get physical wavenumber components
        k_phys = ntuple(d -> ks[d][I[d]], D)

        # Accumulate over all spatial points
        for u_idx in 1:NU
            val = zero(eltype(coeffs))
            for j in 1:N
                phi = zero(eltype(k_phys[1]))
                for d in 1:D
                    phi += k_phys[d] * coords[d][j]
                end
                phi = -iflag * phi
                W = cis(phi)
                val += fields[u_idx][j] * W
            end
            @inbounds coeffs[I..., u_idx] = val
        end
    end
end

@inline function _ka_flat_to_cartesian(flat_idx::Int, ms::Tuple, D::Int)
    idx = flat_idx - 1  # 0-based
    ntuple(D) do d
        stride = 1
        for i in d+1:D
            stride *= ms[i]
        end
        i = div(idx, stride) + 1
        idx = mod(idx, stride)
        i
    end
end

# =============================================================================
# GPU Spherical Direct Sum
# =============================================================================

function _calculate_spectrum_spherical_gpu!(
    coeffs::AbstractArray{Complex{FT}},
    backend::KA.Backend,
    coords_dev::Tuple,
    fields_dev::Tuple,
    lmax::Int,
    weights_dev,
) where {FT}
    N = length(coords_dev[1])
    NU = length(fields_dev)
    Nθ = lmax + 1
    Nφ = 2 * lmax + 1

    # Zero out coefficients
    fill!(coeffs, zero(Complex{FT}))

    θ_dev = coords_dev[1]
    φ_dev = coords_dev[2]

    # Reinterpret complex array as a real array to support atomic operations
    coeffs_real = reinterpret(reshape, FT, coeffs)

    # Launch kernel - one thread per point
    kernel! = _spherical_direct_sum_kernel!(backend)
    kernel!(coeffs_real, θ_dev, φ_dev, fields_dev, weights_dev, lmax, Nφ, NU; ndrange = N)
    KA.synchronize(backend)

    return (0:lmax, -lmax:lmax)
end

KA.@kernel function _spherical_direct_sum_kernel!(
    coeffs_real,
    @Const(θ),
    @Const(φ),
    @Const(fields),
    @Const(w),
    lmax::Int,
    Nφ::Int,
    NU::Int,
)
    j = @index(Global)
    
    if j <= length(θ)
        θj = θ[j]
        φj = φ[j]
        wj = w[j]

        xj = cos(θj)
        sj = sin(θj)

        for l in 0:lmax
            for m in -l:l
                abs_m = abs(m)
                P_l_m = _ka_normalized_legendre(l, abs_m, xj, sj)

                factor = (m < 0 && isodd(abs_m)) ? -one(typeof(P_l_m)) : one(typeof(P_l_m))
                phase = cis(m * φj)
                Y_lm = factor * P_l_m * phase

                row = l - abs_m + 1
                col = m == 0 ? 1 : (m < 0 ? 2 * abs_m : 2 * m + 1)
                
                fj_conj_Ylm_wj = conj(Y_lm) * wj
                for u_idx in 1:NU
                    contrib = fields[u_idx][j] * fj_conj_Ylm_wj
                    # Use atomics for thread-safe accumulation on real and imaginary parts
                    @atomic coeffs_real[1, row, col, u_idx] += real(contrib)
                    @atomic coeffs_real[2, row, col, u_idx] += imag(contrib)
                end
            end
        end
    end
end

@inline function _ka_normalized_legendre(l::Int, m::Int, x::FT, s::FT)::FT where FT
    m > l && return zero(FT)

    # Sectoral recurrence
    P_mm = one(FT) / sqrt(FT(4π))
    for mm in 1:m
        P_mm *= -sqrt(FT(2mm + 1) / (2mm)) * s
    end

    l == m && return P_mm

    # P_{m+1}^m
    P_lm = x * sqrt(FT(2m + 3)) * P_mm
    P_lminus1_m = P_mm

    l == m + 1 && return P_lm

    # Recurrence for higher l
    for ll in (m+2):l
        coeff1 = sqrt(FT(4ll^2 - 1) / (ll^2 - m^2))
        coeff2 = sqrt(FT(2ll + 1) * ((ll - 1)^2 - m^2) / ((2ll - 3) * (ll^2 - m^2)))
        P_lminus1_m, P_lm = P_lm, x * coeff1 * P_lm - coeff2 * P_lminus1_m
    end

    return P_lm
end

end # module FlowFieldSpectraKernelAbstractionsExt
