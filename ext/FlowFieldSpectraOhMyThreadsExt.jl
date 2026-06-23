module FlowFieldSpectraOhMyThreadsExt

using OhMyThreads: OhMyThreads as OMT
using FlowFieldSpectra: FlowFieldSpectra as FFS

# =============================================================================
# Threaded entry points — coordinate system is fixed by the caller (grid type
# dispatch happens in core), so there is no coordinate heuristic here.
# =============================================================================

function FFS._calculate_spectrum_threaded_cartesian!(
    coeffs::AbstractArray{Complex{FT}},
    coords_vecs::Tuple,
    fields_vecs::Tuple,
    ms::Tuple,
    iflag::Int,
    domain_size,
) where {FT}
    N = length(coords_vecs[1])
    for d in 1:length(coords_vecs)
        length(coords_vecs[d]) == N || throw(DimensionMismatch("Coordinates length mismatch"))
    end
    for u_idx in 1:length(fields_vecs)
        length(fields_vecs[u_idx]) == N || throw(DimensionMismatch("Field length mismatch"))
    end
    return _calculate_spectrum_cartesian_threaded!(coeffs, coords_vecs, fields_vecs, ms, iflag, domain_size)
end

function FFS._calculate_spectrum_threaded_spherical!(
    coeffs::AbstractArray{Complex{FT}},
    coords_vecs::Tuple,
    fields_vecs::Tuple,
    lmax::Int,
    weights,
) where {FT}
    N = length(coords_vecs[1])
    for u_idx in 1:length(fields_vecs)
        length(fields_vecs[u_idx]) == N || throw(DimensionMismatch("Field length mismatch"))
    end
    return _calculate_spectrum_spherical_threaded!(coeffs, coords_vecs, fields_vecs, lmax, weights)
end

# =============================================================================
# Threaded Cartesian Direct Sum
# =============================================================================

function _calculate_spectrum_cartesian_threaded!(
    coeffs::AbstractArray{Complex{FT}},
    coords_vecs::Tuple,
    fields_vecs::Tuple,
    ms::Tuple,
    iflag::Int,
    domain_size::Union{Nothing, Tuple} = nothing,
) where {FT}
    D = length(coords_vecs)
    NU = length(fields_vecs)
    N = length(coords_vecs[1])

    # Coordinate ranges for physical wavenumbers
    ranges = ntuple(Val(D)) do d
        if domain_size !== nothing
            return domain_size[d]
        else
            min_x, max_x = extrema(coords_vecs[d])
            return max_x - min_x
        end
    end

    # Generate physical wavenumbers
    ks_phys = ntuple(
        d ->
            range(FT(-ms[d] ÷ 2), stop = FT((ms[d] - 1) ÷ 2), length = ms[d]) .*
            (FT(2π) / (ranges[d] == 0 ? one(FT) : ranges[d])),
        Val(D),
    )

    # Zero out coeffs
    fill!(coeffs, zero(Complex{FT}))

    # Threaded accumulation using OMT.@tasks (race-condition safe)
    OMT.@tasks for I in CartesianIndices(ms)
        @inbounds for j in 1:N
            phi = zero(FT)
            for d in 1:D
                phi += ks_phys[d][I[d]] * coords_vecs[d][j]
            end
            phi = -iflag * phi

            W = cis(phi)

            for u_idx in 1:NU
                coeffs[I, u_idx] += fields_vecs[u_idx][j] * W
            end
        end
    end

    coeffs ./= N
    return ks_phys
end

# =============================================================================
# Threaded Spherical Direct Sum — parallelize over point chunks with per-task
# accumulators (O(nthreads) buffers, not O(N)) and the shared Legendre table.
# =============================================================================

function _calculate_spectrum_spherical_threaded!(
    coeffs::AbstractArray{Complex{FT}},
    coords_vecs::Tuple,
    fields_vecs::Tuple,
    lmax::Int,
    weights::Union{Nothing, AbstractVector},
) where {FT}
    N = length(coords_vecs[1])
    NU = length(fields_vecs)
    Nθ = lmax + 1
    Nφ = 2 * lmax + 1

    expected_size = (Nθ, Nφ, NU)
    size(coeffs) == expected_size || throw(DimensionMismatch("coeffs size $(size(coeffs)) != expected $expected_size"))

    θ = coords_vecs[1]
    φ = coords_vecs[2]
    w = weights === nothing ? fill(FT(4π) / N, N) : weights

    fill!(coeffs, zero(Complex{FT}))
    N == 0 && return (0:lmax, -lmax:lmax)

    tables = FFS.SphericalKernels.legendre_tables(FT, lmax)

    # Contiguous point chunks, one parallel task each.
    nt = max(1, min(Threads.nthreads(), N))
    chunks = [(div((c - 1) * N, nt) + 1):(div(c * N, nt)) for c in 1:nt]

    accs = OMT.tmap(chunks) do chunk
        acc = zeros(Complex{FT}, Nθ, Nφ, NU)
        Plm = Matrix{FT}(undef, lmax + 1, lmax + 1)
        @inbounds for j in chunk
            xj = cos(θ[j])
            sj = sin(θ[j])
            φj = φ[j]
            wj = w[j]
            FFS.SphericalKernels.fill_legendre!(Plm, tables, xj, sj, lmax)
            for l in 0:lmax
                for m in -l:l
                    abs_m = abs(m)
                    P_l_m = Plm[l+1, abs_m+1]
                    factor = (m < 0 && isodd(abs_m)) ? -one(FT) : one(FT)
                    Y_lm = factor * P_l_m * cis(m * φj)
                    idx = FFS.sph_mode_index(l, m)
                    g = conj(Y_lm) * wj
                    for u_idx in 1:NU
                        acc[idx, u_idx] += fields_vecs[u_idx][j] * g
                    end
                end
            end
        end
        return acc
    end

    @inbounds for acc in accs
        coeffs .+= acc
    end

    return (0:lmax, -lmax:lmax)
end

end # module FlowFieldSpectraOhMyThreadsExt
