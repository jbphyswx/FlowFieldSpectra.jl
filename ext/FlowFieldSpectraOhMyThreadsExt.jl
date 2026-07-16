module FlowFieldSpectraOhMyThreadsExt

using OhMyThreads: OhMyThreads as OMT
using FlowFieldSpectra: FlowFieldSpectra as FFS

# =============================================================================
# ThreadedBackend execution of the DirectSum transform (forward + inverse), tensor-native. Cartesian
# forward parallelizes over spectral modes (each mode owns its coefficient row → race-free); spherical
# forward parallelizes over point chunks with per-task accumulators (each point touches every mode).
# Inverses parallelize over the independent output points. The batch is the inner contiguous loop.
# =============================================================================

# ---- Cartesian forward (tensor-product) ----
function FFS._directsum_cartesian!(::FFS.ThreadedBackend, coeffs::AbstractArray{Complex{FT}},
        g::Union{FFS.UniformCartesianGrid{FT, D}, FFS.NonuniformCartesianGrid{FT, D}},
        field::AbstractArray, ms::NTuple{D, Int}, iflag::Int) where {FT, D}
    axes = g.axes
    ss = map(length, axes)
    Npts = prod(ss)
    M = prod(ms)
    B = length(coeffs) ÷ M
    ks = FFS.Grids.physical_wavenumbers(g.domain_size, ms, FT)
    F = reshape(field, Npts, B)
    C = reshape(coeffs, M, B)
    fill!(C, zero(Complex{FT}))
    modes = CartesianIndices(ms)
    spat = CartesianIndices(ss)
    OMT.@tasks for mi in 1:M
        @inbounds begin
            I = modes[mi]
            for (pj, P) in enumerate(spat)
                phi = zero(FT)
                for d in 1:D
                    phi += ks[d][I[d]] * FT(axes[d][P[d]])
                end
                W = cis(-iflag * phi)
                for b in 1:B
                    C[mi, b] += F[pj, b] * W
                end
            end
        end
    end
    C ./= Npts
    return ks
end

# ---- Cartesian forward (scattered) ----
function FFS._directsum_cartesian!(::FFS.ThreadedBackend, coeffs::AbstractArray{Complex{FT}},
        g::FFS.ScatteredCartesianGrid{FT, D}, field::AbstractArray, ms::NTuple{D, Int}, iflag::Int) where {FT, D}
    coords = g.coords
    N = length(coords[1])
    M = prod(ms)
    B = length(coeffs) ÷ M
    ks = FFS.Grids.physical_wavenumbers(g.domain_size, ms, FT)
    F = reshape(field, N, B)
    C = reshape(coeffs, M, B)
    fill!(C, zero(Complex{FT}))
    modes = CartesianIndices(ms)
    OMT.@tasks for mi in 1:M
        @inbounds begin
            I = modes[mi]
            for j in 1:N
                phi = zero(FT)
                for d in 1:D
                    phi += ks[d][I[d]] * FT(coords[d][j])
                end
                W = cis(-iflag * phi)
                for b in 1:B
                    C[mi, b] += F[j, b] * W
                end
            end
        end
    end
    C ./= N
    return ks
end

# ---- Spherical forward (point chunks + per-task accumulators) ----
function FFS._directsum_spherical!(::FFS.ThreadedBackend, coeffs::AbstractArray{Complex{FT}},
        g::FFS.AbstractSphericalGrid{FT}, field::AbstractArray, lmax::Int) where {FT}
    θpt, φpt, wpt = FFS.DirectSum._sph_point_data(g)
    N = length(θpt)
    Nθc = lmax + 1
    Nφc = 2 * lmax + 1
    B = length(coeffs) ÷ (Nθc * Nφc)
    F = reshape(field, N, B)
    fill!(coeffs, zero(Complex{FT}))
    C = reshape(coeffs, Nθc, Nφc, B)
    N == 0 && return (0:lmax, -lmax:lmax)
    tables = FFS.SphericalKernels.legendre_tables(FT, lmax)
    nt = max(1, min(Threads.nthreads(), N))
    chunks = [(div((c - 1) * N, nt) + 1):(div(c * N, nt)) for c in 1:nt]
    accs = OMT.tmap(chunks) do chunk
        acc = zeros(Complex{FT}, Nθc, Nφc, B)
        Plm = Matrix{FT}(undef, lmax + 1, lmax + 1)
        @inbounds for p in chunk
            xj = cos(θpt[p])
            sj = sin(θpt[p])
            φp = φpt[p]
            wp = wpt[p]
            FFS.SphericalKernels.fill_legendre!(Plm, tables, xj, sj, lmax)
            for l in 0:lmax
                for m in -l:l
                    abs_m = abs(m)
                    factor = (m < 0 && isodd(abs_m)) ? -one(FT) : one(FT)
                    Ylm = factor * Plm[l+1, abs_m+1] * cis(m * φp)
                    idx = FFS.sph_mode_index(l, m)
                    gw = conj(Ylm) * wp
                    for b in 1:B
                        acc[idx, b] += F[p, b] * gw
                    end
                end
            end
        end
        return acc
    end
    @inbounds for acc in accs
        C .+= acc
    end
    return (0:lmax, -lmax:lmax)
end

# ---- Cartesian inverse (parallel over independent output points) ----
function FFS._synthesize_cartesian!(::FFS.ThreadedBackend, out::AbstractArray{Complex{FT}},
        g::Union{FFS.UniformCartesianGrid{FT, D}, FFS.NonuniformCartesianGrid{FT, D}},
        coeffs::AbstractArray, ms::NTuple{D, Int}, iflag::Int) where {FT, D}
    axes = g.axes
    ss = map(length, axes)
    Npts = prod(ss)
    M = prod(ms)
    B = length(out) ÷ Npts
    ks = FFS.Grids.physical_wavenumbers(g.domain_size, ms, FT)
    O = reshape(out, Npts, B)
    C = reshape(coeffs, M, B)
    fill!(O, zero(Complex{FT}))
    modes = CartesianIndices(ms)
    spat = CartesianIndices(ss)
    OMT.@tasks for pj in 1:Npts
        @inbounds begin
            P = spat[pj]
            for (mi, I) in enumerate(modes)
                phi = zero(FT)
                for d in 1:D
                    phi += ks[d][I[d]] * FT(axes[d][P[d]])
                end
                W = cis(iflag * phi)
                for b in 1:B
                    O[pj, b] += C[mi, b] * W
                end
            end
        end
    end
    return out
end

function FFS._synthesize_cartesian!(::FFS.ThreadedBackend, out::AbstractArray{Complex{FT}},
        g::FFS.ScatteredCartesianGrid{FT, D}, coeffs::AbstractArray, ms::NTuple{D, Int}, iflag::Int) where {FT, D}
    coords = g.coords
    N = length(coords[1])
    M = prod(ms)
    B = length(out) ÷ N
    ks = FFS.Grids.physical_wavenumbers(g.domain_size, ms, FT)
    O = reshape(out, N, B)
    C = reshape(coeffs, M, B)
    fill!(O, zero(Complex{FT}))
    modes = CartesianIndices(ms)
    OMT.@tasks for j in 1:N
        @inbounds begin
            for (mi, I) in enumerate(modes)
                phi = zero(FT)
                for d in 1:D
                    phi += ks[d][I[d]] * FT(coords[d][j])
                end
                W = cis(iflag * phi)
                for b in 1:B
                    O[j, b] += C[mi, b] * W
                end
            end
        end
    end
    return out
end

# ---- Spherical inverse (parallel over independent output points, per-task Legendre buffer) ----
function FFS._synthesize_spherical!(::FFS.ThreadedBackend, out::AbstractArray{Complex{FT}},
        g::FFS.AbstractSphericalGrid{FT}, coeffs::AbstractArray, lmax::Int) where {FT}
    θpt, φpt, _ = FFS.DirectSum._sph_point_data(g)
    N = length(θpt)
    Nθc = lmax + 1
    Nφc = 2 * lmax + 1
    B = length(out) ÷ N
    O = reshape(out, N, B)
    C = reshape(coeffs, Nθc, Nφc, B)
    fill!(O, zero(Complex{FT}))
    tables = FFS.SphericalKernels.legendre_tables(FT, lmax)
    OMT.@tasks for p in 1:N
        @local Plm = Matrix{FT}(undef, lmax + 1, lmax + 1)
        @inbounds begin
            xj = cos(θpt[p])
            sj = sin(θpt[p])
            φp = φpt[p]
            FFS.SphericalKernels.fill_legendre!(Plm, tables, xj, sj, lmax)
            for l in 0:lmax
                for m in -l:l
                    abs_m = abs(m)
                    factor = (m < 0 && isodd(abs_m)) ? -one(FT) : one(FT)
                    Ylm = factor * Plm[l+1, abs_m+1] * cis(m * φp)
                    idx = FFS.sph_mode_index(l, m)
                    for b in 1:B
                        O[p, b] += C[idx, b] * Ylm
                    end
                end
            end
        end
    end
    return out
end

end # module FlowFieldSpectraOhMyThreadsExt
