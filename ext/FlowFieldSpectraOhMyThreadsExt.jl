module FlowFieldSpectraOhMyThreadsExt

using OhMyThreads: OhMyThreads as OMT
using FlowFieldSpectra: FlowFieldSpectra as FFS
using ComputationalBackends: ComputationalBackends
using FlowGeometries: FlowGeometries

# =============================================================================
# ThreadedBackend execution of the DirectSum transform (forward + inverse), tensor-native. Cartesian
# forward parallelizes over spectral modes (each mode owns its coefficient row → race-free); spherical
# forward parallelizes over point chunks with per-task accumulators (each point touches every mode).
# Inverses parallelize over the independent output points. The batch is the inner contiguous loop.
# =============================================================================

# ---- Cartesian forward (structured tensor-product) ----
# Separable and BLAS-threaded, so the ThreadedBackend runs the same factorized path as Serial.
FFS._directsum_cartesian!(::ComputationalBackends.AbstractThreadedBackend, coeffs::AbstractArray{Complex{FT}},
        g::FlowGeometries.Grids.AbstractStructuredGrid{<:FlowGeometries.Geometry.AbstractCartesianGeometry},
        field::AbstractArray, ms::NTuple{D, Int}, iflag::Int) where {FT, D} =
    FFS.DirectSum._calculate_spectrum_cartesian_direct!(coeffs, g, field, ms, iflag)

# ---- Cartesian forward (unstructured / scattered) ----
# A point cloud does not factorize; parallelize the direct sum over the packed modes (each mode owns its
# coefficient row → race-free). A real field computes only the packed half.
function FFS._directsum_cartesian!(::ComputationalBackends.AbstractThreadedBackend, coeffs::AbstractArray{Complex{FT}},
        g::FFS.Grids.PointwiseCartesian,
        field::AbstractArray, ms::NTuple{D, Int}, iflag::Int) where {FT, D}
    l = FFS.DirectSum.CloudCartesian()
    T = eltype(field) <: Real ? FT : Complex{FT}
    s = FFS.DirectSum.cart_setup(l, g, T, ms, FFS.Grids.field_batch_shape(g, field), iflag)
    _cloud_sum_threaded!(coeffs, s.ks, s.coords, s.pms, s.N, iflag, field, FT)
    return FFS.DirectSum._attach_twin(s.twin, l, s, field)
end

# The same sum the serial path runs, parallel over the mode set.
function _cloud_sum_threaded!(dest::AbstractArray{Complex{FT}}, kaxes::Tuple, coords::Tuple,
        pms::NTuple{D, Int}, N::Int, iflag::Int, field::AbstractArray, ::Type{FT}) where {FT, D}
    M = prod(pms)
    B = M == 0 ? 0 : length(dest) ÷ M
    C = reshape(dest, M, B)
    F = reshape(field, N, B)
    fill!(C, zero(Complex{FT}))
    modes = CartesianIndices(pms)
    OMT.@tasks for mi in 1:M
        @inbounds begin
            I = modes[mi]
            for j in 1:N
                W = cis(-iflag * FFS.DirectSum._phase_scattered(kaxes, coords, I, j, FT))
                for b in 1:B
                    C[mi, b] += F[j, b] * W
                end
            end
        end
    end
    B == 0 || (C ./= N)
    return dest
end

# ---- Spherical forward: routes on the grid's declared layout, as the serial path does ----
FFS._directsum_spherical!(::ComputationalBackends.AbstractThreadedBackend, coeffs::AbstractArray{<:Number},
        g::FlowGeometries.Grids.AbstractGrid{<:FFS.Grids.SphericalHarmonicGeometry},
        field::AbstractArray, lmax::Int; kwargs...) =
    _sph_threaded!(FFS.Grids._sph_layout(g), coeffs, g, field, lmax; kwargs...)

# ---- Spherical forward (tensor product: factored, parallel over latitude chunks) ----
function _sph_threaded!(::FFS.Grids.TensorSphere, coeffs::AbstractArray{CT}, g,
        field::AbstractArray, lmax::Int; sampling = nothing, weights = nothing) where {CT <: Number}
    FT = real(float(CT))
    λ, θ, wθ, nlon, nlat, _ = FFS.DirectSum._sph_structured_setup(g, FT; sampling = sampling,
        weights = weights, lmax = lmax)
    Nθc = lmax + 1
    Nφc = 2 * lmax + 1
    B = length(coeffs) ÷ (Nθc * Nφc)
    fhat, fhat2 = FFS.DirectSum._sph_longitude_dft(CT, λ, field, lmax, nlon, nlat, B)
    C = reshape(coeffs, Nθc, Nφc, B)
    fill!(coeffs, zero(CT))
    nlat == 0 && return (0:lmax, -lmax:lmax)
    tables = FFS.SphericalKernels.legendre_tables(FT, lmax)
    nt = max(1, min(Threads.nthreads(), nlat))
    chunks = [(div((c - 1) * nlat, nt) + 1):(div(c * nlat, nt)) for c in 1:nt]
    accs = OMT.tmap(chunks) do rows
        acc = zeros(CT, Nθc, Nφc, B)
        Plm = Matrix{FT}(undef, lmax + 1, lmax + 1)
        FFS.DirectSum._sph_theta_accumulate!(acc, fhat, fhat2, θ, wθ, tables, rows, lmax, B, Plm)
        return acc
    end
    @inbounds for acc in accs
        C .+= acc
    end
    return (0:lmax, -lmax:lmax)
end

# ---- Spherical forward (iso-latitude rings: ring chunks + per-task accumulators) ----
# Each task stages a `(lmax+1, |rows|, B)` block covering only the rings it owns, sums their longitudes
# into it, and contracts that block.
function _sph_threaded!(::FFS.Grids.RingSphere, coeffs::AbstractArray{CT}, g,
        field::AbstractArray, lmax::Int; sampling = nothing, weights = nothing) where {CT <: Number}
    FT = real(float(CT))
    rt = FFS.Grids._ring_table(g, FT; lmax = lmax)
    nr = length(rt.ranges)
    Nθc = lmax + 1
    Nφc = 2 * lmax + 1
    B = length(coeffs) ÷ (Nθc * Nφc)
    N = length(rt.λ)
    F = reshape(field, N, B)
    C = reshape(coeffs, Nθc, Nφc, B)
    fill!(coeffs, zero(CT))
    nr == 0 && return (0:lmax, -lmax:lmax)
    tables = FFS.SphericalKernels.legendre_tables(FT, lmax)
    nt = max(1, min(Threads.nthreads(), nr))
    chunks = [(div((c - 1) * nr, nt) + 1):(div(c * nr, nt)) for c in 1:nt]
    accs = OMT.tmap(chunks) do rows
        acc = zeros(CT, Nθc, Nφc, B)
        Plm = Matrix{FT}(undef, lmax + 1, lmax + 1)
        nrows = length(rows)
        fh = zeros(Complex{FT}, lmax + 1, nrows, B)
        fh2 = FFS.DirectSum._partner_buffer(CT, FT, lmax + 1, nrows, B)
        FFS.DirectSum._fill_partner!(fh2)
        FFS.DirectSum._sph_ring_longitude!(fh, fh2, F, rt, rows, lmax, B, FT)
        FFS.DirectSum._sph_theta_accumulate!(acc, fh, fh2, view(rt.θ, rows), view(rt.w, rows), tables,
            1:nrows, lmax, B, Plm)
        return acc
    end
    @inbounds for acc in accs
        C .+= acc
    end
    return (0:lmax, -lmax:lmax)
end

# ---- Spherical forward (scattered: point chunks + per-task accumulators) ----
function _sph_threaded!(::FFS.Grids.ScatteredSphere, coeffs::AbstractArray{CT}, g,
        field::AbstractArray, lmax::Int; sampling = nothing, weights = nothing) where {CT <: Number}
    FT = real(float(CT))
    θpt, φpt, wpt = FFS.DirectSum._sph_point_data(g, FT; sampling = sampling, weights = weights)
    N = length(θpt)
    Nθc = lmax + 1
    Nφc = 2 * lmax + 1
    B = length(coeffs) ÷ (Nθc * Nφc)
    F = reshape(field, N, B)
    fill!(coeffs, zero(CT))
    C = reshape(coeffs, Nθc, Nφc, B)
    N == 0 && return (0:lmax, -lmax:lmax)
    tables = FFS.SphericalKernels.legendre_tables(FT, lmax)
    nt = max(1, min(Threads.nthreads(), N))
    chunks = [(div((c - 1) * N, nt) + 1):(div(c * N, nt)) for c in 1:nt]
    accs = OMT.tmap(chunks) do chunk
        acc = zeros(CT, Nθc, Nφc, B)
        Plm = Matrix{FT}(undef, lmax + 1, lmax + 1)
        @inbounds for p in chunk
            wp = wpt[p]
            # A node of zero weight contributes nothing, and skipping it keeps a masked cell's value out
            # of the sum entirely: masked data is commonly NaN, and `0 * NaN` is NaN.
            iszero(wp) && continue
            xj = cos(θpt[p])
            sj = sin(θpt[p])
            φp = φpt[p]
            FFS.SphericalKernels.fill_legendre!(Plm, tables, xj, sj, lmax)
            for l in 0:lmax
                for m in -l:l
                    Ylm = FFS.DirectSum._real_sph(Plm, l, m, φp)   # real SH (FSH convention)
                    idx = FFS.sph_mode_index(l, m)
                    gw = Ylm * wp
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

# ---- Cartesian real inverse from the packed half (parallel over independent output points) ----
# Same two-pass native cover as `DirectSum._synthesize_packed_direct!` (see the derivation there), with
# the point loop parallelized.
function FFS._synthesize_packed!(::ComputationalBackends.AbstractThreadedBackend, out::AbstractArray{FT},
        g::FlowGeometries.Grids.AbstractStructuredGrid{<:FlowGeometries.Geometry.AbstractCartesianGeometry},
        coeffs::AbstractArray, ms::NTuple{D, Int}, iflag::Int, twin) where {FT, D}
    axes = FlowGeometries.Grids.coordinates(g)
    ss = size(g)
    Npts = prod(ss)
    ks = FFS.Grids.physical_wavenumbers(g, ms, Val(true))
    pms = FFS.Packing.packed_size(ms, Val(true))
    M = prod(pms)
    B = length(out) ÷ Npts
    O = reshape(out, Npts, B)
    fill!(O, zero(FT))
    modes = CartesianIndices(pms)
    spat = CartesianIndices(ss)
    OMT.@tasks for pj in 1:Npts
        @inbounds begin
            P = spat[pj]
            for (mi, I) in enumerate(modes)
                nyq = FFS.Packing.is_nyquist(ks[1], I[1])
                W = nyq ? zero(Complex{FT}) : cis(iflag * FFS.DirectSum._phase_tensor(ks, axes, I, P, FT))
                Wn = I[1] > 1 ? cis(iflag * FFS.DirectSum._phase_tensor_neg(ks, axes, I, P, FT)) : zero(Complex{FT})
                for b in 1:B
                    off = (b - 1) * M
                    acc = nyq ? zero(FT) : real(coeffs[mi + off] * W)
                    I[1] > 1 && (acc += real(FFS.DirectSum._neg_row_value(coeffs, ks, twin, I, off, pms, b) * Wn))
                    O[pj, b] += acc
                end
            end
        end
    end
    return out
end

function FFS._synthesize_packed!(::ComputationalBackends.AbstractThreadedBackend, out::AbstractArray{FT},
        g::FFS.Grids.PointwiseCartesian,
        coeffs::AbstractArray, ms::NTuple{D, Int}, iflag::Int, twin) where {FT, D}
    coords = FlowGeometries.Grids.coordinates(g)
    N = length(coords[1])
    ks = FFS.Grids.physical_wavenumbers(g, ms, Val(true))
    pms = FFS.Packing.packed_size(ms, Val(true))
    M = prod(pms)
    B = length(out) ÷ N
    O = reshape(out, N, B)
    fill!(O, zero(FT))
    modes = CartesianIndices(pms)
    OMT.@tasks for j in 1:N
        @inbounds begin
            for (mi, I) in enumerate(modes)
                nyq = FFS.Packing.is_nyquist(ks[1], I[1])
                W = nyq ? zero(Complex{FT}) : cis(iflag * FFS.DirectSum._phase_scattered(ks, coords, I, j, FT))
                Wn = I[1] > 1 ? cis(iflag * FFS.DirectSum._phase_scattered_neg(ks, coords, I, j, FT)) : zero(Complex{FT})
                for b in 1:B
                    off = (b - 1) * M
                    acc = nyq ? zero(FT) : real(coeffs[mi + off] * W)
                    I[1] > 1 && (acc += real(FFS.DirectSum._neg_row_value(coeffs, ks, twin, I, off, pms, b) * Wn))
                    O[j, b] += acc
                end
            end
        end
    end
    return out
end

# ---- Cartesian inverse (parallel over independent output points) ----
function FFS._synthesize_cartesian!(::ComputationalBackends.AbstractThreadedBackend, out::AbstractArray{Complex{FT}},
        g::FlowGeometries.Grids.AbstractStructuredGrid{<:FlowGeometries.Geometry.AbstractCartesianGeometry},
        coeffs::AbstractArray, ms::NTuple{D, Int}, iflag::Int) where {FT, D}
    axes = FlowGeometries.Grids.coordinates(g)
    ss = size(g)
    Npts = prod(ss)
    M = prod(ms)
    B = length(out) ÷ Npts
    ks = FFS.Grids.physical_wavenumbers(g, ms, Val(false))   # native order, matching the forward's output
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

function FFS._synthesize_cartesian!(::ComputationalBackends.AbstractThreadedBackend, out::AbstractArray{Complex{FT}},
        g::FFS.Grids.PointwiseCartesian,
        coeffs::AbstractArray, ms::NTuple{D, Int}, iflag::Int) where {FT, D}
    coords = FlowGeometries.Grids.coordinates(g)
    N = length(coords[1])
    M = prod(ms)
    B = length(out) ÷ N
    ks = FFS.Grids.physical_wavenumbers(g, ms, Val(false))   # native order, matching the forward's output
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
function FFS._synthesize_spherical!(::ComputationalBackends.AbstractThreadedBackend, out::AbstractArray{OT},
        g::FlowGeometries.Grids.AbstractGrid{<:FFS.Grids.SphericalHarmonicGeometry}, coeffs::AbstractArray, lmax::Int) where {OT <: Number}
    FT = real(float(OT))
    θraw, φraw = FFS.Grids._sph_points(g)   # synthesis reads nodes only; it carries no quadrature
    θpt = FT.(θraw)
    φpt = FT.(φraw)
    N = length(θpt)
    Nθc = lmax + 1
    Nφc = 2 * lmax + 1
    B = length(out) ÷ N
    O = reshape(out, N, B)
    C = reshape(coeffs, Nθc, Nφc, B)
    fill!(O, zero(OT))
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
                    Ylm = FFS.DirectSum._real_sph(Plm, l, m, φp)   # real SH (FSH convention)
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
