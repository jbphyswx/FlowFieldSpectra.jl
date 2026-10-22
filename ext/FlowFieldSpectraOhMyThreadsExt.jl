module FlowFieldSpectraOhMyThreadsExt

using OhMyThreads: OhMyThreads as OMT
using FlowFieldSpectra: FlowFieldSpectra as FFS
using ComputationalBackends: ComputationalBackends
using FlowGeometries: FlowGeometries

# =============================================================================
# ThreadedBackend execution of the DirectSum transform (forward + inverse), tensor-native. The forward
# runs against the setup `DirectSum.cart_setup` / `sph_setup` builds, which a one-shot builds per call and a
# plan holds. A point set's Cartesian forward parallelizes over spectral modes (each mode owns its
# coefficient row → race-free); the spherical forward over latitude rows, rings or point chunks with
# per-task accumulators (each point touches every mode). Inverses parallelize over the independent output
# points. The batch is the inner contiguous loop.
# =============================================================================

# Contiguous chunks of `1:n`, one per task.
_task_chunks(n::Int) =
    (nt = max(1, min(Threads.nthreads(), n)); [(div((c - 1) * n, nt) + 1):div(c * n, nt) for c in 1:nt])

# ---- Cartesian forward (unstructured / scattered) ----
# A point cloud does not factorize over a shared axis; each task takes a block of the last axis's modes,
# every one accumulating all points in order (`DirectSum._cloud_block!`), so the blocks write disjoint
# coefficients and the sums are the serial ones. A real field computes only the packed half, the modes
# outside its computed box and their mirror at zero. A tensor grid contracts through BLAS on either host
# backend (`FFS._cart_run!`).
function FFS._cart_run!(::ComputationalBackends.AbstractThreadedBackend, ::FFS.DirectSum.CloudCartesian,
        coeffs::AbstractArray{Complex{FT}}, s, field::AbstractArray) where {FT}
    _cloud_sum_threaded!(coeffs, s.ks, s.coords, s.pms, s.N, s.iflag, field)
    FFS.DirectSum._zero_outside!(coeffs, s.outside, prod(s.pms))
    return s.ks
end

function _cloud_sum_threaded!(dest::AbstractArray{Complex{FT}}, kaxes::Tuple, coords::Tuple,
        pms::NTuple{D, Int}, N::Int, iflag::Int, field::AbstractArray) where {FT, D}
    fill!(dest, zero(Complex{FT}))
    prod(pms) == 0 && return dest
    nD = pms[D]
    nt = max(1, min(Threads.nthreads(), nD))
    OMT.@tasks for c in 1:nt
        blk = (div((c - 1) * nD, nt) + 1):div(c * nD, nt)
        E = FFS.DirectSum._axis_scratch(FT, pms)
        FFS.DirectSum._cloud_block!(dest, kaxes, coords, pms, N, FT(-iflag), field, E, blk)
    end
    dest ./= N
    return dest
end

# ---- Spherical forward: routes on the grid's declared layout, as the serial `DirectSum.sph_run!` does ----

# The per-task accumulators summed into `coeffs` `(lmax+1, 2lmax+1, B)`.
function _sum_accumulators!(coeffs::AbstractArray, accs, lmax::Int, B::Int)
    C = reshape(coeffs, lmax + 1, 2 * lmax + 1, B)
    @inbounds for acc in accs
        C .+= acc
    end
    return (0:lmax, -lmax:lmax)
end

# Tensor product: one longitude DFT for every latitude, then the θ contraction over latitude chunks.
function FFS._sph_run!(::ComputationalBackends.AbstractThreadedBackend, ::FlowGeometries.Grids.TensorRings,
        coeffs::AbstractArray{CT}, s, field::AbstractArray, lmax::Int, B::Int) where {CT <: Number}
    FT = real(float(CT))
    FFS.DirectSum._sph_longitude_dft!(s.fhat, s.Wφ, field, s.nlon, s.nlat, B)
    FFS.DirectSum._sph_longitude_dft!(s.fhat2, s.Wφc, field, s.nlon, s.nlat, B)
    fill!(coeffs, zero(CT))
    s.nlat == 0 && return (0:lmax, -lmax:lmax)
    accs = OMT.tmap(_task_chunks(s.nlat)) do rows
        acc = zeros(CT, lmax + 1, 2 * lmax + 1, B)
        Plm = Matrix{FT}(undef, lmax + 1, lmax + 1)
        FFS.DirectSum._sph_theta_accumulate!(acc, s.fhat, s.fhat2, s.θ, s.wθ, s.tables, rows, lmax, B, Plm)
        return acc
    end
    return _sum_accumulators!(coeffs, accs, lmax, B)
end

# Iso-latitude rings: each task stages a `(lmax+1, |rows|, B)` block covering only the rings it owns, sums
# their longitudes into it, and contracts that block.
function FFS._sph_run!(::ComputationalBackends.AbstractThreadedBackend, ::FlowGeometries.Grids.IsoLatitudeRings,
        coeffs::AbstractArray{CT}, s, field::AbstractArray, lmax::Int, B::Int) where {CT <: Number}
    FT = real(float(CT))
    rt = s.rt
    F = reshape(field, s.N, B)
    fill!(coeffs, zero(CT))
    s.nr == 0 && return (0:lmax, -lmax:lmax)
    accs = OMT.tmap(_task_chunks(s.nr)) do rows
        acc = zeros(CT, lmax + 1, 2 * lmax + 1, B)
        Plm = Matrix{FT}(undef, lmax + 1, lmax + 1)
        nrows = length(rows)
        fh = zeros(Complex{FT}, lmax + 1, nrows, B)
        fh2 = FFS.DirectSum._partner_buffer(CT, FT, lmax + 1, nrows, B)
        FFS.DirectSum._fill_partner!(fh2)
        FFS.DirectSum._sph_ring_longitude!(fh, fh2, F, rt, rows, lmax, B, FT)
        FFS.DirectSum._sph_theta_accumulate!(acc, fh, fh2, view(rt.θ, rows), nothing, s.tables,
            1:nrows, lmax, B, Plm)
        return acc
    end
    return _sum_accumulators!(coeffs, accs, lmax, B)
end

# No rings: point chunks, each task projecting its points into its own accumulator.
function FFS._sph_run!(::ComputationalBackends.AbstractThreadedBackend, ::FlowGeometries.Grids.NoRings,
        coeffs::AbstractArray{CT}, s, field::AbstractArray, lmax::Int, B::Int) where {CT <: Number}
    FT = real(float(CT))
    F = reshape(field, s.N, B)
    fill!(coeffs, zero(CT))
    s.N == 0 && return (0:lmax, -lmax:lmax)
    accs = OMT.tmap(_task_chunks(s.N)) do chunk
        acc = zeros(CT, lmax + 1, 2 * lmax + 1, B)
        Plm = Matrix{FT}(undef, lmax + 1, lmax + 1)
        trig = FFS.DirectSum._lon_scratch(FT, lmax)
        @inbounds for p in chunk
            FFS.DirectSum._sph_project_point!(acc, F, Plm, trig, s.tables, s.θpt[p], s.φpt[p], s.wpt[p], p,
                lmax, B)
        end
        return acc
    end
    return _sum_accumulators!(coeffs, accs, lmax, B)
end

# ---- Cartesian real inverse from the packed half ----
# A tensor grid's synthesis is per-axis matrix products, BLAS-threaded, so the ThreadedBackend runs the
# same factorized path as Serial. A point set is parallel over its independent output points, each the
# fold-weighted sum `DirectSum._packed_point!` takes.
FFS._synthesize_packed!(::ComputationalBackends.AbstractThreadedBackend, out::AbstractArray{FT},
        g::FlowGeometries.Grids.AbstractStructuredGrid{<:FlowGeometries.Geometry.AbstractCartesianGeometry},
        coeffs::AbstractArray, ms::NTuple{D, Int}, iflag::Int, geo = nothing) where {FT, D} =
    FFS.DirectSum._synthesize_packed_direct!(out, g, coeffs, ms, iflag, geo)

function FFS._synthesize_packed!(::ComputationalBackends.AbstractThreadedBackend, out::AbstractArray{FT},
        g::FFS.Grids.PointwiseCartesian,
        coeffs::AbstractArray, ms::NTuple{D, Int}, iflag::Int, geo = nothing) where {FT, D}
    s = geo === nothing ? FFS.DirectSum.synth_setup(FFS.DirectSum.CloudCartesian(), g, FT, ms, (), iflag) : geo
    N = length(s.xs[1])
    pms = map(length, s.ks)
    B = length(out) ÷ N
    fill!(out, zero(FT))
    OMT.@tasks for j in 1:N
        @local E = FFS.DirectSum._axis_scratch(FT, pms)
        FFS.DirectSum._packed_point!(out, coeffs, E, s.ks, s.xs, pms, j, FT(iflag), N, B)
    end
    return out
end

# ---- Cartesian inverse: factorized on a tensor grid, parallel over output points on a point set ----
FFS._synthesize_cartesian!(::ComputationalBackends.AbstractThreadedBackend, out::AbstractArray{Complex{FT}},
        g::FlowGeometries.Grids.AbstractStructuredGrid{<:FlowGeometries.Geometry.AbstractCartesianGeometry},
        coeffs::AbstractArray, ms::NTuple{D, Int}, iflag::Int, geo = nothing) where {FT, D} =
    FFS.DirectSum._synthesize_cartesian_direct!(out, g, coeffs, ms, iflag, geo)

function FFS._synthesize_cartesian!(::ComputationalBackends.AbstractThreadedBackend, out::AbstractArray{Complex{FT}},
        g::FFS.Grids.PointwiseCartesian,
        coeffs::AbstractArray, ms::NTuple{D, Int}, iflag::Int, geo = nothing) where {FT, D}
    s = geo === nothing ?
        FFS.DirectSum.synth_setup(FFS.DirectSum.CloudCartesian(), g, Complex{FT}, ms, (), iflag) : geo
    N = length(s.xs[1])
    B = length(out) ÷ N
    fill!(out, zero(Complex{FT}))
    OMT.@tasks for j in 1:N
        @local E = FFS.DirectSum._axis_scratch(FT, ms)
        FFS.DirectSum._native_point!(out, coeffs, E, s.ks, s.xs, ms, j, FT(iflag), N, B)
    end
    return out
end

# ---- Spherical inverse (parallel over independent output points, per-task Legendre buffer) ----
function FFS._synthesize_spherical!(::ComputationalBackends.AbstractThreadedBackend, out::AbstractArray{OT},
        g::FlowGeometries.Grids.AbstractGrid{<:FFS.Grids.SphericalHarmonicGeometry}, coeffs::AbstractArray, lmax::Int) where {OT <: Number}
    FT = real(float(OT))
    θraw, φraw = FFS.Grids._sph_points(g)
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
        @local begin
            Plm = Matrix{FT}(undef, lmax + 1, lmax + 1)
            trig = FFS.DirectSum._lon_scratch(FT, lmax)
        end
        @inbounds FFS.DirectSum._sph_synth_point!(O, C, Plm, trig, tables, θpt[p], φpt[p], p, lmax, B)
    end
    return out
end

end # module FlowFieldSpectraOhMyThreadsExt
