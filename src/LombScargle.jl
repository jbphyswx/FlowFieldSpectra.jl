module LombScargle

export lomb_scargle

"""
    lomb_scargle(t, y, freqs; center=true) -> Vector

Lomb–Scargle periodogram of an **irregularly-sampled** 1D series `y` at sample times/locations
`t`, evaluated at the (strictly positive) frequencies `freqs`. This is the standard estimator for
gappy / non-uniformly sampled records (moorings, drifters, satellite tracks, astronomical time
series) where an FFT cannot be applied directly.

With `center=true` (default) the sample mean is removed first. The classic time-shift `τ` is
chosen per frequency to make the cosine and sine bases orthogonal, giving a periodogram that is
invariant to time translation:

```math
P(f) = \\tfrac12\\left[ \\frac{(\\sum_j y_j\\cos\\omega(t_j-\\tau))^2}{\\sum_j\\cos^2\\omega(t_j-\\tau)}
                     + \\frac{(\\sum_j y_j\\sin\\omega(t_j-\\tau))^2}{\\sum_j\\sin^2\\omega(t_j-\\tau)} \\right],
\\quad \\omega = 2\\pi f .
```

This is the direct ``O(N\\,M)`` reference implementation (`N` samples, `M` frequencies); it needs
no FFT. `freqs` must be positive (the `f=0` term is undefined).
"""
function lomb_scargle(t::AbstractVector{T}, y::AbstractVector{T}, freqs::AbstractVector;
        center::Bool = true) where {T<:AbstractFloat}
    N = length(t)
    length(y) == N || throw(DimensionMismatch("t and y must have equal length"))
    ȳ = center ? sum(y) / N : zero(T)
    P = Vector{T}(undef, length(freqs))

    @inbounds for (i, f) in enumerate(freqs)
        f > 0 || throw(ArgumentError("frequencies must be strictly positive (got $f)"))
        ω = 2 * T(π) * T(f)
        # Orthogonalizing time shift τ.
        s2 = zero(T)
        c2 = zero(T)
        for j in 1:N
            s2 += sin(2ω * t[j])
            c2 += cos(2ω * t[j])
        end
        τ = atan(s2, c2) / (2ω)
        num_c = zero(T)
        den_c = zero(T)
        num_s = zero(T)
        den_s = zero(T)
        for j in 1:N
            arg = ω * (t[j] - τ)
            ct = cos(arg)
            st = sin(arg)
            yj = y[j] - ȳ
            num_c += yj * ct
            den_c += ct * ct
            num_s += yj * st
            den_s += st * st
        end
        pc = den_c > 0 ? num_c^2 / den_c : zero(T)
        ps = den_s > 0 ? num_s^2 / den_s : zero(T)
        P[i] = T(0.5) * (pc + ps)
    end
    return P
end

end # module LombScargle
