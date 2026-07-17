module FlowFieldSpectraCairoMakieExt

using CairoMakie: CairoMakie
using FlowFieldSpectra: FlowFieldSpectra as FFS

# =============================================================================
# Plotting, tensor-native. Coefficients are `(ms…, batch…)`; `u_idx` selects a slice from the
# flattened trailing batch (component/level/time/…). 1D → |c(k)| line; 2D → |c(kx,ky)| heatmap.
# =============================================================================

# Batch slice `u_idx` of a `(spectral…, batch…)` array, as a dense `spectral…`-shaped array.
@inline function _batch_slice(coeffs::AbstractArray, D::Int, u_idx::Int)
    sp = ntuple(d -> size(coeffs, d), D)
    flat = reshape(coeffs, prod(sp), :)
    return reshape(flat[:, u_idx], sp)
end

"""
    plot_spectrum(k_ranges::Tuple, coeffs; title="Power Spectrum", u_idx=1, kwargs...)

Plot the magnitude of the spectral coefficients (1D line or 2D heatmap) for batch slice `u_idx`.
"""
function FFS.plot_spectrum(k_ranges::Tuple, coeffs::AbstractArray; title = "Power Spectrum", u_idx::Int = 1, kwargs...)
    D = length(k_ranges)
    if D == 1
        c = abs.(_batch_slice(coeffs, 1, u_idx))
        fig = CairoMakie.Figure()
        ax = CairoMakie.Axis(fig[1, 1]; title = title, xlabel = "k", ylabel = "|c(k)|", kwargs...)
        CairoMakie.lines!(ax, k_ranges[1], c)
        return fig
    elseif D == 2
        c = abs.(_batch_slice(coeffs, 2, u_idx))
        fig = CairoMakie.Figure()
        ax = CairoMakie.Axis(fig[1, 1]; title = title, xlabel = "kx", ylabel = "ky", kwargs...)
        hm = CairoMakie.heatmap!(ax, k_ranges[1], k_ranges[2], c)
        CairoMakie.Colorbar(fig[1, 2], hm)
        return fig
    end
    error("plot_spectrum supports 1D and 2D results (got D = $D)")
end

"""
    compare_spectra(results; u_idx=1, title="Spectral Analysis Comparison", kwargs...)

Plot multiple spectra side by side. `results` is an iterable of `label => (coeffs, k_ranges)`.
"""
function FFS.compare_spectra(results; u_idx::Int = 1, title = "Spectral Analysis Comparison", kwargs...)
    n = length(results)
    fig = CairoMakie.Figure(; size = (600 * n, 550))
    CairoMakie.Label(fig[0, :], title; fontsize = 20, font = :bold)
    for (i, (label, (coeffs, k_ranges))) in enumerate(results)
        D = length(k_ranges)
        ax = CairoMakie.Axis(fig[1, i]; title = label, xlabel = D == 1 ? "k" : "kx", ylabel = D == 1 ? "|c(k)|" : "ky")
        if D == 1
            CairoMakie.lines!(ax, k_ranges[1], abs.(_batch_slice(coeffs, 1, u_idx)))
        else
            hm = CairoMakie.heatmap!(ax, k_ranges[1], k_ranges[2], abs.(_batch_slice(coeffs, 2, u_idx)))
            i == n && CairoMakie.Colorbar(fig[1, i + 1], hm)
        end
    end
    return fig
end

"""
    compare_spectral_analysis(input_data, results; u_idx=1, title="Spectral Analysis Comparison", kwargs...)

Prepend the input field to a [`compare_spectra`](@ref)-style comparison. `input_data` is
`(x_coords, field)` where `field` is `(spatial…, batch…)`.
"""
function FFS.compare_spectral_analysis(input_data, results; u_idx::Int = 1,
        title = "Spectral Analysis Comparison", kwargs...)
    x_coords, field = input_data
    D = length(x_coords)
    n = length(results)
    fig = CairoMakie.Figure(; size = (600 * (n + 1), 550))
    CairoMakie.Label(fig[0, :], title; fontsize = 20, font = :bold)
    ax_field = CairoMakie.Axis(fig[1, 1]; title = "Input field", xlabel = "x", ylabel = D == 1 ? "u" : "y")
    u_plot = real.(_batch_slice(field, D == 1 ? 1 : 2, u_idx))
    if D == 1
        CairoMakie.lines!(ax_field, x_coords[1], vec(u_plot))
    else
        CairoMakie.heatmap!(ax_field, x_coords[1], x_coords[2], u_plot)
    end
    for (i, (label, (coeffs, k_ranges))) in enumerate(results)
        Dk = length(k_ranges)
        ax = CairoMakie.Axis(fig[1, i + 1]; title = label, xlabel = Dk == 1 ? "k" : "kx", ylabel = Dk == 1 ? "|c(k)|" : "ky")
        if Dk == 1
            CairoMakie.lines!(ax, k_ranges[1], abs.(_batch_slice(coeffs, 1, u_idx)))
        else
            hm = CairoMakie.heatmap!(ax, k_ranges[1], k_ranges[2], abs.(_batch_slice(coeffs, 2, u_idx)))
            i == n && CairoMakie.Colorbar(fig[1, i + 2], hm)
        end
    end
    return fig
end

end # module FlowFieldSpectraCairoMakieExt
