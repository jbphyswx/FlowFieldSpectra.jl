using Documenter: Documenter
# `using FlowFieldSpectra` (bare) is intentional: Documenter's `@docs` blocks resolve the exported
# (and submodule) names in this module's scope. Nothing else is needed — the docs env is deliberately
# lean (just Documenter + FlowFieldSpectra). The pages are static and embed figures pre-rendered by
# `generate_assets/` (its own env carries CairoMakie + the transform backends); the build never runs
# a live transform, so no backend packages are loaded here.
using FlowFieldSpectra

Documenter.makedocs(;
    modules  = [FlowFieldSpectra],
    sitename = "FlowFieldSpectra.jl",
    authors  = "Jordan Benjamin",
    format   = Documenter.HTML(;
        prettyurls  = get(ENV, "CI", "false") == "true",
        canonical   = "https://jbphyswx.github.io/FlowFieldSpectra.jl",
        edit_link   = "main",
    ),
    pages = [
        "Home"                   => "index.md",
        "Backends & Extensions"  => "backends.md",
        "Examples"               => [
            "Cartesian (FFT)"          => "examples/cartesian.md",
            "NUFFT & coastline cutout" => "examples/nufft_coastline.md",
            "4D fixed-grid spectra"    => "examples/horizontal_4d.md",
            "Derived quantities"       => "examples/derived_quantities.md",
            "Cross-spectra & coherence" => "examples/cross_spectra.md",
            "Wavenumber–frequency E(k,ω)" => "examples/komega.md",
            "Irregular & windowed"     => "examples/estimation.md",
            "Spherical harmonics"      => "examples/spherical.md",
        ],
        "API Reference"          => "api.md",
        "Internals"              => "internals.md",
    ],
)

Documenter.deploydocs(;
    repo   = "github.com/jbphyswx/FlowFieldSpectra.jl",
    target = "build",
    branch = "gh-pages",
    devbranch = "main",
)
