using Documenter: Documenter
# `using FlowFieldSpectra` (bare) is intentional: Documenter's `@docs` blocks resolve the
# exported names in this module's scope. Every other package is loaded module-qualified
# (`using X: X`) purely to activate its extension, so no foreign exports leak in to clash.
using FlowFieldSpectra
using FFTW: FFTW
using FINUFFT: FINUFFT
using FastSphericalHarmonics: FastSphericalHarmonics
using NUFSHT: NUFSHT
using CairoMakie: CairoMakie

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
