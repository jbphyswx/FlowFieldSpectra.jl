using Documenter
using FlowFieldSpectra
using FFTW
using FINUFFT
using FastSphericalHarmonics
using NUFSHT
using CairoMakie

makedocs(;
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
            "Cross-spectra"            => "examples/cross_spectra.md",
            "Spherical harmonics"      => "examples/spherical.md",
        ],
        "API Reference"          => "api.md",
    ],
    warnonly = [:missing_docs, :docs_block],
)

deploydocs(;
    repo   = "github.com/jbphyswx/FlowFieldSpectra.jl",
    target = "build",
    branch = "gh-pages",
    devbranch = "main",
)
