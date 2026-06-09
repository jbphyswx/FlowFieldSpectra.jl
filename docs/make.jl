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
