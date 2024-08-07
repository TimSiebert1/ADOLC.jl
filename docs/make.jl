using Documenter, DocumenterCitations
using ADOLC

bib = CitationBibliography(joinpath(@__DIR__, "src", "citations.bib"); style=:numeric)

DocMeta.setdocmeta!(ADOLC, :DocTestSetup, :(using ADOLC); recursive=true)

makedocs(;
    sitename="ADOLC.jl",
    authors="Tim Siebert",
    plugins=[bib],
    repo="github.com/TimSiebert1/ADOLC.jl",
    modules=[ADOLC],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
        "Guides" => [
            "Low Level Control" => "lib/guides/low_level_control.md",
            "Seed Matrix" => "lib/guides/seed_matrix.md",
            "Univariate Taylor Polynomial Propagation" => "lib/guides/utpp.md",
            "Tape Management" => "lib/guides/tape_management.md",
            "Working with C++ Memory" => "lib/guides/ww_cxx_mem.md",
            "Performance Tips" => "lib/guides/performance_tips.md",
            ],
        "Derivative Modes" => "lib/derivative_modes.md",
        "Reference" => "lib/reference.md",
        "Wrapped Functions" => "lib/wrapped_fcts.md",
    ],
)

deploydocs(; repo="github.com/TimSiebert1/ADOLC.jl", devbranch="master")
