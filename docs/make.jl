using Documenter 
using ADOLC


DocMeta.setdocmeta!(ADOLC, :DocTestSetup, :(using ADOLC); recursive=true)

makedocs(;
    sitename="ADOLC.jl",
    authors="Tim Siebert",
    #plugins = [bib], 
    repo="github.com/TimSiebert1/ADOLC.jl",
    modules=[ADOLC],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
        "Guides" => "lib/guides.md",
        "Derivative Modes" => "lib/derivative_modes.md",
        "Reference" => "lib/reference.md",
        "Wrapped Functions" => "lib/wrapped_fcts.md",
    ],
)

deploydocs(; repo="github.com/TimSiebert1/ADOLC.jl", devbranch="master")
