using Documenter
using ADOLC

DocMeta.setdocmeta!(
    ADOLC,
    :DocTestSetup,
    :(using ADOLC);
    recursive=true,
)

makedocs(;
    sitename = "ADOLC.jl",
    authors="Tim Siebert",
    repo="github.com/TimSiebert1/ADOLC.jl",
    modules=[ADOLC],
    format=Documenter.HTML(),
    pages=["Home" => "index.md", "Reference" => "lib/reference.md"],
)

deploydocs(; repo="github.com/TimSiebert1/ADOLC.jl", devbranch="master")