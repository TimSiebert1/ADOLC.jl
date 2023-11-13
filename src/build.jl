using Pkg


Pkg.add(url="git@github.com:TimSiebert1/ADOLC_jll.jl.git")
Pkg.add("libcxxwrap_julia_jll")
Pkg.add("CxxWrap")

using ADOLC_jll
using libcxxwrap_julia_jll

build_DIR = "build"
src_DIR = "."

ADOLC_DIR = ADOLC_jll.artifact_dir
JlCxx_DIR = joinpath(libcxxwrap_julia_jll.artifact_dir, "lib", "cmake", "JlCxx")

# Build!
run(`cmake -DADOLC_DIR=$(ADOLC_DIR) -DJlCxx_DIR=$(JlCxx_DIR) -S$(src_DIR) -B$(build_DIR)`)
run(`cmake --build $(build_DIR)`)


