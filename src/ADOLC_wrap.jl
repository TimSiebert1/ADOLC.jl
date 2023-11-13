module ADOLC_wrap
  using CxxWrap

  # one need to specify the location of adolc_wrap.{so, dylib}
  total_build_dir = joinpath(@__DIR__, "build")
  @wrapmodule(() -> joinpath(total_build_dir,"libadolc_wrap"))

  function __init__()
    @initcxx
  end
  include("ops.jl")
export trace_on, trace_off, adouble, getValue, forward, myalloc2, alloc_vec, getindex_mat, setindex_mat, getindex_vec, setindex_vec, reverse2
end # module ADOLC_wrap
