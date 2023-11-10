module CppHello
  using CxxWrap
  @wrapmodule(() -> joinpath("build/","libtest_power"))

  function __init__()
    @initcxx
  end
  include("ops.jl")
export trace_on, trace_off, adouble, getValue, forward, myalloc2, alloc_vec, getindex_mat, setindex_mat, getindex_vec, setindex_vec, reverse2
end