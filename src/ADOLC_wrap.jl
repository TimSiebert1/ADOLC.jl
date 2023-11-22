module ADOLC_wrap
  using CxxWrap
  
  # one need to specify the location of adolc_wrap.{so, dylib}
  total_build_dir = joinpath(@__DIR__, "build")
  @wrapmodule(() -> joinpath(total_build_dir,"libadolc_wrap"))
  
  function __init__()
    @initcxx
  end
  include("ops.jl")


function gradient(tape_num::Int64, x::Vector{Float64})
  num_ind = length(x)
  g = Vector{Float64}(undef, num_ind)
  gradient(tape_num, num_ind, x, g)
  return g
end

function gradient(tape_num::Int64, num_ind::Int64, x::Vector{Float64})
  g = Vector{Float64}(undef, num_ind)
  gradient(tape_num, num_ind, x, g)
  return g
end

export adouble, getValue

# allocators
export myalloc2, alloc_vec

# adolc utils
export trace_on, trace_off, forward, reverse2, gradient

# matrix operations 
export getindex_mat, setindex_mat, getindex_vec, setindex_vec

# current base operations:
# max, abs, exp, sqrt, *, +, -, ^


end # module ADOLC_wrap
