module Adouble
    using CxxWrap
    
    # one need to specify the location of adolc_wrap.{so, dylib}
    total_build_dir = joinpath(@__DIR__, "build")
    @wrapmodule(() -> joinpath(total_build_dir,"libadolc_wrap"), :define_julia_module)
    
    function __init__()
      @initcxx
    end

# convient inits for vector of independant and dependant 
function Base.:<<(a::Vector{Main.ADOLC_wrap.Adouble.adoubleAllocated}, x::AbstractVector)
  for i in eachindex(x)
      a[i] << x[i]
  end
end


function Base.:>>(a::Vector{Main.ADOLC_wrap.Adouble.adoubleAllocated}, x::AbstractVector)
  for i in eachindex(x)
      a[i] >> x[i]
  end
end

function Base.:>>(a::Main.ADOLC_wrap.Adouble.adoubleAllocated, x::Vector{Float64})
  if length(x) != 1
    throw("DimensionMismatch: Length of x ($x) should be 1!")
  end
  return [a] >> x
end

function Base.:>>(a::Vector{Main.ADOLC_wrap.Adouble.adoubleAllocated}, x::Float64)
  if length(a) != 1
    throw("DimensionMismatch: Length of a ($a) should be 1!")
  end
  return a >> [x]
end


function Base.:*(a::Main.ADOLC_wrap.Adouble.adoubleAllocated, x::AbstractVector)
  return map((x_i)->a*x_i, x)
end


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

  # adolc utils
  export trace_on, trace_off, forward, reverse2, gradient

  # matrix operations 
  export getindex_mat, setindex_mat, getindex_vec, setindex_vec


  # point-wise smooth utils 
  export enableMinMaxUsingAbs, get_num_switches, zos_pl_forward, fos_pl_forward, fov_pl_forward, abs_normal
  # current base operations:
  # max, abs, exp, sqrt, *, +, -, ^

end # module adouble