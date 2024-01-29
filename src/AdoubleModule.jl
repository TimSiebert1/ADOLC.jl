module AdoubleModule
    using ADOLC_jll
    using CxxWrap
    
    # one need to specify the location of adolc_wrap.{so, dylib}
    #total_build_dir = joinpath(ADOLC_jll.artifact_dir, "lib64")
    #@wrapmodule(() -> joinpath(total_build_dir,"libadolc_wrap"), :Adouble_module)
    @wrapmodule(() -> libadolc_wrap, :Adouble_module)
   
    function __init__()
        @initcxx
        println("init adouble")
    end

# convient inits for vector of independant and dependant 
function Base.:<<(a::Vector{AdoubleModule.AdoubleCxxAllocated}, x::AbstractVector)
  for i in eachindex(x)
      a[i] << x[i]
  end
end


function Base.:>>(a::Vector{AdoubleModule.AdoubleCxxAllocated}, x::AbstractVector)
  for i in eachindex(x)
      a[i] >> x[i]
  end
end

function Base.:>>(a::AdoubleModule.AdoubleCxxAllocated, x::Vector{Float64})
  if length(x) != 1
    throw("DimensionMismatch: Length of x ($x) should be 1!")
  end
  return [a] >> x
end

function Base.:>>(a::Vector{AdoubleModule.AdoubleCxxAllocated}, x::Float64)
  if length(a) != 1
    throw("DimensionMismatch: Length of a ($a) should be 1!")
  end
  return a >> [x]
end


function Base.:*(a::AdoubleModule.AdoubleCxxAllocated, x::AbstractVector)
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


  export AdoubleCxx, getValue

  # general adolc
  export trace_on, trace_off, ad_forward, ad_reverse, gradient

  # more low level function
  export zos_forward, fos_forward, hos_forward, fov_forward, hov_forward
  export              fos_reverse, hos_reverse, fov_reverse, hov_reverse

  # easy to use higher order driver
  export tensor_eval, tensor_address

  # more drivers
  export   jacobian, hessian, vec_jac, jac_vec, hess_vec, hess_mat, lagra_hess_vec, jac_solv


  # point-wise smooth utils 
  export enableMinMaxUsingAbs, get_num_switches, zos_pl_forward, fos_pl_forward, fov_pl_forward, abs_normal
  # current base operations:
  # max, abs, exp, sqrt, *, +, -, ^

end # module adouble
