module TbadoubleModule
    using ADOLC_jll
    using CxxWrap
    
    @wrapmodule(() -> libadolc_wrap, :Tbadouble_module)
    function __init__()
      @initcxx
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


  # C++ version of adouble
  export TbadoubleCxx #, getValue

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
