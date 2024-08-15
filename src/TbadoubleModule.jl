module TbadoubleModule
using ADOLC_jll
using CxxWrap

@wrapmodule(() -> libadolc_wrap, :Tbadouble_module)
function __init__()
    @initcxx
end
# C++ version of adouble
export TbadoubleCxx #, getValue

# general adolc
export trace_on, trace_off

# more low level function
export zos_forward, fos_forward, hos_forward, fov_forward, hov_forward, hov_wk_forward
export fos_reverse, hos_reverse, fov_reverse, hov_reverse

# univariate TPP
export ad_forward, ad_reverse

# higher-order drivers
export tensor_eval

# more drivers
export jacobian, hessian, vec_jac, jac_vec, hess_vec, hess_mat, lagra_hess_vec, jac_solv

# abs-smooth utils 
export enableMinMaxUsingAbs,
    get_num_switches, zos_pl_forward, fos_pl_forward, fov_pl_forward, abs_normal

export cbrt

end # module adouble
