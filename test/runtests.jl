using ADOLC
using ADOLC.TbadoubleModule
using ADOLC.array_types
using Test


include("test_abs_normal.jl")
include("test_adouble.jl")
include("test_gradient.jl")
include("test_hess_vec.jl")
include("test_reverse.jl")
include("test_taylor_coeff.jl")
include("test_tensor_eval.jl")
