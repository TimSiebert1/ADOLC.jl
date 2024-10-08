using ADOLC
using ADOLC.TbadoubleModule
using ADOLC.array_types
using Test
using CxxWrap
using SpecialFunctions: SpecialFunctions

include("test_adouble.jl")
include("test_array_types.jl")
include("test_arithmetics.jl")
include("test_univariate_tpp.jl")
include("first_order/test_derivative.jl")
include("first_order/test_derivative!.jl")
include("first_order/test_function_and_derivative_value!.jl")
include("second_order/test_derivative.jl")
include("second_order/test_derivative!.jl")
include("higher_order/test_derivative.jl")
include("higher_order/test_derivative!.jl")
include("abs_normal/test_derivative.jl")
include("abs_normal/test_derivative!.jl")
include("param/test_param.jl")
