module ADOLC
include("array_types.jl")
include("TbadoubleModule.jl")
include("TladoubleModule.jl")



using .array_types
using .TbadoubleModule
using .TladoubleModule


TbAlloc = TbadoubleModule.TbadoubleCxxAllocated
TlAlloc = TladoubleModule.TladoubleCxxAllocated

include("Adouble.jl")
export TbAlloc, TlAlloc, Adouble, getValue, get_gradient


include("arithmetics.jl")

include("abs_normal.jl")
export abs_normal!, AbsNormalForm

include("utils.jl")
include("univariate_tpp.jl")
include("derivative.jl")
export derivative!

export gradient, _gradient_tape_based, _gradient_tape_less
export _higher_order, tensor_address2, build_tensor, create_cxx_identity
export taylor_coeff, check_input_taylor_coeff
export erf, eps

end # module ADOLC
