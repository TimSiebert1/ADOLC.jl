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
include("derivative.jl")
export derivative!

export tensor_address, create_cxx_identity
export erf, eps

end # module ADOLC
