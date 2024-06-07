module ADOLC

using CxxWrap
using ADOLC_jll
using LinearAlgebra
using SpecialFunctions: SpecialFunctions

include("array_types.jl")
include("abs_normal.jl")
include("TbadoubleModule.jl")
include("TladoubleModule.jl")
include("Adouble.jl")
include("arithmetics.jl")
include("utils.jl")
include("derivative.jl")


using .array_types
using .TbadoubleModule
using .TladoubleModule




export TbAlloc, TlAlloc, Adouble, getValue, get_gradient

export abs_normal!, AbsNormalForm


export derivative!

export tensor_address, create_cxx_identity
export erf, eps

end # module ADOLC
