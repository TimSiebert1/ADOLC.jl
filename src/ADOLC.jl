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

export partial_to_adolc_format,
    partial_to_adolc_format!,
    create_cxx_identity,
    create_partial_cxx_identity,
    seed_idxs_partial_format,
    seed_idxs_adolc_format,
    partial_format_to_seed_space,
    adolc_format_to_seed_space,
    myalloc2,
    CxxVector,
    CxxMatrix,
    CxxTensor

export TbAlloc, TlAlloc, Adouble, getValue, get_gradient

export abs_normal!, AbsNormalForm

export derivative,
    derivative!,
    allocator,
    jl_allocator,
    deallocator!,
    create_independent,
    dependent,
    cxx_res_to_jl_res!

export erf, eps

export gradient

end # module ADOLC
