module ADOLC
# using ADOLC_jll
adolc_interface_lib = "/Users/timsiebert/Projects/ADOLCInterface/ADOLCInterface.jl/lib/libADOLCInterface.dylib"
using CxxWrap
using LinearAlgebra
using SpecialFunctions: SpecialFunctions

include("array_types.jl")
using .array_types
include("abs_normal.jl")
include("Adouble.jl")
include("arithmetics.jl")
include("utils.jl")
include("derivative.jl")
include("univariate_tpp.jl")
include("tape_handling.jl")

export partial_to_adolc_format,
    partial_to_adolc_format!,
    create_cxx_identity,
    create_partial_cxx_identity,
    seed_idxs_partial_format,
    seed_idxs_adolc_format,
    partial_format_to_seed_space,
    adolc_format_to_seed_space,
    CxxVector,
    CxxMatrix,
    CxxTensor

export TapeBasedAD,
    TapeLessAD,
    Adouble,
    get_value,
    get_ad_value,
    get_ad_values,
    set_value,
    set_ad_value,
    set_num_dir
export abs_normal!, AbsNormalForm

export derivative,
    derivative!,
    function_and_derivative_value!,
    univariate_tpp!,
    univariate_tpp,
    allocator,
    jl_allocator,
    create_tape,
    create_independent,
    create_dependent,
    init_abs_normal_form,
    tensor_address,
    set_param_vec,
    fov_forward!,
    fov_reverse!

export erf, eps
export mkparam
export ADOLC_JLL_PATH
end # module ADOLC
