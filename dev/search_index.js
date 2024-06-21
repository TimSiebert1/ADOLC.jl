var documenterSearchIndex = {"docs":
[{"location":"lib/reference/#API-reference","page":"Reference","title":"API reference","text":"","category":"section"},{"location":"lib/reference/","page":"Reference","title":"Reference","text":"CollapsedDocStrings = true","category":"page"},{"location":"lib/reference/","page":"Reference","title":"Reference","text":"ADOLC.derivative\nADOLC.derivative!\nADOLC.tensor_address\nADOLC.partial_to_adolc_format\nADOLC.partial_to_adolc_format!\nADOLC.create_cxx_identity\nADOLC.create_partial_cxx_identity\nADOLC.seed_idxs_partial_format\nADOLC.seed_idxs_adolc_format\nADOLC.partial_format_to_seed_space\nADOLC.adolc_format_to_seed_space\nADOLC.create_independent\nADOLC.allocator\nADOLC.jl_allocator\nADOLC.deallocator!\nADOLC.cxx_mat_to_jl_mat!\nADOLC.cxx_vec_to_jl_vec!\nADOLC.cxx_tensor_to_jl_tensor!\nADOLC.cxx_res_to_jl_res!","category":"page"},{"location":"lib/reference/#ADOLC.derivative","page":"Reference","title":"ADOLC.derivative","text":"derivative(\n    f::Function,\n    x::Union{Float64,Vector{Float64}},\n    mode::Symbol;\n    dir::Union{Vector{Float64},Matrix{Float64}}=Vector{Float64}(),\n    weights::Union{Vector{Float64},Matrix{Float64}}=Vector{Float64}(),\n    tape_id::Integer=0,\n    reuse_tape::Bool=false,\n)\n\nA variant of the derivative driver, which can be used to compute first-order and second-order  derivatives, as well as the abs-normal-form  of the given function f at the point x. The available modes are listed here. The formulas in the tables define weights (left multiplier) and dir (right multiplier). Most modes leverage a tape, which has the identifier tape_id. If there is already a valid  tape for the function f at the selected point x use reuse_tape=true and set the tape_id accordingly to avoid the re-creation of the tape.\n\nExamples:\n\nFirst-Order:\n\nf(x) = sin(x)\nres = derivative(f, float(π), :jac)\n\n# output\n\n1-element Vector{Float64}:\n -1.0\n\nSecond-Order:\n\nf(x) = [x[1]*x[2]^2, x[1]^2*x[3]^3]\nx = [1.0, 2.0, -1.0]\ndir = [1.0, 0.0, 0.0]\nweights = [1.0, 1.0]\nres = derivative(f, x, :vec_hess_vec, dir=dir, weights=weights)\n\n# output\n\n3-element Vector{Float64}:\n -2.0\n  4.0\n  6.0\n\nAbs-Normal-Form:\n\nf(x) = max(x[1]*x[2], x[1]^2)\nx = [1.0, 1.0]\nres = derivative(f, x, :abs_normal)\n\n# output\n\nAbsNormalForm(0, 1, 2, 1, [1.0, 1.0], [1.0], [0.0], [0.0], [1.0], [1.5 0.5], [0.5;;], [1.0 -1.0], [0.0;;])\n\n\n\n\n\nderivative(\n    f::Function,\n    x::Union{Float64,Vector{Float64}},\n    partials::Vector{Vector{Int64}};\n    tape_id::Int64=0,\n    reuse_tape::Bool=false,\n    id_seed::Bool=false,\n    adolc_format::Bool=false,\n)\n\nA variant of the derivative driver, which can be used to compute higher-order derivatives of the function f  at the point x. The derivatives are specified as mixed-partials in the partials vector. To define the partial-derivatives use either the Partial-Format or the ADOLC-Format and set adolc_format accordingly. The flag id_seed is used to specify the method for seed-matrix generation. The underlying method leverages a tape, which has the identifier tape_id. If there is already a valid  tape for the function f at the selected point x use reuse_tape=true and set the tape_id accordingly to avoid the re-creation of the tape.\n\nExamples:\n\nPartial-Format:\n\nf(x) = [x[1]^2*x[2]^2, x[3]^2*x[4]^2]\nx = [1.0, 2.0, 3.0, 4.0]\npartials = [[1, 1, 0, 0], [0, 0, 1, 1], [2, 2, 0, 0]]\nres = derivative(f, x, partials)\n\n# output\n\n2×3 Matrix{Float64}:\n 8.0   0.0  4.0\n 0.0  48.0  0.0\n\nADOLC-Format:\n\nf(x) = [x[1]^2*x[2]^2, x[3]^2*x[4]^2]\nx = [1.0, 2.0, 3.0, 4.0]\npartials = [[2, 1, 0, 0], [4, 3, 0, 0], [2, 2, 1, 1]]\nres = derivative(f, x, partials, adolc_format=true)\n\n# output\n\n2×3 Matrix{Float64}:\n 8.0   0.0  4.0\n 0.0  48.0  0.0\n\n\n\n\n\nderivative(\n    f::Function,\n    x::Union{Float64,Vector{Float64}},\n    partials::Vector{Vector{Int64}},\n    seed::Matrix{Float64};\n    tape_id::Int64=0,\n    reuse_tape::Bool=false,\n    adolc_format::Bool=false,\n)\n\nVariant of the derivative driver for the computation of higher-order derivatives, that requires a seed. Details on the idea behind seed can be found  here.\n\nExample:\n\nf(x) = [x[1]^4, x[2]^3*x[1]]\nx = [1.0, 2.0]\npartials = [[1], [2], [3]]\nseed = [[1.0, 1.0];;]\nres = derivative(f, x, partials, seed)\n\n\n# output\n\n2×3 Matrix{Float64}:\n  4.0  12.0  24.0\n 20.0  36.0  42.0\n\n\n\n\n\n","category":"function"},{"location":"lib/reference/#ADOLC.derivative!","page":"Reference","title":"ADOLC.derivative!","text":"derivative!(\n    res,\n    f::Function,\n    m::Int64,\n    n::Int64,\n    x::Union{Float64,Vector{Float64}},\n    mode::Symbol;\n    dir::Union{Vector{Float64},Matrix{Float64}}=Vector{Float64}(),\n    weights::Union{Vector{Float64},Matrix{Float64}}=Vector{Float64}(),\n    tape_id::Int64=0,\n    reuse_tape::Bool=false,\n)\n\nA variant of the derivative driver for first-, second-order and abs-normal-form  computations that allows the user to provide a pre-allocated container for the result res.  In addition to the arguments of derivative, the output dimension m and  input dimension n of the function f is required. If there is already a valid  tape for the function f at the selected point x use reuse_tape=true and set the tape_id accordingly to avoid the re-creation of the tape.\n\nnote: Note\nres must to be C++ memory and should be allocated by allocator.  Since the memory is not managed by Julia (only the pointer to it) at the moment,  it has to be manually destroyed by the use of deallocator!. There is a guide on how to work on these CxxPtr types. \n\nExample:\n\nf(x) = [cos(x[1]), x[2]*x[3]]\nx = [0.0, 1.5, -1.0]\nmode = :hess_mat\ndir = [[1.0, -1.0, 1.0] [0.5, -0.5, 1.0]]\nm = 2\nn = 3\nres =  allocator(m, n, mode, size(dir, 2)[1], 0)\nderivative!(res, f, m, n, x, mode, dir=dir)\nfor i in 1:m\n    for j in 1:n\n        for k in 1:size(dir, 2)\n            print(res[i, j, k], \" \")\n        end\n        println(\"\")\n    end\n    println(\"\")\nend\ndeallocator!(res, m, mode)\n\n# output\n\n-1.0 -0.5 \n0.0 0.0 \n0.0 0.0 \n\n0.0 0.0 \n1.0 1.0 \n-1.0 -0.5 \n\n\n\n\n\nderivative!(\n    res,\n    f,\n    m::Int64,\n    n::Int64,\n    x::Union{Float64,Vector{Float64}},\n    partials::Vector{Vector{Int64}};\n    tape_id::Int64=0,\n    reuse_tape::Bool=false,\n    id_seed::Bool=false,\n    adolc_format::Bool=false,\n)\n\nA variant of the derivative driver for the computation of  higher-order derivatives that allows the user to provide  a pre-allocated container for the result res. In addition to the arguments of  derivative, the output dimension m and input dimension n of the function f is required. If there is already a valid tape for the function f at the  selected point x use reuse_tape=true and set the tape_id accordingly to  avoid the re-creation of the tape.\n\nnote: Note\nIn contrast to the derivative! method for the first- and second-order computations res is of type Matrix{Float64} with the dimensions (m, length(partials).\n\nExample: \n\nf(x) = x[1]^4*x[2]*x[3]*x[4]^2\nx = [3.0, -1.5, 1.5, -2.0]\npartials = [[4, 0, 0, 0], [3, 0, 1, 2]]\nm = 1\nn = 4\nres = Matrix{Float64}(undef, m, length(partials))\nderivative!(res, f, m, n, x, partials)\nres\n\n# output\n\n1×2 Matrix{Float64}:\n -216.0  -216.0\n\n\n\n\n\nderivative!(\n    res,\n    f,\n    m::Int64,\n    n::Int64,\n    x::Union{Float64,Vector{Float64}},\n    partials::Vector{Vector{Int64}},\n    seed::Matrix{Float64};\n    tape_id::Int64=0,\n    reuse_tape::Bool=false,\n    adolc_format::Bool=false,\n)\n\nVariant of the derivative! driver for the computation of higher-order derivatives, that requires a seed. Details on the idea behind seed can be found  here.\n\nExample:\n\nf(x) = [x[1]^4, x[2]^3*x[1]]\nx = [1.0, 2.0]\npartials = [[1], [2], [3]]\nseed = [[1.0, 1.0];;]\nm = 2\nn = 2\nres = Matrix{Float64}(undef, m, length(partials))\nderivative!(res, f, m, n, x, partials, seed)\nres\n\n# output\n\n2×3 Matrix{Float64}:\n  4.0  12.0  24.0\n 20.0  36.0  42.0\n\n\n\n\n\n","category":"function"},{"location":"lib/reference/#ADOLC.tensor_address","page":"Reference","title":"ADOLC.tensor_address","text":"tensor_address(degree::I, adolc_partial::Vector{I}) where I <: Integer\ntensor_address(degree::Cint, adolc_partial::Vector{I}) where I <: Integer\ntensor_address(degree::I, adolc_partial::Vector{Cint}) where I <: Integer\ntensor_address(degree::Cint, adolc_partial::Vector{Cint})\n\nGenerates the index (address) of the mixed-partial specified by adolc_partial in an higher-order derivative tensor of derivative order degree.\n\nnote: Note\nThe partial has to be in ADOLC-Format.\n\n\n\n\n\n","category":"function"},{"location":"lib/reference/#ADOLC.partial_to_adolc_format","page":"Reference","title":"ADOLC.partial_to_adolc_format","text":"partial_to_adolc_format(partial::Vector{I_1}, degree::I_2) where {I_1<:Integer, I_2<:Integer}\n\nTransforms a given partial to the ADOLC-Format. \n\nnote: Note\npartial is required to be in the Partial-format\n\nExample:\n\n\npartial = [1, 0, 4]\ndegree = sum(partial)\npartial_to_adolc_format(partial, degree)\n\n# output\n\n5-element Vector{Int32}:\n 3\n 3\n 3\n 3\n 1\n\n\n\n\n\n","category":"function"},{"location":"lib/reference/#ADOLC.partial_to_adolc_format!","page":"Reference","title":"ADOLC.partial_to_adolc_format!","text":"partial_to_adolc_format!(res::Vector{Cint}, partial::Vector{I_1}, degree::I_2) where {I_1<:Integer, I_2<:Integer}\npartial_to_adolc_format!(res::Vector{Cint}, partial::Vector{Cint}, degree::I) where I <: Integer\n\nVariant of partial_to_adolc_format that writes the result to res.\n\nExample:\n\npartial = [1, 3, 2, 0]\ndegree = sum(partial)\nres = zeros(Int32, degree)\npartial_to_adolc_format!(res, partial, degree)\n\n# output\n\n6-element Vector{Int32}:\n 3\n 3\n 2\n 2\n 2\n 1\n\n\n\n\n\n","category":"function"},{"location":"lib/reference/#ADOLC.create_cxx_identity","page":"Reference","title":"ADOLC.create_cxx_identity","text":"create_cxx_identity(n::I_1, m::I_2) where {I_1 <: Integer, I_2 <: Integer}\n\nCreates a identity matrix of shape (n, m) of type CxxPtr{CxxPtr{Float64}} (wrapper of C++'s double**).\n\nExample\n\nid = create_cxx_identity(2, 4)\nfor i in 1:2\n    for j in 1:4\n        print(id[i, j], \" \")\n    end\n    println(\"\")\nend\n\n# output\n\n1.0 0.0 0.0 0.0 \n0.0 1.0 0.0 0.0\n\n\n\n\n\n","category":"function"},{"location":"lib/reference/#ADOLC.create_partial_cxx_identity","page":"Reference","title":"ADOLC.create_partial_cxx_identity","text":"create_partial_cxx_identity(n::I_1, idxs::Vector{I_2}) where {I_1 <: Integer, I_2 <: Integer}\n\nCreates a matrix of shape (n, length(idxs)) of type CxxPtr{CxxPtr{Float64}} (wrapper of C++'s double**). The columns are canonical basis vectors corresponding to the entries of idxs. The order of the basis vectors is defined by the order of the indices in idxs. Details about the application can be found in this guide.\n\nwarning: Warning\nThe number of rows n must be smaller than the maximal index of idxs!\n\nwarning: Warning\nThe values of idxs must be non-negative!\n\nExamples\n\nn = 4\nidxs = [1, 3]\nid = create_partial_cxx_identity(n, idxs)\nfor i in 1:4\n    for j in 1:length(idxs)\n        print(id[i, j], \" \")\n    end\n    println(\"\")\nend\n\n# output\n\n1.0 0.0 \n0.0 0.0\n0.0 1.0\n0.0 0.0\n\nThe order in idxs defines the order of the basis vectors.\n\nn = 3\nidxs = [3, 0, 1]\nid = create_partial_cxx_identity(n, idxs)\nfor i in 1:3\n    for j in 1:length(idxs)\n        print(id[i, j], \" \")\n    end\n    println(\"\")\nend\n\n# output\n\n0.0 0.0 1.0\n0.0 0.0 0.0\n1.0 0.0 0.0\n\n\n\n\n\n","category":"function"},{"location":"lib/reference/#ADOLC.seed_idxs_partial_format","page":"Reference","title":"ADOLC.seed_idxs_partial_format","text":"seed_idxs_partial_format(partials::Vector{Vector{I}}) where I <: Integer\n\nExtracts the actually required derivative directions of partials and returns them  ascendet sorted. \n\nnote: Note\npartials has to be in Partial-Format.\n\nExample\n\n\npartials = [[1, 0, 0, 0, 3], [1, 0, 1, 0, 0], [0, 0, 3, 0, 0]]\nseed_idxs_partial_format(partials)\n\n# output\n\n3-element Vector{Int64}:\n 1\n 3\n 5\n\n\n\n\n\n","category":"function"},{"location":"lib/reference/#ADOLC.seed_idxs_adolc_format","page":"Reference","title":"ADOLC.seed_idxs_adolc_format","text":"seed_idxs_adolc_format(partials::Vector{Vector{I}}) where I <: Integer\n\nExtracts the actually required derivative directions of partials and returns them  ascendet sorted. \n\nnote: Note\npartials has to be in ADOLC-Format.\n\nExample\n\n\npartials = [[5, 5, 5, 1], [3, 1, 0, 0], [3, 3, 3, 0]]\nseed_idxs_adolc_format(partials)\n\n# output\n\n3-element Vector{Int64}:\n 1\n 3\n 5\n\n\n\n\n\n","category":"function"},{"location":"lib/reference/#ADOLC.partial_format_to_seed_space","page":"Reference","title":"ADOLC.partial_format_to_seed_space","text":"partial_format_to_seed_space(partials::Vector{Vector{I_1}}, seed_idxs::Vector{I_2}) where {I_1 <: Integer, I_2 <: Integer}\npartial_format_to_seed_space(partials::Vector{Vector{I}}) where I <: Integer\n\nConverts partials in Partial-Format to partials of the same format but with (possible) reduced number  of derivatives directions. The seed_idxs is expected to store the result of seed_idxs_partial_format(seed_idxs). Without seed_idxs the function first calls seed_idxs_partial_format(seed_idxs) to get the indices.\n\nExamples\n\n\npartials = [[0, 1, 1], [0, 2, 0]]\nseed_idxs = seed_idxs_partial_format(partials)\npartial_format_to_seed_space(partials, seed_idxs)\n\n# output\n\n2-element Vector{Vector{Int64}}:\n [1, 1]\n [2, 0]\n\nWithout seed_idxs\n\n\npartials = [[0, 1, 1], [0, 2, 0]]\npartial_format_to_seed_space(partials)\n\n# output\n\n2-element Vector{Vector{Int64}}:\n [1, 1]\n [2, 0]\n\n\n\n\n\n","category":"function"},{"location":"lib/reference/#ADOLC.adolc_format_to_seed_space","page":"Reference","title":"ADOLC.adolc_format_to_seed_space","text":"adolc_format_to_seed_space(partials::Vector{Vector{I_1}}, seed_idxs::Vector{I_2}) where {I_1 <: Integer, I_2 <: Integer}\nadolc_format_to_seed_space(partials::Vector{Vector{I}}) where I <: Integer\n\nSame as partial_format_to_seed_space but with ADOLC-Format.\n\nExamples\n\n\npartials = [[3, 2], [2, 2]]\nseed_idxs = seed_idxs_adolc_format(partials)\nadolc_format_to_seed_space(partials, seed_idxs)\n\n# output\n\n2-element Vector{Vector{Int64}}:\n [2, 1]\n [1, 1]\n\nWithout seed_idxs\n\n\npartials = [[3, 2], [2, 2]]\nseed_idxs = seed_idxs_adolc_format(partials)\nadolc_format_to_seed_space(partials, seed_idxs)\n\n# output\n\n2-element Vector{Vector{Int64}}:\n [2, 1]\n [1, 1]\n\n\n\n\n\n","category":"function"},{"location":"lib/reference/#ADOLC.create_independent","page":"Reference","title":"ADOLC.create_independent","text":"create_independent(x::Union{Float64, Vector{Float64}})\n\n\n\n\n\n","category":"function"},{"location":"lib/reference/#ADOLC.allocator","page":"Reference","title":"ADOLC.allocator","text":"allocator(m::Int64, n::Int64, mode::Symbol, num_dir::Int64, num_weights::Int64)\n\n\n\n\n\n","category":"function"},{"location":"lib/reference/#ADOLC.jl_allocator","page":"Reference","title":"ADOLC.jl_allocator","text":"jl_allocator(m::Int64, n::Int64, mode::Symbol, num_dir::Int64, num_weights::Int64)\n\n\n\n\n\n","category":"function"},{"location":"lib/reference/#ADOLC.deallocator!","page":"Reference","title":"ADOLC.deallocator!","text":"deallocator(res, m::Int64, mode::Symbol)\n\n\n\n\n\n","category":"function"},{"location":"lib/reference/#ADOLC.array_types.cxx_mat_to_jl_mat!","page":"Reference","title":"ADOLC.array_types.cxx_mat_to_jl_mat!","text":"cxx_mat_to_jl_mat!(\n    jl_mat::Matrix{Float64}, mat_cxx::CxxPtr{CxxPtr{Float64}}, dim1, dim2\n)\n\n\n\n\n\n","category":"function"},{"location":"lib/reference/#ADOLC.array_types.cxx_vec_to_jl_vec!","page":"Reference","title":"ADOLC.array_types.cxx_vec_to_jl_vec!","text":"cxx_vec_to_jl_vec!(jl_vec::Vector{Float64}, cxx_vec::CxxPtr{Float64}, dim)\n\n\n\n\n\n","category":"function"},{"location":"lib/reference/#ADOLC.array_types.cxx_tensor_to_jl_tensor!","page":"Reference","title":"ADOLC.array_types.cxx_tensor_to_jl_tensor!","text":"cxx_tensor_to_jl_tensor!(\n    jl_tensor::Array{Float64,3},\n    cxx_tensor::CxxPtr{CxxPtr{CxxPtr{Float64}}},\n    dim1,\n    dim2,\n    dim3,\n)\n\n\n\n\n\n","category":"function"},{"location":"lib/reference/#ADOLC.array_types.cxx_res_to_jl_res!","page":"Reference","title":"ADOLC.array_types.cxx_res_to_jl_res!","text":"cxx_res_to_jl_res!(\n    jl_res, cxx_res, m::Int64, n::Int64, mode::Symbol, num_dir::Int64, num_weights::Int64\n)\n\n\n\n\n\n","category":"function"},{"location":"lib/wrapped_fcts/#List-of-wrapped-ADOL-C-drivers","page":"Wrapped Functions","title":"List of wrapped ADOL-C drivers","text":"","category":"section"},{"location":"lib/wrapped_fcts/#TbadoubleModule","page":"Wrapped Functions","title":"TbadoubleModule","text":"","category":"section"},{"location":"lib/wrapped_fcts/","page":"Wrapped Functions","title":"Wrapped Functions","text":"getValue\ngradient\njacobian\nhessian\nvec_jac\njac_vec\nhess_vec\nhess_mat\nlagra_hess_vec\njac_solv\n\nad_forward(short tag, int m, int n, int d, int keep, double **X, double **Y) (in ADOL-C: forward)\nad_reverse(short tag, int m, int n, int d, double *u, double **Z) (in ADOL-C: reverse)\n\nzos_forward\nfos_forward\nhos_forward\nhov_wk_forward\n\nfov_forward\nhov_forward\n\nfos_reverse\nhos_reverse\n\nfov_reverse\nhov_reverse\ntensor_address\ntensor_eval","category":"page"},{"location":"lib/wrapped_fcts/#Abs-Smooth-Utilities","page":"Wrapped Functions","title":"Abs-Smooth Utilities","text":"","category":"section"},{"location":"lib/wrapped_fcts/","page":"Wrapped Functions","title":"Wrapped Functions","text":"enableMinMaxUsingAbs\nget_num_switches\nzos_pl_forward\nfos_pl_forward\nfov_pl_forward\nabs_normal","category":"page"},{"location":"lib/wrapped_fcts/#Tape-Utilities","page":"Wrapped Functions","title":"Tape Utilities","text":"","category":"section"},{"location":"lib/wrapped_fcts/","page":"Wrapped Functions","title":"Wrapped Functions","text":"<< (in ADOL-C: <<=)\n>> (in ADOL-C: =>>)\ntrace_on(int tag)\ntrace_on(int tag, int keep)\ntrace_off(int file)\ntrace_off()","category":"page"},{"location":"lib/wrapped_fcts/#TladoubleModule","page":"Wrapped Functions","title":"TladoubleModule","text":"","category":"section"},{"location":"lib/wrapped_fcts/","page":"Wrapped Functions","title":"Wrapped Functions","text":"setNumDir(int const &n) \ngetValue()                      \ngetADValue(int const &i)\nsetADValue(double const &val)\nsetADValue(double const val, int const &i)","category":"page"},{"location":"lib/wrapped_fcts/#Arithmethics","page":"Wrapped Functions","title":"Arithmethics","text":"","category":"section"},{"location":"lib/wrapped_fcts/","page":"Wrapped Functions","title":"Wrapped Functions","text":"+ \n- \n* \n/ \n^","category":"page"},{"location":"lib/wrapped_fcts/#Comparison","page":"Wrapped Functions","title":"Comparison","text":"","category":"section"},{"location":"lib/wrapped_fcts/","page":"Wrapped Functions","title":"Wrapped Functions","text":"<\n>\n>=\n<=\n==","category":"page"},{"location":"lib/wrapped_fcts/#Unary-Functions","page":"Wrapped Functions","title":"Unary Functions","text":"","category":"section"},{"location":"lib/wrapped_fcts/","page":"Wrapped Functions","title":"Wrapped Functions","text":"abs\nsqrt\nsin\ncos\ntan\nasin\nacos\natan\nexp\nlog\nlog10\nsinh\ncosh\ntanh\nasinh\nacosh\natanh\nceil\nfloor\nmax\nmin\nldexp\nfrexp\nerf\ncbrt","category":"page"},{"location":"lib/guides/#Working-with-C-Memory","page":"Guides","title":"Working with C++ Memory","text":"","category":"section"},{"location":"lib/guides/","page":"Guides","title":"Guides","text":"ADOLC.jl is a wrapper of the C/C++ library ADOL-C. Wrapper means data from Julia is passed to C++, and calls to functions in Julia trigger C++ function calls to get output data in Julia. The communication between Julia and ADOL-C is handled by CxxWrap.jl, which, for example, allows to pass a Cint from Julia into a int in C++ automatically. Most functions  of ADOL-C modify pre-allocated memory declared as double*, double**, or double*** to store a functions' results. Two apparent options exist for providing the pre-allocated data to the C++ functions called from Julia. The first option would be to write wrappers in C++, which allocate the memory every time before the actual ADOL-C function call. This would cease control over the allocations from Julia, but it would be easier to work with on the Julia side, and the C++ side would have full control over the memory. The second option is to allocate  C++-owned data from julia by calling ADOL-C's allocation methods from Julia, to pass these data to ADOL-C's functions mutating the allocated data, and to access the mutated values from Julia. The second option allows more control over the data, but a user has to be aware of some critical aspects: ","category":"page"},{"location":"lib/guides/","page":"Guides","title":"Guides","text":"C++ owned memory is not automatically garbage collected and can lead to memory leaks quickly if not released. For example, having a Julia function that allocates a double** in C++ and binds this pointer to a variable in a Julia would release the  bound variable when going out of the functions' scope, but the C++ memory would still be  there, you just cannot access it anymore.\nThere is no out-of-bounds error checking to prevent accessing or setting of  data outside of the allocated area, which may lead to segmentation faults and program crashes.\nIf you want to do computations with the C++ data, you either have to copy these to a corresponding Julia type or write access  methods to work with the C++  allocated data.","category":"page"},{"location":"lib/guides/","page":"Guides","title":"Guides","text":"ADOLC.jl implements the second option. The critical aspects can still be avoided using the driver derivative, which handles the C++ allocated memory for you. However, the best performance is obtained when using derivative!. For first- and second-order derivative computations, the derivative! driver requires  a pre-allocated container of C++ allocated data. The allocation is done using allocator. This function allocates C++ memory for your specific case (i.e., for the problem-specific parameters m, n, mode, num_dir, num_weights). Thus, the computation of the derivative just  utilizes derivative. For example:","category":"page"},{"location":"lib/guides/","page":"Guides","title":"Guides","text":"using ADOLC\nf(x) = (x[1] - x[2])^2\nx = [3.0, 7.5]\ndir = [1/3, 1/7]\nm = 1\nn = 2\nmode = :jac_vec\nnum_dir = size(dir, 2)[1]\nnum_weights = 0\ncxx_res = allocator(m, n, mode, num_dir, num_weights)\nderivative!(cxx_res, f, m, n, x, mode, dir=dir)\ndeallocator!(cxx_res, m, mode)","category":"page"},{"location":"lib/guides/","page":"Guides","title":"Guides","text":"The first critical point is tackled using the deallocator! function, which handles the release of the C++ memory. Of course, one wants to conduct computations with cxx_res. The recommended way to do so is to pre-allocate a corresponding Julia container (Vector{Float64}, Matrix{Float64} or Array{Float64, 3}) obtained from jl_allocator and copy the data from cxx_res the Julia storage jl_res by leveraging cxx_res_to_jl_res!:","category":"page"},{"location":"lib/guides/","page":"Guides","title":"Guides","text":"using ADOLC\nf(x) = (x[1] - x[2])^2\nx = [3.0, 7.5]\ndir = [1/3, 1/7]\nm = 1\nn = 2\nmode = :jac_vec\nnum_dir = size(dir, 2)[1]\nnum_weights = 0\n\n# pre-allocation \njl_res = jl_allocator(m, n, mode, num_dir, num_weights)\ncxx_res = allocator(m, n, mode, num_dir, num_weights)\n\nderivative!(cxx_res, f, m, n, x, mode, dir=dir)\n\n# conversion \ncxx_res_to_jl_res!(jl_res, cxx_res, m, n, mode, num_dir, num_weights)\n\n# do computations .... \n\n\ndeallocator!(cxx_res, m, mode)","category":"page"},{"location":"lib/guides/","page":"Guides","title":"Guides","text":"Since you work with Julia data, the procedure above avoids the second and third points of the critical aspects but includes an additional allocation.  ","category":"page"},{"location":"lib/guides/","page":"Guides","title":"Guides","text":"warning: Warning\ncxx_res still stores a pointer. The corresponding memory is destroyed, but cxx_res is managed by Julia's garbage collector. Do not use it.","category":"page"},{"location":"lib/guides/","page":"Guides","title":"Guides","text":"In the future, the plan is to implement a struct that combines the Julia and C++ arrays with a finalizer that enables Julia's garbage collector to manage the C++ memory. ","category":"page"},{"location":"lib/guides/#Seed-Matrix","page":"Guides","title":"Seed-Matrix","text":"","category":"section"},{"location":"lib/guides/","page":"Guides","title":"Guides","text":"This guide is related to the higher-order derivative computation with  derivative or derivative!. Internally, the drivers are based on the propagation of univariate Taylor polynomials [1]. The underlying method leverages a seed matrix Sin mathbbR^n times p to compute mixed-partials of arbitrary order for a function fmathbbR^n to mathbbR^m in the form: ","category":"page"},{"location":"lib/guides/","page":"Guides","title":"Guides","text":"    fracpartial^k f(x + Sz)partial^k zbig_z=0 ","category":"page"},{"location":"lib/guides/","page":"Guides","title":"Guides","text":"for some z in mathbbR^p. Usually, S is the identity or the partial identity (see create_partial_cxx_identity), which is also the case, when no seed is passed to the driver. To switch between both identity options the flag id_seed can be used. In the case of identity, the formula above boils down to ","category":"page"},{"location":"lib/guides/","page":"Guides","title":"Guides","text":"    fracpartial^k f(x + Sz)partial^k zbig_z=0= fracpartial^k f(x)partial^k x","category":"page"},{"location":"lib/guides/","page":"Guides","title":"Guides","text":"Moreover, the partial identity results in the same but is more efficient. Leveraging the partial identity ensures that only the derivatives of the requested derivative directions are computed, and this is explained briefly in the following paragraph.   ","category":"page"},{"location":"lib/guides/","page":"Guides","title":"Guides","text":"Assume we want to compute the derivatives specified in the Partial-Format: [[4, 0, 0, 3], [2, 0, 0, 4], [1, 0, 0, 1]].   Obviously, none of the derivatives includes x_2 and x_3. To avoid unnecessary computations (i.e., the propagation of unnecessary univariate Polynomials), the partial identity is created, stacking only those canonical basis vectors that are related to the requested derivative directions. In our case, the partial identity looks like this:  ","category":"page"},{"location":"lib/guides/","page":"Guides","title":"Guides","text":"left\n    beginmatrix\n    1  0 \n    0  0 \n    0  0 \n    0  1 \n    endmatrix\n right","category":"page"},{"location":"lib/guides/","page":"Guides","title":"Guides","text":"As you can see, the directions are reduced from four to two. In general, the number of required univariate Polynomial propagations to compute all mixed-partials up to degree d of for f is left( beginmatrix n - 1 + d  d endmatrix right). Leveraging the seed S reduces this number to left( beginmatrix p - 1 + d  d endmatrix right), where p is often much smaller than n. In addition, S can be used as a subspace projection. For example, if S=1 dots 1^T, you could compute the sum of the different univariate Taylor coefficients:","category":"page"},{"location":"lib/guides/","page":"Guides","title":"Guides","text":"using ADOLC\nf(x) = x[1]^3*x[2]^2 - x[2]^3\nx = [1.0, 1.0]\npartials = [[1], [2], [3]]\nseed = [[1.0, 1.0];;]\nres = derivative(f, x, partials, seed)\n\n# output\n\n1×3 Matrix{Float64}:\n 2.0  14.0  54.0","category":"page"},{"location":"lib/guides/#Tape-Management","page":"Guides","title":"Tape Management","text":"","category":"section"},{"location":"lib/guides/#Performance-Tips","page":"Guides","title":"Performance Tips","text":"","category":"section"},{"location":"lib/guides/","page":"Guides","title":"Guides","text":"The following tips are meant to decrease the derivative computation's runtime complexity, especially when derivatives of the same function are needed repeatedly. There are two major modifications for all kinds of derivatives: ","category":"page"},{"location":"lib/guides/","page":"Guides","title":"Guides","text":"Use the derivative! driver, and work with allocator, deallocator!, and [jl_res_to_cxx_res!] as explained in the guide Working with C++ Memory\nReuse the tape as often as possible. derivative! (derivative) supply the flag reuse_tape, which if set to true suppresses the creation of the tape, in addition the identifiyer of an existing tape must be provided as the parameter tape_id. More details can be found here.","category":"page"},{"location":"lib/guides/","page":"Guides","title":"Guides","text":"An example could look like this:","category":"page"},{"location":"lib/guides/","page":"Guides","title":"Guides","text":"using ADOLC\n\n# problem setup\nf(x) = (x[1] - x[2])^2\nx = [3.0, 7.5]\ndir = [1/3, 1/7]\nm = 1\nn = 2\nmode = :jac_vec\nnum_dir = size(dir, 2)[1]\nnum_weights = 0\ntape_id = 1\nnum_iters = 100\n\n# pre-allocation \njl_res = jl_allocator(m, n, mode, num_dir, num_weights)\ncxx_res = allocator(m, n, mode, num_dir, num_weights)\n\nderivative!(cxx_res, f, m, n, x, mode, dir=dir, tape_id=tape_id)\n\n# conversion \ncxx_res_to_jl_res!(jl_res, cxx_res, m, n, mode, num_dir, num_weights)\n# do computations ....\n\nfor i in 1:num_iters\n    # update x\n    derivative!(cxx_res, f, m, n, x, mode, dir=dir, tape_id=tape_id, reuse_tape=true)\n    cxx_res_to_jl_res!(jl_res, cxx_res, m, n, mode, num_dir, num_weights)\n    # do computations ... \nend\ndeallocator!(cxx_res, m, mode)","category":"page"},{"location":"lib/guides/","page":"Guides","title":"Guides","text":"Moreover, for higher-order derivatives you might consider the generation of a seed. However, if you do not pass a seed to the derivative! (derivative) driver, the partial identity is created as the seed automatically (see here). ","category":"page"},{"location":"lib/derivative_modes/#Derivative-Modes","page":"Derivative Modes","title":"Derivative Modes","text":"","category":"section"},{"location":"lib/derivative_modes/","page":"Derivative Modes","title":"Derivative Modes","text":"The following sections provide some details about the available modes and possibilities for the computation of derivatives with derivative or derivative!.","category":"page"},{"location":"lib/derivative_modes/#First-Order","page":"Derivative Modes","title":"First-Order","text":"","category":"section"},{"location":"lib/derivative_modes/","page":"Derivative Modes","title":"Derivative Modes","text":"A list of the available modes for first-order derivative computation using derivative or derivative! is presented below.  ","category":"page"},{"location":"lib/derivative_modes/","page":"Derivative Modes","title":"Derivative Modes","text":"Mode Formula Output Space\n:jac Df(x) mathbbR^m times n\n:jac_vec Df(x)dotv mathbbR^m\n:jac_mat Df(x)dotV mathbbR^m times p\n:vec_jac barz^T Df(x) mathbbR^n\n:mat_jac barZ^T Df(x) mathbbR^q times n","category":"page"},{"location":"lib/derivative_modes/","page":"Derivative Modes","title":"Derivative Modes","text":"Each mode's formula symbolizes the underlying computation. A user can read-off the dimension of the result from the last column, where it is assumed that fmathbbR^n to mathbbR^m, dotv in mathbbR^n, dotV in mathbbR^n times p, barz  in mathbbR^m and barZ in mathbbR^m times q.","category":"page"},{"location":"lib/derivative_modes/#Second-Order","page":"Derivative Modes","title":"Second-Order","text":"","category":"section"},{"location":"lib/derivative_modes/","page":"Derivative Modes","title":"Derivative Modes","text":"A list of the available modes for second-order derivative computation using derivative or derivative! is presented below.  ","category":"page"},{"location":"lib/derivative_modes/","page":"Derivative Modes","title":"Derivative Modes","text":"Mode Formula Output Space\n:hess D^2f(x) mathbbR^m times n times n\n:hess_vec D^2f(x) dotv mathbbR^m times n\n:hess_mat D^2f(x)  dotV mathbbR^m times n times p\n:vec_hess barz^T D^2f(x) mathbbR^n times n\n:mat_hess barZ^T D^2f(x) mathbbR^q times n times n\n:vec_hess_vec barz^T D^2f(x)  dotv mathbbR^n\n:vec_hess_mat barz^T D^2f(x)  dotV mathbbR^n times p\n:mat_hess_mat barZ^T D^2f(x)  dotV mathbbR^q times n times p\n:mat_hess_vec barZ^T D^2f(x)  dotv mathbbR^q times n","category":"page"},{"location":"lib/derivative_modes/","page":"Derivative Modes","title":"Derivative Modes","text":"Each mode's formula symbolizes the underlying computation. A user can read-off the dimension of the result from the last column, where it is assumed that fmathbbR^n to mathbbR^m, dotv in mathbbR^n, dotV in mathbbR^n times p, barz  in mathbbR^m and barZ in mathbbR^m times q.","category":"page"},{"location":"lib/derivative_modes/#Abs-Normal-Form","page":"Derivative Modes","title":"Abs-Normal-Form","text":"","category":"section"},{"location":"lib/derivative_modes/","page":"Derivative Modes","title":"Derivative Modes","text":"There is one mode to compute the abs-normal-form of a function using derivative or derivative!:","category":"page"},{"location":"lib/derivative_modes/","page":"Derivative Modes","title":"Derivative Modes","text":"Mode Formula\n:abs_norm Delta f(x)","category":"page"},{"location":"lib/derivative_modes/","page":"Derivative Modes","title":"Derivative Modes","text":"The theory behind this method can be found in [2].","category":"page"},{"location":"lib/derivative_modes/#Higher-Order","page":"Derivative Modes","title":"Higher-Order","text":"","category":"section"},{"location":"lib/derivative_modes/","page":"Derivative Modes","title":"Derivative Modes","text":"The goal of the following explanations is to familiarize the reader with  the possibilities for computing higher-order derivatives that are included in ADOLC.jl. In the context of ADOLC.jl, higher-order derivatives are given as a Vector of  arbitrary-order mixed-partials. For example, let fmathbbR^n to mathbbR^m and we want to compute the mixed-partials","category":"page"},{"location":"lib/derivative_modes/","page":"Derivative Modes","title":"Derivative Modes","text":"leftfracpartial^3partial^2 f(x)partial^3 x_2 partial^2 x_1 fracpartial^4 f(x)partial^4 x_3 fracpartial^2 f(x)partial^2 x_1right","category":"page"},{"location":"lib/derivative_modes/","page":"Derivative Modes","title":"Derivative Modes","text":"leveraging the derivative (derivative!) driver. After defining the function f and the point for the derivative evaluation x, we have to select the format of the partials. There exist two options explained below that use Vector{Int64} to define a partial derivative.","category":"page"},{"location":"lib/derivative_modes/#ADOLC-Format","page":"Derivative Modes","title":"ADOLC-Format","text":"","category":"section"},{"location":"lib/derivative_modes/","page":"Derivative Modes","title":"Derivative Modes","text":"The ADOLC-Format repeats the index i of a derivative direction x_i up to the derivative order of this index: fracpartial^4 f(x)partial^4 x_3 to 3 3 3 3. Additionally, the resulting vector is sorted descendent; if the vector's length is less than the total derivative degree, it is filled with zeros. The requested mixed-partials results in:","category":"page"},{"location":"lib/derivative_modes/","page":"Derivative Modes","title":"Derivative Modes","text":"\n 2 2 2 1 1\n 3 3 3 3 0\n 1 1 0 0 0\n","category":"page"},{"location":"lib/derivative_modes/#Partial-Format","page":"Derivative Modes","title":"Partial-Format","text":"","category":"section"},{"location":"lib/derivative_modes/","page":"Derivative Modes","title":"Derivative Modes","text":"The Partial-Format mimics the notation of the mixed-partial, as used above. The entry of the vector at index i is the derivative degree corresponding to the derivative direction x_i. Therefore, partials is given as","category":"page"},{"location":"lib/derivative_modes/","page":"Derivative Modes","title":"Derivative Modes","text":"\n 2 3 0 0\n 0 0 4 0\n 2 0 0 0\n","category":"page"},{"location":"lib/derivative_modes/","page":"Derivative Modes","title":"Derivative Modes","text":"note: Note\nInternally, there is, at some point, a conversion from Partial-Format to ADOLC-Format since the access to the higher-order tensor computed with ADOL-C is based on the ADOLC-Format. However, only one entry is converted at a time, meaning that the benefits of both modes, as explained below, are still valid.","category":"page"},{"location":"lib/derivative_modes/","page":"Derivative Modes","title":"Derivative Modes","text":"note: Note\nBoth formats have their benefits. The ADOLC-Format should be used if the total derivative degree is small compared to the number of independents n. Otherwise, Partial-Format should be used.","category":"page"},{"location":"lib/derivative_modes/","page":"Derivative Modes","title":"Derivative Modes","text":"There are utilities to convert between the formats: partial_to_adolc_format","category":"page"},{"location":"#ADOLC.jl","page":"Home","title":"ADOLC.jl","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"A Julia wrapper of the automatic differentiation package ADOL-C","category":"page"},{"location":"","page":"Home","title":"Home","text":"DocTestSetup = quote\n    using ADOLC\nend","category":"page"},{"location":"","page":"Home","title":"Home","text":"To use this package, type","category":"page"},{"location":"","page":"Home","title":"Home","text":"using Pkg; Pkg.add(url=\"https://github.com/TimSiebert1/ADOLC.jl\")\nusing ADOLC","category":"page"},{"location":"","page":"Home","title":"Home","text":"First- and second-order derivatives can be calculated as follows","category":"page"},{"location":"","page":"Home","title":"Home","text":"f(x) = [x[1]*x[2]^2, x[1]^2*x[3]^3]\nx = [1.0, 2.0, -1.0]\ndir = [1.0, 0.0, 0.0]\nweights = [1.0, 1.0]\nres = derivative(f, x, :vec_hess_vec, dir=dir, weights=weights)\n\n# output\n\n3-element Vector{Float64}:\n -2.0\n  4.0\n  6.0","category":"page"},{"location":"","page":"Home","title":"Home","text":"There are various available modes for first- and second-order calculations. The computation of higher-order derivatives is explained here and works as sketched below","category":"page"},{"location":"","page":"Home","title":"Home","text":"f(x) = [x[1]^2*x[2]^2, x[3]^2*x[4]^2]\nx = [1.0, 2.0, 3.0, 4.0]\npartials = [[1, 1, 0, 0], [0, 0, 1, 1], [2, 2, 0, 0]]\nres = derivative(f, x, partials)\n\n# output\n\n2×3 Matrix{Float64}:\n 8.0   0.0  4.0\n 0.0  48.0  0.0","category":"page"},{"location":"","page":"Home","title":"Home","text":"For advanced user, there is a list of all functions, wrapped from ADOL-C.","category":"page"},{"location":"#API-Reference","page":"Home","title":"API Reference","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Pages = [\"lib/reference.md\"]","category":"page"},{"location":"","page":"Home","title":"Home","text":"A. Griewank, J. Utke and A. Walther. Evaluating higher derivative tensors by forward propagation of univariate Taylor series. Mathematics of Computation 69 (1999).\n\n\n\nA. Griewank. On stable piecewise linearization and generalized algorithmic differentiation. Optimization Methods and Software 28 (2013).\n\n\n\n","category":"page"}]
}
