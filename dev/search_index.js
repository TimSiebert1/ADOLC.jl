var documenterSearchIndex = {"docs":
[{"location":"lib/reference/#API-reference","page":"Reference","title":"API reference","text":"","category":"section"},{"location":"lib/reference/","page":"Reference","title":"Reference","text":"ADOLC.derivative\nADOLC.tensor_address\nADOLC.partial_to_adolc_format\nADOLC.partial_to_adolc_format!\nADOLC.create_cxx_identity\nADOLC.create_partial_cxx_identity\nADOLC.seed_idxs_partial_format\nADOLC.seed_idxs_adolc_format\nADOLC.partial_format_to_seed_space\nADOLC.adolc_format_to_seed_space","category":"page"},{"location":"lib/reference/#ADOLC.derivative","page":"Reference","title":"ADOLC.derivative","text":"derivative(\n    f::Function,\n    m::Integer,\n    n::Integer,\n    x::Union{Float64,Vector{Float64}},\n    mode::Symbol;\n    dir::Union{Vector{Float64},Matrix{Float64}}=Vector{Float64}(),\n    weights::Union{Vector{Float64},Matrix{Float64}}=Vector{Float64}(),\n    tape_id::Integer=0,\n    reuse_tape::Bool=false,\n)\n\nA variant of the derivative driver, which can be used to compute first-order and second-order  derivatives, as well as the abs-normal-form  of the given function f with output dimension m, input dimension n  at the point x. The mode has to be choosen from derivative modes. The corresponding formulas define weights (left multiplier) and dir (right multiplier). Most modes leverage a tape, which has the identifier tape_id. If there is already a valid  tape for the function f at the selected point x use reuse_tape=true and set the tape_id accordingly to avoid the re-creation of the tape.\n\nExamples:\n\nFirst-Order:\n\nf(x) = sin(x)\nres = derivative(f, 1, 1, float(π), :jac)\n\n# output\n\n1-element Vector{Float64}:\n -1.0\n\nSecond-Order:\n\nf(x) = [x[1]*x[2]^2, x[1]^2*x[3]^3]\nx = [1.0, 2.0, -1.0]\ndir = [1.0, 0.0, 0.0]\nweights = [1.0, 1.0]\nres = derivative(f, 2, 3, x, :vec_hess_vec, dir=dir, weights=weights)\n\n# output\n\n3-element Vector{Float64}:\n -2.0\n  4.0\n  6.0\n\nAbs-Normal-Form:\n\nf(x) = max(x[1]*x[2], x[1]^2)\nx = [1.0, 1.0]\nres = derivative(f, 1, 2, x, :abs_normal)\n\n# output\n\nAbsNormalForm(0, 1, 2, 1, [1.0, 1.0], [1.0], [0.0], [0.0], [1.0], [1.5 0.5], [0.5;;], [1.0 -1.0], [0.0;;])\n\n\n\n\n\nderivative(\n    f::Function,\n    m::Integer,\n    n::Integer,\n    x::Union{Float64,Vector{Float64}},\n    partials::Vector{Vector{Int64}};\n    tape_id::Int64=0,\n    reuse_tape::Bool=false,\n    id_seed::Bool=false,\n    adolc_format::Bool=false,\n)\n\nA variant of the derivative driver, which can be used to compute higher-order derivatives of the function f with output  dimension m, input dimension n at the point x. The derivatives are specified as mixed-partials in the partials vector. To define the partial-derivatives use either the Partial-Format or the ADOLC-Format and set adolc_format accordingly. The flag id_seed is used to specify the method for seed-matrix generation. The underlying method leverages a tape, which has the identifier tape_id. If there is already a valid  tape for the function f at the selected point x use reuse_tape=true and set the tape_id accordingly to avoid the re-creation of the tape.\n\nExamples:\n\nPartial-Format:\n\nf(x) = [x[1]^2*x[2]^2, x[3]^2*x[4]^2]\nx = [1.0, 2.0, 3.0, 4.0]\npartials = [[1, 1, 0, 0], [0, 0, 1, 1], [2, 2, 0, 0]]\nres = derivative(f, 2, 4, x, partials)\n\n# output\n\n2×3 Matrix{Float64}:\n 8.0   0.0  4.0\n 0.0  48.0  0.0\n\nADOLC-Format:\n\nf(x) = [x[1]^2*x[2]^2, x[3]^2*x[4]^2]\nx = [1.0, 2.0, 3.0, 4.0]\npartials = [[2, 1, 0, 0], [4, 3, 0, 0], [2, 2, 1, 1]]\nres = derivative(f, 2, 4, x, partials, adolc_format=true)\n\n# output\n\n2×3 Matrix{Float64}:\n 8.0   0.0  4.0\n 0.0  48.0  0.0\n\n\n\n\n\nderivative(\n    f::Function,\n    m::Int64,\n    n::Int64,\n    x::Union{Float64,Vector{Float64}},\n    partials::Vector{Vector{Int64}},\n    seed::Matrix{Float64};\n    tape_id::Int64=0,\n    reuse_tape::Bool=false,\n    adolc_format::Bool=false,\n)\n\n\n\n\n\n","category":"function"},{"location":"lib/reference/#ADOLC.tensor_address","page":"Reference","title":"ADOLC.tensor_address","text":"tensor_address(degree::I, adolc_partial::Vector{I}) where I <: Integer\ntensor_address(degree::Cint, adolc_partial::Vector{I}) where I <: Integer\ntensor_address(degree::I, adolc_partial::Vector{Cint}) where I <: Integer\ntensor_address(degree::Cint, adolc_partial::Vector{Cint})\n\nGenerates the index (address) of the mixed-partial specified by adolc_partial in an higher-order derivative tensor of derivative order degree.\n\nnote: Note\nThe partial has to be in ADOLC-Format.\n\n\n\n\n\n","category":"function"},{"location":"lib/reference/#ADOLC.partial_to_adolc_format","page":"Reference","title":"ADOLC.partial_to_adolc_format","text":"partial_to_adolc_format(partial::Vector{I_1}, degree::I_2) where {I_1<:Integer, I_2<:Integer}\n\nTransforms a given partial to the ADOLC-Format. \n\nnote: Note\npartial is required to be in the Partial-format\n\nExample:\n\n\npartial = [1, 0, 4]\ndegree = sum(partial)\npartial_to_adolc_format(partial, degree)\n\n# output\n\n5-element Vector{Int32}:\n 3\n 3\n 3\n 3\n 1\n\n\n\n\n\n","category":"function"},{"location":"lib/reference/#ADOLC.partial_to_adolc_format!","page":"Reference","title":"ADOLC.partial_to_adolc_format!","text":"partial_to_adolc_format!(res::Vector{Cint}, partial::Vector{I_1}, degree::I_2) where {I_1<:Integer, I_2<:Integer}\npartial_to_adolc_format!(res::Vector{Cint}, partial::Vector{Cint}, degree::I) where I <: Integer\n\nVariant of partial_to_adolc_format that writes the result to res.\n\nExample:\n\npartial = [1, 3, 2, 0]\ndegree = sum(partial)\nres = zeros(Int32, degree)\npartial_to_adolc_format!(res, partial, degree)\n\n# output\n\n6-element Vector{Int32}:\n 3\n 3\n 2\n 2\n 2\n 1\n\n\n\n\n\n","category":"function"},{"location":"lib/reference/#ADOLC.create_cxx_identity","page":"Reference","title":"ADOLC.create_cxx_identity","text":"create_cxx_identity(n::I_1, m::I_2) where {I_1 <: Integer, I_2 <: Integer}\n\nCreates a identity matrix of shape (n, m) of type CxxPtr{CxxPtr{Float64}} (wrapper of C++'s double**).\n\nExample\n\nid = create_cxx_identity(2, 4)\nfor i in 1:2\n    for j in 1:4\n        print(id[i, j], \" \")\n    end\n    println(\"\")\nend\n\n# output\n\n1.0 0.0 0.0 0.0 \n0.0 1.0 0.0 0.0\n\n\n\n\n\n","category":"function"},{"location":"lib/reference/#ADOLC.create_partial_cxx_identity","page":"Reference","title":"ADOLC.create_partial_cxx_identity","text":"create_partial_cxx_identity(n::I_1, idxs::Vector{I_2}) where {I_1 <: Integer, I_2 <: Integer}\n\nCreates a matrix of shape (n, length(idxs)) of type CxxPtr{CxxPtr{Float64}} (wrapper of C++'s double**). The columns are canonical basis vectors corresponding to the entries of idxs. The order of the basis vectors is defined by the order of the indices in idxs.\n\nwarning: Warning\nThe number of rows n must be smaller than the maximal index of idxs!\n\nwarning: Warning\nThe values of idxs must be non-negative!\n\nExamples\n\nn = 4\nidxs = [1, 3]\nid = create_partial_cxx_identity(n, idxs)\nfor i in 1:4\n    for j in 1:length(idxs)\n        print(id[i, j], \" \")\n    end\n    println(\"\")\nend\n\n# output\n\n1.0 0.0 \n0.0 0.0\n0.0 1.0\n0.0 0.0\n\nThe order in idxs defines the order of the basis vectors.\n\nn = 3\nidxs = [3, 0, 1]\nid = create_partial_cxx_identity(n, idxs)\nfor i in 1:3\n    for j in 1:length(idxs)\n        print(id[i, j], \" \")\n    end\n    println(\"\")\nend\n\n# output\n\n0.0 0.0 1.0\n0.0 0.0 0.0\n1.0 0.0 0.0\n\n\n\n\n\n","category":"function"},{"location":"lib/reference/#ADOLC.seed_idxs_partial_format","page":"Reference","title":"ADOLC.seed_idxs_partial_format","text":"seed_idxs_partial_format(partials::Vector{Vector{I}}) where I <: Integer\n\nExtracts the actually required derivative directions of partials and returns them  ascendet sorted. \n\nnote: Note\npartials has to be in Partial-Format.\n\nExample\n\n\npartials = [[1, 0, 0, 0, 3], [1, 0, 1, 0, 0], [0, 0, 3, 0, 0]]\nseed_idxs_partial_format(partials)\n\n# output\n\n3-element Vector{Int64}:\n 1\n 3\n 5\n\n\n\n\n\n","category":"function"},{"location":"lib/reference/#ADOLC.seed_idxs_adolc_format","page":"Reference","title":"ADOLC.seed_idxs_adolc_format","text":"seed_idxs_adolc_format(partials::Vector{Vector{I}}) where I <: Integer\n\nExtracts the actually required derivative directions of partials and returns them  ascendet sorted. \n\nnote: Note\npartials has to be in ADOLC-Format.\n\nExample\n\n\npartials = [[5, 5, 5, 1], [3, 1, 0, 0], [3, 3, 3, 0]]\nseed_idxs_adolc_format(partials)\n\n# output\n\n3-element Vector{Int64}:\n 1\n 3\n 5\n\n\n\n\n\n","category":"function"},{"location":"lib/reference/#ADOLC.partial_format_to_seed_space","page":"Reference","title":"ADOLC.partial_format_to_seed_space","text":"partial_format_to_seed_space(partials::Vector{Vector{I_1}}, seed_idxs::Vector{I_2}) where {I_1 <: Integer, I_2 <: Integer}\npartial_format_to_seed_space(partials::Vector{Vector{I}}) where I <: Integer\n\nConverts partials in Partial-Format to partials of the same format but with (possible) reduced number  of derivatives directions. The seed_idxs is expected to store the result of seed_idxs_partial_format(seed_idxs). Without seed_idxs the function first calls seed_idxs_partial_format(seed_idxs) to get the indices.\n\nExamples\n\n\npartials = [[0, 1, 1], [0, 2, 0]]\nseed_idxs = seed_idxs_partial_format(partials)\npartial_format_to_seed_space(partials, seed_idxs)\n\n# output\n\n2-element Vector{Vector{Int64}}:\n [1, 1]\n [2, 0]\n\nWithout seed_idxs\n\n\npartials = [[0, 1, 1], [0, 2, 0]]\npartial_format_to_seed_space(partials)\n\n# output\n\n2-element Vector{Vector{Int64}}:\n [1, 1]\n [2, 0]\n\n\n\n\n\n","category":"function"},{"location":"lib/reference/#ADOLC.adolc_format_to_seed_space","page":"Reference","title":"ADOLC.adolc_format_to_seed_space","text":"adolc_format_to_seed_space(partials::Vector{Vector{I_1}}, seed_idxs::Vector{I_2}) where {I_1 <: Integer, I_2 <: Integer}\nadolc_format_to_seed_space(partials::Vector{Vector{I}}) where I <: Integer\n\nSame as partial_format_to_seed_space but with ADOLC-Format.\n\nExamples\n\n\npartials = [[3, 2], [2, 2]]\nseed_idxs = seed_idxs_adolc_format(partials)\nadolc_format_to_seed_space(partials, seed_idxs)\n\n# output\n\n2-element Vector{Vector{Int64}}:\n [2, 1]\n [1, 1]\n\nWithout seed_idxs\n\n\npartials = [[3, 2], [2, 2]]\nseed_idxs = seed_idxs_adolc_format(partials)\nadolc_format_to_seed_space(partials, seed_idxs)\n\n# output\n\n2-element Vector{Vector{Int64}}:\n [2, 1]\n [1, 1]\n\n\n\n\n\n","category":"function"},{"location":"lib/wrapped_fcts/#List-of-wrapped-ADOL-C-drivers","page":"Wrapped Functions","title":"List of wrapped ADOL-C drivers","text":"","category":"section"},{"location":"lib/wrapped_fcts/#TbadoubleModule","page":"Wrapped Functions","title":"TbadoubleModule","text":"","category":"section"},{"location":"lib/wrapped_fcts/","page":"Wrapped Functions","title":"Wrapped Functions","text":"getValue\ngradient\njacobian\nhessian\nvec_jac\njac_vec\nhess_vec\nhess_mat\nlagra_hess_vec\njac_solv\n\nad_forward(short tag, int m, int n, int d, int keep, double **X, double **Y) (in ADOL-C: forward)\nad_reverse(short tag, int m, int n, int d, double *u, double **Z) (in ADOL-C: reverse)\n\nzos_forward\nfos_forward\nhos_forward\nhov_wk_forward\n\nfov_forward\nhov_forward\n\nfos_reverse\nhos_reverse\n\nfov_reverse\nhov_reverse\ntensor_address\ntensor_eval","category":"page"},{"location":"lib/wrapped_fcts/#Abs-Smooth-Utilities","page":"Wrapped Functions","title":"Abs-Smooth Utilities","text":"","category":"section"},{"location":"lib/wrapped_fcts/","page":"Wrapped Functions","title":"Wrapped Functions","text":"enableMinMaxUsingAbs\nget_num_switches\nzos_pl_forward\nfos_pl_forward\nfov_pl_forward\nabs_normal","category":"page"},{"location":"lib/wrapped_fcts/#Tape-Utilities","page":"Wrapped Functions","title":"Tape Utilities","text":"","category":"section"},{"location":"lib/wrapped_fcts/","page":"Wrapped Functions","title":"Wrapped Functions","text":"<< (in ADOL-C: <<=)\n>> (in ADOL-C: =>>)\ntrace_on(int tag)\ntrace_on(int tag, int keep)\ntrace_off(int file)\ntrace_off()","category":"page"},{"location":"lib/wrapped_fcts/#TladoubleModule","page":"Wrapped Functions","title":"TladoubleModule","text":"","category":"section"},{"location":"lib/wrapped_fcts/","page":"Wrapped Functions","title":"Wrapped Functions","text":"setNumDir(int const &n) \ngetValue()                      \ngetADValue(int const &i)\nsetADValue(double const &val)\nsetADValue(double const val, int const &i)","category":"page"},{"location":"lib/wrapped_fcts/#Arithmethics","page":"Wrapped Functions","title":"Arithmethics","text":"","category":"section"},{"location":"lib/wrapped_fcts/","page":"Wrapped Functions","title":"Wrapped Functions","text":"+ \n- \n* \n/ \n^","category":"page"},{"location":"lib/wrapped_fcts/#Comparison","page":"Wrapped Functions","title":"Comparison","text":"","category":"section"},{"location":"lib/wrapped_fcts/","page":"Wrapped Functions","title":"Wrapped Functions","text":"<\n>\n>=\n<=\n==","category":"page"},{"location":"lib/wrapped_fcts/#Unary-Functions","page":"Wrapped Functions","title":"Unary Functions","text":"","category":"section"},{"location":"lib/wrapped_fcts/","page":"Wrapped Functions","title":"Wrapped Functions","text":"abs\nsqrt\nsin\ncos\ntan\nasin\nacos\natan\nexp\nlog\nlog10\nsinh\ncosh\ntanh\nasinh\nacosh\natanh\nceil\nfloor\nmax\nmin\nldexp\nfrexp\nerf\ncbrt","category":"page"},{"location":"lib/derivative_modes/#Derivative-Modes","page":"Derivative Modes","title":"Derivative Modes","text":"","category":"section"},{"location":"lib/derivative_modes/#First-Order","page":"Derivative Modes","title":"First-Order","text":"","category":"section"},{"location":"lib/derivative_modes/","page":"Derivative Modes","title":"Derivative Modes","text":"Mode Formula\njac Df(x)\njac_vec Df(x)dotv\njac_mat Df(x)dotV\nvec_jac barz^T Df(x)\nmat_jac barZ^T Df(x)","category":"page"},{"location":"lib/derivative_modes/#Second-Order","page":"Derivative Modes","title":"Second-Order","text":"","category":"section"},{"location":"lib/derivative_modes/","page":"Derivative Modes","title":"Derivative Modes","text":"Mode Formula\nhess D^2f(x)\nhess_vec D^2f(x) dotv\nhess_mat D^2f(x)  dotV\nvec_hess barz^T D^2f(x)\nmat_hess barZ^T D^2f(x)\nvechessvec barz^T D^2f(x)  dotv\nvechessmat barz^T D^2f(x)  dotV\nmathessmat barZ^T D^2f(x)  dotV\nmathessvec barZ^T D^2f(x)  dotv","category":"page"},{"location":"lib/derivative_modes/#Abs-Normal-Form","page":"Derivative Modes","title":"Abs-Normal-Form","text":"","category":"section"},{"location":"lib/derivative_modes/","page":"Derivative Modes","title":"Derivative Modes","text":"Mode Formula\nabs_norm Delta f(x)","category":"page"},{"location":"lib/derivative_modes/#Higher-Order","page":"Derivative Modes","title":"Higher-Order","text":"","category":"section"},{"location":"lib/derivative_modes/","page":"Derivative Modes","title":"Derivative Modes","text":"The goal of the following explanations is to familiarize the reader with  the possibilities for computing higher-order derivatives that are included in ADOLC.jl. In the context of ADOLC.jl, higher-order derivatives are given as a Vector of  arbitrary-order mixed-partials. For example, let fmathbbR^n to mathbbR^m and we want to compute the mixed-partials","category":"page"},{"location":"lib/derivative_modes/","page":"Derivative Modes","title":"Derivative Modes","text":"leftfracpartial^3partial^2 f(x)partial^3 x_2 partial^2 x_1 fracpartial^4 f(x)partial^4 x_3 fracpartial^2 f(x)partial^2 x_1right","category":"page"},{"location":"lib/derivative_modes/","page":"Derivative Modes","title":"Derivative Modes","text":"leveraging the derivative driver. After defining the function f and the point for the derivative evaluation x, we have to select the format of the partials. There exist two options explained below that use Vector{Int64} to define a partial derivative.","category":"page"},{"location":"lib/derivative_modes/#ADOLC-Format","page":"Derivative Modes","title":"ADOLC-Format","text":"","category":"section"},{"location":"lib/derivative_modes/","page":"Derivative Modes","title":"Derivative Modes","text":"The ADOLC-Format repeats the index i of a derivative direction x_i up to the derivative order of this index: fracpartial^4 f(x)partial^4 x_3 to 3 3 3 3. Additionally, the resulting vector is sorted descendent; if the vector's length is less than the total derivative degree, it is filled with zeros. The requested mixed-partials results in:","category":"page"},{"location":"lib/derivative_modes/","page":"Derivative Modes","title":"Derivative Modes","text":"\n 2 2 2 1 1\n 3 3 3 3 0\n 1 1 0 0 0\n","category":"page"},{"location":"lib/derivative_modes/#Partial-Format","page":"Derivative Modes","title":"Partial-Format","text":"","category":"section"},{"location":"lib/derivative_modes/","page":"Derivative Modes","title":"Derivative Modes","text":"The Partial-Format mimics the notation of the mixed-partial, as used above. The entry of the vector at index i is the derivative degree corresponding to the derivative direction x_i. Therefore, partials is given as","category":"page"},{"location":"lib/derivative_modes/","page":"Derivative Modes","title":"Derivative Modes","text":"\n 2 3 0 0\n 0 0 4 0\n 2 0 0 0\n","category":"page"},{"location":"lib/derivative_modes/","page":"Derivative Modes","title":"Derivative Modes","text":"note: Note\nInternally, there is, at some point, a conversion from Partial-Format to ADOLC-Format since the access to the higher-order tensor computed with ADOL-C is based on the ADOLC-Format. However, only one entry is converted at a time, meaning that the benefits of both modes, as explained below, are still valid.","category":"page"},{"location":"lib/derivative_modes/","page":"Derivative Modes","title":"Derivative Modes","text":"note: Note\nBoth formats have their benefits. The ADOLC-Format should be used if the total derivative degree is small compared to the number of independents n. Otherwise, Partial-Format should be used.","category":"page"},{"location":"lib/derivative_modes/","page":"Derivative Modes","title":"Derivative Modes","text":"There are utilities to convert between the formats: partial_to_adolc_format","category":"page"},{"location":"lib/derivative_modes/#Seed-Space","page":"Derivative Modes","title":"Seed-Space","text":"","category":"section"},{"location":"lib/derivative_modes/#Memory-handling","page":"Derivative Modes","title":"Memory handling","text":"","category":"section"},{"location":"#ADOLC.jl","page":"Home","title":"ADOLC.jl","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"A Julia wrapper of the algorithmic differentiation package ADOL-C","category":"page"},{"location":"","page":"Home","title":"Home","text":"To use this package, use ","category":"page"},{"location":"","page":"Home","title":"Home","text":"using Pkg; Pkg.add(url=\"https://github.com/TimSiebert1/ADOLC.jl\")\nusing ADOLC","category":"page"},{"location":"","page":"Home","title":"Home","text":"First- and second-order derivatives can be calculated as follows","category":"page"},{"location":"","page":"Home","title":"Home","text":"f(x) = [x[1]*x[2]^2, x[1]^2*x[3]^3]\nx = [1.0, 2.0, -1.0]\ndir = [1.0, 0.0, 0.0]\nweights = [1.0, 1.0]\nres = derivative(f, 2, 3, x, :vec_hess_vec, dir=dir, weights=weights)","category":"page"},{"location":"","page":"Home","title":"Home","text":"There are various available modes for first- and second-order calculations. The computation of higher-order derivatives is explained here and works as sketched below","category":"page"},{"location":"","page":"Home","title":"Home","text":"f(x) = [x[1]^2*x[2]^2, x[3]^2*x[4]^2]\nx = [1.0, 2.0, 3.0, 4.0]\npartials = [[1, 1, 0, 0], [0, 0, 1, 1], [2, 2, 0, 0]]\nres = derivative(f, 2, 4, x, partials)","category":"page"},{"location":"","page":"Home","title":"Home","text":"For advanced user, there is a list of all functions, wrapped from ADOL-C.","category":"page"},{"location":"#API-Reference","page":"Home","title":"API Reference","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Pages = [\"lib/reference.md\"]","category":"page"}]
}
