"""

    derivative(
        f::Function,
        x::Union{Float64,Vector{Float64}},
        mode::Symbol;
        dir::Union{Vector{Float64},Matrix{Float64}}=Vector{Float64}(),
        weights::Union{Vector{Float64},Matrix{Float64}}=Vector{Float64}(),
        tape_id::Integer=0,
        reuse_tape::Bool=false,
    )

A variant of the [`derivative`](@ref) driver, which can be used to compute
[first-order](@ref "First-Order") and [second-order](@ref "Second-Order") 
derivatives, as well as the [abs-normal-form](@ref "Abs-Normal-Form") 
of the given function `f` at the point `x`. The mode has to be choosen from [`derivative modes`](@ref "Derivative Modes").
The corresponding formulas define `weights` (left multiplier) and `dir` (right multiplier).
Most modes leverage a tape, which has the identifier `tape_id`. If there is already a valid 
tape for the function `f` at the selected point `x` use `reuse_tape=true` and set the `tape_id`
accordingly to avoid the re-creation of the tape.

# Examples:

First-Order:
```jldoctest
f(x) = sin(x)
res = derivative(f, float(π), :jac)

# output

1-element Vector{Float64}:
 -1.0
```

Second-Order:
```jldoctest
f(x) = [x[1]*x[2]^2, x[1]^2*x[3]^3]
x = [1.0, 2.0, -1.0]
dir = [1.0, 0.0, 0.0]
weights = [1.0, 1.0]
res = derivative(f, x, :vec_hess_vec, dir=dir, weights=weights)

# output

3-element Vector{Float64}:
 -2.0
  4.0
  6.0
```

Abs-Normal-Form:
```jldoctest
f(x) = max(x[1]*x[2], x[1]^2)
x = [1.0, 1.0]
res = derivative(f, x, :abs_normal)

# output

AbsNormalForm(0, 1, 2, 1, [1.0, 1.0], [1.0], [0.0], [0.0], [1.0], [1.5 0.5], [0.5;;], [1.0 -1.0], [0.0;;])
```
"""
function derivative(
    f::Function,
    x::Union{Float64,Vector{Float64}},
    mode::Symbol;
    dir::Union{Vector{Float64},Matrix{Float64}}=Vector{Float64}(),
    weights::Union{Vector{Float64},Matrix{Float64}}=Vector{Float64}(),
    tape_id::Integer=0,
    reuse_tape::Bool=false,
)
    if !reuse_tape
        if mode == :abs_normal
            m, n = create_tape(f, x, tape_id, enableMinMaxUsingAbs=true)
        else
            m, n = create_tape(f, x, tape_id)
        end
        reuse_tape = true
    else
        m = TbadoubleModule.num_dependents(tape_id)
        n = TbadoubleModule.num_independents(tape_id)
    end

    cxx_res = if mode === :abs_normal
        init_abs_normal_form(f, x; tape_id=tape_id, reuse_tape=reuse_tape)
    else
        allocator(m, n, mode, size(dir, 2)[1], size(weights, 1)[1])
    end
    # allocator creates tape in case of :abs_normal
    derivative!(
        cxx_res,
        f,
        m,
        n,
        x,
        mode;
        dir=dir,
        weights=weights,
        tape_id=mode === :abs_normal ? cxx_res.tape_id : tape_id,
        reuse_tape=mode === :abs_normal ? true : reuse_tape,
    )
    jl_res = if mode === :abs_normal
        cxx_res
    else
        cxx_res_to_julia_res(cxx_res, m, n, mode, size(dir, 2)[1], size(weights, 1)[1])
    end
    deallocator(cxx_res, m, mode)
    return jl_res
end


"""
    derivative(
        f::Function,
        x::Union{Float64,Vector{Float64}},
        partials::Vector{Vector{Int64}};
        tape_id::Int64=0,
        reuse_tape::Bool=false,
        id_seed::Bool=false,
        adolc_format::Bool=false,
    )
A variant of the [`derivative`](@ref) driver, which can be used to compute
[higher-order](@ref "Higher-Order") derivatives of the function `f` 
at the point `x`. The derivatives are
specified as mixed-partials in the `partials` vector. To define the partial-derivatives use either
the [Partial-Format](@ref) or the [ADOLC-Format](@ref) and set `adolc_format` accordingly.
The flag `id_seed` is used to specify the method for [seed-matrix generation](@ref "Seed-Space").
The underlying method leverages a tape, which has the identifier `tape_id`. If there is already a valid 
tape for the function `f` at the selected point `x` use `reuse_tape=true` and set the `tape_id`
accordingly to avoid the re-creation of the tape.


# Examples:


Partial-Format:
```jldoctest
f(x) = [x[1]^2*x[2]^2, x[3]^2*x[4]^2]
x = [1.0, 2.0, 3.0, 4.0]
partials = [[1, 1, 0, 0], [0, 0, 1, 1], [2, 2, 0, 0]]
res = derivative(f, x, partials)

# output

2×3 Matrix{Float64}:
 8.0   0.0  4.0
 0.0  48.0  0.0
```

ADOLC-Format:
```jldoctest
f(x) = [x[1]^2*x[2]^2, x[3]^2*x[4]^2]
x = [1.0, 2.0, 3.0, 4.0]
partials = [[2, 1, 0, 0], [4, 3, 0, 0], [2, 2, 1, 1]]
res = derivative(f, x, partials, adolc_format=true)

# output

2×3 Matrix{Float64}:
 8.0   0.0  4.0
 0.0  48.0  0.0
```
"""
function derivative(
    f,
    x::Union{Float64,Vector{Float64}},
    partials::Vector{Vector{Int64}};
    tape_id::Int64=0,
    reuse_tape::Bool=false,
    id_seed::Bool=false,
    adolc_format::Bool=false,
)
    if !reuse_tape 
        m, n = create_tape(f, x, tape_id)
        reuse_tape = true
    else
        m = TbadoubleModule.num_dependents(tape_id)
        n = TbadoubleModule.num_independents(tape_id)
    end
    res = Matrix{Float64}(undef, m, length(partials))
    derivative!(
        res,
        f,
        m,
        n,
        x,
        partials;
        tape_id=tape_id,
        reuse_tape=reuse_tape,
        id_seed=id_seed,
        adolc_format=adolc_format,
    )
    return res
end


"""
    derivative(
        f::Function,
        x::Union{Float64,Vector{Float64}},
        partials::Vector{Vector{Int64}},
        seed::Matrix{Float64};
        tape_id::Int64=0,
        reuse_tape::Bool=false,
        adolc_format::Bool=false,
    )

"""
function derivative(
    f,
    x::Union{Float64,Vector{Float64}},
    partials::Vector{Vector{Int64}},
    seed::Matrix{Float64};
    tape_id::Int64=0,
    reuse_tape::Bool=false,
    adolc_format::Bool=false,
)
    if !reuse_tape
        m, n = create_tape(f, x, tape_id)
        reuse_tape = true
    else
        m = TbadoubleModule.num_dependents(tape_id)
        n = TbadoubleModule.num_independents(tape_id)
    end

    res = Matrix{Float64}(undef, m, length(partials))
    derivative!(
        res,
        f,
        m,
        n,
        x,
        partials,
        seed;
        tape_id=tape_id,
        reuse_tape=reuse_tape,
        adolc_format=adolc_format,
    )
    return res
end


"""
    derivative!(
        res,
        f::Function,
        m,
        n,
        x::Union{Float64,Vector{Float64}},
        mode::Symbol;
        dir::Union{Vector{Float64},Matrix{Float64}}=Vector{Float64}(),
        weights::Union{Vector{Float64},Matrix{Float64}}=Vector{Float64}(),
        tape_id::Int64=0,
        reuse_tape::Bool=false,
    )
"""
function derivative!(
    res,
    f::Function,
    m,
    n,
    x::Union{Float64,Vector{Float64}},
    mode::Symbol;
    dir::Union{Vector{Float64},Matrix{Float64}}=Vector{Float64}(),
    weights::Union{Vector{Float64},Matrix{Float64}}=Vector{Float64}(),
    tape_id::Int64=0,
    reuse_tape::Bool=false,
)
    if mode === :jac
        jac!(res, f, m, n, x, tape_id, reuse_tape)
    elseif mode === :hess
        hessian!(res, f, m, n, x, tape_id, reuse_tape)

    elseif mode === :jac_vec
        fos_forward!(res, f, m, n, x, dir, tape_id, reuse_tape)
    elseif mode === :jac_mat
        fov_forward!(res, f, m, n, x, dir, tape_id, reuse_tape)
    elseif mode === :hess_vec
        hess_vec!(res, f, m, n, x, dir, tape_id, reuse_tape)
    elseif mode === :hess_mat
        hess_mat!(res, f, m, n, x, dir, tape_id, reuse_tape)

    elseif mode === :vec_jac
        fos_reverse!(res, f, m, n, x, weights, tape_id, reuse_tape)
    elseif mode === :mat_jac
        fov_reverse!(res, f, m, n, x, weights, tape_id, reuse_tape)
    elseif mode === :vec_hess
        vec_hess!(res, f, m, n, x, weights, tape_id, reuse_tape)
    elseif mode === :mat_hess
        mat_hess!(res, f, m, n, x, weights, tape_id, reuse_tape)

    elseif mode === :vec_hess_vec
        vec_hess_vec!(res, f, m, n, x, dir, weights, tape_id, reuse_tape)
    elseif mode === :mat_hess_vec
        mat_hess_vec!(res, f, m, n, x, dir, weights, tape_id, reuse_tape)
    elseif mode === :vec_hess_mat
        vec_hess_mat!(res, f, m, n, x, dir, weights, tape_id, reuse_tape)
    elseif mode === :mat_hess_mat
        mat_hess_mat!(res, f, m, n, x, dir, weights, tape_id, reuse_tape)

    elseif mode === :abs_normal
        abs_normal!(res, f, m, n, x, tape_id, reuse_tape)

    else
        throw("mode $mode not implemented!")
    end
end

function derivative!(
    res,
    f,
    m::Int64,
    n::Int64,
    x::Union{Float64,Vector{Float64}},
    partials::Vector{Vector{Int64}};
    tape_id::Int64=0,
    reuse_tape::Bool=false,
    id_seed::Bool=false,
    adolc_format::Bool=false,
)
    if id_seed
        seed = create_cxx_identity(n, n)
    else
        seed_idxs = if adolc_format
            seed_idxs_adolc_format(partials)
        else
            seed_idxs_partial_format(partials)
        end
        partials = if adolc_format
            adolc_format_to_seed_space(partials, seed_idxs)
        else
            partial_format_to_seed_space(partials, seed_idxs)
        end
        seed = create_partial_cxx_identity(n, seed_idxs)
    end
    higher_order!(res, f, m, n, x, partials, seed, n, tape_id, reuse_tape, adolc_format)
    return myfree2(seed)
end

function derivative!(
    res,
    f,
    m::Int64,
    n::Int64,
    x::Union{Float64,Vector{Float64}},
    partials::Vector{Vector{Int64}},
    seed::Matrix{Float64};
    tape_id::Int64=0,
    reuse_tape::Bool=false,
    adolc_format::Bool=false,
)
    seed_cxx = julia_mat_to_cxx_mat(seed)
    higher_order!(
        res,
        f,
        m,
        n,
        x,
        partials,
        seed_cxx,
        size(seed, 2),
        tape_id,
        reuse_tape,
        adolc_format,
    )
    return myfree2(seed_cxx)
end

function jac!(
    res,
    f,
    m::Int64,
    n::Int64,
    x::Union{Float64,Vector{Float64}},
    tape_id::Int64,
    reuse_tape,
)
    if m == 1
        gradient!(res, f, n, x, tape_id, reuse_tape)
    else
        if n / 2 < m
            tape_less_forward!(res, f, n, x)
        else
            weights = create_cxx_identity(m, m)
            fov_reverse!(res, f, m, n, m, x, weights, tape_id, reuse_tape)
        end
    end
end

function gradient!(
    res, f, n::Int64, x::Union{Float64,Vector{Float64}}, tape_id::Int64, reuse_tape
)
    if !reuse_tape
        create_tape(f, x, tape_id)
    end
    return TbadoubleModule.gradient(tape_id, n, x, res)
end

function tape_less_forward!(res, f, n::Int64, x::Union{Float64,Vector{Float64}})
    TladoubleModule.set_num_dir(n)
    a = Adouble{TlAlloc}(x, true)
    init_gradient(a)
    b = f(a)
    return gradient(n, b, res)
end

function fos_reverse!(
    res,
    f,
    m::Int64,
    n::Int64,
    x::Union{Float64,Vector{Float64}},
    weights::Vector{Float64},
    tape_id::Int64,
    reuse_tape,
)
    if !reuse_tape
        create_tape(f, x, tape_id; keep=1)
    else
        TbadoubleModule.zos_forward(tape_id, m, n, 1, x, [0.0 for _ in 1:m])
    end
    return TbadoubleModule.fos_reverse(tape_id, m, n, weights, res)
end

function fov_reverse!(
    res,
    f,
    m::Int64,
    n::Int64,
    x::Union{Float64,Vector{Float64}},
    weights::Matrix{Float64},
    tape_id::Int64,
    reuse_tape,
)
    num_dir = size(weights, 2)
    weights_cxx = julia_mat_to_cxx_mat(weights)
    fov_reverse!(res, f, m, n, num_dir, x, weights_cxx, tape_id, reuse_tape)
    return myfree2(weights_cxx)
end

function fov_reverse!(
    res,
    f,
    m::Int64,
    n::Int64,
    num_dir::Int64,
    x::Union{Float64,Vector{Float64}},
    weights::CxxPtr{CxxPtr{Float64}},
    tape_id::Int64,
    reuse_tape,
)
    if !reuse_tape
        create_tape(f, x, tape_id; keep=1)
    else
        TbadoubleModule.zos_forward(tape_id, m, n, 1, x, [0.0 for _ in 1:m])
    end

    return TbadoubleModule.fov_reverse(tape_id, m, n, num_dir, weights, res)
end

function fos_forward!(
    res,
    f,
    m::Int64,
    n::Int64,
    x::Union{Float64,Vector{Float64}},
    dir::Vector{Float64},
    tape_id::Int64,
    reuse_tape,
)
    if !reuse_tape
        create_tape(f, x, tape_id)
    end
    return TbadoubleModule.fos_forward(
        tape_id, m, n, 0, x, dir, [0.0 for _ in 1:m], res
    )
end

function fov_forward!(
    res,
    f,
    m::Int64,
    n::Int64,
    x::Union{Float64,Vector{Float64}},
    dir::Matrix{Float64},
    tape_id,
    reuse_tape,
)
    num_dir = size(dir, 2)
    dir_cxx = julia_mat_to_cxx_mat(dir)
    fov_forward!(res, f, m, n, x, dir_cxx, num_dir, tape_id, reuse_tape)
    return myfree2(dir_cxx)
end

function fov_forward!(
    res,
    f,
    m::Int64,
    n::Int64,
    x::Union{Float64,Vector{Float64}},
    dir::CxxPtr{CxxPtr{Float64}},
    num_dir::Int64,
    tape_id,
    reuse_tape,
)
    if !reuse_tape
        create_tape(f, x, tape_id)
    end
    return TbadoubleModule.fov_forward(
        tape_id, m, n, num_dir, x, dir, [0.0 for _ in 1:m], res
    )
end

function check_resue_abs_normal_problem(
    tape_id::Int64, m, n, abs_normal_problem::AbsNormalForm
)
    if abs_normal_problem.tape_id != tape_id
        throw(
            "Tape_id mistmatch ($(abs_normal_problem.tape_id) vs. $tape_id)! The tape id has to be the same when reusing abs_normal_problem!",
        )
    end
    if abs_normal_problem.m != m
        throw(
            "Outputdimension mismatch ($(abs_normal_problem.m) vs. $m)! The dimensions has to remain the same when resuing abs_normal_problem!",
        )
    end
    if abs_normal_problem.n != n
        throw(
            "Inputdimension mismatch ($(abs_normal_problem.n) vs. $n)! The dimensions has to remain the same when resuing abs_normal_problem!",
        )
    end
    if get_num_switches(tape_id) != abs_normal_problem.num_switches
        throw(
            "NumSwitches mistmacht ($(abs_normal_problem.num_switches) vs. $(get_num_switches(tape_id)))! The number of switches has to remain the same when reusing abs_normal_problem!",
        )
    end
end

function abs_normal!(
    abs_normal_problem,
    f,
    m,
    n,
    x::Union{Float64,Vector{Float64}},
    tape_id::Int64,
    reuse_tape::Bool,
)
    if !reuse_tape
        create_tape(f, x, tape_id; enableMinMaxUsingAbs=true)
    else
        check_resue_abs_normal_problem(tape_id, m, n, abs_normal_problem)
        vec_to_cxx(abs_normal_problem.x, x)
    end
    return abs_normal!(abs_normal_problem)
end

function init_abs_normal_form(
    f, x::Union{Float64,Vector{Float64}}; tape_id::Int64=0, reuse_tape=false
)
    if !reuse_tape
        m, n = create_tape(f, x, tape_id; enableMinMaxUsingAbs=true)
        reuse_tape=true
    else
        m = TbadoubleModule.num_dependents(tape_id)
        n = TbadoubleModule.num_independents(tape_id)
    end
    return AbsNormalForm(tape_id, m, n, x, [0.0 for _ in 1:m])
end

function vec_hess_vec!(
    res,
    f,
    m::Int64,
    n::Int64,
    x::Union{Float64,Vector{Float64}},
    dir::Vector{Float64},
    weights::Vector{Float64},
    tape_id::Int64,
    reuse_tape::Bool,
)
    if !(reuse_tape)
        create_tape(f, x, tape_id)
    end
    return TbadoubleModule.lagra_hess_vec(tape_id, m, n, x, dir, weights, res)
end

function vec_hess_mat!(
    res::CxxPtr{CxxPtr{Float64}},
    f,
    m::Int64,
    n::Int64,
    x::Union{Float64,Vector{Float64}},
    dir::Matrix{Float64},
    weights::Vector{Float64},
    tape_id::Int64,
    reuse_tape::Bool,
)
    if !(reuse_tape)
        create_tape(f, x, tape_id)
    end
    res_tmp = alloc_vec_double(n)
    for i in axes(dir, 2)
        vec_hess_vec!(res_tmp, f, m, n, x, dir[:, i], weights, tape_id, true)
        for j in 1:n
            res[j, i] = res_tmp[j]
            res_tmp[j] = 0.0
        end
    end
    return free_vec_double(res_tmp)
end

function vec_hess!(
    res::CxxPtr{CxxPtr{Float64}},
    f,
    m::Int64,
    n::Int64,
    x::Union{Float64,Vector{Float64}},
    weights::Vector{Float64},
    tape_id::Int64,
    reuse_tape::Bool,
)
    dir = Matrix{Float64}(I, n, n)
    return vec_hess_mat!(res, f, m, n, x, dir, weights, tape_id, reuse_tape)
end

function mat_hess_vec!(
    res,
    f,
    m::Int64,
    n::Int64,
    x::Union{Float64,Vector{Float64}},
    dir::Vector{Float64},
    weights::Matrix{Float64},
    tape_id::Int64,
    reuse_tape::Bool,
)
    num_weights = size(weights, 1)
    weights_cxx = julia_mat_to_cxx_mat(weights)
    return mat_hess_vec!(
        res, f, m, n, x, dir, weights_cxx, num_weights, tape_id, reuse_tape
    )
end

function mat_hess_vec!(
    res,
    f,
    m::Int64,
    n::Int64,
    x::Union{Float64,Vector{Float64}},
    dir::Vector{Float64},
    weights::CxxPtr{CxxPtr{Float64}},
    num_weights::Int64,
    tape_id::Int64,
    reuse_tape::Bool;
    res_fos_tmp::Union{CxxPtr{Float64},Nothing}=nothing,
    nz_tmp::Union{CxxPtr{CxxPtr{Int16}},Nothing}=nothing,
    res_hov_tmp::Union{CxxPtr{CxxPtr{CxxPtr{Float64}}},Nothing}=nothing,
)
    if !(reuse_tape)
        create_tape(f, x, tape_id)
    end
    free_res_fos_tmp = false
    free_nz_tmp = false
    free_res_hov_tmp = false

    if res_fos_tmp === nothing
        res_fos_tmp = alloc_vec_double(m)
        free_res_fos_tmp = true
    end
    if nz_tmp === nothing
        nz_tmp = alloc_mat_short(num_weights, n)
        free_nz_tmp = true
    end
    degree = 1
    keep = degree + 1
    if res_hov_tmp === nothing
        res_hov_tmp = myalloc3(num_weights, n, degree + 1)
        free_res_hov_tmp = true
    end
    TbadoubleModule.fos_forward(
        tape_id, m, n, keep, x, dir, [0.0 for _ in 1:m], res_fos_tmp
    )
    TbadoubleModule.hov_reverse(
        tape_id, m, n, degree, num_weights, weights, res_hov_tmp, nz_tmp
    )
    for i in 1:num_weights
        for j in 1:n
            res[i, j] = res_hov_tmp[i, j, degree + 1]
            res_hov_tmp[i, j, degree + 1] = 0.0
        end
    end
    if free_res_fos_tmp
        free_vec_double(res_fos_tmp)
    end
    if free_nz_tmp
        free_mat_short(nz_tmp, num_weights)
    end
    if free_res_hov_tmp
        myfree3(res_hov_tmp)
    end
end

function mat_hess_mat!(
    res::CxxPtr{CxxPtr{CxxPtr{Float64}}},
    f,
    m::Int64,
    n::Int64,
    x::Union{Float64,Vector{Float64}},
    dir::Matrix{Float64},
    weights::Matrix{Float64},
    tape_id::Int64,
    reuse_tape::Bool,
)
    num_weights = size(weights, 1)
    weights_cxx = julia_mat_to_cxx_mat(weights)
    return mat_hess_mat!(
        res, f, m, n, x, dir, weights_cxx, num_weights, tape_id, reuse_tape
    )
end

function mat_hess_mat!(
    res::CxxPtr{CxxPtr{CxxPtr{Float64}}},
    f,
    m::Int64,
    n::Int64,
    x::Union{Float64,Vector{Float64}},
    dir::Matrix{Float64},
    weights::CxxPtr{CxxPtr{Float64}},
    num_weights::Int64,
    tape_id::Int64,
    reuse_tape::Bool;
    lower_triag::Bool=false,
)
    if !(reuse_tape)
        create_tape(f, x, tape_id)
    end
    res_tmp = myalloc2(num_weights, n)
    res_fos_tmp = alloc_vec_double(m)
    nz_tmp = alloc_mat_short(num_weights, n)
    res_hov_tmp = myalloc3(num_weights, n, 2)
    for i in axes(dir, 2)
        mat_hess_vec!(
            res_tmp,
            f,
            m,
            n,
            x,
            dir[:, i],
            weights,
            num_weights,
            tape_id,
            true;
            res_fos_tmp=res_fos_tmp,
            nz_tmp=nz_tmp,
            res_hov_tmp=res_hov_tmp,
        )
        if lower_triag
            for j in 1:m
                for k in 1:n
                    if i <= k
                        res[j, k, i] = res_tmp[j, k]
                        res_tmp[j, k] = 0.0
                    end
                end
            end
        else
            for j in 1:num_weights
                for k in 1:n
                    res[j, k, i] = res_tmp[j, k]
                    res_tmp[j, k] = 0.0
                end
            end
        end
    end
    myfree2(res_tmp)
    free_vec_double(res_fos_tmp)
    free_mat_short(nz_tmp, num_weights)
    return myfree3(res_hov_tmp)
end

function hessian!(
    res::CxxPtr{CxxPtr{CxxPtr{Float64}}},
    f::Function,
    m::Int64,
    n::Int64,
    x::Union{Float64,Vector{Float64}},
    tape_id::Int64,
    reuse_tape::Bool,
)
    dir = Matrix{Float64}(I, n, n)
    weights = create_cxx_identity(m, m)
    return mat_hess_mat!(
        res, f, m, n, x, dir, weights, m, tape_id, reuse_tape; lower_triag=true
    )
end

function hess_vec!(
    res,
    f,
    m::Int64,
    n::Int64,
    x::Union{Float64,Vector{Float64}},
    dir::Vector{Float64},
    tape_id::Int64,
    reuse_tape::Bool,
)
    weights = Matrix{Float64}(I, m, m)
    return mat_hess_vec!(res, f, m, n, x, dir, weights, tape_id, reuse_tape)
end

function mat_hess!(
    res::CxxPtr{CxxPtr{CxxPtr{Float64}}},
    f,
    m::Int64,
    n::Int64,
    x::Union{Float64,Vector{Float64}},
    weights::Matrix{Float64},
    tape_id::Int64,
    reuse_tape::Bool,
)
    dir = Matrix{Float64}(I, n, n)
    return mat_hess_mat!(res, f, m, n, x, dir, weights, tape_id, reuse_tape)
end

function hess_mat!(
    res::CxxPtr{CxxPtr{CxxPtr{Float64}}},
    f,
    m::Int64,
    n::Int64,
    x::Union{Float64,Vector{Float64}},
    dir::Matrix{Float64},
    tape_id::Int64,
    reuse_tape::Bool,
)
    weights = create_cxx_identity(m, m)
    return mat_hess_mat!(res, f, m, n, x, dir, weights, m, tape_id, reuse_tape)
end

function higher_order!(
    res,
    f,
    m::Int64,
    n::Int64,
    x::Vector{Float64},
    partials::Vector{Vector{Int64}},
    seed::CxxPtr{CxxPtr{Float64}},
    num_seeds::Int64,
    tape_id::Int64,
    reuse_tape::Bool,
    adolc_format::Bool,
)
    if !reuse_tape
        create_tape(f, x, tape_id)
    end
    if adolc_format
        degree = length(partials[1])
    else
        degree = maximum(map(sum, partials))
    end
    res_tmp = myalloc2(m, binomial(num_seeds + degree, degree))
    tensor_eval(tape_id, m, n, degree, num_seeds, x, res_tmp, seed)
    if !adolc_format
        adolc_partial = zeros(Cint, degree)
    end
    for (i, partial) in enumerate(partials)
        if !adolc_format
            partial_to_adolc_format!(adolc_partial, partial, degree)
        end
        for j in 1:m
            if !adolc_format
                res[j, i] = res_tmp[j, tensor_address(degree, adolc_partial)]
            else
                res[j, i] = res_tmp[j, tensor_address(degree, partial)]
            end
        end
    end
    return myfree2(res_tmp)
end

function create_tape(
    f,
    x::Union{Float64,Vector{Float64}},
    tape_id::Int64;
    keep::Int64=0,
    enableMinMaxUsingAbs=false,
)
    if enableMinMaxUsingAbs
        TbadoubleModule.enableMinMaxUsingAbs()
    end

    trace_on(tape_id, keep)
    a = create_independent(x)
    b = f(a)
    dependent(b)
    trace_off()

    return length(b), length(x)
end

"""
    create_independent(x::Union{Float64, Vector{Float64}})

"""
function create_independent(x)
    n = length(x)
    a = n == 1 ? Adouble{TbAlloc}() : [Adouble{TbAlloc}() for _ in 1:n]
    a << x
    return a  
end


function dependent(b)
    m = length(b)
    y = m == 1 ? 0.0 : [0.0 for _ in 1:m]
    b >> y 
end