using CxxWrap
function derivative!(
    res,
    f::Function,
    m::Int64,
    n::Int64,
    x::Union{Float64,Vector{Float64}},
    mode::Symbol;
    dir::Union{Vector{Float64},Matrix{Float64}} = Vector{Float64}(),
    weights::Union{Vector{Float64},Matrix{Float64}} = Vector{Float64}(),
    partials::Vector{Int64} = Vector{Int64}(),
    tape_id::Int64 = 0,
    reuse_tape::Bool = false,
)
    if mode === :jac
        jac!(res, f, m, n, x, tape_id, reuse_tape)
    elseif mode === :jac_vec
        fos_forward!(res, f, m, n, x, dir, tape_id, reuse_tape)
    elseif mode === :jac_mat
        fov_forward!(res, f, m, n, x, dir, tape_id, reuse_tape)
    elseif mode === :vec_jac
        fos_reverse!(res, f, m, n, x, weights, tape_id, reuse_tape)
    elseif mode === :mat_jac
        fov_reverse!(res, f, m, n, x, weights, tape_id, reuse_tape)
    elseif mode === :abs_normal
        abs_normal!(res, f, m, n, x, tape_id, reuse_tape)

    elseif mode === :hess
        hessian!(res, f, m, n, x, tape_id, reuse_tape)
    elseif mode === :hess_vec
        hess_vec!(res, f, m, n, x, dir, tape_id, reuse_tape)
    elseif mode === :hess_mat
        hess_mat!(res, f, m, n, x, dir, tape_id, reuse_tape)
    elseif mode === :vec_hess_vec
        vec_hess_vec!(res, f, m, n, x, dir, weights, tape_id, reuse_tape)
    elseif mode === :mat_hess_vec
        mat_hess_vec!(res, f, m, n, x, dir, weights, tape_id, reuse_tape)
    elseif mode === :vec_hess
        vec_hess!(res, f, m, n, x, weights, tape_id, reuse_tape)
    else
        throw("mode $mode not implemented!")
    end
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
    res,
    f,
    n::Int64,
    x::Union{Float64,Vector{Float64}},
    tape_id::Int64,
    reuse_tape,
)
    if !reuse_tape
        create_tape(f, 1, n, x, tape_id)
    end
    ADOLC.TbadoubleModule.gradient(tape_id, n, x, res)
end

function tape_less_forward!(res, f, n::Int64, x::Union{Float64,Vector{Float64}})
    ADOLC.TladoubleModule.set_num_dir(n)
    a = Adouble{TlAlloc}(x, true)
    ADOLC.init_gradient(a)
    b = f(a)
    ADOLC.gradient(n, b, res)
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
        create_tape(f, m, n, x, tape_id, keep = 1)
    else
        ADOLC.TbadoubleModule.zos_forward(tape_id, m, n, 1, x, [0.0 for _ = 1:m])
    end
    ADOLC.TbadoubleModule.fos_reverse(tape_id, m, n, weights, res)
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
    weights_cxx = myalloc2(size(weights)...)
    for i = 1:size(weights, 1)
        for j = 1:size(weights, 2)
            weights_cxx[i, j] = weights[i, j]
        end
    end
    fov_reverse!(res, f, m, n, num_dir, x, weights_cxx, tape_id, reuse_tape)
    myfree2(weights_cxx)
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
        create_tape(f, m, n, x, tape_id, keep = 1)
    else
        ADOLC.TbadoubleModule.zos_forward(tape_id, m, n, 1, x, [0.0 for _ = 1:m])
    end

    ADOLC.TbadoubleModule.fov_reverse(tape_id, m, n, num_dir, weights, res)
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
        create_tape(f, m, n, x, tape_id)
    end
    ADOLC.TbadoubleModule.fos_forward(tape_id, m, n, 0, x, dir, [0.0 for _ = 1:m], res)
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
    dir_cxx = myalloc2(size(dir)...)
    for i = 1:size(dir, 1)
        for j = 1:size(dir, 2)
            dir_cxx[i, j] = dir[i, j]
        end
    end
    fov_forward!(res, f, m, n, x, dir_cxx, num_dir, tape_id, reuse_tape)
    myfree2(dir_cxx)
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
        create_tape(f, m, n, x, tape_id)
    end
    ADOLC.TbadoubleModule.fov_forward(
        tape_id,
        m,
        n,
        num_dir,
        x,
        dir,
        [0.0 for _ = 1:m],
        res,
    )
end


function check_resue_abs_normal_problem(
    tape_id::Int64,
    m::Int64,
    n::Int64,
    abs_normal_problem::AbsNormalForm,
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
    m::Int64,
    n::Int64,
    x::Union{Float64,Vector{Float64}},
    tape_id::Int64,
    reuse_tape::Bool,
)
    if !reuse_tape
        create_tape(f, m, n, x, tape_id, enableMinMaxUsingAbs = true)
    else
        check_resue_abs_normal_problem(tape_id, m, n, abs_normal_problem)
        ADOLC.array_types.vec_to_cxx(abs_normal_problem.x, x)
    end
    ADOLC.abs_normal!(abs_normal_problem)
end


function init_abs_normal_form(
    f,
    m::Int64,
    n::Int64,
    x::Union{Float64,Vector{Float64}};
    tape_id::Int64 = 0,
)
    create_tape(f, m, n, x, tape_id, enableMinMaxUsingAbs = true)
    return ADOLC.AbsNormalForm(tape_id, m, n, x, [0.0 for _ = 1:m])
end


function hessian!(
    res::CxxPtr{CxxPtr{Float64}},
    f::Function,
    m::Int64,
    n::Int64,
    x::Union{Float64,Vector{Float64}},
    tape_id::Int64,
    reuse_tape::Bool,
)
    if !reuse_tape
        create_tape(f, m, n, x, tape_id)
    end
    ADOLC.TbadoubleModule.hessian(tape_id, n, x, res)
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
    if !reuse_tape
        create_tape(f, m, n, x, tape_id)
    end
    weights = create_cxx_identity(m, m)
    res_tmp = myalloc2(m, n)
    for i = 1:n
        dir = zeros(Float64, n)
        dir[i] = 1.0

        mat_hess_vec!(res_tmp, f, 2, 3, x, dir, weights, m, tape_id, true)
        for j = 1:m
            for k = 1:n
                res[j, k, i] = res_tmp[j, k]
                res_tmp[j, k] = 0.0
            end
        end
    end
    myfree2(weights)
    myfree2(res_tmp)
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
    if !(reuse_tape)
        create_tape(f, m, n, x, tape_id)
    end

    degree = 1
    keep = degree + 1

    weights = create_cxx_identity(m, m)
    tmp = alloc_vec_double(m)

    nz = ADOLC.array_types.alloc_mat_short(m, n)
    res_tmp = ADOLC.array_types.myalloc3(m, n, degree + 1)
    ADOLC.TbadoubleModule.fos_forward(tape_id, m, n, keep, x, dir, [0.0 for _ = 1:m], tmp)
    ADOLC.TbadoubleModule.hov_reverse(tape_id, m, n, degree, m, weights, res_tmp, nz)

    for i = 1:m
        for j = 1:n
            res[i, j] = res_tmp[i, j, degree+1]
        end
    end

    free_mat_short(nz, m)
    free_vec_double(tmp)
    myfree2(weights)
    myfree3(res_tmp)

end


function hess_mat!(
    res::CxxPtr{CxxPtr{Float64}},
    f,
    m::Int64,
    n::Int64,
    x::Union{Float64,Vector{Float64}},
    dir::Matrix{Float64},
    tape_id::Int64,
    reuse_tape::Bool,
)

    num_dir = size(dir, 2)
    dir_cxx = myalloc2(size(dir)...)
    for i = 1:size(dir, 1)
        for j = 1:size(dir, 2)
            dir_cxx[i, j] = dir[i, j]
        end
    end
    hess_mat!(res, f, m, n, x, dir_cxx, num_dir, tape_id, reuse_tape)
    myfree2(dir_cxx)
end

function hess_mat!(
    res::CxxPtr{CxxPtr{Float64}},
    f,
    m::Int64,
    n::Int64,
    x::Union{Float64,Vector{Float64}},
    dir::CxxPtr{CxxPtr{Float64}},
    num_dir::Int64,
    tape_id::Int64,
    reuse_tape::Bool,
)
    if !(reuse_tape)
        create_tape(f, m, n, x, tape_id)
    end
    ADOLC.TbadoubleModule.hess_mat(tape_id, n, num_dir, x, dir, res)
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
    if !(reuse_tape)
        create_tape(f, m, n, x, tape_id)
    end
    num_dir = size(dir, 2)
    weights = create_cxx_identity(m, m)
    res_tmp = myalloc2(m, num_dir)
    for i = 1:num_dir
        mat_hess_vec!(res_tmp, f, 2, 3, x, dir[:, i], weights, m, tape_id, reuse_tape)
        for j = 1:m
            for k = 1:n
                res[j, k, i] = res_tmp[j, k]
                res_tmp[j, k] = 0.0
            end
        end
    end
    myfree2(weights)
    myfree2(res_tmp)
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
        create_tape(f, m, n, x, tape_id)
    end
    ADOLC.TbadoubleModule.lagra_hess_vec(tape_id, m, n, x, dir, weights, res)
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
    weights_cxx = myalloc2(size(weights)...)
    for i = 1:size(weights, 1)
        for j = 1:size(weights, 2)
            weights_cxx[i, j] = weights[i, j]
        end
    end
    mat_hess_vec!(res, f, m, n, x, dir, weights_cxx, num_weights, tape_id, reuse_tape)
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
    reuse_tape::Bool,
)
    if !(reuse_tape)
        create_tape(f, m, n, x, tape_id)
    end
    tmp = alloc_vec_double(m)
    degree = 1
    keep = degree + 1
    nz = ADOLC.array_types.alloc_mat_short(num_weights, n)
    res_tmp = ADOLC.array_types.myalloc3(num_weights, n, degree + 1)
    ADOLC.TbadoubleModule.fos_forward(tape_id, m, n, keep, x, dir, [0.0 for _ = 1:m], tmp)
    ADOLC.TbadoubleModule.hov_reverse(
        tape_id,
        m,
        n,
        degree,
        num_weights,
        weights,
        res_tmp,
        nz,
    )

    for i = 1:num_weights
        for j = 1:n
            res[i, j] = res_tmp[i, j, degree+1]
        end
    end

    free_mat_short(nz, num_weights)
    myfree3(res_tmp)
    free_vec_double(tmp)
end

function vec_hess!(res::CxxPtr{CxxPtr{Float64}}, f, m::Int64, n::Int64, x::Union{Float64, Vector{Float64}}, weights::Vector{Float64}, tape_id::Int64, reuse_tape::Bool)
    if !(reuse_tape)
        create_tape(f, m, n, x, tape_id)
    end

    res_tmp = alloc_vec_double(n)
    dir = zeros(Float64, n)
    for i = 1:n
        dir[i] = 1.0
        vec_hess_vec!(res_tmp, f, m, n, x, dir, weights, tape_id, true)
        for j in 1:n
            res[j, i] = res_tmp[j]
        end
        dir[i] = 0.0
    end
    free_vec_double(res_tmp)
end

function create_tape(
    f,
    m::Int64,
    n::Int64,
    x::Union{Float64,Vector{Float64}},
    tape_id::Int64;
    keep::Int64 = 0,
    enableMinMaxUsingAbs = false,
)
    if enableMinMaxUsingAbs
        ADOLC.TbadoubleModule.enableMinMaxUsingAbs()
    end
    a = n == 1 ? Adouble{TbAlloc}() : [Adouble{TbAlloc}() for _ = 1:n]
    b = m == 1 ? Adouble{TbAlloc}() : [Adouble{TbAlloc}() for _ = 1:m]

    y = m == 1 ? 0.0 : [0.0 for _ = 1:m]
    trace_on(tape_id, keep)
    a << x
    b = f(a)
    b >> y
    trace_off()
end
