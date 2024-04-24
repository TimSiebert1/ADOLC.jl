using ADOLC, ADOLC.array_types, ADOLC.TbadoubleModule, CxxWrap

function derivative(
    f::Function,
    m::Int64,
    n::Int64,
    x::Union{Float64,Vector{Float64}},
    mode::Symbol;
    d::Int64=1,
    dir::Union{Vector{Float64},Matrix{Float64}}=Vector{Float64}(),
    weights::Union{Vector{Float64},Matrix{Float64}}=Vector{Float64}(),
    partials::Vector{Int64}=Vector{Int64}(),
    tape_id::Int64=0,
    res::Union{Vector{Float64},CxxPtr{Float64},CxxPtr{CxxPtr{Float64}},AbsNormalProblem}=Vector{Float64}(),
)
    if d == 1
        first_order(f, m, n, x, mode, dir, weights, tape_id, res)
    else
        throw("derivative_order: $d not implemented!")
    end
end

function derivative(
    tape_id::Int64,
    m::Int64,
    n::Int64,
    x::Union{Float64,Vector{Float64}},
    mode::Symbol;    
    d::Int64=1,
    dir::Union{Vector{Float64},Matrix{Float64}}=Vector{Float64}(),
    weights::Union{Vector{Float64},Matrix{Float64}}=Vector{Float64}(),
    partials::Vector{Int64}=Vector{Int64}(),
    res::Union{Vector{Float64},CxxPtr{Float64},CxxPtr{CxxPtr{Float64}},AbsNormalProblem}=Vector{Float64}(),
)
    if d == 1
        first_order(tape_id, m, n, x, mode, dir, weights, res)
    else
        throw("derivative_order: $d not implemented!")
    end
end

function first_order(
    f,
    m::Int64,
    n::Int64,
    x::Union{Float64,Vector{Float64}},
    mode::Symbol,
    dir::Union{Vector{Float64},Matrix{Float64}},
    weights::Union{Vector{Float64},Matrix{Float64}},
    tape_id::Int64,
    res,
)
    if mode === :jac
        jac(f, m, n, x, tape_id, res)
    elseif mode === :jac_vec
        fos_forward(f, m, n, x, dir, tape_id, res)
    elseif mode === :jac_mat
        fov_forward(f, m, n, x, dir, tape_id, res)
    elseif mode === :vec_jac
        fos_reverse(f, m, n, x, weights, tape_id, res)
    elseif mode === :mat_jac
        fov_reverse(f, m, n, x, weights, tape_id, res)
    elseif mode === :abs_normal
        abs_normal(f, m, n, x, tape_id)
    else
        throw("mode: $mode not implemented!")
    end
end

function first_order(
    tape_id::Int64,
    m::Int64,
    n::Int64,
    x::Union{Float64,Vector{Float64}},
    mode::Symbol,
    dir::Union{Vector{Float64},Matrix{Float64}},
    weights::Union{Vector{Float64},Matrix{Float64}},
    res,
)
    if mode === :abs_normal
        abs_normal(tape_id, m, n, x, res)
    else
        throw("mode: $mode not implemented!")
    end
end


function jac(f, m::Int64, n::Int64, x::Union{Float64,Vector{Float64}}, tape_id::Int64, res)
    if m == 1
        gradient(f, n, x, tape_id, res)
    else
        if n / 2 < m
            throw("track not implemented!")
            #tape_less_forward()
        else
            weights = create_cxx_identity(m, m)
            fov_reverse(f, m, n, m, x, weights, tape_id, res)
        end
    end
end


function gradient(f, n::Int64, x::Union{Float64,Vector{Float64}}, tape_id::Int64, res)
    _ = create_tape(f, 1, n, x, tape_id)
    ADOLC.TbadoubleModule.gradient(tape_id, n, x, res)
end


function fos_reverse(
    f,
    m::Int64,
    n::Int64,
    x::Union{Float64,Vector{Float64}},
    weights::Vector{Float64},
    tape_id::Int64,
    res,
)
    _ = create_tape(f, m, n, x, tape_id, keep = 1)
    ADOLC.TbadoubleModule.fos_reverse(tape_id, m, n, weights, res)
end



function fov_reverse(
    f,
    m::Int64,
    n::Int64,
    x::Union{Float64,Vector{Float64}},
    weights::Matrix{Float64},
    tape_id::Int64,
    res,
)
    num_dir = size(weights, 2)
    weights_cxx = myalloc2(size(weights)...)
    for i = 1:size(weights, 1)
        for j = 1:size(weights, 2)
            weights_cxx[i, j] = weights[i, j]
        end
    end
    fov_reverse(f, m, n, num_dir, x, weights_cxx, tape_id, res)
    myfree2(weights_cxx)
end

function fov_reverse(
    f,
    m::Int64,
    n::Int64,
    num_dir::Int64,
    x::Union{Float64,Vector{Float64}},
    weights::CxxPtr{CxxPtr{Float64}},
    tape_id::Int64,
    res,
)
    _ = create_tape(f, m, n, x, tape_id, keep = 1)
    ADOLC.TbadoubleModule.fov_reverse(tape_id, m, n, num_dir, weights, res)
end


function fos_forward(
    f,
    m::Int64,
    n::Int64,
    x::Union{Float64,Vector{Float64}},
    dir::Vector{Float64},
    tape_id,
    res,
)
    y = create_tape(f, m, n, x, tape_id)
    ADOLC.TbadoubleModule.fos_forward(tape_id, m, n, 0, x, dir, y, res)
end


function fov_forward(
    f,
    m::Int64,
    n::Int64,
    x::Union{Float64,Vector{Float64}},
    dir::Matrix{Float64},
    tape_id,
    res,
)
    num_dir = size(dir, 2)
    dir_cxx = myalloc2(size(dir)...)
    for i = 1:size(dir, 1)
        for j = 1:size(dir, 2)
            dir_cxx[i, j] = dir[i, j]
        end
    end
    fov_forward(f, m, n, num_dir, x, dir_cxx, tape_id, res)
    myfree2(dir_cxx)
end


function fov_forward(
    f,
    m::Int64,
    n::Int64,
    num_dir::Int64,
    x::Union{Float64,Vector{Float64}},
    dir::CxxPtr{CxxPtr{Float64}},
    tape_id,
    res,
)
    y = create_tape(f, m, n, x, tape_id)
    ADOLC.TbadoubleModule.fov_forward(tape_id, m, n, num_dir, x, dir, y, res)
end



function abs_normal(
    f,
    m::Int64,
    n::Int64,
    x::Union{Float64,Vector{Float64}},
    tape_id::Int64,
)
    y = create_tape(f, m, n, x, tape_id, enableMinMaxUsingAbs = true)
    abs_normal_problem = ADOLC.AbsNormalProblem{Float64}(tape_id, m, n, x, m == 1 ? [y] : y)
    ADOLC.abs_normal!(abs_normal_problem)
    return abs_normal_problem
end

function check_resue_abs_normal_problem(
    tape_id::Int64,
    m::Int64,
    n::Int64,
    abs_normal_problem::AbsNormalProblem,
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

function abs_normal(
    tape_id::Int64,
    m::Int64,
    n::Int64,
    x::Union{Float64,Vector{Float64}},
    abs_normal_problem::AbsNormalProblem,
)
    check_resue_abs_normal_problem(tape_id, m, n, abs_normal_problem)
    ADOLC.array_types.vec_to_cxx(abs_normal_problem.x, x)
    ADOLC.abs_normal!(abs_normal_problem)
end


"""
function tape_less_forward(func, init_point::Vector{Float64})
    a = Adouble{TlAlloc}(init_point)
    b = func(a)
    return get_gradient(b, length(init_point)), getValue(b)
end
"""
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
    y = b >> y
    trace_off()
    return y
end
