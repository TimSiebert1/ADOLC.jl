
"""
    univariate_tpp(
        f,
        degree::Integer,
        x::Union{Cdouble,Vector{Cdouble}};
        keep::Bool=false,
        tape_id::Integer=0,
        reuse_tape::Bool=false,
    )
"""
function univariate_tpp(
    f,
    degree::Integer,
    x::Union{Cdouble,Vector{Cdouble}};
    keep::Bool=false,
    tape_id::Integer=0,
    reuse_tape::Bool=false,
)
    if !reuse_tape
        m, n = create_tape(f, x, tape_id)
    else
        m = TbadoubleModule.num_dependents(tape_id)
        n = TbadoubleModule.num_independents(tape_id)
    end
    init_tp = CxxMatrix(zeros(Cdouble, n, degree + 1))
    for j in 1:n
        for i in 1:n
            if j == 1
                init_tp[i, j] = x[i]
            elseif j == 2
                init_tp[i, j] = 1.0
            end
        end
    end
    res = CxxMatrix(m, degree + 1)
    univariate_tpp!(res, f, degree, x, init_tp; keep=keep, tape_id=tape_id, reuse_tape=true)
    return res
end


"""
    univariate_tpp(
        f,
        degree::Integer,
        x::Union{Cdouble,Vector{Cdouble}},
        init_tp::CxxMatrix;
        keep::Bool=false,
        tape_id::Integer=0,
        reuse_tape::Bool=false,
    )
"""
function univariate_tpp(
    f,
    degree::Integer,
    x::Union{Cdouble,Vector{Cdouble}},
    init_tp::CxxMatrix;
    keep::Bool=false,
    tape_id::Integer=0,
    reuse_tape::Bool=false,
)
    if !reuse_tape
        m, n = create_tape(f, x, tape_id)
    else
        m = TbadoubleModule.num_dependents(tape_id)
        n = TbadoubleModule.num_independents(tape_id)
    end
    res = CxxMatrix(m, degree + 1)
    univariate_tpp!(res, f, degree, x, init_tp; keep=keep, tape_id=tape_id, reuse_tape=true)
    return res
end

"""
    univariate_tpp!(
        res::CxxMatrix,
        f,
        degree::Integer,
        x::Union{Cdouble,Vector{Cdouble}},
        init_tp::CxxMatrix;
        keep::Bool=false,
        tape_id::Integer=0,
        reuse_tape::Bool=false,
    )
"""
function univariate_tpp!(
    res::CxxMatrix,
    f,
    degree::Integer,
    x::Union{Cdouble,Vector{Cdouble}},
    init_tp::CxxMatrix;
    keep::Bool=false,
    tape_id::Integer=0,
    reuse_tape::Bool=false,
)
    if !reuse_tape
        m, n = create_tape(f, x, tape_id)
    else
        m = TbadoubleModule.num_dependents(tape_id)
        n = TbadoubleModule.num_independents(tape_id)
    end
    return TbadoubleModule.ad_forward(tape_id, m, n, degree, keep, init_tp.data, res.data)
end
