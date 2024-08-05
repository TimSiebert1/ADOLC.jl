
"""
    univariate_tpp(
        f,
        x::Union{Cdouble,Vector{Cdouble}},
        degree::Integer;
        keep::Bool=false,
        tape_id::Integer=0,
        reuse_tape::Bool=false,
    )
The driver propagates univariate Taylor polynomials through the given function `f` at the point 
`x` up to `degree`. The `keep` flag is used to prepare the tape for subsequent reverse-mode computations
on the Taylor polynomial. The `tape_id` specifies the identifier of the tape and the flag `reuse_tape` should 
be used for suppressing the tape creation.
More information is given in the guide: [Univariate Taylor Polynomial Propagation](@ref).
"""
function univariate_tpp(
    f,
    x::Union{Cdouble,Vector{Cdouble}},
    degree::Integer;
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
    for i in 1:n
        init_tp[i, 1] = x[i]
        init_tp[i, 2] = 1.0
    end
    res = CxxMatrix(m, degree + 1)
    univariate_tpp!(res, f, x, degree, init_tp; keep=keep, tape_id=tape_id, reuse_tape=true)
    return res
end


"""
    univariate_tpp(
        f,
        x::Union{Cdouble,Vector{Cdouble}},
        degree::Integer,
        init_tp::CxxMatrix;
        keep::Bool=false,
        tape_id::Integer=0,
        reuse_tape::Bool=false,
    )
Version of the [`univariate_tpp`](@ref) driver, which allows additional control over the initial Taylor polynomial. 
More information is given in the guide: [Univariate Taylor Polynomial Propagation](@ref).
"""
function univariate_tpp(
    f,
    x::Union{Cdouble,Vector{Cdouble}},
    degree::Integer,
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
    univariate_tpp!(res, f, x, degree, init_tp; keep=keep, tape_id=tape_id, reuse_tape=true)
    return res
end

"""
    univariate_tpp!(
        res::CxxMatrix,
        f,
        x::Union{Cdouble,Vector{Cdouble}},
        degree::Integer,
        init_tp::CxxMatrix;
        keep::Bool=false,
        tape_id::Integer=0,
        reuse_tape::Bool=false,
    )
A version of the [`univariate_tpp`](@ref) driver that allows a user to pass in a pre-allocated [CxxMatrix](@ref "Working with C++ Memory").
More information is given in the guide: [Univariate Taylor Polynomial Propagation](@ref).
"""
function univariate_tpp!(
    res::CxxMatrix,
    f,
    x::Union{Cdouble,Vector{Cdouble}},
    degree::Integer,
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
