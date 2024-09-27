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
        _, m, n = create_tape(f, x, tape_id)
    else
        m = ccall((:num_dependent, ADOLC_JLL_PATH), Cuint, (Cshort,), tape_id)
        n = ccall((:num_independent, ADOLC_JLL_PATH), Cuint, (Cshort,), tape_id)
    end
    init_tp = CxxMatrix(zeros(Cdouble, n, degree + 1))
    for i in 1:n
        init_tp[i, 1] = x[i]
        init_tp[i, 2] = 1.0
    end
    res = CxxMatrix(m, degree + 1)
    univariate_tpp!(res, tape_id, m, n, degree, init_tp; keep=keep)
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
        y, m, n = create_tape(f, x, tape_id)
    else
        m = ccall((:num_dependent, ADOLC_JLL_PATH), Cuint, (Cshort,), tape_id)
        n = ccall((:num_independent, ADOLC_JLL_PATH), Cuint, (Cshort,), tape_id)
    end
    res = CxxMatrix(m, degree + 1)
    univariate_tpp!(res, tape_id, m, n, degree, init_tp; keep=keep)
    return res
end

"""
    univariate_tpp!(
        res::CxxMatrix,
        tape_id::Integer,
        m::Integer,
        n::Integer,
        degree::Integer,
        init_tp::CxxMatrix;
        keep::Bool=false
    )
A version of the [`univariate_tpp`](@ref) driver that allows a user to pass in a pre-allocated [CxxMatrix](@ref "Working with C++ Memory").
More information is given in the guide: [Univariate Taylor Polynomial Propagation](@ref).
"""
function univariate_tpp!(
    res, tape_id, m::Integer, n::Integer, degree::Integer, init_tp; keep::Bool=false
)
    return ccall(
        (:forward1, ADOLC_JLL_PATH),
        Cvoid,
        (Cshort, Cint, Cint, Cint, Cint, Ptr{Ptr{Cdouble}}, Ptr{Ptr{Cdouble}}),
        tape_id,
        m,
        n,
        degree,
        keep,
        init_tp,
        res,
    )
end
