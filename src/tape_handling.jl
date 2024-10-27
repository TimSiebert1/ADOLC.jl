#-------------------- create independent ---------------------

function Base.:<<(a::TapeBasedAD, x::Cdouble)
    return ccall(
        (:create_independent, adolc_interface_lib), Cvoid, (Ptr{Cvoid}, Cdouble), a.adouble, x
    )
end
function Base.:<<(a::Adouble{TapeBasedAD}, x::Cdouble)
    check_is_diff(a)
    return a.val << x
end

function Base.:<<(a::Vector{Adouble{TapeBasedAD}}, x::Vector{Cdouble})
    @assert length(a) == length(x)
    for i in eachindex(x)
        a[i].val << x[i]
    end
end

function Base.:<<(a::Adouble{TapeBasedAD}, x::Vector{Cdouble})
    @assert length(a) == length(x)
    return a.val << x[1]
end

function Base.:<<(a::Vector{Adouble{TapeBasedAD}}, x::Cdouble)
    @assert length(a) == length(x)
    return a[1].val << x
end

function create_independent(x::Union{Cdouble,Vector{Cdouble}})
    a = if isa(x, Number)
        Adouble{TapeBasedAD}(0.0; is_diff=true)
    else
        [Adouble{TapeBasedAD}(0.0; is_diff=true) for _ in eachindex(x)]
    end
    a << x
    return a
end

#------------------- create dependent -------------------------------

function Base.:>>(a::TapeBasedAD, x_ref::Base.RefArray{Cdouble})
    return ccall(
        (:create_dependent, adolc_interface_lib), Cvoid, (Ptr{Cvoid}, Ptr{Cdouble}), a, x_ref
    )
end
function Base.:>>(a::Adouble{TapeBasedAD}, x::Vector{Cdouble})
    @assert length(a) == length(x)
    return a.val >> Ref(x, 1)
end

function Base.:>>(a::Vector{Adouble{TapeBasedAD}}, x::Vector{Cdouble})
    @assert length(a) == length(x)
    for i in eachindex(x)
        a[i].val >> Ref(x, i)
    end
end

function create_dependent(b)
    y = Vector{Cdouble}(undef, length(b))
    b >> y
    return y
end

# ---------------------- open and close tape -----------------------

function trace_on(tag; keep=0)
    return ccall(
        (:c_trace_on, adolc_interface_lib), Cint, (Cshort, Cint), Cshort(tag), Cint(keep)
    )
end
function trace_off(; flag=0)
    return ccall((:c_trace_off, adolc_interface_lib), Cint, (Cint,), Cint(flag))
end

# --------------------- create tape -----------------------------
function create_tape(
    f,
    x::Union{Cdouble,Vector{Cdouble}},
    tape_id;
    keep=0,
    flag=0,
    enableMinMaxUsingAbs=false,
)
    if enableMinMaxUsingAbs
        ccall((:enableMinMaxUsingAbs, adolc_interface_lib), Cvoid, ())
    end
    trace_on(tape_id; keep=keep)
    a = create_independent(x)
    b = f(a)
    y = create_dependent(b)
    trace_off(; flag=flag)
    if enableMinMaxUsingAbs
        ccall((:disableMinMaxUsingAbs, adolc_interface_lib), Cvoid, ())
    end
    return y, length(y), length(x)
end

function create_tape(
    f,
    x::Union{Cdouble,Vector{Cdouble}},
    param::Union{Cdouble,Vector{Cdouble}},
    tape_id::Integer;
    keep=0,
    flag=0,
    enableMinMaxUsingAbs=false,
)
    if enableMinMaxUsingAbs
        ccall((:enableMinMaxUsingAbs, adolc_interface_lib), Cvoid, ())
    end
    trace_on(tape_id; keep=keep)
    a, a_param = create_independent(x, param)
    b = f(a, a_param)
    y = create_dependent(b)
    trace_off(; flag=flag)
    if enableMinMaxUsingAbs
        ccall((:disableMinMaxUsingAbs, adolc_interface_lib), Cvoid, ())
    end
    return y, length(y), length(x)
end

"""
    create_independent(x::Union{Cdouble, Vector{Cdouble}}, param::Union{Cdouble,Vector{Cdouble}}) 

The argument `x` is stored as differentiable `Adouble{TbAlloc}` and marked as independent. `param` is
marked as parameters on the tape to be changeble without retaping.
"""
function create_independent(
    x::Union{Cdouble,Vector{Cdouble}}, param::Union{Cdouble,Vector{Cdouble}}
)
    a = create_independent(x)
    a_param = mkparam(param)
    return a, a_param
end

# -------------------- set param vec

function set_param_vec(tape_id, param)
    return ccall(
        (:set_param_vec, adolc_interface_lib),
        Cvoid,
        (Cshort, Cuint, Ptr{Cdouble}),
        tape_id,
        length(param),
        isa(param, Number) ? [param] : param,
    )
end
