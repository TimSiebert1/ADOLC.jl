struct TapeBasedAD
    adouble::Ptr{Cvoid}
end
struct TapeLessAD
    adouble::Ptr{Cvoid}
end

mutable struct Adouble{T<:Union{TapeBasedAD,TapeLessAD}} <: AbstractFloat
    val::Union{Cdouble,T}
    is_diff::Bool

    function Adouble{T}() where {T<:Union{TapeBasedAD,TapeLessAD}}
        adouble = new{T}(0.0, false)
        return finalizer(a -> (a = nothing), adouble)
    end
    function Adouble{TapeBasedAD}(a::TapeBasedAD)
        adouble = new{TapeBasedAD}(a, true)
        return finalizer(
            a -> ccall((:free_tb_adouble, adolc_interface_lib), Cvoid, (Ptr{Cvoid},), a.val),
            adouble,
        )
    end
    function Adouble{TapeBasedAD}(val::V; is_diff::Bool=false) where {V<:Number}
        if is_diff
            adouble = new{TapeBasedAD}(
                ccall(
                    (:create_tb_adouble, adolc_interface_lib),
                    TapeBasedAD,
                    (Cdouble,),
                    Cdouble(val),
                ),
                is_diff,
            )
            finalizer(
                a -> ccall((:free_tb_adouble, adolc_interface_lib), Cvoid, (Ptr{Cvoid},), a.val),
                adouble,
            )
        else
            adouble = new{TapeBasedAD}(val, is_diff)
            finalizer(a -> (a = nothing), adouble)
        end
    end
    function Adouble{TapeLessAD}(a::TapeLessAD)
        adouble = new{TapeLessAD}(a, true)
        return finalizer(
            a -> ccall((:free_tl_adouble, adolc_interface_lib), Cvoid, (Ptr{Cvoid},), a.val),
            adouble,
        )
    end
    function Adouble{TapeLessAD}(val::V; is_diff::Bool=false) where {V<:Number}
        if is_diff
            adouble = new{TapeLessAD}(
                ccall(
                    (:create_tl_adouble, adolc_interface_lib),
                    TapeLessAD,
                    (Cdouble,),
                    Cdouble(val),
                ),
                is_diff,
            )
            finalizer(
                a -> ccall((:free_tl_adouble, adolc_interface_lib), Cvoid, (Ptr{Cvoid},), a.val),
                adouble,
            )
        else
            adouble = new{TapeLessAD}(val, is_diff)
            finalizer(a -> (a = nothing), adouble)
        end
    end
    function Adouble{TapeLessAD}(val::V, ad_val::W) where {V<:Real,W<:Real}
        adouble = new{TapeLessAD}(
            ccall(
                (:create_tl_adouble_with_ad, adolc_interface_lib),
                TapeLessAD,
                (Cdouble, Ptr{Cdouble}),
                Cdouble(val),
                Ref(Cdouble(ad_val)),
            ),
            true,
        )
        return finalizer(
            a -> ccall((:free_tl_adouble, adolc_interface_lib), Cvoid, (Ptr{Cvoid},), a.val),
            adouble,
        )
    end
    function Adouble{TapeLessAD}(val::V, ad_val::Vector{W}) where {V<:Real,W<:Real}
        adouble = new{TapeLessAD}(
            ccall(
                (:create_tl_adouble_with_ad, adolc_interface_lib),
                TapeLessAD,
                (Cdouble, Ptr{Vector{Cdouble}}),
                Cdouble(val),
                convert(Vector{Cdouble}, ad_val),
            ),
            true,
        )
        return finalizer(
            a -> ccall((:free_tl_adouble, adolc_interface_lib), Cvoid, (Ptr{Cvoid},), a.val),
            adouble,
        )
    end
end

function check_is_diff(a::Adouble{T}) where {T<:Union{TapeBasedAD,TapeLessAD}}
    if !a.is_diff
        throw("Given variable is not differentiable and has no AD value!")
    end
end

function Base.unsafe_convert(
    ::Type{Ptr{Cvoid}}, a::T
) where {T<:Union{TapeBasedAD,TapeLessAD}}
    return a.adouble
end

function get_value(a::TapeBasedAD)
    return ccall((:get_tb_value, adolc_interface_lib), Cdouble, (Ptr{Cvoid},), a)
end

function get_value(a::TapeLessAD)
    return ccall((:get_tl_value, adolc_interface_lib), Cdouble, (Ptr{Cvoid},), a)
end

function get_ad_value(a::TapeLessAD)
    return unsafe_load(
        ccall((:get_tl_ad_values, adolc_interface_lib), Ptr{Cdouble}, (Ptr{Cvoid},), a)
    )
end
function get_ad_value(a::Adouble{TapeLessAD})
    check_is_diff(a)
    return get_ad_value(a.val)
end

function get_ad_value(a::TapeLessAD, idx::Integer)
    return ccall(
        (:get_tl_ad_value_idx, adolc_interface_lib), Cdouble, (Ptr{Cvoid}, Cint), a, idx - 1
    )
end
function get_ad_value(a::Adouble{TapeLessAD}, idx::Integer)
    check_is_diff(a)
    return get_ad_value(a.val, idx)
end

function get_ad_values(a::TapeLessAD, n::Integer)
    ptr_vals = ccall((:get_tl_ad_values, adolc_interface_lib), Ptr{Cdouble}, (Ptr{Cvoid},), a)
    return unsafe_wrap(Vector{Cdouble}, ptr_vals, n)
end
function get_ad_values(a::Adouble{TapeLessAD}, n::Integer)
    check_is_diff(a)
    return get_ad_values(a.val, n)
end

function get_value(a::Adouble{T}) where {T<:Union{TapeBasedAD,TapeLessAD}}
    return get_value(a.val)
end
function get_value(x::Cdouble)
    return x
end
function get_value(a::Vector{Adouble{T}}) where {T<:Union{TapeBasedAD,TapeLessAD}}
    res = Vector{Cdouble}(undef, length(a))
    for i in eachindex(a)
        res[i] = get_value(a[i].val)
    end
    return res
end

function set_value(a::TapeLessAD, val::Cdouble)
    return ccall((:set_tl_value, adolc_interface_lib), Cvoid, (Ptr{Cvoid}, Cdouble), a, val)
end
function set_value(a::Adouble{TapeLessAD}, val::Cdouble)
    check_is_diff(a)
    return set_value(a.val, val)
end

function set_ad_value(a::TapeLessAD, val::Cdouble)
    return ccall(
        (:set_tl_ad_value, adolc_interface_lib), Cvoid, (Ptr{Cvoid}, Ptr{Cdouble}), a, Ref(val)
    )
end
function set_ad_value(a::TapeLessAD, val::Vector{Cdouble})
    return ccall(
        (:set_tl_ad_value, adolc_interface_lib),
        Cvoid,
        (Ptr{Cvoid}, Ptr{Vector{Cdouble}}),
        a,
        val,
    )
end
function set_ad_value(a::Adouble{TapeLessAD}, val::Union{Cdouble,Vector{Cdouble}})
    check_is_diff(a)
    return set_ad_value(a.val, val)
end

function set_ad_value(a::TapeLessAD, idx::Integer, val::Cdouble)
    return ccall(
        (:set_tl_ad_value_idx, adolc_interface_lib),
        Cvoid,
        (Ptr{Cvoid}, Cint, Cdouble),
        a,
        idx - 1,
        val,
    )
end
function set_ad_value(a::Adouble{TapeLessAD}, idx::Integer, val::Cdouble)
    check_is_diff(a)
    return set_ad_value(a.val, idx, val)
end

function set_num_dir(n::Integer)
    return ccall((:set_num_dir, adolc_interface_lib), Cvoid, (Cint,), n)
end

#-------- utilities for type handling ----------
function Base.promote(x::V, y::Adouble{T}) where {T<:Union{TapeBasedAD,TapeLessAD},V<:Real}
    return Adouble{T}(Cdouble(x); is_diff=false)
end
Base.promote(x::Adouble{T}, y::Adouble{T}) where {T<:Union{TapeBasedAD,TapeLessAD}} = x

function Base.promote_rule(
    ::Type{Adouble{T}}, ::Type{V}
) where {T<:Union{TapeBasedAD,TapeLessAD},V<:Real}
    return Adouble{T}
end

function Base.promote_op(
    f, ::Type{Adouble{T}}, ::Type{Adouble{T}}
) where {T<:Union{TapeBasedAD,TapeLessAD}}
    return Adouble{T}
end
function Base.promote_op(
    f, ::Type{V}, ::Type{Adouble{T}}
) where {T<:Union{TapeBasedAD,TapeLessAD},V<:Real}
    return Adouble{T}
end
function Base.promote_op(
    f, ::Type{Adouble{T}}, ::Type{V}
) where {T<:Union{TapeBasedAD,TapeLessAD},V<:Real}
    return Adouble{T}
end
function Base.convert(
    ::Type{Adouble{T}}, x::V
) where {T<:Union{TapeBasedAD,TapeLessAD},V<:Real}
    return Adouble{T}(Cdouble(x); is_diff=false)
end
Base.convert(::Type{Adouble{T}}, x::Adouble{T}) where {T<:Union{TapeBasedAD,TapeLessAD}} = x

# ----------------------- parameter ----------------------

function mkparam(x::Cdouble)
    return Adouble{TapeBasedAD}(
        ccall((:mkparam_, adolc_interface_lib), TapeBasedAD, (Cdouble,), x)
    )
end

function mkparam(x::Vector{Cdouble})
    return [
        Adouble{TapeBasedAD}(
            ccall((:mkparam_, adolc_interface_lib), TapeBasedAD, (Cdouble,), x_i)
        ) for x_i in x
    ]
end
