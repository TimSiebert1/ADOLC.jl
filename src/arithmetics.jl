using SpecialFunctions: SpecialFunctions
#--------------- Operation: * -------------------
function Base.:*(a::TapeBasedAD, b::TapeBasedAD)
    return ccall(
        (:mult_tb_adouble, ADOLC_JLL_PATH), TapeBasedAD, (TapeBasedAD, TapeBasedAD), a, b
    )
end
function Base.:*(val::V, b::TapeBasedAD) where {V<:Real}
    return ccall(
        (:mult_double_tb_adouble, ADOLC_JLL_PATH),
        TapeBasedAD,
        (Cdouble, TapeBasedAD),
        Cdouble(val),
        b,
    )
end
function Base.:*(a::TapeBasedAD, val::V) where {V<:Real}
    return ccall(
        (:mult_tb_adouble_double, ADOLC_JLL_PATH),
        TapeBasedAD,
        (TapeBasedAD, Cdouble),
        a,
        Cdouble(val),
    )
end

function Base.:*(a::TapeLessAD, b::TapeLessAD)
    return ccall(
        (:mult_tl_adouble, ADOLC_JLL_PATH), TapeLessAD, (TapeLessAD, TapeLessAD), a, b
    )
end
function Base.:*(val::V, b::TapeLessAD) where {V<:Real}
    return ccall(
        (:mult_double_tl_adouble, ADOLC_JLL_PATH),
        TapeLessAD,
        (Cdouble, TapeLessAD),
        Cdouble(val),
        b,
    )
end
function Base.:*(a::TapeLessAD, val::V) where {V<:Real}
    return ccall(
        (:mult_tl_adouble_double, ADOLC_JLL_PATH),
        TapeLessAD,
        (TapeLessAD, Cdouble),
        a,
        Cdouble(val),
    )
end

function Base.:*(a::Adouble{T}, b::Adouble{T}) where {T<:Union{TapeBasedAD,TapeLessAD}}
    return Adouble{T}(a.val * b.val)
end
function Base.:*(a::Adouble{T}, x::V) where {T<:Union{TapeBasedAD,TapeLessAD},V<:Real}
    return Adouble{T}(a.val * Cdouble(x))
end
function Base.:*(x::V, a::Adouble{T}) where {T<:Union{TapeBasedAD,TapeLessAD},V<:Real}
    return Adouble{T}(Cdouble(x) * a.val)
end

function Base.:*(
    a::Adouble{T}, x::AbstractVector{Cdouble}
) where {T<:Union{TapeBasedAD,TapeLessAD}}
    return a .* x
end

#--------------- Operation: + -------------------
function Base.:+(a::TapeBasedAD, b::TapeBasedAD)
    return ccall(
        (:add_tb_adouble, ADOLC_JLL_PATH), TapeBasedAD, (TapeBasedAD, TapeBasedAD), a, b
    )
end
function Base.:+(val::V, b::TapeBasedAD) where {V<:Real}
    return ccall(
        (:add_double_tb_adouble, ADOLC_JLL_PATH),
        TapeBasedAD,
        (Cdouble, TapeBasedAD),
        Cdouble(val),
        b,
    )
end
function Base.:+(a::TapeBasedAD, val::V) where {V<:Real}
    return ccall(
        (:add_tb_adouble_double, ADOLC_JLL_PATH),
        TapeBasedAD,
        (TapeBasedAD, Cdouble),
        a,
        Cdouble(val),
    )
end

function Base.:+(a::TapeLessAD, b::TapeLessAD)
    return ccall(
        (:add_tl_adouble, ADOLC_JLL_PATH), TapeLessAD, (TapeLessAD, TapeLessAD), a, b
    )
end
function Base.:+(val::V, b::TapeLessAD) where {V<:Real}
    return ccall(
        (:add_double_tl_adouble, ADOLC_JLL_PATH),
        TapeLessAD,
        (Cdouble, TapeLessAD),
        Cdouble(val),
        b,
    )
end
function Base.:+(a::TapeLessAD, val::V) where {V<:Real}
    return ccall(
        (:add_tl_adouble_double, ADOLC_JLL_PATH),
        TapeLessAD,
        (TapeLessAD, Cdouble),
        a,
        Cdouble(val),
    )
end

function Base.:+(a::Adouble{T}, b::Adouble{T}) where {T<:Union{TapeBasedAD,TapeLessAD}}
    return Adouble{T}(a.val + b.val)
end
function Base.:+(a::Adouble{T}, x::V) where {T<:Union{TapeBasedAD,TapeLessAD},V<:Real}
    return Adouble{T}(a.val + Cdouble(x))
end
function Base.:+(x::V, a::Adouble{T}) where {T<:Union{TapeBasedAD,TapeLessAD},V<:Real}
    return Adouble{T}(Cdouble(x) + a.val)
end

#--------------- Operation: - -------------------

function Base.:-(a::TapeBasedAD, b::TapeBasedAD)
    return ccall(
        (:subtr_tb_adouble, ADOLC_JLL_PATH), TapeBasedAD, (TapeBasedAD, TapeBasedAD), a, b
    )
end
function Base.:-(val::V, b::TapeBasedAD) where {V<:Real}
    return ccall(
        (:subtr_double_tb_adouble, ADOLC_JLL_PATH),
        TapeBasedAD,
        (Cdouble, TapeBasedAD),
        Cdouble(val),
        b,
    )
end
function Base.:-(a::TapeBasedAD, val::V) where {V<:Real}
    return ccall(
        (:subtr_tb_adouble_double, ADOLC_JLL_PATH),
        TapeBasedAD,
        (TapeBasedAD, Cdouble),
        a,
        Cdouble(val),
    )
end

function Base.:-(a::TapeLessAD, b::TapeLessAD)
    return ccall(
        (:subtr_tl_adouble, ADOLC_JLL_PATH), TapeLessAD, (TapeLessAD, TapeLessAD), a, b
    )
end
function Base.:-(val::V, b::TapeLessAD) where {V<:Real}
    return ccall(
        (:subtr_double_tl_adouble, ADOLC_JLL_PATH),
        TapeLessAD,
        (Cdouble, TapeLessAD),
        Cdouble(val),
        b,
    )
end
function Base.:-(a::TapeLessAD, val::V) where {V<:Real}
    return ccall(
        (:subtr_tl_adouble_double, ADOLC_JLL_PATH),
        TapeLessAD,
        (TapeLessAD, Cdouble),
        a,
        Cdouble(val),
    )
end

function Base.:-(x::V, a::Adouble{T}) where {T<:Union{TapeBasedAD,TapeLessAD},V<:Real}
    return Adouble{T}(Cdouble(x) - a.val)
end
function Base.:-(a::Adouble{T}, x::V) where {T<:Union{TapeBasedAD,TapeLessAD},V<:Real}
    return Adouble{T}(a.val - Cdouble(x))
end
function Base.:-(a::Adouble{T}, b::Adouble{T}) where {T<:Union{TapeBasedAD,TapeLessAD}}
    return Adouble{T}(a.val - b.val)
end

Base.:-(a::Adouble{T}) where {T<:Union{TapeBasedAD,TapeLessAD}} = (-1) * a

#--------------- Operation: / -------------------
function Base.:/(a::TapeBasedAD, b::TapeBasedAD)
    return ccall(
        (:div_tb_adouble, ADOLC_JLL_PATH), TapeBasedAD, (TapeBasedAD, TapeBasedAD), a, b
    )
end
function Base.:/(val::V, b::TapeBasedAD) where {V<:Real}
    return ccall(
        (:div_double_tb_adouble, ADOLC_JLL_PATH),
        TapeBasedAD,
        (Cdouble, TapeBasedAD),
        Cdouble(val),
        b,
    )
end
function Base.:/(a::TapeBasedAD, val::V) where {V<:Real}
    return ccall(
        (:div_tb_adouble_double, ADOLC_JLL_PATH),
        TapeBasedAD,
        (TapeBasedAD, Cdouble),
        a,
        Cdouble(val),
    )
end

function Base.:/(a::TapeLessAD, b::TapeLessAD)
    return ccall(
        (:div_tl_adouble, ADOLC_JLL_PATH), TapeLessAD, (TapeLessAD, TapeLessAD), a, b
    )
end
function Base.:/(val::V, b::TapeLessAD) where {V<:Real}
    return ccall(
        (:div_double_tl_adouble, ADOLC_JLL_PATH),
        TapeLessAD,
        (Cdouble, TapeLessAD),
        Cdouble(val),
        b,
    )
end
function Base.:/(a::TapeLessAD, val::V) where {V<:Real}
    return ccall(
        (:div_tl_adouble_double, ADOLC_JLL_PATH),
        TapeLessAD,
        (TapeLessAD, Cdouble),
        a,
        Cdouble(val),
    )
end

function Base.:/(x::V, a::Adouble{T}) where {T<:Union{TapeBasedAD,TapeLessAD},V<:Real}
    return Adouble{T}(Cdouble(x) / a.val)
end
function Base.:/(a::Adouble{T}, x::V) where {T<:Union{TapeBasedAD,TapeLessAD},V<:Real}
    return Adouble{T}(a.val / Cdouble(x))
end
function Base.:/(a::Adouble{T}, b::Adouble{T}) where {T<:Union{TapeBasedAD,TapeLessAD}}
    return Adouble{T}(a.val / b.val)
end

#-------------- Functions: max -----------------
function Base.max(a::TapeBasedAD, b::TapeBasedAD)
    return ccall(
        (:max_tb_adouble, ADOLC_JLL_PATH), TapeBasedAD, (TapeBasedAD, TapeBasedAD), a, b
    )
end
function Base.max(val::V, b::TapeBasedAD) where {V<:Real}
    return ccall(
        (:max_double_tb_adouble, ADOLC_JLL_PATH),
        TapeBasedAD,
        (Cdouble, TapeBasedAD),
        Cdouble(val),
        b,
    )
end
function Base.max(a::TapeBasedAD, val::V) where {V<:Real}
    return ccall(
        (:max_tb_adouble_double, ADOLC_JLL_PATH),
        TapeBasedAD,
        (TapeBasedAD, Cdouble),
        a,
        Cdouble(val),
    )
end
function Base.max(a::TapeLessAD, b::TapeLessAD)
    return ccall(
        (:max_tl_adouble, ADOLC_JLL_PATH), TapeLessAD, (TapeLessAD, TapeLessAD), a, b
    )
end
function Base.max(val::V, b::TapeLessAD) where {V<:Real}
    return ccall(
        (:max_double_tl_adouble, ADOLC_JLL_PATH),
        TapeLessAD,
        (Cdouble, TapeLessAD),
        Cdouble(val),
        b,
    )
end
function Base.max(a::TapeLessAD, val::V) where {V<:Real}
    return ccall(
        (:max_tl_adouble_double, ADOLC_JLL_PATH),
        TapeLessAD,
        (TapeLessAD, Cdouble),
        a,
        Cdouble(val),
    )
end

function Base.max(x::V, a::Adouble{T}) where {T<:Union{TapeBasedAD,TapeLessAD},V<:Real}
    return Adouble{T}(max(a.val, x))
end
function Base.max(a::Adouble{T}, x::V) where {T<:Union{TapeBasedAD,TapeLessAD},V<:Real}
    return Adouble{T}(max(x, a.val))
end
function Base.max(a::Adouble{T}, b::Adouble{T}) where {T<:Union{TapeBasedAD,TapeLessAD}}
    return Adouble{T}(max(a.val, b.val))
end

# ------------ Functions: min -----------
function Base.min(a::TapeBasedAD, b::TapeBasedAD)
    return ccall(
        (:min_tb_adouble, ADOLC_JLL_PATH), TapeBasedAD, (TapeBasedAD, TapeBasedAD), a, b
    )
end
function Base.min(val::V, b::TapeBasedAD) where {V<:Real}
    return ccall(
        (:min_double_tb_adouble, ADOLC_JLL_PATH),
        TapeBasedAD,
        (Cdouble, TapeBasedAD),
        Cdouble(val),
        b,
    )
end
function Base.min(a::TapeBasedAD, val::V) where {V<:Real}
    return ccall(
        (:min_tb_adouble_double, ADOLC_JLL_PATH),
        TapeBasedAD,
        (TapeBasedAD, Cdouble),
        a,
        Cdouble(val),
    )
end
function Base.min(a::TapeLessAD, b::TapeLessAD)
    return ccall(
        (:min_tl_adouble, ADOLC_JLL_PATH), TapeLessAD, (TapeLessAD, TapeLessAD), a, b
    )
end
function Base.min(val::V, b::TapeLessAD) where {V<:Real}
    return ccall(
        (:min_double_tl_adouble, ADOLC_JLL_PATH),
        TapeLessAD,
        (Cdouble, TapeLessAD),
        Cdouble(val),
        b,
    )
end
function Base.min(a::TapeLessAD, val::V) where {V<:Real}
    return ccall(
        (:min_tl_adouble_double, ADOLC_JLL_PATH),
        TapeLessAD,
        (TapeLessAD, Cdouble),
        a,
        Cdouble(val),
    )
end

function Base.min(x::V, a::Adouble{T}) where {T<:Union{TapeBasedAD,TapeLessAD},V<:Real}
    return Adouble{T}(min(a.val, x))
end
function Base.min(a::Adouble{T}, x::V) where {T<:Union{TapeBasedAD,TapeLessAD},V<:Real}
    return Adouble{T}(min(x, a.val))
end
function Base.min(a::Adouble{T}, b::Adouble{T}) where {T<:Union{TapeBasedAD,TapeLessAD}}
    return Adouble{T}(min(a.val, b.val))
end
#--------------- Operation: ^ -------------------
function Base.:^(a::TapeBasedAD, b::TapeBasedAD)
    return ccall(
        (:pow_tb_adouble, ADOLC_JLL_PATH), TapeBasedAD, (TapeBasedAD, TapeBasedAD), a, b
    )
end
function Base.:^(a::TapeBasedAD, val::V) where {V<:Real}
    return ccall(
        (:pow_tb_adouble_double, ADOLC_JLL_PATH),
        TapeBasedAD,
        (TapeBasedAD, Cdouble),
        a,
        Cdouble(val),
    )
end
function Base.:^(a::TapeLessAD, b::TapeLessAD)
    return ccall(
        (:pow_tl_adouble, ADOLC_JLL_PATH), TapeLessAD, (TapeLessAD, TapeLessAD), a, b
    )
end
function Base.:^(a::TapeLessAD, val::V) where {V<:Real}
    return ccall(
        (:pow_tl_adouble_double, ADOLC_JLL_PATH),
        TapeLessAD,
        (TapeLessAD, Cdouble),
        a,
        Cdouble(val),
    )
end
function Base.:^(a::Adouble{T}, b::Adouble{T}) where {T<:Union{TapeBasedAD,TapeLessAD}}
    return Adouble{T}(a.val^b.val)
end
function Base.:^(a::Adouble{T}, val::V) where {T<:Union{TapeBasedAD,TapeLessAD},V<:Real}
    return Adouble{T}(a.val^val)
end

function Base.:^(a::Adouble{T}, val::Integer) where {T<:Union{TapeBasedAD,TapeLessAD}}
    return Adouble{T}(a.val^Cdouble(val))
end

#--------------- Operation: >= -------------------
function Base.:>=(a::TapeBasedAD, b::TapeBasedAD)
    return ccall((:ge_tb_adouble, ADOLC_JLL_PATH), Bool, (TapeBasedAD, TapeBasedAD), a, b)
end
function Base.:>=(val::V, b::TapeBasedAD) where {V<:Real}
    return ccall(
        (:ge_double_tb_adouble, ADOLC_JLL_PATH),
        Bool,
        (Cdouble, TapeBasedAD),
        Cdouble(val),
        b,
    )
end
function Base.:>=(a::TapeBasedAD, val::V) where {V<:Real}
    return ccall(
        (:ge_tb_adouble_double, ADOLC_JLL_PATH),
        Bool,
        (TapeBasedAD, Cdouble),
        a,
        Cdouble(val),
    )
end
function Base.:>=(a::TapeLessAD, b::TapeLessAD)
    return ccall((:ge_tl_adouble, ADOLC_JLL_PATH), Bool, (TapeLessAD, TapeLessAD), a, b)
end
function Base.:>=(val::V, b::TapeLessAD) where {V<:Real}
    return ccall(
        (:ge_double_tl_adouble, ADOLC_JLL_PATH),
        Bool,
        (Cdouble, TapeLessAD),
        Cdouble(val),
        b,
    )
end
function Base.:>=(a::TapeLessAD, val::V) where {V<:Real}
    return ccall(
        (:ge_tl_adouble_double, ADOLC_JLL_PATH),
        Bool,
        (TapeLessAD, Cdouble),
        a,
        Cdouble(val),
    )
end
Base.:>=(x::V, a::Adouble{T}) where {T<:Union{TapeBasedAD,TapeLessAD},V<:Real} = x >= a.val
Base.:>=(a::Adouble{T}, x::V) where {T<:Union{TapeBasedAD,TapeLessAD},V<:Real} = a.val >= x
function Base.:>=(a::Adouble{T}, b::Adouble{T}) where {T<:Union{TapeBasedAD,TapeLessAD}}
    return a.val >= b.val
end

#--------------- Operation: > -------------------
function Base.:>(a::TapeBasedAD, b::TapeBasedAD)
    return ccall((:g_tb_adouble, ADOLC_JLL_PATH), Bool, (TapeBasedAD, TapeBasedAD), a, b)
end
function Base.:>(val::V, b::TapeBasedAD) where {V<:Real}
    return ccall(
        (:g_double_tb_adouble, ADOLC_JLL_PATH),
        Bool,
        (Cdouble, TapeBasedAD),
        Cdouble(val),
        b,
    )
end
function Base.:>(a::TapeBasedAD, val::V) where {V<:Real}
    return ccall(
        (:g_tb_adouble_double, ADOLC_JLL_PATH),
        Bool,
        (TapeBasedAD, Cdouble),
        a,
        Cdouble(val),
    )
end
function Base.:>(a::TapeLessAD, b::TapeLessAD)
    return ccall((:g_tl_adouble, ADOLC_JLL_PATH), Bool, (TapeLessAD, TapeLessAD), a, b)
end
function Base.:>(val::V, b::TapeLessAD) where {V<:Real}
    return ccall(
        (:g_double_tl_adouble, ADOLC_JLL_PATH), Bool, (Cdouble, TapeLessAD), Cdouble(val), b
    )
end
function Base.:>(a::TapeLessAD, val::V) where {V<:Real}
    return ccall(
        (:g_tl_adouble_double, ADOLC_JLL_PATH), Bool, (TapeLessAD, Cdouble), a, Cdouble(val)
    )
end
Base.:>(x::V, a::Adouble{T}) where {T<:Union{TapeBasedAD,TapeLessAD},V<:Real} = x > a.val
Base.:>(a::Adouble{T}, x::V) where {T<:Union{TapeBasedAD,TapeLessAD},V<:Real} = a.val > x
function Base.:>(a::Adouble{T}, b::Adouble{T}) where {T<:Union{TapeBasedAD,TapeLessAD}}
    return a.val > b.val
end

#--------------- Operation: <= -------------------
function Base.:<=(a::TapeBasedAD, b::TapeBasedAD)
    return ccall((:le_tb_adouble, ADOLC_JLL_PATH), Bool, (TapeBasedAD, TapeBasedAD), a, b)
end
function Base.:<=(val::V, b::TapeBasedAD) where {V<:Real}
    return ccall(
        (:le_double_tb_adouble, ADOLC_JLL_PATH),
        Bool,
        (Cdouble, TapeBasedAD),
        Cdouble(val),
        b,
    )
end
function Base.:<=(a::TapeBasedAD, val::V) where {V<:Real}
    return ccall(
        (:le_tb_adouble_double, ADOLC_JLL_PATH),
        Bool,
        (TapeBasedAD, Cdouble),
        a,
        Cdouble(val),
    )
end
function Base.:<=(a::TapeLessAD, b::TapeLessAD)
    return ccall((:le_tl_adouble, ADOLC_JLL_PATH), Bool, (TapeLessAD, TapeLessAD), a, b)
end
function Base.:<=(val::V, b::TapeLessAD) where {V<:Real}
    return ccall(
        (:le_double_tl_adouble, ADOLC_JLL_PATH),
        Bool,
        (Cdouble, TapeLessAD),
        Cdouble(val),
        b,
    )
end
function Base.:<=(a::TapeLessAD, val::V) where {V<:Real}
    return ccall(
        (:le_tl_adouble_double, ADOLC_JLL_PATH),
        Bool,
        (TapeLessAD, Cdouble),
        a,
        Cdouble(val),
    )
end
Base.:<=(x::V, a::Adouble{T}) where {T<:Union{TapeBasedAD,TapeLessAD},V<:Real} = x <= a.val
Base.:<=(a::Adouble{T}, x::V) where {T<:Union{TapeBasedAD,TapeLessAD},V<:Real} = a.val <= x
function Base.:<=(a::Adouble{T}, b::Adouble{T}) where {T<:Union{TapeBasedAD,TapeLessAD}}
    return a.val <= b.val
end

#--------------- Operation: < -------------------
function Base.:<(a::TapeBasedAD, b::TapeBasedAD)
    return ccall((:l_tb_adouble, ADOLC_JLL_PATH), Bool, (TapeBasedAD, TapeBasedAD), a, b)
end
function Base.:<(val::V, b::TapeBasedAD) where {V<:Real}
    return ccall(
        (:l_double_tb_adouble, ADOLC_JLL_PATH),
        Bool,
        (Cdouble, TapeBasedAD),
        Cdouble(val),
        b,
    )
end
function Base.:<(a::TapeBasedAD, val::V) where {V<:Real}
    return ccall(
        (:l_tb_adouble_double, ADOLC_JLL_PATH),
        Bool,
        (TapeBasedAD, Cdouble),
        a,
        Cdouble(val),
    )
end
function Base.:<(a::TapeLessAD, b::TapeLessAD)
    return ccall((:l_tl_adouble, ADOLC_JLL_PATH), Bool, (TapeLessAD, TapeLessAD), a, b)
end
function Base.:<(val::V, b::TapeLessAD) where {V<:Real}
    return ccall(
        (:l_double_tl_adouble, ADOLC_JLL_PATH), Bool, (Cdouble, TapeLessAD), Cdouble(val), b
    )
end
function Base.:<(a::TapeLessAD, val::V) where {V<:Real}
    return ccall(
        (:l_tl_adouble_double, ADOLC_JLL_PATH), Bool, (TapeLessAD, Cdouble), a, Cdouble(val)
    )
end
Base.:<(x::V, a::Adouble{T}) where {T<:Union{TapeBasedAD,TapeLessAD},V<:Real} = x < a.val
Base.:<(a::Adouble{T}, x::V) where {T<:Union{TapeBasedAD,TapeLessAD},V<:Real} = a.val < x
function Base.:<(a::Adouble{T}, b::Adouble{T}) where {T<:Union{TapeBasedAD,TapeLessAD}}
    return a.val < b.val
end

#--------------- Operation: == -------------------

function Base.:(==)(a::TapeBasedAD, b::TapeBasedAD)
    return ccall((:eq_tb_adouble, ADOLC_JLL_PATH), Bool, (TapeBasedAD, TapeBasedAD), a, b)
end
function Base.:(==)(val::V, b::TapeBasedAD) where {V<:Real}
    return ccall(
        (:eq_double_tb_adouble, ADOLC_JLL_PATH),
        Bool,
        (Cdouble, TapeBasedAD),
        Cdouble(val),
        b,
    )
end
function Base.:(==)(a::TapeBasedAD, val::V) where {V<:Real}
    return ccall(
        (:eq_tb_adouble_double, ADOLC_JLL_PATH),
        Bool,
        (TapeBasedAD, Cdouble),
        a,
        Cdouble(val),
    )
end
function Base.:(==)(a::TapeLessAD, b::TapeLessAD)
    return ccall((:eq_tl_adouble, ADOLC_JLL_PATH), Bool, (TapeLessAD, TapeLessAD), a, b)
end
function Base.:(==)(val::V, b::TapeLessAD) where {V<:Real}
    return ccall(
        (:eq_double_tl_adouble, ADOLC_JLL_PATH),
        Bool,
        (Cdouble, TapeLessAD),
        Cdouble(val),
        b,
    )
end
function Base.:(==)(a::TapeLessAD, val::V) where {V<:Real}
    return ccall(
        (:eq_tl_adouble_double, ADOLC_JLL_PATH),
        Bool,
        (TapeLessAD, Cdouble),
        a,
        Cdouble(val),
    )
end
function Base.:(==)(x::V, a::Adouble{T}) where {T<:Union{TapeBasedAD,TapeLessAD},V<:Real}
    return x == a.val
end
function Base.:(==)(a::Adouble{T}, x::V) where {T<:Union{TapeBasedAD,TapeLessAD},V<:Real}
    return a.val == x
end
function Base.:(==)(a::Adouble{T}, b::Adouble{T}) where {T<:Union{TapeBasedAD,TapeLessAD}}
    return a.val == b.val
end

#-------------- unary Functions
function Base.abs(a::TapeBasedAD)
    return ccall((:tb_abs, ADOLC_JLL_PATH), TapeBasedAD, (TapeBasedAD,), a)
end
function Base.abs(a::TapeLessAD)
    return ccall((:tl_abs, ADOLC_JLL_PATH), TapeLessAD, (TapeLessAD,), a)
end
Base.abs(a::Adouble{T}) where {T<:Union{TapeBasedAD,TapeLessAD}} = Adouble{T}(abs(a.val))

function Base.sqrt(a::TapeBasedAD)
    return ccall((:tb_sqrt, ADOLC_JLL_PATH), TapeBasedAD, (TapeBasedAD,), a)
end
function Base.sqrt(a::TapeLessAD)
    return ccall((:tl_sqrt, ADOLC_JLL_PATH), TapeLessAD, (TapeLessAD,), a)
end
Base.sqrt(a::Adouble{T}) where {T<:Union{TapeBasedAD,TapeLessAD}} = Adouble{T}(sqrt(a.val))

function Base.exp(a::TapeBasedAD)
    return ccall((:tb_exp, ADOLC_JLL_PATH), TapeBasedAD, (TapeBasedAD,), a)
end
function Base.exp(a::TapeLessAD)
    return ccall((:tl_exp, ADOLC_JLL_PATH), TapeLessAD, (TapeLessAD,), a)
end
Base.exp(a::Adouble{T}) where {T<:Union{TapeBasedAD,TapeLessAD}} = Adouble{T}(exp(a.val))

function Base.log(a::TapeBasedAD)
    return ccall((:tb_log, ADOLC_JLL_PATH), TapeBasedAD, (TapeBasedAD,), a)
end
function Base.log(a::TapeLessAD)
    return ccall((:tl_log, ADOLC_JLL_PATH), TapeLessAD, (TapeLessAD,), a)
end
Base.log(a::Adouble{T}) where {T<:Union{TapeBasedAD,TapeLessAD}} = Adouble{T}(log(a.val))

function Base.log10(a::TapeBasedAD)
    return ccall((:tb_log10, ADOLC_JLL_PATH), TapeBasedAD, (TapeBasedAD,), a)
end
function Base.log10(a::TapeLessAD)
    return ccall((:tl_log10, ADOLC_JLL_PATH), TapeLessAD, (TapeLessAD,), a)
end
function Base.log10(a::Adouble{T}) where {T<:Union{TapeBasedAD,TapeLessAD}}
    return Adouble{T}(log10(a.val))
end

function Base.sin(a::TapeBasedAD)
    return ccall((:tb_sin, ADOLC_JLL_PATH), TapeBasedAD, (TapeBasedAD,), a)
end
function Base.sin(a::TapeLessAD)
    return ccall((:tl_sin, ADOLC_JLL_PATH), TapeLessAD, (TapeLessAD,), a)
end
Base.sin(a::Adouble{T}) where {T<:Union{TapeBasedAD,TapeLessAD}} = Adouble{T}(sin(a.val))

function Base.cos(a::TapeBasedAD)
    return ccall((:tb_cos, ADOLC_JLL_PATH), TapeBasedAD, (TapeBasedAD,), a)
end
function Base.cos(a::TapeLessAD)
    return ccall((:tl_cos, ADOLC_JLL_PATH), TapeLessAD, (TapeLessAD,), a)
end
Base.cos(a::Adouble{T}) where {T<:Union{TapeBasedAD,TapeLessAD}} = Adouble{T}(cos(a.val))

function Base.tan(a::TapeBasedAD)
    return ccall((:tb_tan, ADOLC_JLL_PATH), TapeBasedAD, (TapeBasedAD,), a)
end
function Base.tan(a::TapeLessAD)
    return ccall((:tl_tan, ADOLC_JLL_PATH), TapeLessAD, (TapeLessAD,), a)
end
Base.tan(a::Adouble{T}) where {T<:Union{TapeBasedAD,TapeLessAD}} = Adouble{T}(tan(a.val))

function Base.asin(a::TapeBasedAD)
    return ccall((:tb_asin, ADOLC_JLL_PATH), TapeBasedAD, (TapeBasedAD,), a)
end
function Base.asin(a::TapeLessAD)
    return ccall((:tl_asin, ADOLC_JLL_PATH), TapeLessAD, (TapeLessAD,), a)
end
Base.asin(a::Adouble{T}) where {T<:Union{TapeBasedAD,TapeLessAD}} = Adouble{T}(asin(a.val))

function Base.acos(a::TapeBasedAD)
    return ccall((:tb_acos, ADOLC_JLL_PATH), TapeBasedAD, (TapeBasedAD,), a)
end
function Base.acos(a::TapeLessAD)
    return ccall((:tl_acos, ADOLC_JLL_PATH), TapeLessAD, (TapeLessAD,), a)
end
Base.acos(a::Adouble{T}) where {T<:Union{TapeBasedAD,TapeLessAD}} = Adouble{T}(acos(a.val))

function Base.atan(a::TapeBasedAD)
    return ccall((:tb_atan, ADOLC_JLL_PATH), TapeBasedAD, (TapeBasedAD,), a)
end
function Base.atan(a::TapeLessAD)
    return ccall((:tl_atan, ADOLC_JLL_PATH), TapeLessAD, (TapeLessAD,), a)
end
Base.atan(a::Adouble{T}) where {T<:Union{TapeBasedAD,TapeLessAD}} = Adouble{T}(atan(a.val))

function Base.sinh(a::TapeBasedAD)
    return ccall((:tb_sinh, ADOLC_JLL_PATH), TapeBasedAD, (TapeBasedAD,), a)
end
function Base.sinh(a::TapeLessAD)
    return ccall((:tl_sinh, ADOLC_JLL_PATH), TapeLessAD, (TapeLessAD,), a)
end
Base.sinh(a::Adouble{T}) where {T<:Union{TapeBasedAD,TapeLessAD}} = Adouble{T}(sinh(a.val))

function Base.cosh(a::TapeBasedAD)
    return ccall((:tb_cosh, ADOLC_JLL_PATH), TapeBasedAD, (TapeBasedAD,), a)
end
function Base.cosh(a::TapeLessAD)
    return ccall((:tl_cosh, ADOLC_JLL_PATH), TapeLessAD, (TapeLessAD,), a)
end
Base.cosh(a::Adouble{T}) where {T<:Union{TapeBasedAD,TapeLessAD}} = Adouble{T}(cosh(a.val))

function Base.tanh(a::TapeBasedAD)
    return ccall((:tb_tanh, ADOLC_JLL_PATH), TapeBasedAD, (TapeBasedAD,), a)
end
function Base.tanh(a::TapeLessAD)
    return ccall((:tl_tanh, ADOLC_JLL_PATH), TapeLessAD, (TapeLessAD,), a)
end
Base.tanh(a::Adouble{T}) where {T<:Union{TapeBasedAD,TapeLessAD}} = Adouble{T}(tanh(a.val))

function Base.asinh(a::TapeBasedAD)
    return ccall((:tb_asinh, ADOLC_JLL_PATH), TapeBasedAD, (TapeBasedAD,), a)
end
function Base.asinh(a::TapeLessAD)
    return ccall((:tl_asinh, ADOLC_JLL_PATH), TapeLessAD, (TapeLessAD,), a)
end
function Base.asinh(a::Adouble{T}) where {T<:Union{TapeBasedAD,TapeLessAD}}
    return Adouble{T}(asinh(a.val))
end

function Base.acosh(a::TapeBasedAD)
    return ccall((:tb_acosh, ADOLC_JLL_PATH), TapeBasedAD, (TapeBasedAD,), a)
end
function Base.acosh(a::TapeLessAD)
    return ccall((:tl_acosh, ADOLC_JLL_PATH), TapeLessAD, (TapeLessAD,), a)
end
function Base.acosh(a::Adouble{T}) where {T<:Union{TapeBasedAD,TapeLessAD}}
    return Adouble{T}(acosh(a.val))
end

function Base.atanh(a::TapeBasedAD)
    return ccall((:tb_atanh, ADOLC_JLL_PATH), TapeBasedAD, (TapeBasedAD,), a)
end
function Base.atanh(a::TapeLessAD)
    return ccall((:tl_atanh, ADOLC_JLL_PATH), TapeLessAD, (TapeLessAD,), a)
end
function Base.atanh(a::Adouble{T}) where {T<:Union{TapeBasedAD,TapeLessAD}}
    return Adouble{T}(atanh(a.val))
end

function Base.ceil(a::TapeBasedAD)
    return ccall((:tb_ceil, ADOLC_JLL_PATH), TapeBasedAD, (TapeBasedAD,), a)
end
function Base.ceil(a::TapeLessAD)
    return ccall((:tl_ceil, ADOLC_JLL_PATH), TapeLessAD, (TapeLessAD,), a)
end
Base.ceil(a::Adouble{T}) where {T<:Union{TapeBasedAD,TapeLessAD}} = Adouble{T}(ceil(a.val))

function Base.floor(a::TapeBasedAD)
    return ccall((:tb_floor, ADOLC_JLL_PATH), TapeBasedAD, (TapeBasedAD,), a)
end
function Base.floor(a::TapeLessAD)
    return ccall((:tl_floor, ADOLC_JLL_PATH), TapeLessAD, (TapeLessAD,), a)
end
function Base.floor(a::Adouble{T}) where {T<:Union{TapeBasedAD,TapeLessAD}}
    return Adouble{T}(floor(a.val))
end

function Base.ldexp(a::TapeBasedAD, n::Integer)
    return ccall((:tb_ldexp, ADOLC_JLL_PATH), TapeBasedAD, (TapeBasedAD, Cint), a, Cint(n))
end
function Base.ldexp(a::TapeLessAD, n::Integer)
    return ccall((:tl_ldexp, ADOLC_JLL_PATH), TapeLessAD, (TapeLessAD, Cint), a, Cint(n))
end
function Base.ldexp(a::Adouble{T}, n::Integer) where {T<:Union{TapeBasedAD,TapeLessAD}}
    return Adouble{T}(ldexp(a.val, n))
end

"""
function Base.frexp(a::Adouble{T}, n::Cint) where {T<:Union{TapeBasedAD,TapeLessAD}}
    return Adouble{T}(frexp(a.val, n); adouble=true)
end
"""

function erf(a::TapeBasedAD)
    return ccall((:tb_erf, ADOLC_JLL_PATH), TapeBasedAD, (TapeBasedAD,), a)
end
function erf(a::TapeLessAD)
    return ccall((:tl_erf, ADOLC_JLL_PATH), TapeLessAD, (TapeLessAD,), a)
end
erf(a::Adouble{T}) where {T<:Union{TapeBasedAD,TapeLessAD}} = Adouble{T}(erf(a.val))
SpecialFunctions.erf(a::Adouble{T}) where {T<:Union{TapeBasedAD,TapeLessAD}} = erf(a)
erf(x::Cdouble) = SpecialFunctions.erf(x)
SpecialFunctions.erfc(a::Adouble{T}) where {T<:Union{TapeBasedAD,TapeLessAD}} = 1.0 - erf(a)

Base.eps(::Type{Adouble{T}}) where {T<:Union{TapeBasedAD,TapeLessAD}} = eps(Cdouble)
