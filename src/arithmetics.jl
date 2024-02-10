

#--------------- Operation: * -------------------

function Base.:*(a::T, x::AbstractVector{Float64}) where T <: Union{TbAlloc, TlAlloc}
    return map((x_i)->a*x_i, x)
end

Base.:*(a::T, x::AbstractVector{Float64}) where T <: Union{Tb, Tl} = a.val * x



Base.:*(a::Adouble{T}, x::V) where {T <: Union{TbAlloc, TlAlloc}, V<:Real}  = Adouble{T}(a.val * Float64(x))
Base.:*(x::V, a::Adouble{T}) where {T <: Union{TbAlloc, TlAlloc}, V<:Real} = Adouble{T}(Float64(x) * a.val)
Base.:*(a::Adouble{T}, b::Adouble{T}) where T <: Union{TbAlloc, TlAlloc} = Adouble{T}(a.val * b.val)

Base.:*(a::Adouble{T}, x::Bool) where T <: Union{TbAlloc, TlAlloc} = Adouble{T}(a.val * Float64(x))
Base.:*(x::Bool, a::Adouble{T}) where T <: Union{TbAlloc, TlAlloc} = Adouble{T}(Float64(x) * a.val)



#--------------- Operation: * -------------------

Base.:+(x::V, a::Adouble{T}) where {T <: Union{TbAlloc, TlAlloc}, V<:Real} = Adouble{T}(Float64(x) + a.val)
Base.:+(a::Adouble{T}, x::V) where {T <: Union{TbAlloc, TlAlloc}, V<:Real} = Adouble{T}(a.val + Float64(x))
Base.:+(a::Adouble{T}, b::Adouble{T}) where T <: Union{TbAlloc, TlAlloc} = Adouble{T}(a.val + b.val)


#--------------- Operation: - -------------------
Base.:-(x::V, a::Adouble{T}) where {T <: Union{TbAlloc, TlAlloc}, V<:Real} = Adouble{T}(Float64(x) - a.val)
Base.:-(a::Adouble{T}, x::V) where {T <: Union{TbAlloc, TlAlloc}, V<:Real} = Adouble{T}(a.val - Float64(x))
Base.:-(a::Adouble{T}, b::Adouble{T}) where T <: Union{TbAlloc, TlAlloc} = Adouble{T}(a.val - b.val)

Base.:-(a::Adouble{T}) where T <: Union{TbAlloc, TlAlloc} = (-1) * a


#--------------- Operation: / -------------------

Base.:/(x::V, a::Adouble{T}) where {T <: Union{TbAlloc, TlAlloc}, V<:Real} = Adouble{T}(Float64(x) / a.val)
Base.:/(a::Adouble{T}, x::V) where {T <: Union{TbAlloc, TlAlloc}, V<:Real} = Adouble{T}(a.val / Float64(x))
Base.:/(a::Adouble{T}, b::Adouble{T}) where T <: Union{TbAlloc, TlAlloc} = Adouble{T}(a.val / b.val)



#--------------- Operation: >= -------------------
Base.:>=(x::V, a::Adouble{T}) where {T <: Union{TbAlloc, TlAlloc}, V<:Real} = x >= a.val
Base.:>=(a::Adouble{T}, x::V) where {T <: Union{TbAlloc, TlAlloc}, V<:Real} = a.val >= x
Base.:>=(a::Adouble{T}, b::Adouble{T}) where T <: Union{TbAlloc, TlAlloc} = a.val >= b.val

#--------------- Operation: > -------------------
Base.:>(x::V, a::Adouble{T}) where {T <: Union{TbAlloc, TlAlloc}, V<:Real} = x > a.val
Base.:>(a::Adouble{T}, x::V) where {T <: Union{TbAlloc, TlAlloc}, V<:Real}= a.val > x
Base.:>(a::Adouble{T}, b::Adouble{T}) where T <: Union{TbAlloc, TlAlloc} = a.val > b.val

#--------------- Operation: <= -------------------
Base.:<=(x::V, a::Adouble{T}) where {T <: Union{TbAlloc, TlAlloc}, V<:Real} = x <= a.val
Base.:<=(a::Adouble{T}, x::V) where {T <: Union{TbAlloc, TlAlloc}, V<:Real} = a.val <= x
Base.:<=(a::Adouble{T}, b::Adouble{T}) where T <: Union{TbAlloc, TlAlloc} = a.val <= b.val


#--------------- Operation: < -------------------
Base.:<(x::V, a::Adouble{T}) where {T <: Union{TbAlloc, TlAlloc}, V<:Real} = x < a.val
Base.:<(a::Adouble{T}, x::V) where {T <: Union{TbAlloc, TlAlloc}, V<:Real} = a.val < x
Base.:<(a::Adouble{T}, b::Adouble{T}) where T <: Union{TbAlloc, TlAlloc} = a.val < b.val


#--------------- Operation: == -------------------
Base.:(==)(x::V, a::Adouble{T}) where {T <: Union{TbAlloc, TlAlloc}, V<:Real} = x == a.val
Base.:(==)(a::Adouble{T}, x::V) where {T <: Union{TbAlloc, TlAlloc}, V<:Real} = a.val == x
Base.:(==)(a::Adouble{T}, b::Adouble{T}) where T <: Union{TbAlloc, TlAlloc} = a.val == b.val


#-------------- Functions: max -----------------

Base.max(x::V, a::Adouble{T}) where {T <: Union{TbAlloc, TlAlloc}, V<:Real} = Adouble{T}(max(a.val, x))
Base.max(a::Adouble{T}, x::V) where {T <: Union{TbAlloc, TlAlloc}, V<:Real}= Adouble{T}(max(x, a.val))
Base.max(a::Adouble{T}, b::Adouble{T}) where T <: Union{TbAlloc, TlAlloc} = Adouble{T}(max(a.val, b.val))

# ------------ Functions: min -----------

Base.min(x::V, a::Adouble{T}) where {T <: Union{TbAlloc, TlAlloc}, V<:Real} = Adouble{T}(min(a.val, x))
Base.min(a::Adouble{T}, x::V) where {T <: Union{TbAlloc, TlAlloc}, V<:Real} = Adouble{T}(min(x, a.val))
Base.min(a::Adouble{T}, b::Adouble{T}) where T <: Union{TbAlloc, TlAlloc} = Adouble{T}(min(a.val, b.val))




#-------------- unary Functions

Base.abs(a::Adouble{T}) where T <: Union{TbAlloc, TlAlloc} = Adouble{T}(abs(a.val))
Base.sqrt(a::Adouble{T}) where T <: Union{TbAlloc, TlAlloc} = Adouble{T}(sqrt(a.val))
Base.exp(a::Adouble{T}) where T <: Union{TbAlloc, TlAlloc} = Adouble{T}(exp(a.val))
Base.log(a::Adouble{T}) where T <: Union{TbAlloc, TlAlloc} = Adouble{T}(log(a.val))

Base.sin(a::Adouble{T}) where T <: Union{TbAlloc, TlAlloc} = Adouble{T}(sin(a.val))
Base.cos(a::Adouble{T}) where T <: Union{TbAlloc, TlAlloc} = Adouble{T}(cos(a.val))
Base.tan(a::Adouble{T}) where T <: Union{TbAlloc, TlAlloc} = Adouble{T}(tan(a.val))

Base.asin(a::Adouble{T}) where T <: Union{TbAlloc, TlAlloc} = Adouble{T}(asin(a.val))
Base.acos(a::Adouble{T}) where T <: Union{TbAlloc, TlAlloc} = Adouble{T}(acos(a.val))
Base.atan(a::Adouble{T}) where T <: Union{TbAlloc, TlAlloc} = Adouble{T}(atan(a.val))

Base.log10(a::Adouble{T}) where T <: Union{TbAlloc, TlAlloc} = Adouble{T}(log10(a.val))

Base.sinh(a::Adouble{T}) where T <: Union{TbAlloc, TlAlloc} = Adouble{T}(sinh(a.val))
Base.cosh(a::Adouble{T}) where T <: Union{TbAlloc, TlAlloc} = Adouble{T}(cosh(a.val))
Base.tanh(a::Adouble{T}) where T <: Union{TbAlloc, TlAlloc} = Adouble{T}(tanh(a.val))

Base.asinh(a::Adouble{T}) where T <: Union{TbAlloc, TlAlloc} = Adouble{T}(asinh(a.val))
Base.acosh(a::Adouble{T}) where T <: Union{TbAlloc, TlAlloc} = Adouble{T}(acosh(a.val))
Base.atanh(a::Adouble{T}) where T <: Union{TbAlloc, TlAlloc} = Adouble{T}(atanh(a.val))

Base.ceil(a::Adouble{T}) where T <: Union{TbAlloc, TlAlloc} = Adouble{T}(ceil(a.val))
Base.floor(a::Adouble{T}) where T <: Union{TbAlloc, TlAlloc} = Adouble{T}(floor(a.val))



Base.ldexp(a::Adouble{T}) where T <: Union{TbAlloc, TlAlloc} = Adouble{T}(ldexp(a.val))
Base.frexp(a::Adouble{T}) where T <: Union{TbAlloc, TlAlloc} = Adouble{T}(frexp(a.val))


erf(a::Adouble{T}) where T <: Union{TbAlloc, TlAlloc} = Adouble{T}(erf(a.val))

##############################################################


#-------- utilities for type handling ----------
Base.promote(x::V, y::Adouble{T}) where {T <: Union{TbAlloc, TlAlloc}, V <: Real} = Adouble{T}(Float64(x), false)

# not sure if needed 
Base.promote(x::Adouble{T}, y::Adouble{T}) where T <: Union{TbAlloc, TlAlloc} = x

Base.promote_rule(::Type{Adouble{T}}, ::Type{V}) where {T <: Union{TbAlloc, TlAlloc}, V <: Real} = Adouble{T}

# since every operation where an argument is a adouble have to return a adouble
Base.promote_op(f, ::Type{Adouble{T}}, ::Type{Adouble{T}}) where T <: Union{TbAlloc, TlAlloc} = Adouble{T}
Base.promote_op(f, ::Type{V}, ::Type{Adouble{T}}) where {T <: Union{TbAlloc, TlAlloc}, V <: Real} = Adouble{T}
Base.promote_op(f, ::Type{Adouble{T}}, ::Type{V}) where {T <: Union{TbAlloc, TlAlloc}, V <: Real} = Adouble{T}



Base.convert(::Type{Adouble{T}}, x::V) where {T <: Union{TbAlloc, TlAlloc}, V <: Real} = Adouble{T}(Float64(x), false)


# this is called e.g. when creating a vector of Adouble{T}s ... not sure why
Base.convert(::Type{Adouble{T}}, x::Adouble{T}) where T <: Union{TbAlloc, TlAlloc} = x

