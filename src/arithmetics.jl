
# ----- only Adouble{TbAlloc} 

function check_input(a, x)
    if length(a) > length(x)
        throw("DimensionMismatch: Length of a should be less equal lenght of x!")
    end
    if length(a) < length(x)
        println(
            "Warning: length of a is less than length of x. Initialize the derivative values 
                up to length of a!",
        )
    end
end

# convient inits for vector of independant and dependant 
function Base.:<<(a::Vector{Adouble{TbAlloc}}, x::Vector{Float64})
    check_input(a, x)
    for i in eachindex(x)
        a[i].val << x[i]
    end
end

function Base.:>>(a::Vector{Adouble{TbAlloc}}, x::Vector{Float64})
    check_input(a, x)
    for i in eachindex(x)
        a[i].val >> x[i]
    end
    return x
end

function Base.:>>(a::Adouble{TbAlloc}, x::Vector{Float64})
    if length(x) != 1
        throw("DimensionMismatch: Length of x ($x) should be 1!")
    end
    return [a] >> x
end

function Base.:>>(a::Vector{Adouble{TbAlloc}}, x::Float64)
    if length(a) != 1
        throw("DimensionMismatch: Length of a ($a) should be 1!")
    end
    return a >> [x]
end

function Base.:>>(a::Adouble{TbAlloc}, x::Float64)
    return a.val >> x
end

function Base.:<<(a::Adouble{TbAlloc}, x::Float64)
    return a.val << x
end

#--------------- Operation: * -------------------

function Base.:*(
    a::Adouble{T}, x::AbstractVector{Float64}
) where {T<:Union{TbAlloc,TlAlloc}}
    return map((x_i) -> a * x_i, x)
end

function Base.:*(a::Adouble{T}, x::V) where {T<:Union{TbAlloc,TlAlloc},V<:Real}
    return Adouble{T}(a.val * Float64(x))
end
function Base.:*(x::V, a::Adouble{T}) where {T<:Union{TbAlloc,TlAlloc},V<:Real}
    return Adouble{T}(Float64(x) * a.val)
end
function Base.:*(a::Adouble{T}, b::Adouble{T}) where {T<:Union{TbAlloc,TlAlloc}}
    return Adouble{T}(a.val * b.val)
end

function Base.:*(a::Adouble{T}, x::Bool) where {T<:Union{TbAlloc,TlAlloc}}
    return Adouble{T}(a.val * Float64(x))
end
function Base.:*(x::Bool, a::Adouble{T}) where {T<:Union{TbAlloc,TlAlloc}}
    return Adouble{T}(Float64(x) * a.val)
end

#--------------- Operation: * -------------------

function Base.:+(x::V, a::Adouble{T}) where {T<:Union{TbAlloc,TlAlloc},V<:Real}
    return Adouble{T}(Float64(x) + a.val)
end
function Base.:+(a::Adouble{T}, x::V) where {T<:Union{TbAlloc,TlAlloc},V<:Real}
    return Adouble{T}(a.val + Float64(x))
end
function Base.:+(a::Adouble{T}, b::Adouble{T}) where {T<:Union{TbAlloc,TlAlloc}}
    return Adouble{T}(a.val + b.val)
end

#--------------- Operation: - -------------------
function Base.:-(x::V, a::Adouble{T}) where {T<:Union{TbAlloc,TlAlloc},V<:Real}
    return Adouble{T}(Float64(x) - a.val)
end
function Base.:-(a::Adouble{T}, x::V) where {T<:Union{TbAlloc,TlAlloc},V<:Real}
    return Adouble{T}(a.val - Float64(x))
end
function Base.:-(a::Adouble{T}, b::Adouble{T}) where {T<:Union{TbAlloc,TlAlloc}}
    return Adouble{T}(a.val - b.val)
end

Base.:-(a::Adouble{T}) where {T<:Union{TbAlloc,TlAlloc}} = (-1) * a

#--------------- Operation: / -------------------

function Base.:/(x::V, a::Adouble{T}) where {T<:Union{TbAlloc,TlAlloc},V<:Real}
    return Adouble{T}(Float64(x) / a.val)
end
function Base.:/(a::Adouble{T}, x::V) where {T<:Union{TbAlloc,TlAlloc},V<:Real}
    return Adouble{T}(a.val / Float64(x))
end
function Base.:/(a::Adouble{T}, b::Adouble{T}) where {T<:Union{TbAlloc,TlAlloc}}
    return Adouble{T}(a.val / b.val)
end

#--------------- Operation: >= -------------------
Base.:>=(x::V, a::Adouble{T}) where {T<:Union{TbAlloc,TlAlloc},V<:Real} = x >= a.val
Base.:>=(a::Adouble{T}, x::V) where {T<:Union{TbAlloc,TlAlloc},V<:Real} = a.val >= x
Base.:>=(a::Adouble{T}, b::Adouble{T}) where {T<:Union{TbAlloc,TlAlloc}} = a.val >= b.val

#--------------- Operation: > -------------------
Base.:>(x::V, a::Adouble{T}) where {T<:Union{TbAlloc,TlAlloc},V<:Real} = x > a.val
Base.:>(a::Adouble{T}, x::V) where {T<:Union{TbAlloc,TlAlloc},V<:Real} = a.val > x
Base.:>(a::Adouble{T}, b::Adouble{T}) where {T<:Union{TbAlloc,TlAlloc}} = a.val > b.val

#--------------- Operation: <= -------------------
Base.:<=(x::V, a::Adouble{T}) where {T<:Union{TbAlloc,TlAlloc},V<:Real} = x <= a.val
Base.:<=(a::Adouble{T}, x::V) where {T<:Union{TbAlloc,TlAlloc},V<:Real} = a.val <= x
Base.:<=(a::Adouble{T}, b::Adouble{T}) where {T<:Union{TbAlloc,TlAlloc}} = a.val <= b.val

#--------------- Operation: < -------------------
Base.:<(x::V, a::Adouble{T}) where {T<:Union{TbAlloc,TlAlloc},V<:Real} = x < a.val
Base.:<(a::Adouble{T}, x::V) where {T<:Union{TbAlloc,TlAlloc},V<:Real} = a.val < x
Base.:<(a::Adouble{T}, b::Adouble{T}) where {T<:Union{TbAlloc,TlAlloc}} = a.val < b.val

#--------------- Operation: == -------------------
Base.:(==)(x::V, a::Adouble{T}) where {T<:Union{TbAlloc,TlAlloc},V<:Real} = x == a.val
Base.:(==)(a::Adouble{T}, x::V) where {T<:Union{TbAlloc,TlAlloc},V<:Real} = a.val == x
Base.:(==)(a::Adouble{T}, b::Adouble{T}) where {T<:Union{TbAlloc,TlAlloc}} = a.val == b.val

#-------------- Functions: max -----------------

function Base.max(x::V, a::Adouble{T}) where {T<:Union{TbAlloc,TlAlloc},V<:Real}
    return Adouble{T}(max(a.val, x))
end
function Base.max(a::Adouble{T}, x::V) where {T<:Union{TbAlloc,TlAlloc},V<:Real}
    return Adouble{T}(max(x, a.val))
end
function Base.max(a::Adouble{T}, b::Adouble{T}) where {T<:Union{TbAlloc,TlAlloc}}
    return Adouble{T}(max(a.val, b.val))
end

# ------------ Functions: min -----------

function Base.min(x::V, a::Adouble{T}) where {T<:Union{TbAlloc,TlAlloc},V<:Real}
    return Adouble{T}(min(a.val, x))
end
function Base.min(a::Adouble{T}, x::V) where {T<:Union{TbAlloc,TlAlloc},V<:Real}
    return Adouble{T}(min(x, a.val))
end
function Base.min(a::Adouble{T}, b::Adouble{T}) where {T<:Union{TbAlloc,TlAlloc}}
    return Adouble{T}(min(a.val, b.val))
end

#-------------- unary Functions

Base.abs(a::Adouble{T}) where {T<:Union{TbAlloc,TlAlloc}} = Adouble{T}(abs(a.val))
Base.sqrt(a::Adouble{T}) where {T<:Union{TbAlloc,TlAlloc}} = Adouble{T}(sqrt(a.val))
Base.exp(a::Adouble{T}) where {T<:Union{TbAlloc,TlAlloc}} = Adouble{T}(exp(a.val))
Base.log(a::Adouble{T}) where {T<:Union{TbAlloc,TlAlloc}} = Adouble{T}(log(a.val))

Base.sin(a::Adouble{T}) where {T<:Union{TbAlloc,TlAlloc}} = Adouble{T}(sin(a.val))
Base.cos(a::Adouble{T}) where {T<:Union{TbAlloc,TlAlloc}} = Adouble{T}(cos(a.val))
Base.tan(a::Adouble{T}) where {T<:Union{TbAlloc,TlAlloc}} = Adouble{T}(tan(a.val))

Base.asin(a::Adouble{T}) where {T<:Union{TbAlloc,TlAlloc}} = Adouble{T}(asin(a.val))
Base.acos(a::Adouble{T}) where {T<:Union{TbAlloc,TlAlloc}} = Adouble{T}(acos(a.val))
Base.atan(a::Adouble{T}) where {T<:Union{TbAlloc,TlAlloc}} = Adouble{T}(atan(a.val))

Base.log10(a::Adouble{T}) where {T<:Union{TbAlloc,TlAlloc}} = Adouble{T}(log10(a.val))

Base.sinh(a::Adouble{T}) where {T<:Union{TbAlloc,TlAlloc}} = Adouble{T}(sinh(a.val))
Base.cosh(a::Adouble{T}) where {T<:Union{TbAlloc,TlAlloc}} = Adouble{T}(cosh(a.val))
Base.tanh(a::Adouble{T}) where {T<:Union{TbAlloc,TlAlloc}} = Adouble{T}(tanh(a.val))

Base.asinh(a::Adouble{T}) where {T<:Union{TbAlloc,TlAlloc}} = Adouble{T}(asinh(a.val))
Base.acosh(a::Adouble{T}) where {T<:Union{TbAlloc,TlAlloc}} = Adouble{T}(acosh(a.val))
Base.atanh(a::Adouble{T}) where {T<:Union{TbAlloc,TlAlloc}} = Adouble{T}(atanh(a.val))

Base.ceil(a::Adouble{T}) where {T<:Union{TbAlloc,TlAlloc}} = Adouble{T}(ceil(a.val))
Base.floor(a::Adouble{T}) where {T<:Union{TbAlloc,TlAlloc}} = Adouble{T}(floor(a.val))

Base.ldexp(a::Adouble{T}) where {T<:Union{TbAlloc,TlAlloc}} = Adouble{T}(ldexp(a.val))
Base.frexp(a::Adouble{T}) where {T<:Union{TbAlloc,TlAlloc}} = Adouble{T}(frexp(a.val))

erf(a::Adouble{TbAlloc}) = Adouble{TbAlloc}(TbadoubleModule.erf(a.val))
erf(a::Adouble{TlAlloc}) = Adouble{TlAlloc}(TladoubleModule.erf(a.val))

Base.eps(::Type{Adouble{T}}) where {T<:Union{TbAlloc,TlAlloc}} = eps(Float64)
#### SpecialFunctions

SpecialFunctions.erf(a::Adouble{T}) where {T<:Union{TbAlloc,TlAlloc}} = erf(a)
SpecialFunctions.erfc(a::Adouble{T}) where {T<:Union{TbAlloc,TlAlloc}} = 1.0 - erf(a)

##############################################################

#-------- utilities for type handling ----------
function Base.promote(x::V, y::Adouble{T}) where {T<:Union{TbAlloc,TlAlloc},V<:Real}
    return Adouble{T}(Float64(x), false)
end

# not sure if needed 
Base.promote(x::Adouble{T}, y::Adouble{T}) where {T<:Union{TbAlloc,TlAlloc}} = x

function Base.promote_rule(
    ::Type{Adouble{T}}, ::Type{V}
) where {T<:Union{TbAlloc,TlAlloc},V<:Real}
    return Adouble{T}
end

# since every operation where an argument is a adouble have to return a adouble
function Base.promote_op(
    f, ::Type{Adouble{T}}, ::Type{Adouble{T}}
) where {T<:Union{TbAlloc,TlAlloc}}
    return Adouble{T}
end
function Base.promote_op(
    f, ::Type{V}, ::Type{Adouble{T}}
) where {T<:Union{TbAlloc,TlAlloc},V<:Real}
    return Adouble{T}
end
function Base.promote_op(
    f, ::Type{Adouble{T}}, ::Type{V}
) where {T<:Union{TbAlloc,TlAlloc},V<:Real}
    return Adouble{T}
end

function Base.convert(::Type{Adouble{T}}, x::V) where {T<:Union{TbAlloc,TlAlloc},V<:Real}
    return Adouble{T}(Float64(x), false)
end

# this is called e.g. when creating a vector of Adouble{T}s ... not sure why
Base.convert(::Type{Adouble{T}}, x::Adouble{T}) where {T<:Union{TbAlloc,TlAlloc}} = x
