function Base.:<<(a::Vector{Adouble{TbAlloc}}, x::Vector{Cdouble})
    @assert length(a) == length(x)
    for i in eachindex(x)
        a[i].val << x[i]
    end
end
function Base.:<<(a::Adouble{TbAlloc}, x::Cdouble)
    return a.val << x
end
function Base.:>>(b::Vector{Adouble{TbAlloc}}, y::Vector{Cdouble})
    @assert length(b) == length(y)
    for i in eachindex(y)
        b[i].val >> y[i]
        y[i] = TbadoubleModule.getValue(b[i].val)
    end
    return y
end
function Base.:>>(b::Adouble{TbAlloc}, y::Cdouble)
    b.val >> y
    return y = TbadoubleModule.getValue(b.val)
end
function Base.:>>(b::Adouble{TbAlloc}, y::Vector{Cdouble})
    @assert(length(y) == 1)
    b.val >> y[1]
    return y[1] = TbadoubleModule.getValue(b.val)
end

#--------------- Operation: * -------------------

function Base.:*(
    a::Adouble{T}, x::AbstractVector{Cdouble}
) where {T<:Union{TbAlloc,TlAlloc}}
    return a .* x
end

function Base.:*(a::Adouble{T}, x::V) where {T<:Union{TbAlloc,TlAlloc},V<:Real}
    return Adouble{T}(a.val * Cdouble(x))
end
function Base.:*(x::V, a::Adouble{T}) where {T<:Union{TbAlloc,TlAlloc},V<:Real}
    return Adouble{T}(Cdouble(x) * a.val)
end
function Base.:*(a::Adouble{T}, b::Adouble{T}) where {T<:Union{TbAlloc,TlAlloc}}
    return Adouble{T}(a.val * b.val)
end

function Base.:*(a::Adouble{T}, x::Bool) where {T<:Union{TbAlloc,TlAlloc}}
    return Adouble{T}(a.val * Cdouble(x))
end
function Base.:*(x::Bool, a::Adouble{T}) where {T<:Union{TbAlloc,TlAlloc}}
    return Adouble{T}(Cdouble(x) * a.val)
end

#--------------- Operation: * -------------------

function Base.:+(x::V, a::Adouble{T}) where {T<:Union{TbAlloc,TlAlloc},V<:Real}
    return Adouble{T}(Cdouble(x) + a.val)
end
function Base.:+(a::Adouble{T}, x::V) where {T<:Union{TbAlloc,TlAlloc},V<:Real}
    return Adouble{T}(a.val + Cdouble(x))
end
function Base.:+(a::Adouble{T}, b::Adouble{T}) where {T<:Union{TbAlloc,TlAlloc}}
    return Adouble{T}(a.val + b.val)
end

#--------------- Operation: - -------------------
function Base.:-(x::V, a::Adouble{T}) where {T<:Union{TbAlloc,TlAlloc},V<:Real}
    return Adouble{T}(Cdouble(x) - a.val)
end
function Base.:-(a::Adouble{T}, x::V) where {T<:Union{TbAlloc,TlAlloc},V<:Real}
    return Adouble{T}(a.val - Cdouble(x))
end
function Base.:-(a::Adouble{T}, b::Adouble{T}) where {T<:Union{TbAlloc,TlAlloc}}
    return Adouble{T}(a.val - b.val)
end

Base.:-(a::Adouble{T}) where {T<:Union{TbAlloc,TlAlloc}} = (-1) * a

#--------------- Operation: / -------------------

function Base.:/(x::V, a::Adouble{T}) where {T<:Union{TbAlloc,TlAlloc},V<:Real}
    return Adouble{T}(Cdouble(x) / a.val)
end
function Base.:/(a::Adouble{T}, x::V) where {T<:Union{TbAlloc,TlAlloc},V<:Real}
    return Adouble{T}(a.val / Cdouble(x))
end
function Base.:/(a::Adouble{T}, b::Adouble{T}) where {T<:Union{TbAlloc,TlAlloc}}
    return Adouble{T}(a.val / b.val)
end

#--------------- Operation: ^ -------------------

function Base.:^(a::Adouble{T}, b::Adouble{T}) where {T<:Union{TbAlloc,TlAlloc}}
    return Adouble{T}(a.val^b.val)
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

function Base.ldexp(a::Adouble{T}, n::Integer) where {T<:Union{TbAlloc,TlAlloc}}
    return Adouble{T}(ldexp(a.val, n))
end
function Base.frexp(a::Adouble{T}, n::Cint) where {T<:Union{TbAlloc,TlAlloc}}
    return Adouble{T}(frexp(a.val, n); adouble=true)
end

erf(a::Adouble{TbAlloc}) = Adouble{TbAlloc}(TbadoubleModule.erf(a.val))
erf(a::Adouble{TlAlloc}) = Adouble{TlAlloc}(TladoubleModule.erf(a.val))

Base.eps(::Type{Adouble{T}}) where {T<:Union{TbAlloc,TlAlloc}} = eps(Cdouble)

SpecialFunctions.erf(a::Adouble{T}) where {T<:Union{TbAlloc,TlAlloc}} = erf(a)
SpecialFunctions.erfc(a::Adouble{T}) where {T<:Union{TbAlloc,TlAlloc}} = 1.0 - erf(a)
