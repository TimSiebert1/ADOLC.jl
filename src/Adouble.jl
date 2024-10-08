
TbAlloc = TbadoubleModule.TbadoubleCxxAllocated
TlAlloc = TladoubleModule.TladoubleCxxAllocated

struct Adouble{T<:Union{TbAlloc,TlAlloc}} <: AbstractFloat
    val::Union{Float64,T}
    function Adouble{T}() where {T<:Union{TbAlloc,TlAlloc}}
        return if T == TbAlloc
            new{T}(TbadoubleModule.TbadoubleCxx())
        else
            new{T}(TladoubleModule.TladoubleCxx())
        end
    end
    Adouble{T}(a::T) where {T<:Union{TbAlloc,TlAlloc}} = new{T}(a)
    function Adouble{T}(
        val::V; adouble::Bool=false
    ) where {V<:Number,T<:Union{TbAlloc,TlAlloc}}
        val = float(val)
        if adouble
            return if T == TbAlloc
                new{TbAlloc}(TbadoubleModule.TbadoubleCxx(val))
            else
                new{TlAlloc}(TladoubleModule.TladoubleCxx(val))
            end
        end
        return new{T}(val)
    end
end

Adouble{T}(a::Adouble{T}) where {T<:Union{TbAlloc,TlAlloc}} = Adouble{T}(a.val)

function Adouble{T}(
    vals::Vector{V}; adouble::Bool=false
) where {V<:Real,T<:Union{TbAlloc,TlAlloc}}
    if adouble
        return [Adouble{T}(val; adouble=true) for val in vals]
    end
    return [Adouble{T}(val) for val in vals]
end

function getValue(a::Adouble{T}) where {T<:Union{TbAlloc,TlAlloc}}
    return if typeof(a.val) == Float64
        a.val
    else
        (T == TbAlloc ? TbadoubleModule.getValue(a.val) : TladoubleModule.getValue(a.val))
    end
end

function getValue(a::Vector{Adouble{T}}) where {T<:Union{TbAlloc,TlAlloc}}
    res = Vector{Float64}(undef, length(a))
    for i in eachindex(a)
        res[i] = getValue(a[i])
    end
    return res
end

function getADValue(a::Adouble{TlAlloc}, i::Integer)
    return TladoubleModule.getADValue(a.val, i)
end

#-------- utilities for type handling ----------
function Base.promote(x::V, y::Adouble{T}) where {T<:Union{TbAlloc,TlAlloc},V<:Real}
    return Adouble{T}(Cdouble(x); adouble=false)
end
Base.promote(x::Adouble{T}, y::Adouble{T}) where {T<:Union{TbAlloc,TlAlloc}} = x

function Base.promote_rule(
    ::Type{Adouble{T}}, ::Type{V}
) where {T<:Union{TbAlloc,TlAlloc},V<:Real}
    return Adouble{T}
end

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
    return Adouble{T}(Cdouble(x); adouble=false)
end
Base.convert(::Type{Adouble{T}}, x::Adouble{T}) where {T<:Union{TbAlloc,TlAlloc}} = x

function mkparam(a::Adouble{TbAlloc})
    @assert typeof(a.val) == Cdouble "mkparam works only if input a.val is of type Cdouble ($Cdouble)!"
    return Adouble{TbAlloc}(TbadoubleModule.mkparam(a.val))
end

function mkparam(x::Cdouble)
    return Adouble{TbAlloc}(TbadoubleModule.mkparam(x))
end

function mkparam(x::Vector{Cdouble})
    return Adouble{TbAlloc}([mkparam(x_i) for x_i in x])
end
