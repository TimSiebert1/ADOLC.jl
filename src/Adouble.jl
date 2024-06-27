
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
