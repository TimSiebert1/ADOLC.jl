
TbAlloc = TbadoubleModule.TbadoubleCxxAllocated
TlAlloc = TladoubleModule.TladoubleCxxAllocated

struct Adouble{T<:Union{TbAlloc,TlAlloc}} <: AbstractFloat
    val::Union{Float64,T}
    Adouble{T}() where {T<:Union{TbAlloc,TlAlloc}} = T == TbAlloc ? new{T}(TbadoubleModule.TbadoubleCxx()) : new{T}(TladoubleModule.TladoubleCxx())
    Adouble{T}(a::T) where {T<:Union{TbAlloc,TlAlloc}} = new{T}(a)
    function Adouble{T}(val::V; adouble::Bool=false) where {V<:Number, T<:Union{TbAlloc,TlAlloc}}
        val = float(val)
        if adouble
            return T == TbAlloc ? new{TbAlloc}(TbadoubleModule.TbadoubleCxx(val)) : new{TlAlloc}(TladoubleModule.TladoubleCxx(val))
        end
        return new{T}(val)
    end
end

Adouble{T}(a::Adouble{T}) where {T<:Union{TbAlloc,TlAlloc}} = Adouble{T}(a.val)

function Adouble{T}(vals::Vector{V}; adouble::Bool=false) where {V<:Real, T<:Union{TbAlloc,TlAlloc}}
    if adouble
        return [Adouble{T}(val, adouble=true) for val in vals]
    end
    return [Adouble{T}(val) for val in vals]
end

function getValue(a::Adouble{T}) where {T<:Union{TbAlloc,TlAlloc}}
    return typeof(a.val) == Float64 ? a.val : (T == TbAlloc ? TbadoubleModule.getValue(a.val) : TladoubleModule.getValue(a.val))
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
