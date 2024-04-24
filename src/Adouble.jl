struct Adouble{T<:Union{TbAlloc, TlAlloc}} <: AbstractFloat
    """
    Wrapper for adouble from c++ but with more flexibility.
    
    @param val: Its a Float64 or a c++ adouble. This gives the possibility
                to have containers of type adouble but without allocating 
                an element which has to be differentiated. 
    """
    val:: Union{Float64, T}


    Adouble{TbAlloc}() = new{TbAlloc}(TbadoubleModule.TbadoubleCxx())
    Adouble{TlAlloc}() = new{TlAlloc}(TladoubleModule.TladoubleCxx())

    Adouble{TbAlloc}(a::TbAlloc) = new{TbAlloc}(a)
    Adouble{TlAlloc}(a::TlAlloc) = new{TlAlloc}(a)

    Adouble{T}(val::V) where {T<:Union{TbAlloc, TlAlloc}, V<:Real} = new(Float64(val))


end
Adouble{TbAlloc}(a::Adouble{TbAlloc}) = Adouble{TbAlloc}(a.val)
Adouble{TlAlloc}(a::Adouble{TlAlloc}) = Adouble{TlAlloc}(a.val)

function Adouble{TbAlloc}(val::V, isadouble::Bool) where V <: Real
    """
    The idea behind this is that when a floating point number is promoted to 
    a adouble e.g. in vcat, then we dont want to create a new "real" adouble since 
    this would require a new derivative calculation.
    """
    if isadouble
        return Adouble{TbAlloc}(TbadoubleModule.TbadoubleCxx(val))
    end
    return Adouble{TbAlloc}(val)
end

Adouble{TbAlloc}(val::Bool, isadouble::Bool) = Adouble{TbAlloc}(float(val), isadouble)


function Adouble{TlAlloc}(val::V, isadouble::Bool) where V <: Real
    """
    The idea behind this is that when a floating point number is promoted to 
    a adouble e.g. in vcat, then we dont want to create a new "real" adouble since 
    this would require a new derivative calculation.
    """
    if isadouble
        return Adouble{TlAlloc}(TladoubleModule.TladoubleCxx(val))
    end
    return Adouble{TlAlloc}(val)
end

Adouble{TlAlloc}(val::Bool, isadouble::Bool) = Adouble{TlAlloc}(float(val), isadouble)



function Adouble{TbAlloc}(vals::Vector{V}, isadouble::Bool) where V <: Real
    """
    The idea behind this is that when a floating point number is promoted to 
    a adouble e.g. in vcat, then we dont want to create a new "real" adouble since 
    this would require a new derivative calculation.
    """
    if isadouble
        return [Adouble{TbAlloc}(TladoubleModule.TbadoubleCxx(val)) for val in vals]
    end
    return [Adouble{TbAlloc}(val) for val in vals]
end

function Adouble{TlAlloc}(vals::Vector{V}, isadouble::Bool) where V <: Real
    """
    The idea behind this is that when a floating point number is promoted to 
    a adouble e.g. in vcat, then we dont want to create a new "real" adouble since 
    this would require a new derivative calculation.
    """
    if isadouble
        return [Adouble{TlAlloc}(TladoubleModule.TladoubleCxx(val)) for val in vals]
    end
    return [Adouble{TlAlloc}(val) for val in vals]
end


getValue(a::Adouble{TbAlloc}) = typeof(a.val) == Float64 ? a.val : TbadoubleModule.getValue(a.val)
getValue(a::Adouble{TlAlloc}) = typeof(a.val) == Float64 ? a.val : TladoubleModule.getValue(a.val)

function getValue(a::Vector{Adouble{T}}) where T <: Union{TbAlloc, TlAlloc}
    res = Vector{Float64}(undef, length(a))
    for i in eachindex(a)
      res[i] = getValue(a[i])
    end
    return res
  end
  

function getADValue(a::Adouble{TlAlloc}, i::Int64)
    return TladoubleModule.getADValue(a.val, i)
end

    
function gradient(n::Int64, a::Adouble{TlAlloc}, res)
    for i in 1:n
        res[i] = getADValue(a, i)
    end
end

function gradient(n::Int64, a::Vector{Adouble{TlAlloc}}, res)
    for i in 1:length(a)
        for j in 1:n
            res[i, j] = getADValue(a[i], j)
        end
    end
end

function init_gradient(a::Vector{Adouble{TlAlloc}})
    for i in eachindex(a)
        TladoubleModule.setADValue(a[i].val, 1.0, i)
    end
end