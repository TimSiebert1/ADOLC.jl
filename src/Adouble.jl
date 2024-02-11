struct Adouble{T<:Union{TbAlloc, TlAlloc}} <: AbstractFloat
    """
    Wrapper for adouble from c++ but with more flexibility.
    
    @param val: Its a Float64 or a c++ adouble. This gives the possibility
                to have containers of type adouble but without allocating 
                an element which has to be differentiated. 
    """
    val:: Union{Real, T}


    Adouble{TbAlloc}() = new{TbAlloc}(TbadoubleModule.TbadoubleCxx())
    Adouble{TlAlloc}() = new{TlAlloc}(TladoubleModule.TladoubleCxx())

    Adouble{TbAlloc}(a::TbAlloc) = new{TbAlloc}(a)
    Adouble{TlAlloc}(a::TlAlloc) = new{TlAlloc}(a)

    Adouble{T}(val::V) where {T<:Union{TbAlloc, TlAlloc}, V<:Real} = new(val)

end


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

getValue(a::Adouble{TbAlloc}) = typeof(a.val) == Float64 ? a.val : TbadoubleModule.getValue(a.val)
getValue(a::Adouble{TlAlloc}) = typeof(a.val) == Float64 ? a.val : TladoubleModule.getValue(a.val)


function getValue(a::Vector{Adouble{T}}) where T <: Union{TbAlloc, TlAlloc}
    res = Vector{Float64}(undef, length(a))
    for i in eachindex(a)
      res[i] = getValue(a[i])
    end
    return res
  end
  

  function Adouble{TlAlloc}(data::Vector{Float64}) 
    """
    Create a vector of Tladouble with val = tladouble(data_entry). 
    """
  
    # c++ function
    tl_a = tl_init_for_gradient(data, length(data))
  
    tl_a_vec = Vector{Adouble{TlAlloc}}(undef, length(data))
    for i in 1:length(data)
      tl_a_vec[i] = Adouble{TlAlloc}(tl_a[i])
    end
    return tl_a_vec
  
    end


    function get_gradient(a::Adouble{TlAlloc}, num_independent::Int64)
        grad = Vector{Float64}(undef, num_independent)
        for i in 1:num_independent
        grad[i] = getADValue(a.val, i)
        end
        return grad
    end
    
