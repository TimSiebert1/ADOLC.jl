module TladoubleModule
    using ADOLC_jll
    using CxxWrap
    
    @wrapmodule(() -> libadolc_wrap, :Tladouble_module)
    
    function __init__()
      @initcxx
    end

  # need to handle c++ opbjec tladouble* (array of tladoubles), if the access is over the bound
  # it returns eps
  Base.getindex(a::CxxWrap.CxxWrapCore.CxxPtr{TladoubleCxx}, i::Int64) = getindex_tl(a, i)

  struct Tladouble <: AbstractFloat
    """
    Wrapper for tladouble from c++ but with more flexibility.
    
    @param val: Its a Float64 or a c++ tladouble. This gives the possibility
                to have containers of type Tladouble but without allocating 
                an element which has to be differentiated. 
    """
    val:: Union{Real, TladoubleModule.TladoubleCxxAllocated}
    Tladouble() = new(TladoubleCxx())
    function Tladouble(x::Real, isadouble::Bool)
        """
        The idea behind this is that when a floating point number is promoted to 
        a adouble e.g. in vcat, then we dont want to create a new adouble since 
        this would require a new derivative calculation.
        """
        if isadouble
            return new(TladoubleCxx(Float64(x)))
        end
        return new(x)
    end
    Tladouble(x::Float64) = new(x)

    # conversion from the c++ type tladouble to the julia type Tladouble
    Tladouble(a::TladoubleModule.TladoubleCxxAllocated) = new(a)
end

getValue(a::Tladouble) = typeof(a.val) == Float64 ? a.val : getValue(a.val)
function getValue(a::Vector{Tladouble})
  res = Vector{Float64}(undef, length(a))
  for i in eachindex(a)
    res[i] = getValue(a[i])
  end
  return res
end
  
#--------------- Operation: * -------------------

Base.:*(a::Tladouble, x::Real) = Tladouble(a.val * Float64(x))
Base.:*(x::Real, a::Tladouble) = Tladouble(Float64(x) * a.val)
Base.:*(a::Tladouble, b::Tladouble) = Tladouble(a.val * b.val)
"""
Base.:*(a::Tladouble, x::Int64) = Tladouble(a.val * Float64(x))
Base.:*(x::Int64, a::Tladouble) = Tladouble(Float64(x) * a.val)
"""
Base.:*(a::Tladouble, x::Bool) = Tladouble(a.val * Float64(x))
Base.:*(x::Bool, a::Tladouble) = Tladouble(Float64(x) * a.val)



##############################################################

#--------------- Operation: + -------------------
Base.:+(x::Real, a::Tladouble) = Tladouble(Float64(x) + a.val)
Base.:+(a::Tladouble, x::Real) = Tladouble(a.val + Float64(x))
Base.:+(a::Tladouble, b::Tladouble) = Tladouble(a.val + b.val)

"""
Base.:+(x::Float64, a::Tladouble) = Tladouble(x + a.val)
Base.:+(a::Tladouble, x::Float64) = Tladouble(a.val + x)

Base.:+(x::Int64, a::Tladouble) = Tladouble(Float64(x) + a.val)
Base.:+(a::Tladouble, x::Int64) = Tladouble(a.val + Float64(x))

Base.:+(x::Bool, a::Tladouble) = Tladouble(Float64(x) + a.val)
Base.:+(a::Tladouble, x::Bool) = Tladouble(a.val + Float64(x))


"""

##############################################################

#--------------- Operation: - -------------------
Base.:-(x::Real, a::Tladouble) = Tladouble(Float64(x) - a.val)
Base.:-(a::Tladouble, x::Real) = Tladouble(a.val - Float64(x))
Base.:-(a::Tladouble, b::Tladouble) = Tladouble(a.val - b.val)

Base.:-(a::Tladouble) = (-1) * a

"""
Base.:-(x::Int64, a::Tladouble) = Tladouble(Float64(x) - a.val)
Base.:-(a::Tladouble, x::Int64) = Tladouble(a.val - Float64(x))

Base.:-(x::Bool, a::Tladouble) = Tladouble(Float64(x) - a.val)
Base.:-(a::Tladouble, x::Bool) = Tladouble(a.val - Float64(x))
"""


##############################################################

#--------------- Operation: / -------------------
Base.:/(x::Real, a::Tladouble) = Tladouble(Float64(x) / a.val)
Base.:/(a::Tladouble, x::Real) = Tladouble(a.val / Float64(x))
Base.:/(a::Tladouble, b::Tladouble) = Tladouble(a.val / b.val)

"""
Base.:/(x::Int64, a::Tladouble) = Tladouble(Float64(x) / a.val)
Base.:/(a::Tladouble, x::Int64) = Tladouble(a.val / Float64(x))

Base.:/(x::Bool, a::Tladouble) = Tladouble(Float64(x) / a.val)
Base.:/(a::Tladouble, x::Bool) = Tladouble(a.val / Float64(x))
"""


##############################################################

#--------------- Operation: >= -------------------
Base.:>=(x::Real, a::Tladouble) = x >= a.val
Base.:>=(a::Tladouble, x::Real) = a.val >= x
Base.:>=(a::Tladouble, b::Tladouble) = a.val >= b.val

"""
Base.:>=(x::Int64, a::Tladouble) = Float64(x) >= a.val
Base.:>=(a::Tladouble, x::Int64) = a.val >= Float64(x)

Base.:>=(x::Bool, a::Tladouble) = Float64(x) >= a.val
Base.:>=(a::Tladouble, x::Bool) = a.val >= Float64(x)
"""


##############################################################

#--------------- Operation: > -------------------
Base.:>(x::Real, a::Tladouble) = x > a.val
Base.:>(a::Tladouble, x::Real) = a.val > x
Base.:>(a::Tladouble, b::Tladouble) = a.val > b.val

"""
Base.:>(x::Int64, a::Tladouble) = Float64(x) > a.val
Base.:>(a::Tladouble, x::Int64) = a.val > Float64(x)

Base.:>(x::Bool, a::Tladouble) = Float64(x) > a.val
Base.:>(a::Tladouble, x::Bool) = a.val > Float64(x)
"""


##############################################################

#--------------- Operation: <= -------------------
Base.:<=(x::Real, a::Tladouble) = x <= a.val
Base.:<=(a::Tladouble, x::Real) = a.val <= x
Base.:<=(a::Tladouble, b::Tladouble) = a.val <= b.val

"""
Base.:<=(x::Int64, a::Tladouble) = Float64(x) <= a.val
Base.:<=(a::Tladouble, x::Int64) = a.val <= Float64(x)

Base.:<=(x::Bool, a::Tladouble) = Float64(x) <= a.val
Base.:<=(a::Tladouble, x::Bool) = a.val <= Float64(x)
"""


##############################################################

#--------------- Operation: < -------------------
Base.:<(x::Real, a::Tladouble) = x < a.val
Base.:<(a::Tladouble, x::Real) = a.val < x
Base.:<(a::Tladouble, b::Tladouble) = a.val < b.val

"""
Base.:<(x::Int64, a::Tladouble) = Float64(x) < a.val
Base.:<(a::Tladouble, x::Int64) = a.val < Float64(x)

Base.:<(x::Bool, a::Tladouble) = Float64(x) < a.val
Base.:<(a::Tladouble, x::Bool) = a.val < Float64(x)
"""


##############################################################

#--------------- Operation: == -------------------
Base.:(==)(x::Real, a::Tladouble) = x == a.val
Base.:(==)(a::Tladouble, x::Real) = a.val == x
Base.:(==)(a::Tladouble, b::Tladouble) = a.val == b.val

"""
Base.:(==)(x::Int64, a::Tladouble) = Float64(x) == a.val
Base.:(==)(a::Tladouble, x::Int64) = a.val == Float64(x)

Base.:(==)(x::Bool, a::Tladouble) = Float64(x) == a.val
Base.:(==)(a::Tladouble, x::Bool) = a.val == Float64(x)
"""


##############################################################

#-------------- Functions: max -----------------

Base.max(x::Real, a::Tladouble) = x > a.val ? Tladouble(x) : a
Base.max(a::Tladouble, x::Real) = max(x, a)
Base.max(a::Tladouble, b::Tladouble) = b.val > a.val ? b : a

"""
Base.max(x::Int64, a::Tladouble) = Float64(x) > a.val ? Tladouble(Float64(x)) : a
Base.max(a::Tladouble, x::Int64) = max(x, a)

Base.max(x::Bool, a::Tladouble) = Float64(x) > a.val ? Tladouble(Float64(x)) : a
Base.max(a::Tladouble, x::Bool) = max(x, a)
"""


##############################################################

#-------------- unary Functions

Base.abs(a::Tladouble) = a.val >= 0 ? a : Tladouble(abs(a.val))
Base.exp(a::Tladouble) = Tladouble(exp(a.val))
Base.sqrt(a::Tladouble) = Tladouble(sqrt(a.val))

erf(a::Tladouble) = Tladouble(erf(a.val))


##############################################################


#-------- utilities for type handling ----------
Base.promote(x::Real, y::Tladouble) = Tladouble(Float64(x), false)

# not sure if needed 
Base.promote(x::Tladouble, y::Tladouble) = x

Base.promote_rule(::Type{Tladouble}, ::Type{Real}) = Tladouble

# since every operation where an argument is a adouble have to return a adouble
Base.promote_op(f::Core.Any, ::Type{Real}, ::Type{Tladouble}) = Tladouble



Base.convert(::Type{Tladouble}, x::Real) = Tladouble(Float64(x), false)


# this is called e.g. when creating a vector of Tladoubles ... not sure why
Base.convert(::Type{Tladouble}, x::Tladouble) = x






###############################################################


function tladouble_vector_init(data::Vector{Float64}) 
  """
  Create a vector of Tladouble with val = tladouble(data_entry). 
  """

  # c++ function
  tl_a = tl_init_for_gradient(data, length(data))

  tl_a_vec = Vector{Tladouble}(undef, length(data))
  for i in 1:length(data)
    tl_a_vec[i] = Tladouble(tl_a[i])
  end
  return tl_a_vec

end

function get_gradient(a::Tladouble, num_independent::Int64)
  grad = Vector{Float64}(undef, num_independent)
  for i in 1:num_independent
    grad[i] = getADValue(a.val, i)
  end
  return grad
end


  # current base operations:
  # max, abs, exp, sqrt, *, +, -, ^
  export TladoubleCxx, getADValue, setADValue, getValue, tl_init_for_gradient, getindex_tl, tladouble_vector_init, Tladouble, get_gradient
  export sin, erf, tanh
end # module adouble

