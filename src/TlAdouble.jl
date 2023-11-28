module TlAdouble
    using CxxWrap
    
    # one need to specify the location of adolc_wrap.{so, dylib}
    total_build_dir = joinpath(@__DIR__, "build")
    @wrapmodule(() -> joinpath(total_build_dir,"libadolc_wrap"), :define_julia_module_tl)
    
    function __init__()
      @initcxx
    end

  # need to handle c++ opbjec tladouble* (array of tladoubles), if the access is over the bound
  # it returns eps
  Base.getindex(a::CxxWrap.CxxWrapCore.CxxPtr{tladouble}, i::Int64) = getindex_tl(a, i)

  struct Tladouble <: AbstractFloat
    """
    Wrapper for tladouble from c++ but with more flexibility.
    
    @param val: Its a Float64 or a c++ tladouble. This gives the possibility
                to have containers of type Tladouble but without allocating 
                an element which has to be differentiated. 
    """
    val:: Union{Float64, Main.ADOLC_wrap.TlAdouble.tladoubleAllocated}
    Tladouble() = new(tladouble())
    function Tladouble(x::Float64, isadouble::Bool)
        """
        The idea behind this is that when a floating point number is promoted to 
        a adouble e.g. in vcat, then we dont want to create a new adouble since 
        this would require a new derivative calculation.
        """
        if isadouble
            return new(tladouble(x))
        end
        return new(x)
    end
    Tladouble(x::Float64) = new(x)

    # conversion from the c++ type tladouble to the julia type Tladouble
    Tladouble(a::Main.ADOLC_wrap.TlAdouble.tladoubleAllocated) = new(a)
end

getValue(a::Tladouble) = typeof(a.val) == Float64 ? a.val : getValue(a.val)

#--------------- Operation: * -------------------

Base.:*(a::Tladouble, x::Float64) = Tladouble(a.val * x)
Base.:*(x::Float64, a::Tladouble) = Tladouble(x * a.val)

Base.:*(a::Tladouble, x::Int64) = Tladouble(a.val * Float64(x))
Base.:*(x::Int64, a::Tladouble) = Tladouble(Float64(x) * a.val)

Base.:*(a::Tladouble, x::Bool) = Tladouble(a.val * Float64(x))
Base.:*(x::Float64, a::Tladouble) = Tladouble(Float64(x) * a.val)

Base.:*(a::Tladouble, b::Tladouble) = Tladouble(a.val * b.val)

##############################################################

#--------------- Operation: + -------------------
Base.:+(x::Float64, a::Tladouble) = Tladouble(x + a.val)
Base.:+(a::Tladouble, x::Float64) = Tladouble(a.val + x)

Base.:+(x::Int64, a::Tladouble) = Tladouble(Float64(x) + a.val)
Base.:+(a::Tladouble, x::Int64) = Tladouble(a.val + Float64(x))

Base.:+(x::Bool, a::Tladouble) = Tladouble(Float64(x) + a.val)
Base.:+(a::Tladouble, x::Bool) = Tladouble(a.val + Float64(x))

Base.:+(a::Tladouble, b::Tladouble) = Tladouble(a.val + b.val)

##############################################################

#--------------- Operation: - -------------------
Base.:-(x::Float64, a::Tladouble) = Tladouble(x - a.val)
Base.:-(a::Tladouble, x::Float64) = Tladouble(a.val - x)

Base.:-(x::Int64, a::Tladouble) = Tladouble(Float64(x) - a.val)
Base.:-(a::Tladouble, x::Int64) = Tladouble(a.val - Float64(x))

Base.:-(x::Bool, a::Tladouble) = Tladouble(Float64(x) - a.val)
Base.:-(a::Tladouble, x::Bool) = Tladouble(a.val - Float64(x))

Base.:-(a::Tladouble, b::Tladouble) = Tladouble(a.val - b.val)

Base.:-(a::Tladouble) = (-1) * a

##############################################################

#--------------- Operation: / -------------------
Base.:/(x::Float64, a::Tladouble) = Tladouble(x / a.val)
Base.:/(a::Tladouble, x::Float64) = Tladouble(a.val / x)

Base.:/(x::Int64, a::Tladouble) = Tladouble(Float64(x) / a.val)
Base.:/(a::Tladouble, x::Int64) = Tladouble(a.val / Float64(x))

Base.:/(x::Bool, a::Tladouble) = Tladouble(Float64(x) / a.val)
Base.:/(a::Tladouble, x::Bool) = Tladouble(a.val / Float64(x))

Base.:/(a::Tladouble, b::Tladouble) = Tladouble(a.val / b.val)

##############################################################

#--------------- Operation: >= -------------------
Base.:>=(x::Float64, a::Tladouble) = x >= a.val
Base.:>=(a::Tladouble, x::Float64) = a.val >= x

Base.:>=(x::Int64, a::Tladouble) = Float64(x) >= a.val
Base.:>=(a::Tladouble, x::Int64) = a.val >= Float64(x)

Base.:>=(x::Bool, a::Tladouble) = Float64(x) >= a.val
Base.:>=(a::Tladouble, x::Bool) = a.val >= Float64(x)

Base.:>=(a::Tladouble, b::Tladouble) = a.val >= b.val

##############################################################

#--------------- Operation: > -------------------
Base.:>(x::Float64, a::Tladouble) = x > a.val
Base.:>(a::Tladouble, x::Float64) = a.val > x

Base.:>(x::Int64, a::Tladouble) = Float64(x) > a.val
Base.:>(a::Tladouble, x::Int64) = a.val > Float64(x)

Base.:>(x::Bool, a::Tladouble) = Float64(x) > a.val
Base.:>(a::Tladouble, x::Bool) = a.val > Float64(x)

Base.:>(a::Tladouble, b::Tladouble) = a.val > b.val

##############################################################

#--------------- Operation: <= -------------------
Base.:<=(x::Float64, a::Tladouble) = x <= a.val
Base.:<=(a::Tladouble, x::Float64) = a.val <= x

Base.:<=(x::Int64, a::Tladouble) = Float64(x) <= a.val
Base.:<=(a::Tladouble, x::Int64) = a.val <= Float64(x)

Base.:<=(x::Bool, a::Tladouble) = Float64(x) <= a.val
Base.:<=(a::Tladouble, x::Bool) = a.val <= Float64(x)

Base.:<=(a::Tladouble, b::Tladouble) = a.val <= b.val

##############################################################

#--------------- Operation: < -------------------
Base.:<(x::Float64, a::Tladouble) = x < a.val
Base.:<(a::Tladouble, x::Float64) = a.val < x

Base.:<(x::Int64, a::Tladouble) = Float64(x) < a.val
Base.:<(a::Tladouble, x::Int64) = a.val < Float64(x)

Base.:<(x::Bool, a::Tladouble) = Float64(x) < a.val
Base.:<(a::Tladouble, x::Bool) = a.val < Float64(x)

Base.:<(a::Tladouble, b::Tladouble) = a.val < b.val

##############################################################

#--------------- Operation: < -------------------
Base.:<(x::Float64, a::Tladouble) = x < a.val
Base.:<(a::Tladouble, x::Float64) = a.val < x

Base.:<(x::Int64, a::Tladouble) = Float64(x) < a.val
Base.:<(a::Tladouble, x::Int64) = a.val < Float64(x)

Base.:<(x::Bool, a::Tladouble) = Float64(x) < a.val
Base.:<(a::Tladouble, x::Bool) = a.val < Float64(x)

Base.:<(a::Tladouble, b::Tladouble) = a.val < b.val

##############################################################

#--------------- Operation: == -------------------
Base.:(==)(x::Float64, a::Tladouble) = x == a.val
Base.:(==)(a::Tladouble, x::Float64) = a.val == x

Base.:(==)(x::Int64, a::Tladouble) = Float64(x) == a.val
Base.:(==)(a::Tladouble, x::Int64) = a.val == Float64(x)

Base.:(==)(x::Bool, a::Tladouble) = Float64(x) == a.val
Base.:(==)(a::Tladouble, x::Bool) = a.val == Float64(x)

Base.:(==)(a::Tladouble, b::Tladouble) = a.val == b.val

##############################################################

#-------------- Functions: max -----------------

Base.max(x::Float64, a::Tladouble) = x > a.val ? Tladouble(x) : a
Base.max(a::Tladouble, x::Float64) = max(x, a)

Base.max(x::Int64, a::Tladouble) = Float64(x) > a.val ? Tladouble(Float64(x)) : a
Base.max(a::Tladouble, x::Int64) = max(x, a)

Base.max(x::Bool, a::Tladouble) = Float64(x) > a.val ? Tladouble(Float64(x)) : a
Base.max(a::Tladouble, x::Bool) = max(x, a)

Base.max(a::Tladouble, b::Tladouble) = b.val > a.val ? b : a

##############################################################

#-------------- Functions: abs, exp, sqrt -----------------

Base.abs(a::Tladouble) = a.val >= 0 ? a : Tladouble(abs(a.val))
Base.exp(a::Tladouble) = Tladouble(exp(a.val))
Base.sqrt(a::Tladouble) = Tladouble(sqrt(a.val))

##############################################################


#-------- utilities for type handling ----------
Base.promote(x::Float64, y::Tladouble) = Tladouble(x, false)
Base.promote_rule(::Type{Tladouble}, ::Type{Float64}) = Tladouble

# since every operation where an arugment is a adouble have to return a adouble
Base.promote_op(f::Core.Any, ::Type{Float64}, ::Type{Tladouble}) = Tladouble
Base.promote_rule(::Type{Tladouble}, ::Type{Main.ADOLC_wrap.TlAdouble.tladoubleAllocated}) = Tladouble


Base.convert(::Type{Tladouble}, x::Float64) = Tladouble(x, false)
Base.convert(::Type{Tladouble}, x::Int64) = Tladouble(Float64(x), false)


Base.zero(::Type{Tladouble}) = Float64(0.0)


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
  # current base operations:
  # max, abs, exp, sqrt, *, +, -, ^
  export tladouble, getADValue, setADValue, getValue, tl_init_for_gradient, getindex_tl, tladouble_vector_init, Tladouble

end # module adouble