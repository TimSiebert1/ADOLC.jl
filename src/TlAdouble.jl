module TlAdouble
    using CxxWrap
    
    # one need to specify the location of adolc_wrap.{so, dylib}
    total_build_dir = joinpath(@__DIR__, "build")
    @wrapmodule(() -> joinpath(total_build_dir,"libadolc_wrap"), :define_julia_module_tl)
    
    function __init__()
      @initcxx
    end
  
function Base.:*(a::Main.ADOLC_wrap.TlAdouble.tladoubleAllocated, x::Vector{Float64})
      return map((x_i)->a*x_i, x)
end


  export tladouble, getADValue, setADValue, getValue, tl_init_for_gradient, getindex_tl, tl_init, isless

  Base.isless(val::Float64, x::Main.ADOLC_wrap.TlAdouble.tladoubleAllocated) = isless(val, x)

  Base.getindex(X::CxxWrap.CxxWrapCore.CxxPtr{tladouble}, row::Int64) = getindex_tl(X, row)

  Base.iterate(x::Main.ADOLC_wrap.TlAdouble.tladoubleAllocated) = (x, nothing)
  Base.iterate(x::Main.ADOLC_wrap.TlAdouble.tladoubleAllocated, y::Nothing) = nothing
  
  function tl_init_for_gradient(data::Vector{Float64}) 
      tl_a = tl_init_for_gradient(data, length(data))
      tl_a_vec = Vector{Any}(undef, length(data))
      for i in 1:length(data)
        tl_a_vec[i] = tl_a[i]
      end
      return tl_a_vec

  end


  # current base operations:
  # max, abs, exp, sqrt, *, +, -, ^

end # module adouble