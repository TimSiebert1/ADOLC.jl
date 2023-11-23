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


  export tladouble, getADValue, setADValue, getValue

  # current base operations:
  # max, abs, exp, sqrt, *, +, -, ^

end # module adouble