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

  export TladoubleCxx, getADValue, setADValue, tl_init_for_gradient, getindex_tl # getValue,
end # module adouble

