module TladoubleModule
using ADOLC_jll
using CxxWrap


@wrapmodule(() -> libadolc_wrap, :Tladouble_module)
function __init__()
    @initcxx
end

export TladoubleCxx, getADValue, setADValue, erf, cbrt, set_num_dir
end # module adouble
