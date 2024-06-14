module TladoubleModule
using ADOLC_jll
using CxxWrap


@wrapmodule(() -> libadolc_wrap, :Tladouble_module)
function __init__()
    @initcxx
end

export TladoubleCxx, setADValue, cbrt, set_num_dir
end # module adouble
