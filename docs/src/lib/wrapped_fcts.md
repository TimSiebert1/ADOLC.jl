# List of wrapped ADOL-C drivers

## TbadoubleModule

    getValue
    gradient
    jacobian
    hessian
    vec_jac
    jac_vec
    hess_vec
    hess_mat
    lagra_hess_vec
    jac_solv

    ad_forward(short tag, int m, int n, int d, int keep, double **X, double **Y) (in ADOL-C: forward)
    ad_reverse(short tag, int m, int n, int d, double *u, double **Z) (in ADOL-C: reverse)

    zos_forward
    fos_forward
    hos_forward
    hov_wk_forward

    fov_forward
    hov_forward

    fos_reverse
    hos_reverse

    fov_reverse
    hov_reverse
    tensor_address
    tensor_eval

## Abs-Smooth Utilities

    enableMinMaxUsingAbs
    get_num_switches
    zos_pl_forward
    fos_pl_forward
    fov_pl_forward
    abs_normal


## Tape Utilities

    << (in ADOL-C: <<=)
    >> (in ADOL-C: =>>)
    trace_on(int tag)
    trace_on(int tag, int keep)
    trace_off(int file)
    trace_off()




## TladoubleModule

    setNumDir(int const &n) 
    getValue()                      
    getADValue(int const &i)
    setADValue(double const &val)
    setADValue(double const val, int const &i)


## Arithmethics

    + 
    - 
    * 
    / 
    ^

## Comparison

    <
    >
    >=
    <=
    ==

## Basic Operations

    abs
    sqrt
    sin
    cos
    tan
    asin
    acos
    atan
    exp
    log
    log10
    sinh
    cosh
    tanh
    asinh
    acosh
    atanh
    ceil
    floor
    max
    min
    ldexp
    frexp
    erf
    cbrt