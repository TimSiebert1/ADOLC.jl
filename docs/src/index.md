# ADOLC.jl

*A Julia wrapper of the automatic differentiation package ADOL-C*



```@meta
DocTestSetup = quote
    using ADOLC
end
```

 
This package wraps the C/C++ automatic differentiation library [ADOL-C](https://github.com/coin-or/ADOL-C) for use in [Julia](https://julialang.org/). 

Currently, the ADOL-C binaries are built for Julia 1.9, 1.10, and 1.11, and they are not compiled for musl libc.  
To add this package, use

To add this package, use
```jl
using Pkg; Pkg.add("ADOLC")
using ADOLC
```
[First-](@ref "First-Order") and [second-order](@ref "Second-Order") derivatives can be calculated as follows
```jldoctest
f(x) = [x[1]*x[2]^2, x[1]^2*x[3]^3]
x = [1.0, 2.0, -1.0]
dir = [1.0, 0.0, 0.0]
weights = [1.0, 1.0]
res = derivative(f, x, :vec_hess_vec, dir=dir, weights=weights)

# output

3-element CxxVector:
 -2.0
  4.0
  6.0
```
There are various available modes for [first-](@ref "First-Order") and [second-order](@ref "Second-Order") calculations. The computation of higher-order derivatives is explained [here](@ref "Higher-Order") and works as sketched below
```jldoctest
f(x) = [x[1]^2*x[2]^2, x[3]^2*x[4]^2]
x = [1.0, 2.0, 3.0, 4.0]
partials = [[1, 1, 0, 0], [0, 0, 1, 1], [2, 2, 0, 0]]
res = derivative(f, x, partials)

# output

2×3 CxxMatrix:
 8.0   0.0  4.0
 0.0  48.0  0.0
```

You can also define parameters (`param`) not used for differentiation, which can be changed 
in subsequent calls without retaping. The given function `f` is expected to have the shape `f(x, param)`:
```jldoctest
function f(x, param)
    x1 = x[1] * param[1]
    return [x1*x[2], x[2]] 
end
x = [-1.0, 1/2]
param = 3.0
dir = [2.0, -2.0]
res = derivative(f, x, param, :jac_vec, dir=dir, tape_id=1)

##res[1] == 9.0
##res[2] == -2.0

param = -3.0
x = [1.0, 1.0]
res = derivative(f, x, param, :jac_vec, dir=dir, tape_id=1, reuse_tape=true)
res 

# output

2-element CxxVector:
  0.0
 -2.0
```

In addition, there is the possibility to compute univariate Taylor polynomials with the [`univariate_tpp`](@ref) driver:
```jldoctest
f(x) = sin(x[1]) + x[2]
x = [pi / 2, 0.5]
d = 2
utp = univariate_tpp(f, x, 2)

# output

1×3 CxxMatrix:
 1.5  1.0  -0.5
```
More information can be found in the `Guides`.


For advanced users, there is a [list](@ref "List of wrapped ADOL-C drivers") of all functions, wrapped from ADOL-C.



## API Reference
```@index
Pages = ["lib/reference.md"]
```

```@bibliography
```

