# ADOLC.jl

[![Build Status](https://github.com/TimSiebert1/ADOLC.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/TimSiebert1/ADOLC.jl/actions?query=branch%3Amaster)
[![Coverage Status](https://codecov.io/github/TimSiebert1/ADOLC.jl/coverage.svg?branch=master)](https://app.codecov.io/gh/timsiebert1/ADOLC.jl)
[![Stable docs](https://img.shields.io/badge/docs-stable-blue.svg)](https://TimSiebert1.github.io/ADOLC.jl/stable/)
[![Dev docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://TimSiebert1.github.io/ADOLC.jl/dev/)

*A Julia wrapper of the automatic differentiation package ADOL-C*
  
This package wraps the C/C++ automatic differentiation library [ADOL-C](https://github.com/coin-or/ADOL-C) for use in [Julia](https://julialang.org/). 

Currently, the ADOL-C binaries are built for Julia 1.9, 1.10, and 1.11, and they are not compiled for musl libc.  
To add this package, use
```jl
using Pkg; Pkg.add("ADOLC")
using ADOLC
```
  
[First-](https://timsiebert1.github.io/ADOLC.jl/dev/lib/derivative_modes/#First-Order) and [second-order](https://timsiebert1.github.io/ADOLC.jl/dev/lib/derivative_modes/#Second-Order) derivatives can be calculated as follows
```jl
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

There are various available modes for [first-](https://timsiebert1.github.io/ADOLC.jl/dev/lib/derivative_modes/#First-Order) and [second-order](https://timsiebert1.github.io/ADOLC.jl/dev/lib/derivative_modes/#Second-Order) calculations. The computation of higher-order derivatives is explained [here](https://timsiebert1.github.io/ADOLC.jl/dev/lib/derivative_modes/#Higher-Order) and works as sketched below
```jl
f(x) = [x[1]^2*x[2]^2, x[3]^2*x[4]^2]
x = [1.0, 2.0, 3.0, 4.0]
partials = [[1, 1, 0, 0], [0, 0, 1, 1], [2, 2, 0, 0]]
res = derivative(f, x, partials)

# output

2×3 CxxMatrix:
 8.0   0.0  4.0
 0.0  48.0  0.0
```

In addition there is the possibility to compute univariate Taylor polynomials with the [`univariate_tpp`](@ref) driver:
```jldoctest
f(x) = sin(x[1]) + x[2]
x = [pi / 2, 0.5]
d = 2
utp = univariate_tpp(f, x, 2)

# output

1×3 CxxMatrix:
 1.5  1.0  -0.5
```
More information can be found in the corresponding guides in the docs.


For advanced users, there is a [list](https://timsiebert1.github.io/ADOLC.jl/dev/lib/wrapped_fcts/) of all functions wrapped from [ADOL-C](https://github.com/coin-or/ADOL-C). 

