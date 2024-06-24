# ADOLC.jl

*A Julia wrapper of the automatic differentiation package ADOL-C*



```@meta
DocTestSetup = quote
    using ADOLC
end
```

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


For advanced users, there is a [list](@ref "List of wrapped ADOL-C drivers") of all functions, wrapped from ADOL-C.



## API Reference
```@index
Pages = ["lib/reference.md"]
```

```@bibliography
```