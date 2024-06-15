## Derivative Modes



# First-Order
| Mode          | Formula                  |
|:---------------:|:--------------------------:|
| jac         | $$Df(x)$$  |
| jac_vec    |  $$Df(x)\dot{v}$$            |
| jac_mat    |  $$Df(x)\dot{V}$$            |
| vec_jac    | $$\bar{z}^T Df(x)$$         |
| mat_jac    | $$\bar{Z}^T Df(x)$$       |


# Second-Order
| Mode             | Formula                       |
|:------------------:|:-------------------------------:|
| hess           | $$D^2f(x)$$               |
| hess_vec      | $$D^2f(x) \dot{v}$$          |
| hess_mat      | $$D^2f(x)  \dot{V}$$         |
| vec_hess      | $$\bar{z}^T D^2f(x)$$       |
| mat_hess      | $$\bar{Z}^T D^2f(x)$$       |
| vec_hess_vec | $$\bar{z}^T D^2f(x)  \dot{v}$$  |
| vec_hess_mat | $$\bar{z}^T D^2f(x)  \dot{V}$$ |
| mat_hess_mat | $$\bar{Z}^T D^2f(x)  \dot{V}$$  |
| mat_hess_vec | $$\bar{Z}^T D^2f(x)  \dot{v}$$  |


# Abs-Normal-Form
| Mode             | Formula                       |
|:------------------:|:-------------------------------:|
| abs_norm           | $$\Delta f(x)$$               |




## Higher-Order 
The goal of the following explanations is to familiarize the reader with 
the possibilities for computing higher-order derivatives that are included in `ADOLC.jl`.
In the context of `ADOLC.jl`, higher-order derivatives are given as a `Vector` of 
arbitrary-order mixed-partials. For example, let $$f:\mathbb{R}^n \to \mathbb{R}^m$$
and we want to compute the mixed-partials
```math
\left[\frac{\partial^3\partial^2 f(x)}{\partial^3 x_2 \partial^2 x_1}, \frac{\partial^4 f(x)}{\partial^4 x_3}, \frac{\partial^2 f(x)}{\partial^2 x_1}\right]
``` 
leveraging the [`derivative`](@ref) driver. After defining the function `f` and the point for the derivative evaluation `x`, we have to select the format of the `partials`. There exist two options explained below that use `Vector{Int64}` to define a partial derivative.

# ADOLC-Format
The ADOLC-Format repeats the index $$i$$ of a derivative direction $$x_i$$ up to the derivative order of this index: $$\frac{\partial^4 f(x)}{\partial^4 x_3} \to [3, 3, 3, 3]$$. Additionally, the resulting vector is sorted descendent; if the vector's length is less than the total derivative degree, it is filled with zeros. The requested mixed-partials results in:
```math 
[
 [2, 2, 2, 1, 1],
 [3, 3, 3, 3, 0],
 [1, 1, 0, 0, 0]
]
```

# Partial-Format
The Partial-Format mimics the notation of the mixed-partial, as used above. The entry of the vector at index $$i$$ is the derivative degree corresponding to the derivative direction $$x_i$$. Therefore, `partials` is given as
```math 
[
 [2, 3, 0, 0],
 [0, 0, 4, 0],
 [2, 0, 0, 0]
].
```
!!! note
    Internally, there is, at some point, a conversion from [Partial-Format](@ref) to [ADOLC-Format](@ref) since the access to the higher-order tensor computed with ADOL-C is based on the [ADOLC-Format](@ref). However, only one entry is converted at a time, meaning that the benefits of both modes, as explained below, are still valid.


!!! note 
    Both formats have their benefits. The [ADOLC-Format](@ref) should be used if the total derivative degree is small compared to the number of independents $n$. Otherwise, [Partial-Format](@ref) should be used.


There are utilities to convert between the formats: [`partial_to_adolc_format`](@ref)




# Seed-Space

## Memory handling
