```@meta
DocTestSetup = quote
    using ADOLC
end
```
# Seed-Matrix
This guide is related to the [higher-order](@ref "Higher-Order") derivative computation with 
[`derivative`](@ref) or [`derivative!`](@ref). Internally, the drivers are based on the propagation of univariate Taylor polynomials [griewank_evaluating_1999](@cite). The underlying method leverages a `seed` matrix $$S\in \mathbb{R}^{n \times p}$$ to compute mixed-partials of arbitrary order for a function $$f:\mathbb{R}^n \to \mathbb{R}^m$$ in the form: 
```math
    \frac{\partial^k f(x + Sz)}{\partial^k z}\big|_{z=0} 
```
for some $$z \in \mathbb{R}^p$$. Usually, $$S$$ is the *identity* or the *partial identity* (see [`create_partial_cxx_identity`](@ref)), which is also the case, when no `seed` is passed to the driver. To switch between both identity options the flag `id_seed` can be used. In the case of identity, the formula above boils down to 
```math
    \frac{\partial^k f(x + Sz)}{\partial^k z}\big|_{z=0}= \frac{\partial^k f(x)}{\partial^k x}.
```
Moreover, the partial identity results in the same but is more efficient. Leveraging the partial identity ensures that only the derivatives of the requested derivative directions are computed, and this is explained briefly in the following paragraph.   

Assume we want to compute the derivatives specified in the [Partial-Format](@ref): [[4, 0, 0, 3], [2, 0, 0, 4], [1, 0, 0, 1]].  
Obviously, none of the derivatives includes $$x_2$$ and $$x_3$$. To avoid unnecessary computations (i.e., the propagation of unnecessary univariate Polynomials), the partial identity is created, stacking only those canonical basis vectors that are related to the requested derivative directions. In our case, the partial identity looks like this:  
```math
\left[
    \begin{matrix}
    1 & 0 \\
    0 & 0 \\
    0 & 0 \\
    0 & 1 
    \end{matrix}
 \right].
```
As you can see, the directions are reduced from four to two. In general, the number of required univariate Polynomial propagations to compute all mixed-partials up to degree $$d$$ of for $$f$$ is $$\left( \begin{matrix} n - 1 + d \\ d \end{matrix} \right)$$. Leveraging the `seed` $$S$$ reduces this number to $$\left( \begin{matrix} p - 1 + d \\ d \end{matrix} \right)$$, where $$p$$ is often much smaller than $$n$$. In addition, $$S$$ can be used as a subspace projection. For example, if $$S=[1, \dots, 1]^T$$, you could compute the sum of the different univariate Taylor coefficients:
```jldoctest
using ADOLC
f(x) = x[1]^3*x[2]^2 - x[2]^3
x = [1.0, 1.0]
partials = [[1], [2], [3]]
seed = CxxMatrix([[1.0, 1.0];;])
res = derivative(f, x, partials, seed)

# output

1×3 CxxMatrix:
 2.0  14.0  54.0
```
