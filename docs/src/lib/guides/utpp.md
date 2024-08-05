```@meta
DocTestSetup = quote
    using ADOLC
end
```
# Univariate Taylor Polynomial Propagation

The univariate Taylor polynomial propagation (UTPP) aims to compute the univariate Taylor polynomial (UTP) $$\varphi_f$$ of a function $$f$$ and is given as a polynomial in $$t \in \mathbb{R}$$:
```math
\varphi_f(t) := \sum_{i=0}^d f^i(x)t^i,
```
where $$d \in \mathbb{N}$$ is the maximal order or degree and $$x \in \mathbb{R}^n$$ the evaluation point of the UTP. $$f^i(x)$$ defines the $$i$$-th uniariate Taylor coefficient (UTC):
```math
f^i(x) := \sum_{\underset{|k|=i}{k \in \mathbb{N}^n}} \frac{1}{k!}\frac{\partial^k f(x)}{\partial^k x}.
```
Since the UTP is computed component-wise (w.r.t the output of $$f$$), it is usually assumed $$f$$ has a one dimensional output. 

To compute the polynomial, UTPP leverages so-called recurrence relations. Recurrence relations are formulas specifying the UTC.
For example: \
Let $$n=2$$, $$f(x) = x_1 + x_2$$ and $$h$$ and $$g$$ suitable functions. The recurrence relation for the $$i$$-th UTC of $$f$$ up to degree $$d=2$$ is given by
```math
f^i(x) = h^i(x) + g^i(x).
```
Therefore, if we know the UTPs of $$h$$ and $$g$$ we can conclude the UTP of $$f$$. These kind of formulas motivate the term "propagation" of input polynomials. Various recurrence relations can be found in table 13.1 and 13.2 of [griewank_13_2008](@cite). AD packages implement the recurrence relations for a set of elemental functions. If a user-defined function is decomposable into these elemental functions, the AD package can computes the UTP automatically.  

ADOLC.jl leverages [ADOL-C](https://github.com/coin-or/ADOL-C)'s utilities for UTPP and wraps it in the [`univariate_tpp`](@ref) driver, which allows the easy computation of Taylor polynomials of functions with arbitrary input and output dimension. To demonstrate the application of [`univariate_tpp`](@ref), lets go back to the example. If we set $$n = 2$$ and $$h(x) = \sin(x_1)$$, $$g(x) = x_2$$ for all $$x=(x_1, x_2) \in \mathbb{R}^2$$, we have by definition of the UTC
```math
f^0(x) = h^0(x) + g^0(x) = \sin(x_1) + x_2 \\
f^1(x) = h^1(x) + g^1(x) = \cos(x_1)x_1' + x_2' \\
f^2(x) = \frac{1}{2}\left(-\sin(x_1)(x_1')^2 + \cos(x_1)x_1''\right) + \frac{1}{2}x_2''.
```
The formulas above emphasize that the values of $$x_i'$$ and $$x_i''$$ must be given. Intuitively one could set them to the derivatives of the projections by interpreting $$x_i = p_i(x)$$. Then $$x_i' = 1$$ and all further derivatives are zero. However, the input values $$x_i$$ could be generated from a complex computation. A user may wants to include the derivative information of this computation in $$x_i', x_i'', \dots$$. To this end, [ADOL-C](https://github.com/coin-or/ADOL-C) allows a user to provide distinct input UTC for each input variable $$x_i$$, where the evaluation point is one dimensional. Those UTC have the form:
```math
\sum_{j=0}^d y^j t^j.
```
The coefficients $$y^j$$ are user-given, which means we need $$d+1$$ coefficients for every input variable $$x_i$$. If the UTC above corresponds to the input variable $$x_1$$, internally the following connection between the coefficient $$y^i$$ and the $$i$$-th derivative of $$x_1$$ is expected
```math
y^i = \frac{1}{i!}x_1^{(i)}.
```
Here it is assumed that $$x_1$$ is provided by some function, which makes sense of the derivative. The output of [`univariate_tpp`](@ref) are the UTC of $$f$$'s UTP. \
In our example, lets first set the input UTCs corresponding to the UTP of the projections $$p_i(x) = x_i$$. For $$x_1$$ we get $$y^0 = x_1$$, $$y^1 = 1$$ and $$y^2 = 0$$. Similarly for $$x_2$$. If we further choose $$x = (\frac{\pi}{2}, \frac{1}{2})$$ we can compute the UTCs $$f^0$$, $$f^1$$ and $$f^2$$ of $$f$$ with ADOLC.jl easily:
```jldoctest
f(x) = sin(x[1]) + x[2]
x = [pi / 2, 0.5]
d = 2
utp = univariate_tpp(f, x, 2)

# output

1×3 CxxMatrix:
 1.5  1.0  -0.5
```

!!! note
    The initialization with the input UTP based on the projections is done automatically if no other initialization UTP is provided to the parameter `init_tp`.


Next, we comput the UTP at the same evaluation point $$x = (\frac{\pi}{2}, \frac{1}{2})$$, but now $$x_2$$ is obtained from the function $$x_2(s) = s^2 + 3s + \frac{1}{2}$$, while $$x_1$$ is still determined by the projection $$p_1$$. Then, $$x_2(0) = \frac{1}{2}$$, $$x_2'(0) = 3$$ and $$x_2''(0) = 2$$. The corresponding UTCs are given as $$y^0 = \frac{1}{2}$$, $$y^1 = 3$$ and $$y^2 = 1$$. The input UTCs are now written to the parameter `init_tp` of [`univariate_tpp`](@ref). The UTCs of $$f$$ are again easily computed:
```jldoctest
f(x) = sin(x[1]) + x[2]
x = [pi / 2, 0.5]
d = 2
init_tp = CxxMatrix([[pi /2, 0.5] [1.0, 3.0] [0.0, 1.0]])
utp = univariate_tpp(f, x, d, init_tp)

# output

1×3 CxxMatrix:
 1.5  3.0  0.5
```

For more information on univariate Taylor polynomial propagation I recommend to read the Chapter 13.2 "Taylor Polynomial Propagation" from [griewank_13_2008](@cite).
