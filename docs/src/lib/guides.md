```@meta
DocTestSetup = quote
    using ADOLC
end
```

## Working with C++ Memory

ADOLC.jl is a wrapper of the C/C++ library [ADOL-C](https://github.com/coin-or/ADOL-C). Wrapper means
data from Julia is passed to C++, and calls to functions in Julia trigger C++ function calls to get output data in Julia. The communication between Julia and [ADOL-C](https://github.com/coin-or/ADOL-C) is handled by [CxxWrap.jl](https://github.com/JuliaInterop/CxxWrap.jl), which, for example, allows to pass a `Cint` from Julia into a `int` in C++ automatically. Most functions 
of [ADOL-C](https://github.com/coin-or/ADOL-C) modify pre-allocated memory declared as `double*`, `double**`, or `double***` to store a functions' results. Two apparent options exist for providing
the pre-allocated data to the C++ functions called from Julia. The first option would be to write wrappers in C++, which allocate the memory every time before the actual [ADOL-C](https://github.com/coin-or/ADOL-C) function call. This would cease control over the allocations from Julia, but it would be easier to work with on the Julia side, and the C++ side would have full control over the memory. The second option is to allocate  C++-owned data from Julia by calling [ADOL-C](https://github.com/coin-or/ADOL-C)'s allocation methods from Julia. This data is then passed from Julia to [ADOL-C](https://github.com/coin-or/ADOL-C)'s functions which mutates the allocated data, and access the mutated values from Julia. The second option allows more control over the data, but a user has to be aware of some critical aspects: 
1. C++ owned memory is not automatically garbage collected and can lead to memory leaks quickly if not released. For example, having a Julia function that allocates a `double**` in C++ and binds this pointer to a variable in a Julia would release the  bound variable when going out of the functions' scope, but the C++ memory would still be  there, you just cannot access it anymore.
2. There is no out-of-bounds error checking to prevent accessing or setting of  data outside of the allocated area, which may lead to segmentation faults and program crashes.
3. If you want to do computations with the C++ data, you either have to copy these to a corresponding Julia type or write access  methods to work with the C++  allocated data.

ADOLC.jl implements the second option and wrapps the C++ memory in a `mutable struct` in Julia: There are three types [`CxxVector`](@ref), [`CxxMatrix`](@ref) and [`CxxTensor`](@ref). The first critical aspect is avoided by attaching the structs with a `finalizer` allowing Julia's garbage collector to release the C++ owned memory. We implement the usual utilities for Array-data to handle the access and tackle the second critical aspect. Since these types are subtypes of `AbstractVector{Cdouble}`, `AbstractMatrix{Cdouble}` and `AbstractArray{Cdouble, 3}` you can work with them like corresponding Julia data. Therefore, point three is also avoided. 
The intended use-case of the wrapper types is shown below.

The [`derivative!`](@ref) driver requires a pre-allocated [`CxxVector`](@ref), [`CxxMatrix`](@ref) or [`CxxTensor`](@ref). For [first-](@ref "First-Order") and [second-order](@ref "Second-Order") computations the problem-specific allocation is done using [`allocator`](@ref). This function allocates the wrapped C++ memory for your specific case (i.e., for the problem-specific parameters `m`, `n`, `mode`, `num_dir`, `num_weights`). For example:
```@example
using ADOLC # hide
f(x) = (x[1] - x[2])^2
x = [3.0, 7.5]
dir = [1/3, 1/7]
m = 1
n = 2
mode = :jac_vec
num_dir = size(dir, 2)[1]
num_weights = 0
cxx_res = allocator(m, n, mode, num_dir, num_weights)
derivative!(cxx_res, f, m, n, x, mode, dir=dir)
```
For [higher-order](@ref "Higher-Order") derivatives one has to allocate `res` as `CxxMatrix`:
```@example
using ADOLC # hide
f(x) = [x[1]^4, x[2]^3*x[1]]
x = [1.0, 2.0]
partials = [[1], [2], [3]]
seed = CxxMatrix([[1.0, 1.0];;])
m = 2
n = 2
res = CxxMatrix(m, length(partials))
derivative!(res, f, m, n, x, partials, seed)
res
```
If you really need the Julia types `Vector{Cdouble}`, `Matrix{Cdouble}` or `AbstractArray{Cdouble, 3}` feel free to use [`jl_allocator`](@ref) and [`cxx_res_to_jl_res!`](@ref)( or [`cxx_res_to_jl_res`](@ref)):
```@example
using ADOLC # hide
f(x) = (x[1] - x[2])^2
x = [3.0, 7.5]
dir = [1/3, 1/7]
m = 1
n = 2
mode = :jac_vec
num_dir = size(dir, 2)[1]
num_weights = 0
cxx_res = allocator(m, n, mode, num_dir, num_weights)
jl_res = jl_allocator(m, n, mode, num_dir, num_weights)
derivative!(cxx_res, f, m, n, x, mode, dir=dir)

# conversion 
cxx_res_to_jl_res!(jl_res, cxx_res)

```

## Seed-Matrix
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

## Tape Management
[ADOL-C](https://github.com/coin-or/ADOL-C) (and therefore ADOLC.jl) leverages a *tape* for its 
derivative calculations except for its tape-less forward-mode, which is applied to compute Jacobians for certain inputs. The tape's task is to save information of all operations in which [ADOL-C]
(https://github.com/coin-or/ADOL-C)'s derivative types are involved, and the connection to other 
operations. Thus, the tape represents a variant of the computational graph of the user-defined function `f` at a given point `x`. The stored information is used for univariate Taylor polynomial propagation computing [higher-order](@ref "Higher-Order") derivatives, and for the application of the reverse-mode in [first-](@ref "First-Order") and [second-order](@ref "Second-Order") calculations. Since the tape only stores the calling sequence for the first given input point `x`, a derivative computation based on the tape could lead to wrong results for other input points. However, the tape's construction is expensive, making it beneficial to create the tape only once and 
recreate it if the computational graph of `f` changes. For example, changes in the computational graph of the `f` might occur if the function is composed of several `if`-statements. In most cases, [ADOL-C](https://github.com/coin-or/ADOL-C) stops the program with an error whenever the tape has to be recreated for a new input. In ADOLC.jl, the [`derivative`](@ref) ([`derivative!`](@ref)) driver builds the tape automatically. Users can set the `resue_tape` flag to suppress this build process. In the first call to the driver, the tape identifier `tape_id` has to be specified. In the following calls of [`derivative`](@ref) ([`derivative!`](@ref)), set the flag `reuse_tape` to `true` and pass the identifier again. As demonstrated in the following example, the application is straightforward.

```@example
using ADOLC # hide 
f(x) = 1x[1]*x[2]^2 - x[3]^3
x = [-1.5, 1.0, -1.5]
dir = [0.0, 0.0, -1.0]
mode = :hess_vec
tape_id = 0
num_iters = 99
res0 = derivative(f, x, mode, dir=dir, tape_id=tape_id)
# update x ....
for i in 1:num_iters
    res = derivative(f, x, mode, dir=dir, tape_id=tape_id, reuse_tape=true)
    # do computations ....
end
```
## Performance Tips
The following tips are meant to decrease the derivative computation's runtime complexity, especially when derivatives of the same function are needed repeatedly. There are two major modifications for all kinds of derivatives: 
1) Use the [`derivative!`](@ref) driver, and work with [`allocator`](@ref) as explained in the guide [Working with C++ Memory](@ref)
2) Reuse the tape as often as possible. [`derivative!`](@ref) ([`derivative`](@ref)) supply the flag `reuse_tape`, which if set to `true` suppresses the creation of the tape, in addition the identifiyer of an existing tape must be provided as the parameter `tape_id`. More details can be found [here](@ref "Tape Management").
An example could look like this:
```@example
using ADOLC # hide 

# problem setup
f(x) = (x[1] - x[2])^2
x = [3.0, 7.5]
dir = [1/3, 1/7]
m = 1
n = 2
mode = :jac_vec
num_dir = size(dir, 2)[1]
num_weights = 0
tape_id = 1
num_iters = 100

# pre-allocation 
cxx_res = allocator(m, n, mode, num_dir, num_weights)

derivative!(cxx_res, f, m, n, x, mode, dir=dir, tape_id=tape_id)

# conversion 

# do computations ....

for i in 1:num_iters
    # update x ...
    derivative!(cxx_res, f, m, n, x, mode, dir=dir, tape_id=tape_id, reuse_tape=true)
    # do computations ... 
end
```
Moreover, for [higher-order](@ref "Higher-Order") derivatives you might consider the generation of a `seed`. However, if you do not pass a `seed` to the [`derivative!`](@ref) ([`derivative`](@ref)) driver, the partial identity is created as the `seed` automatically (see [here](@ref "Seed-Matrix")). 

## Univariate Taylor Polynomial Propagation

The univariate Taylor polynomial propagation (UTPP) aims to compute the univariate Taylor polynomial (UTP) $$\varphi_f$$ of a given function $$f$$ and is given as the following polynomial in $$t \in \mathbb{R}$$:
```math
\varphi_f(t) := \sum_{i=0}^d f^i(x)t^i,
```
where $$d \in \mathbb{N}$$ is the maximal order or degree and $$x \in \mathbb{R}^n$$ the evaluation point of the UTP. $$f^i(x)$$ is the $$i$$-th uniariate Taylor coefficient (UTC)
```math
f^i(x) := \sum_{\underset{|k|=i}{k \in \mathbb{N}^n}} \frac{1}{k!}\frac{\partial^k f(x)}{\partial^k x}.
```
Since UTP is computed component-wise (w.r.t the output of $$f$$), it is usually assumed $$f$$ has a one dimensional output. 

To compute the polynomial, UTPP leverages so-called recurrence relations. Recurrence relations are formulas specifying the UTC.
For example:  

Let $$n=2$$, $$f(x) = x_1 + x_2$$ and $$h$$ and $$g$$ suitable functions. The recurrence relation for the $$i$$-th UTC up to degree $$d=2$$ is
```math
f^i(x) = h^i(x) + g^i(x).
```
Therefore, if we know the UTPs of $$h$$ and $$g$$ we can conclude the UTP of $$f$$. These kind of formulas motivate the term "propagation" of input polynomials. CITE HERE AD packages implement the recurrence relations for a set of elemental functions. If a user-defined function is decomposable into these elemental functions, the AD package can computes the UTP automatically.  

ADOLC.jl leverages [ADOL-C](https://github.com/coin-or/ADOL-C)'s utilities for UTPP and wraps it in the [`univariate_tpp`](@ref) driver, which allows the easy computation of Taylor polynomials of functions with arbitrary input and output dimension. To demonstrate the application of [`univariate_tpp`](@ref), lets go back to the example. If we set $$n = 2$$ and $$h(x) = \sin(x_1)$$, $$g(x) = x_2$$ for all $$x=(x_1, x_2) \in \mathbb{R}^2$$, we have by definition of the UTC
```math
f^0(x) = h^0(x) + g^0(x) = \sin(x_1) + x_2 \\
f^1(x) = h^1(x) + g^1(x) = \cos(x_1)x_1' + x_2' \\
f^2(x) = \frac{1}{2}\left(-\sin(x_1)(x_1')^2 + \cos(x_1)x_1''\right) + \frac{1}{2}x_2''.
```
The formulas above emphasize that the values of $$x_i'$$ and $$x_i''$$ must be given. Intuitively one could set them to the derivatives of the projections by interpreting $$x_i = p_i(x)$$. Then $$x_i' = 1$$ and all further derivatives are zero. However, the input values $$x_i$$ could be passed in from a complex computation. Then it would be helpful to allow the derivatives $$x_i', x_i'', \dots$$ to carry the information of such a function. To this end, [ADOL-C](https://github.com/coin-or/ADOL-C) allows a user to provide distinct input UTC for each input variable $$x_i$$, where the evaluation point is one dimensional. This UTC has the form:
```math
\sum_{j=0}^d y^j t^j.
```
The coefficients $$y^j$$ are user-given, which means we need $$d+1$$ coefficients for every input variable $$x_i$$. If the UTC above corresponds to the input variable $$x_1$$, internally the following connection between the coefficient $$y^i$$ and the $$i$$-th derivative of $$x_1$$ is expected
```math
y^i = \frac{1}{i!}x_1^{(i)}.
```
Again note that in this case $$x_1$$ is provided by some function, which makes sense of the derivative. The output of [`univariate_tpp`](@ref) are the UTC of $$f$$'s UTP.  

In our example, lets first set the values corresponding to the UTP of the projections $$p_i(x) = x_i$$. For $$x_1$$ we get $$y^0 = x_1$$, $$y^1 = 1$$ and $$y^2 = 0$$. Similarly for $$x_2$$. If we further choose $$x = (\frac{\pi}{2}, 0.5)$$ we can compute the UTC $$f^0$$, $$f^1$$ and $$f^2$$ of $$f$$ with ADOLC.jl easily:
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


Next, we comput the UTP at the same evaluation point $$x = (\frac{\pi}{2}, 0.5)$$, but now $$x_2$$ is obtained from the function $$x_2(s) = s^2 + 3s + 0.5$$, while $$x_1$$ is still determined by the projection $$p_1$$. Then, $$x_2(0) = 0.5$$, $$x_2'(0) = 3$$ and $$x_2''(0) = 2$$. The corresponding UTCs are given as $$y^0 = 0.5$$, $$y^1 = 3$$ and $$y^2 = 1$$. In this case the [`univariate_tpp`](@ref) driver gets the `init_tp`, which specifies input UTCs. The UTC's of $$f$$ are again easily computed:
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
