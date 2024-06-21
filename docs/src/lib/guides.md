## Working with C++ Memory

ADOLC.jl is a wrapper of the C/C++ library [ADOL-C](https://github.com/coin-or/ADOL-C). Wrapper means
data from Julia is passed to C++, and calls to functions in Julia trigger C++ function calls to get output data in Julia. The communication between Julia and [ADOL-C](https://github.com/coin-or/ADOL-C) is handled by [CxxWrap.jl](https://github.com/JuliaInterop/CxxWrap.jl), which, for example, allows to pass a `Cint` from Julia into a `int` in C++ automatically. Most functions 
of [ADOL-C](https://github.com/coin-or/ADOL-C) modify pre-allocated memory declared as `double*`, `double**`, or `double***` to store a functions' results. Two apparent options exist for providing
the pre-allocated data to the C++ functions called from Julia. The first option would be to write wrappers in C++, which allocate the memory every time before the actual [ADOL-C](https://github.com/coin-or/ADOL-C) function call. This would cease control over the allocations from Julia, but it would be easier to work with on the Julia side, and the C++ side would have full control over the memory. The second option is to allocate  C++-owned data from julia by calling [ADOL-C](https://github.com/coin-or/ADOL-C)'s allocation methods from Julia, to pass these data to [ADOL-C](https://github.com/coin-or/ADOL-C)'s functions mutating the allocated data, and to access the mutated values from Julia. The second option allows more control over the data, but a user has to be aware of some critical aspects: 
1. C++ owned memory is not automatically garbage collected and can lead to memory leaks quickly if not released. For example, having a Julia function that allocates a `double**` in C++ and binds this pointer to a variable in a Julia would release the  bound variable when going out of the functions' scope, but the C++ memory would still be  there, you just cannot access it anymore.
2. There is no out-of-bounds error checking to prevent accessing or setting of  data outside of the allocated area, which may lead to segmentation faults and program crashes.
3. If you want to do computations with the C++ data, you either have to copy these to a corresponding Julia type or write access  methods to work with the C++  allocated data.

ADOLC.jl implements the second option. The critical aspects can still be avoided using the driver [`derivative`](@ref), which handles the C++ allocated memory for you. However, the best performance is obtained when using [`derivative!`](@ref).
For first- and second-order derivative computations, the [`derivative!`](@ref) driver requires  a pre-allocated container of C++ allocated data. The allocation is done using [`allocator`](@ref). This function allocates C++ memory for your specific case (i.e., for the problem-specific parameters `m`, `n`, `mode`, `num_dir`, `num_weights`). Thus, the computation of the derivative just  utilizes [`derivative`](@ref). For example:
```@example
using ADOLC
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
deallocator!(cxx_res, m, mode)
```
The first critical point is tackled using the [`deallocator!`](@ref) function, which handles the release of the C++ memory. Of course, one wants to conduct computations with `cxx_res`. The recommended way to do so is to pre-allocate a corresponding Julia container (`Vector{Float64}`, `Matrix{Float64}` or `Array{Float64, 3}`) obtained from [`jl_allocator`](@ref) and copy the data from `cxx_res` the Julia storage `jl_res` by leveraging [`cxx_res_to_jl_res!`](@ref):
```@example
using ADOLC
f(x) = (x[1] - x[2])^2
x = [3.0, 7.5]
dir = [1/3, 1/7]
m = 1
n = 2
mode = :jac_vec
num_dir = size(dir, 2)[1]
num_weights = 0

# pre-allocation 
jl_res = jl_allocator(m, n, mode, num_dir, num_weights)
cxx_res = allocator(m, n, mode, num_dir, num_weights)

derivative!(cxx_res, f, m, n, x, mode, dir=dir)

# conversion 
cxx_res_to_jl_res!(jl_res, cxx_res, m, n, mode, num_dir, num_weights)

# do computations .... 


deallocator!(cxx_res, m, mode)
```
Since you work with Julia data, the procedure above avoids the second and third points of the critical aspects but includes an additional allocation.  

!!! warning 
    `cxx_res` still stores a pointer. The corresponding memory is destroyed, but `cxx_res` is managed by Julia's garbage collector. Do not use it.


In the future, the plan is to implement a struct that combines the Julia and C++ arrays with a finalizer that enables Julia's garbage collector to manage the C++ memory. 



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
seed = [[1.0, 1.0];;]
res = derivative(f, x, partials, seed)

# output

1×3 Matrix{Float64}:
 2.0  14.0  54.0
```

## Tape Management


## Performance Tips
The following tips are meant to decrease the derivative computation's runtime complexity, especially when derivatives of the same function are needed repeatedly. There are two major modifications for all kinds of derivatives: 
1) Use the [`derivative!`](@ref) driver, and work with [`allocator`](@ref), [`deallocator!`](@ref), and [`jl_res_to_cxx_res!`] as explained in the guide [Working with C++ Memory](@ref)
2) Reuse the tape as often as possible. [`derivative!`](@ref) ([`derivative`](@ref)) supply the flag `reuse_tape`, which if set to `true` suppresses the creation of the tape, in addition the identifiyer of an existing tape must be provided as the parameter `tape_id`. More details can be found [here](@ref "Tape Management").
An example could look like this:
```@example
using ADOLC

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
jl_res = jl_allocator(m, n, mode, num_dir, num_weights)
cxx_res = allocator(m, n, mode, num_dir, num_weights)

derivative!(cxx_res, f, m, n, x, mode, dir=dir, tape_id=tape_id)

# conversion 
cxx_res_to_jl_res!(jl_res, cxx_res, m, n, mode, num_dir, num_weights)
# do computations ....

for i in 1:num_iters
    # update x
    derivative!(cxx_res, f, m, n, x, mode, dir=dir, tape_id=tape_id, reuse_tape=true)
    cxx_res_to_jl_res!(jl_res, cxx_res, m, n, mode, num_dir, num_weights)
    # do computations ... 
end
deallocator!(cxx_res, m, mode)
```
Moreover, for [higher-order](@ref "Higher-Order") derivatives you might consider the generation of a `seed`. However, if you do not pass a `seed` to the [`derivative!`](@ref) ([`derivative`](@ref)) driver, the partial identity is created as the `seed` automatically (see [here](@ref "Seed-Matrix")). 

