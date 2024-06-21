## Working with C++ Memory

ADOLC.jl is a wrapper of the C/C++ library [ADOL-C](https://github.com/coin-or/ADOL-C). Wrapper means
data from Julia is passed to C++, and calls to functions in Julia trigger C++ function calls to get output data in Julia. The communication between Julia and [ADOL-C](https://github.com/coin-or/ADOL-C) is handled by [CxxWrap.jl](https://github.com/JuliaInterop/CxxWrap.jl), which, for example, allows to pass a `Cint` from Julia into a `int` in C++ automatically. Most functions 
of [ADOL-C](https://github.com/coin-or/ADOL-C) modify pre-allocated memory declared as `double*`, `double**`, or `double***` to store the functions' results. Two apparent options exist for providing
the pre-allocated data to the C++ functions from Julia. The first option would be to write wrappers in C++, which allocates the memory every time before the actual [ADOL-C](https://github.com/coin-or/ADOL-C) function call. This would lose control over the allocations, but it would be easier to work with on the Julia side and the C++ side would have full control of the memory. The second option is to allocate of C++-owned data from julia by calling ADOL-C's allocation methods, to pass these data to [ADOL-C](https://github.com/coin-or/ADOL-C)'s functions mutating the allocated data, and to access the mutated values from
Julia. The second option allows more control over the data, but a user has to be aware of some critical aspects: 
1. C++ owned memory is not automatically garbage collected and can lead to memory leaks quickly if not released. For example, having a Julia function that allocates a `double**` in C++ and binds this pointer to a variable in a Julia would release the  bound variable when going out of the functions' scope, but the C++ memory would still be  there, you just cannot access it anymore.
2. There is no out-of-bounds error checking to prevent access or setting of the data outside of the allocated area, which may lead to segmentation faults and program crashes.
3. If you want to do computations with the C++ data, you either have to copy these to a corresponding Julia type or write access  methods to work with the C++  allocated data.

ADOLC.jl implements the second option. The critical aspects can still be avoided using the driver [`derivative`](@ref), which handles the C++ allocated memory for you. However, the best performance is obtained when using [`derivative!`](@ref).
For first- and second-order derivative computations, the [`derivative!`](@ref) driver requires  a pre-allocated container of C++ allocated data. The allocation is done using [`allocator`](@ref). This function allocates C++ memory for your specific case (i.e., for the problem-specific parameters `m`, `n`, `mode`, `num_dir`, `num_weights`). Thus, the computation of the derivative just 
utilizes [`derivative`](@ref). For example:
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
The first critical point is tackled by using the [`deallocator!`](@ref) function, which handles the release of the C++ memory. Of course, one wants to conduct computations with `cxx_res`. The recommended way to do so is to pre-allocate a corresponding Julia container (`Vector{Float64}`, `Matrix{Float64}` or `Array{Float64, 3}`) obtained from [`jl_allocator`](@ref) and copy the data from `cxx_res` the Julia storage `jl_res` by leveraging [`cxx_res_to_jl_res!`](@ref):
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
Since you actually work with Julia data, the procedure above avoids the second and third point of the ciritical aspects, but includes an additional allocation.  

!!! warning 
    `cxx_res` still stores a pointer. The corresponding memory is destroyed but `cxx_res` itself is managed by Julia's garbage collector. Do not use it.


In the future, the plan is to implement a struct that combines the Julia and C++ arrays with a finalizer that enables Julia's garbage collector to manage the C++ memory. 



## Seed-Matrix
