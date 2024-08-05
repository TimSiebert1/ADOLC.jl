```@meta
DocTestSetup = quote
    using ADOLC
end
```

# Working with C++ Memory

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
