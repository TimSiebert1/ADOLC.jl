```@meta
DocTestSetup = quote
    using ADOLC
end
```
# Performance Tips
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
