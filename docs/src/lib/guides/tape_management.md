```@meta
DocTestSetup = quote
    using ADOLC
end
```
# Tape Management
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