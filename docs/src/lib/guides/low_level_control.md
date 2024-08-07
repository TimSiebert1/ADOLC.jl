```@meta
DocTestSetup = quote
    using ADOLC
end
```

# Low Level Control

The main functionalities of ADOLC.jl (e.g. [`derivative`](@ref)) aim to hide the underlying AD details from the user allowing the user to formulate the required derivative computation in an intuitive way. However, in reality derivative computations with AD tools should be done more carefully, especially when it comes to complex simulations. An informative article about the potential issues with AD was, for example, written by Hückelheim et al. [pitfalls](@cite). \
[ADOL-C](https://github.com/coin-or/ADOL-C) includes various options to provide a user fine-grained control over the derivative computation. ADOLC.jl is going to make them accessible in Julia. At the moment, the ADOLC.jl provides the fine-grained control over the independent and dependent variables tracked on the tape. This option is explained in the following examples. 

### Derivatives of Mutating Functions
Often it is more performant to leverage mutating functions, which stores the results in a pre-allocated container. The example below computes $$A^{-1}x$$ and stores the result in $$x$$ again. We select $$A = \left(\begin{matrix} 0 & 1 \\ 1 & 0 \end{matrix}\right)$$ and $$x=[1.0, 3.0]$$. Therefore, the Jacobian is given by $$A$$ itself. ADOLC.jl can compute this Jacobian easly. Note the drivers used below are a bit more low level (as expected from this chapter). The function `zos_forward` is used to store all values on the tape and the call to `fov_reverse` computes the matrix-Jacobian product $$W^TDf$$, where $$W$$ is given by the `weights` and `Df` denotes the Jacobian.
```jldoctest
using LinearAlgebra

x = [1.0, 3.0]
A = [[0.0, 1.0] [1.0, 0.0]]

tape_id = 1
num_dep = 2
num_indep = 2

ADOLC.TbadoubleModule.trace_on(tape_id)
adoubs = ADOLC.create_independent(x)
fact = lu(A)
ldiv!(adoubs, fact, adoubs)
ADOLC.dependent(adoubs)
ADOLC.TbadoubleModule.trace_off()

result = CxxVector(num_dep)
jac = CxxMatrix(num_dep, num_indep)
weights = CxxMatrix([[1.0, 0.0] [0.0, 1.0]])

ADOLC.TbadoubleModule.zos_forward(tape_id, num_dep, num_indep, 1, x, result.data)
ADOLC.TbadoubleModule.fov_reverse(tape_id, num_dep, num_indep, num_dep, weights.data, jac.data)
result, jac

# output

([3.0, 1.0], [0.0 1.0; 1.0 0.0])
```