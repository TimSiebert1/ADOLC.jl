```@meta
DocTestSetup = quote
    using ADOLC
end
```

# Low Level Control

The main functionalities of ADOLC.jl (e.g. [`derivative`](@ref)) aim to hide the underlying AD details from the user allowing the user to formulate the required derivative computation in an intuitive way. However, in reality derivative computations with AD tools should be done more carefully, especially when it comes to complex simulations. An informative article about the potential issues with AD was, for example, written by Hückelheim et al. [pitfalls](@cite). \
[ADOL-C](https://github.com/coin-or/ADOL-C) includes various options to provide a user fine-grained control over the derivative computation. ADOLC.jl is going to make them accessible in Julia. At the moment, the ADOLC.jl provides the fine-grained control over the independent and dependent variables tracked on the tape. This option is explained in the following examples. 

### Low Level Tape Control

If not all derivatives are required, i.e., only the derivatives of subset of the output parameters with respect to a subset of the input variables is needed, ADOLC.jl provides suitable options. To tape a certain parameter or a `Vector{Cdouble}` of parameters, the [`create_independent`](@ref) method should be used. For output parameters use the [`dependent`](@ref) method. In the following example we compute the derivative of the first entrees of the two cholesky factorization matrizes with respect to a single variable. At first the problem set-up is defined.

```@example 1
using ADOLC # hide
using LinearAlgebra

n = 3
x = 1.0
A = ones(n, n)
A = A + diagm(0 => Adouble{TbAlloc}([10, 10, 10]))

tape_id = 1
```
Now, to create a tape the function `ADOLC.trace_on` is used. Afterwards all calls to `create_independent` and `dependent` will write the corresponding information on the active tape. In our example, we want to mark `x` as independent and introduce the dependece of the component `A[1, 1]` from `x` and compute the cholesky factorization:
```@example 1
ADOLC.trace_on(tape_id)
adoub = create_independent(x)
A[1, 1] = A[1, 1] * adoub
fact = cholesky(A)
```
Finally, we close the tape and the first entries of the factors and, for veryfication reasons, the first entry of `A` are marked as dependent variable.
```@example 1
dependent(fact.L[1, 1])
dependent(fact.U[1, 1])
dependent(A[1, 1])
ADOLC.trace_off()
```
To compute the derivatives, the `fos_forward` method of ADOL-C is leveraged:
```@example 1
result = CxxVector(n)
jac = CxxVector(n)
dir = CxxVector([1.0])

ADOLC.TbadoubleModule.fos_forward(tape_id, 3, 1, 0, [x], dir.data, result.data, jac.data)
```
We check our result to see if everything ran as expected.
```@example 1
using Test
@test jac[1] * fact.U[1, 1] + jac[2] * fact.L[1, 1] == jac[3]

```
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