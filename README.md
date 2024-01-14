This package wraps the C/C++ automatic differentation library [ADOL-C](https://github.com/coin-or/ADOL-C) for the usage in [Julia](https://julialang.org/). 

## How to use this package?

1. git clone the repo
2. use the commands `cd ADOLC.jl/src && julia --project build.jl`
3. check if its works by using `cd .. && julia --project examples/gradient.jl`
4. Use the ADOLC functions by import the package with `include("/path/to/ADOLC.jl/src/ADOLC.jl"); using .ADOLC.AdoubleModule` or `using .ADOLC.TladoubleModule` for tape-based and tape-less ADOLC.

## Example
After including the package,
```julia
include("/path/to/ADOLC.jl/src/ADOLC.jl")
using .ADOLC.AdoubleModule
```
define the function you are planning to differentiate.
```julia
function chained_cresecent1(x)
    return max(x[1]^2 + (x[2] - 1)^2 + x[2] - 1, -x[1]^2 - (x[2] - 1)^2 + x[2] + 1)
end
```
Then define the point for the derivative, initialize a vector of adoubles and initiaize the output value of the function. The output value is in the end needed to speciy the dependent variable.
```julia
x = [-1.5, 2.0]
a = [AdoubleCxx() for _ in 1:2]
y = 0.0
```
Like in ADOLC, you have to use the `trace_on` and `trace_off` functions to start and end the recording of the operations on the tape. The `trace_on` function requires a specifier for the tape, i.e. a corresponding integer. The `trace_off` has to called with 0 or 1 to specify, whether the tape has to be stored or not. For more information you might look into the [ADOLC-manual](https://usermanual.wiki/Pdf/adolcmanual.230286982/view). The tape is then used to calculate the derivative. Between the `trace_on` and `trace_off` function you have to specify the independent variables with `<<` and dependent variables by `>>`. These operators also set the values of `a` and `y`.
```julia
trace_on(1)
a << x
b = chained_cresecent1(a)
b >> y
trace_off(0)
```
Finally, you can calculate the derivative with respect to all independent variables at the point `x`. To use the right tape, this function requires to set the tape specifier (1 in our case). The `gradient` function calculates the derivatives.
```julia
g = gradient(1, x)
println("Crescent: ", g[1], ", ", g[2])
```

You can also evaluate the derivate at an other point.

```julia
g = gradient(1, [-4.5, 2.0])
println("Crescent: ", g[1], ", ", g[2])
```

However, this should be done carefully. Since the tape is not rebuild it could lead to wrong results if the function contains conditional statements. If the control flow differs at the new point you will see a warning like ```ADOL-C Warning: Branch switch detected in comparison (operator le_zero). Forward sweep aborted! Retaping recommended!```. Then you have to reevaluate the whole function for the new point and create a new tape. If the control flow isn't changing, everything will work fine. 
Note, in contrast to someones expectation some functions like `max` does not contain conditional statements due to implementation "tricks". 


## Example for abs-normal interface
For this example we utilize the abs-normal interface of ADOLC. To use the interface load the libraries
```julia
include("/PATH/TO/ADOLC/src/ADOLC.jl")
using .ADOLC.AdoubleModule
using .ADOLC
```
Enable the abs-normal functionalities by calling
```julia
enableMinMaxUsingAbs()
```
Specify your function. For this example, the following is used.
```julia
function func_eval(x)
    return (max(-x[1]-x[2], -x[1]-x[2]+x[1]^2+x[2]^2-1) + max(-x[2]-x[3], -x[2]-x[3]+x[2]^2+x[3]^2-1))
end 
```
Select a start point and allocate the memory for the output and vector of adoubles with
```julia
x = [-0.500000, -0.500000, -0.500000]
n = length(x)
y = Vector{Float64}(undef, 1)
m = length(y)
a = [AdoubleCxx() for _ in 1:length(x)]
b = [AdoubleCxx() for _ in 1:length(y)]
```
Like in the previous example, the function is evaluated by calling it with the vector of adoubles `a` and everything is recored on the tape.
```julia
tape_num = 0
trace_on(tape_num)
a << x
b[1] = func_eval(a)
b >> y
trace_off(0)
```
For the application of the `abs_normal` driver, specify the `AbsNormalProblem` 
```julia
abs_normal_problem =
    AbsNormalProblem{Float64}(tape_num, m, n, x, y)
```
which handles the matrix allocations and stores the relevant problem information. 
```julia
    m::Int64
    n::Int64
    num_switches::Int32

    x::CxxVector{T}
    y::CxxVector{T}
    z::CxxVector{T}

    cz::CxxVector{T}
    cy::CxxVector{T}

    Y::CxxMatrix{T} 
    J::CxxMatrix{T}
    Z::CxxMatrix{T} 
    L::CxxMatrix{T}
```
Then call 
```julia
abs_normal!(abs_normal_problem, tape_num)
```
and test the results
```julia
using Test
@test abs_normal_problem.Y[1, 1] == -1.5
@test abs_normal_problem.Y[1, 2] == -3.0
@test abs_normal_problem.Y[1, 3] == -1.5

@test abs_normal_problem.J[1, 1] == 0.5
@test abs_normal_problem.J[1, 2] == 0.5

@test abs_normal_problem.Z[1, 1] == -1.0
@test abs_normal_problem.Z[1, 2] == -1.0
@test abs_normal_problem.Z[1, 3] == 0.0
@test abs_normal_problem.Z[2, 1] == 0.0
@test abs_normal_problem.Z[2, 2] == -1.0
@test abs_normal_problem.Z[2, 3] == -1.0

@test abs_normal_problem.L[1, 1] == 0.0
@test abs_normal_problem.L[1, 2] == 0.0
@test abs_normal_problem.L[2, 1] == 0.0
@test abs_normal_problem.L[2, 2] == 0.0
```

This and further examples can be found in the `examples` file.
