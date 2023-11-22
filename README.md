This package wraps the C/C++ automatic differentation library [ADOL-C](https://github.com/coin-or/ADOL-C) for the usage in [Julia](https://julialang.org/). 

## How to use this package?

1. git clone the repo
2. use the commands `cd ADOLC_wrap/src && julia --project build.jl`
3. check if its works by using `julia --project test.jl`
4. Use the ADOLC functions by import the package with `include("/path/to/ADOLC_wrap/src/ADOLC_wrap.jl"); using .ADOLC_wrap`

## Example
After including the package, define the function you are planning to differentiate.
```julia
function chained_cresecent1(x)
    return max(x[1]^2 + (x[2] - 1)^2 + x[2] - 1, -x[1]^2 - (x[2] - 1)^2 + x[2] + 1)
end
```
Then define the point for the derivative, initialize a vector of adoubles and initiaize the output value of the function. The output value is in the end needed to speciy the dependent variable.
```julia
x = [-1.5, 2.0]
a = [adouble() for _ in 1:2]
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

However, this should be done carefully. Since the tape is not rebuild it could lead to wrong results if the function contains conditional statements. If the control flow differs at the new point you will see a Warning like ```ADOL-C Warning: Branch switch detected in comparison (operator le_zero). Forward sweep aborted! Retaping recommended!```. Then you have to reevaluate the whole function for the new point. If the control flow isn't changing, everything will work fine. 
Note, in contrast to someones expectation some functions like the `max` function does not contain conditional statements due to implementation "tricks". 
