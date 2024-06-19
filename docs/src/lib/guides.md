## Working with C++ Memory

ADOLC.jl is a wrapper of the C/C++ library [ADOL-C](https://github.com/coin-or/ADOL-C). This means
data from Julia is passed to C++ and calls to functions in Julia trigger C++ function calls with
the goal to get output data in Julia. The communication between Julia and [ADOL-C](https://github.com/coin-or/ADOL-C) is handled by [CxxWrap.jl](https://github.com/JuliaInterop/CxxWrap.jl), which, for example, converts automatically an `Cint` from Julia into a `int` in C++. Most functions 
of [ADOL-C](https://github.com/coin-or/ADOL-C) modify pre-allocated memory in form of `double*`, `double**` or `double***` to store the functions' results


## Seed-Matrix
