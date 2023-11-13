This package wraps the C/C++ automatic differentation library [ADOL-C](https://github.com/coin-or/ADOL-C) for the usage in [Julia](https://julialang.org/). 

## How to use this package?

1. git clone the package
2. use the commands "cd ADOLC_wrap/src && julia --project build.jl"
3. check if its working by using "julia test.jl"


## Troubleshooting
When you obtain issues from the cmake command, your system might not found the julia package. To tackle this problem 
the folder "find_julia" includes a script to test whether you find julia or not. Firstly you should run inside "find_julia" "cmake -S. -Bbuild". If there is an error which says the cmake cant find julia you might find it by setting the paths "Julia_PREFIX" and "Julia_EXECTUABLE". The way how to do it is already include in the "CMAKELists.txt" of "find_julia". Find the right paths and uncomment the cmake commands. Run again "cmake -S. -Bbuild" inside "find_julia". If everything works fine now open the "build.jl" file inside the "src" dir. Uncomment the "Julia_PREFIX" and "Julia_EXECUTABLE". Specify the paths. Finally uncomment the "run" command that includes both path's and comment first the "run" command below. Try to run "build.jl" again. If they are still issues, please feel free to open an issue.