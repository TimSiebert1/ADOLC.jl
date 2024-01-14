include("../src/ADOLC.jl")
using .ADOLC.AdoubleModule

function chained_lq(x)
    return (
        max(-x[1] - x[2], -x[1] - x[2] + x[1]^2 + x[2]^2 - 1) +
        max(-x[2] - x[3], -x[2] - x[3] + x[2]^2 + x[3]^2 - 1)
    )
end

x = [0.0, 2.0, -1.0]
a = [AdoubleCxx() for _ in 1:3]
y = 0.0
trace_on(1)

a << x
b = chained_lq(a)
b >> y
trace_off(0)


g = gradient(1, x)

println("Should be -1.0: ", g[1])
println("Should be 6.0: ", g[2])
println("Should be -3.0: ", g[3])
