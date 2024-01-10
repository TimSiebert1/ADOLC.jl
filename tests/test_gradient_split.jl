include("../src/ADOLC_wrap.jl")
using .ADOLC_wrap
using .ADOLC_wrap.TladoubleModule
using .ADOLC_wrap.Adouble
using Test

# Chained LQ

function chained_lq(x)
    return (
        max(-x[1] - x[2], -x[1] - x[2] + x[1]^2 + x[2]^2 - 1) +
        max(-x[2] - x[3], -x[2] - x[3] + x[2]^2 + x[3]^2 - 1)
    )
end

x = [0.0, 2.0, -1.0]
g = Main.gradient(chained_lq, x, 1, switch_point=100)
g1 = Main.gradient(chained_lq, x, 1, switch_point=100, mode=:tape_based)


@test g[1]==-1.0
@test g[2]==6.0
@test g[3]==-3.0

@test g[1]==g1[1]
@test g[2]==g1[2]
@test g[3]==g1[3]


