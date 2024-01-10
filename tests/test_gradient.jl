include("../src/ADOLC_wrap.jl")
using .ADOLC_wrap
using Test
# Chained LQ

function chained_lq(x)
    return (
        max(-x[1] - x[2], -x[1] - x[2] + x[1]^2 + x[2]^2 - 1) +
        max(-x[2] - x[3], -x[2] - x[3] + x[2]^2 + x[3]^2 - 1)
    )
end

x = [0.0, 2.0, -1.0]
g = gradient(chained_lq, x, 1)

@test g[1]==-1.0
@test g[2]==6.0
@test g[3]==-3.0



################################# reverse mode test ###################################


function func(x)
    return [
        x[1] * x[2]^3,
        x[1] + x[3] / x[2]
    ]
end

init_point = [-1.0, 2.0, -1.0]
num_dependent = 2
Z = gradient(func, init_point, num_dependent, mode=:tape_based)

@test Z[1, 1] == 8.0
@test Z[2, 1] == 1.0
@test Z[1, 2] == -12.0
@test Z[2, 2] == 0.25
@test Z[1, 3] == 0.0
@test Z[2, 3] == 0.5




Z = gradient(func, init_point, num_dependent, mode=:tape_based, derivative_order=2)


@test Z[1][1, 1, 1] == 8.0
@test Z[1][1, 2, 1] == -12.0
@test Z[1][1, 3, 1] == 0.0
@test Z[1][1, 1, 2] == 0.0
@test Z[1][1, 2, 2] == 12.0
@test Z[1][1, 3, 2] == 0.0

@test Z[1][2, 1, 1] == 1.0
@test Z[1][2, 2, 1] == 0.25
@test Z[1][2, 3, 1] == 0.5
@test Z[1][2, 1, 2] == 0.0
@test Z[1][2, 2, 2] == 0.0
@test Z[1][2, 3, 2] == 0.0


@test Z[2][1, 1, 1] == 8.0
@test Z[2][1, 2, 1] == -12.0
@test Z[2][1, 3, 1] == 0.0
@test Z[2][1, 1, 2] == 12.0
@test Z[2][1, 2, 2] == -12.0
@test Z[2][1, 3, 2] == 0.0

@test Z[2][2, 1, 1] == 1.0
@test Z[2][2, 2, 1] == 0.25
@test Z[2][2, 3, 1] == 0.5
@test Z[2][2, 1, 2] == 0.0
@test Z[2][2, 2, 2] == 
@test Z[2][2, 3, 2] == 0.0