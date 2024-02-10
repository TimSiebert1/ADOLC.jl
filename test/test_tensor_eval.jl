using ADOLC
using ADOLC.TbadoubleModule
using Test


function func(x)
    return [
        x[1] * x[2]^3,
        x[1] + x[3] / x[2]
    ]
end

init_point = [-1.0, 2.0, -1.0]

num_independent = length(init_point)
num_dependent = 2
y = Vector{Float64}(undef, num_dependent)
a = [Adouble{TbAlloc}() for _ in 1:num_independent]
tape_num = 1
keep = 1
trace_on(tape_num, keep)
a << init_point
b = func(a)
b >> y
trace_off(0)

jac = _higher_order(tape_num, init_point, 2, 3, 1, compressed_out=false)


@test jac[1, 1] == 8.0
@test jac[2, 1] == -12.0
@test jac[3, 1] == 0.0

@test jac[1, 2] == 1.0
@test jac[2, 2] == 0.25
@test jac[3, 2] == 0.5

################################## second order ################################

function func(x)
    return [
        x[1] * x[2]^3,
        x[1] + x[3] / x[2]
    ]
end

x0 = [-1.0, 2.0, -1.0]

jac, _ = ADOLC.gradient(func, x0, 2, derivative_order=2, compressed_out=false)

@test jac[1, 1, 1] == 0.0
@test jac[1, 2, 1] == 12.0
@test jac[1, 3, 1] == 0.0

@test jac[1, 2, 1] == 12.0
@test jac[2, 2, 1] == -12.0
@test jac[3, 2, 1] == 0.0

@test jac[1, 3, 1] == 0.0
@test jac[2, 3, 1] == 0.0
@test jac[3, 3, 1] == 0.0

@test jac[1, 1, 2] == 0.0
@test jac[2, 1, 2] == 0.0
@test jac[3, 1, 2] == 0.0

@test jac[1, 2, 2] == 0.0
@test jac[2, 2, 2] == -0.25
@test jac[3, 2, 2] == -0.25

@test jac[1, 3, 2] == 0.0
@test jac[2, 3, 2] == -0.25
@test jac[3, 3, 2] == 0.0

println("Done")