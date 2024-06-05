using ADOLC
using ADOLC.TbadoubleModule

enableMinMaxUsingAbs()

function func_eval(x)
    return (
        max(-x[1] - x[2], -x[1] - x[2] + x[1]^2 + x[2]^2 - 1) +
        max(-x[2] - x[3], -x[2] - x[3] + x[2]^2 + x[3]^2 - 1)
    )
end

x = [-0.500000, -0.500000, -0.500000]
n = length(x)
y = Vector{Float64}(undef, 1)
m = length(y)
a = [Adouble{TbAlloc}() for _ in 1:length(x)]
b = [Adouble{TbAlloc}() for _ in 1:length(y)]

tape_num = 0
trace_on(tape_num, 1)
a << x
b[1] = func_eval(a)
b >> y
trace_off(0)

abs_normal_problem = AbsNormalProblem{Float64}(tape_num, m, n, x, y)

abs_normal!(abs_normal_problem, tape_num)

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

print(abs_normal_problem.L[2, 2])

x = -1.0
# call the abs_normal driver
n = length(x)
y = Vector{Float64}(undef, 1)
m = length(y)
a = Adouble{TbAlloc}()

tape_num = 0
trace_on(tape_num, 1)
a << x
b = abs(a)
y = b >> y
trace_off()

abs_normal_problem = AbsNormalProblem{Float64}(tape_num, m, n, [x], y)
abs_normal!(abs_normal_problem, tape_num)

abs_normal_problem = AbsNormalProblem{Float64}(tape_num, m, n, [1.0], y)
abs_normal!(abs_normal_problem, tape_num)
