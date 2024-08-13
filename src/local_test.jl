using Test

function f(x, y)
    a = x*y
    return a^2
end
x = 3.0
y = -1.0

tape_id = 0
m = 1
n = 1

ADOLC.trace_on(tape_id)
adoubs = create_independent(x)
y = mkparam(y)
b = f(adoubs, y)
dependent(b)
ADOLC.trace_off()
jac = CxxVector(1)
ADOLC.TbadoubleModule.zos_forward(tape_id, m, n, 1, [x], [0.0])
ADOLC.TbadoubleModule.fos_reverse(tape_id, m, n, [1.0], jac.data)


@test jac[1] == 2*x*y^2

y = -4.5
ADOLC.TbadoubleModule.set_param_vec(tape_id, 1, [y])
ADOLC.TbadoubleModule.zos_forward(tape_id, m, n, 1, [x], [0.0])
ADOLC.TbadoubleModule.fos_reverse(tape_id, m, n, [1.0], jac.data)
@test jac[1] == 2*x*y^2



################################
active = [1]

ADOLC.trace_on(tape_id+1)
adoubs, y = create_independent(x, y)
b = f(adoubs, y)
dependent(b)
ADOLC.trace_off()
jac = CxxVector(1)
ADOLC.TbadoubleModule.zos_forward(tape_id, m, n, 1, [x], [0.0])
ADOLC.TbadoubleModule.fos_reverse(tape_id, m, n, [1.0], jac.data)


@test jac[1] == 2*x*y^2

y = -4.5
ADOLC.TbadoubleModule.set_param_vec(tape_id, 1, [y])
ADOLC.TbadoubleModule.zos_forward(tape_id, m, n, 1, [x], [0.0])
ADOLC.TbadoubleModule.fos_reverse(tape_id, m, n, [1.0], jac.data)
@test jac[1] == 2*x*y^2


##########################################################
active = [1]
x = 3.0
param = -1.0
tape_id=1111
res = derivative(f, x, param, :jac, tape_id=tape_id)
@test res[1] == 2*x*param^2


param = -4.5
res = derivative(f, x, param, :jac, reuse_tape=true, tape_id=tape_id)
@test res[1] == 2*x*param^2