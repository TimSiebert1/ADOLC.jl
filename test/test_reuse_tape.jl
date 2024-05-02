function speelpenning(x::Vector{Adouble{T}}) where {T<:Union{TbAlloc,TlAlloc}}
    y = Adouble{T}(1.0, true)
    for i in eachindex(x)
        y *= x[i]
    end
    return y
end


function compute_grad(tape_id, m, n, x, y, weights, grad)
    zos_forward(tape_id, m, n, 1, x, y)
    fov_reverse(tape_id, m, n, m, weights, grad)
end


function test_reuse_tape_local(n)
    x = [(i + 1.0) / (2.0 + i) for i = 1:n]
    a = [Adouble{TbAlloc}() for _ in eachindex(x)]
    b = [Adouble{TbAlloc}()]
    y = 0.0

    tape_id = 0
    trace_on(tape_id)
    a << x
    b = speelpenning(a)
    y = b >> y
    trace_off()

    m = length(y)
    weights = myalloc2(m, m)
    for i = 1:m
        for j = 1:m
            weights[i, j] = 0.0
            if i == j
                weights[i, i] = 1.0
            end
        end
    end

    res = myalloc2(m, n)
    for i = 1:10
        compute_grad(tape_id, m, n, x .+ i, y, weights, res)
    end
    return res
end

function create_tape(tape_id, f, x)
    a = [Adouble{TbAlloc}() for _ in eachindex(x)]
    b = [Adouble{TbAlloc}()]
    y = 0.0

    trace_on(tape_id)
    a << x
    b = f(a)
    y = b >> y
    trace_off()
    return y
end

function test_reuse_tape_outside(n)
    tape_id = 0
    x = [(i + 1.0) / (2.0 + i) for i = 1:n]
    y = create_tape(tape_id, speelpenning, x)

    m = length(y)

    weights = myalloc2(m, m)
    for i = 1:m
        for j = 1:m
            weights[i, j] = 0.0
            if i == j
                weights[i, i] = 1.0
            end
        end
    end

    res = myalloc2(m, n)
    for i = 1:10
        compute_grad(tape_id, m, n, x .+ i, y, weights, res)
    end
    return res
end




enableMinMaxUsingAbs()

function func_eval(x)
    return (
        max(-x[1] - x[2], -x[1] - x[2] + x[1]^2 + x[2]^2 - 1) +
        max(-x[2] - x[3], -x[2] - x[3] + x[2]^2 + x[3]^2 - 1)
    )
end

function create_tape_abs_norm(tape_id, f, x)
    y = Vector{Float64}(undef, 1)
    a = [Adouble{TbAlloc}() for _ = 1:length(x)]
    b = [Adouble{TbAlloc}() for _ = 1:length(y)]

    trace_on(tape_id, 1)
    a << x
    b[1] = f(a)
    y = b >> y
    trace_off()
    return y
end

function create_abs_normal()
    tape_id = 0
    x = [-0.500000, -0.500000, -0.500000]
    y = create_tape_abs_norm(tape_id, func_eval, x)
    n = length(x)
    m = length(y)
    abs_normal_form = ADOLC.AbsNormalForm(tape_id, m, n, x, y)

    for _ = 1:10
        abs_normal!(abs_normal_form)
    end
    return abs_normal_form
end

@testset "reuse_abs_normal" begin
    abs_normal_form = create_abs_normal()
    @test abs_normal_form.Y[1, 1] == -1.5
    @test abs_normal_form.Y[1, 2] == -3.0
    @test abs_normal_form.Y[1, 3] == -1.5

    @test abs_normal_form.J[1, 1] == 0.5
    @test abs_normal_form.J[1, 2] == 0.5

    @test abs_normal_form.Z[1, 1] == -1.0
    @test abs_normal_form.Z[1, 2] == -1.0
    @test abs_normal_form.Z[1, 3] == 0.0
    @test abs_normal_form.Z[2, 1] == 0.0
    @test abs_normal_form.Z[2, 2] == -1.0
    @test abs_normal_form.Z[2, 3] == -1.0

    @test abs_normal_form.L[1, 1] == 0.0
    @test abs_normal_form.L[1, 2] == 0.0
    @test abs_normal_form.L[2, 1] == 0.0
    @test abs_normal_form.L[2, 2] == 0.0
end
