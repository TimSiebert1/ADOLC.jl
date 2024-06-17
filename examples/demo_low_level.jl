using ADOLC

function demo_low_level()
    function f(x)
        return [x[1]^2 + x[2], x[3]^3]
    end

    m = 2
    n = 30

    x = [1.0 for _ in 1:30]
    a = [Adouble{TbAlloc}() for _ in 1:n]
    y = [0.0 for _ in 1:m]
    b = [Adouble{TbAlloc}() for _ in 1:m]

    tape_id = 1
    ADOLC.trace_on(tape_id)
    a << x
    b = f(a)
    b >> y
    ADOLC.trace_off()

    res = ADOLC.myalloc2(2, 3)
    ADOLC.jacobian(tape_id, m, n, x, res)
    for i in 1:2
        println("dim $i")
        for j in 1:3
            print(res[i, j], " ")
        end
        println("")
        println("")
    end
    return ADOLC.myfree2(res)
end

demo_low_level()
