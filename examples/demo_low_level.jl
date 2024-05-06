using ADOLC


function demo_low_level()
    function f(x)
        return [x[1]^2 + x[2], x[3]^3]
    end

    m = 2
    n = 3

    x = [1.0, 1.0, 2.0]
    a = [Adouble{TbAlloc}() for _ = 1:n]
    y = [0.0 for _ = 1:m]
    b = [Adouble{TbAlloc}() for _ = 1:m]

    tape_id = 0
    ADOLC.trace_on(tape_id)
    a << x
    b = f(a)
    b >> y
    ADOLC.trace_off()

    res = ADOLC.myalloc2(2, 3)
    ADOLC.jacobian(tape_id, m, n, x, res)
    for i = 1:2
        println("dim $i")
        for j = 1:3
            print(res[i, j], " ")
        end
        println("")
        println("")
    end
    ADOLC.myfree2(res)
end

demo_low_level()
