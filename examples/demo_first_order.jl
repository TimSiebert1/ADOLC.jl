using ADOLC


function demo_first_order()
    function f(x)
        return [x[1]^2 + x[2], x[3]^3]
    end
    res = ADOLC.myalloc2(2, 3)
    dir = [[1.0, 0.0, 0.0] [-1.0, 1.0, 0.0] [0.0, 0.0, 1.0]]

    derivative!(res, f, 2, 3, [1.0, 1.0, 2.0], :jac_mat, dir = dir)
    for i in 1:2
        println("dim $i")
        for j in 1:3
            print(res[i, j], " ")
        end
        println("")
        println("")
    end
end

demo_first_order()