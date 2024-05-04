using ADOLC

function demo_second_order()
    function f(x)
        return [x[1]^2 * x[2], x[1] * x[3]^3]
    end

    x = [1.0, 2.0, 2.0]
    dir = [[1.0, 2.0, 3.0] [-1.0, -2.0, -3.0]]
    weights = [[1.0, 0.0, 0.0] [0.0, 1.0, -1.0]]

    res = ADOLC.myalloc3(3, 3, 2)

    derivative!(res, f, 2, 3, x, :mat_hess_mat, dir = dir, weights = weights)

    for i = 1:3
        println("dim $i: ")
        for j = 1:3
            for k = 1:2
                if res[i, j, k] >= 0
                    print(" ", res[i, j, k], " ")
                else
                    print(res[i, j, k], " ")
                end
            end
            println("")
        end
        println("")
    end

end

demo_second_order()
