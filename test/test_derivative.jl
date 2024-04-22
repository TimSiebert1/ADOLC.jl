

function test_first_order_jac()
    function f(x)
        return [x[1]^2 + x[2], x[3]^3]
    end
    res = myalloc(2, 3)
    derivative(f, 2, 3, [1, 1, 2], :jac, res)

    @test res[1, 1] = 2.0
    @test res[1, 2] = 1.0
    @test res[1, 3] = 0.0

    @test res[2, 1] = 0.0
    @test res[2, 2] = 0.0
    @test res[2, 3] = 12.0
end


function test_first_order_jac_vec()
    function f(x)
        return [x[1]^2 + x[2], x[3]^3]
    end
    res = myalloc(1, 2)
    derivative(f, 2, 3, [1, 1, 2], :jac_vec, [-1, 1, 0], res)

    @test res[1, 1] = -1.0
    @test res[1, 2] = 0.0
end





function test_first_order_jac_mat()
    function f(x)
        return [x[1]^2 + x[2], x[3]^3]
    end
    res = myalloc(1, 3)
    dir = Matrix{Float64}(undef, 3, 3)
    for i in 1:3
        for j in 1:3
            dir[i, j] = 0.0
            if i == j
                dir[i, i] = 1.0
            end
        end
    end
    dir[1, 2] = -1.0

    derivative(f, 2, 3, [1, 1, 2], :jac_mat, dir, res)

    @test res[1, 1] = 2.0
    @test res[1, 2] = -1.0
    @test res[1, 3] = 0.0

    @test res[2, 1] = 0.0
    @test res[2, 2] = 0.0
    @test res[2, 3] = 3.0
end




function test_first_order_vec_jac()
    function f(x)
        return [x[1]^2 + x[2], x[3]^3]
    end
    res = myalloc(1, 3)
    derivative(f, 2, 3, [1, 1, 2], :vec_jac, [-1, 1], res)

    @test res[1, 1] = -2
    @test res[1, 2] = -1
    @test res[1, 3] = 3
end