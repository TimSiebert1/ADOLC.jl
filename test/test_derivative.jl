

####### first_order ########

function test_first_order_jac()
    # m = 1
    function f(x)
        return x[1]^2 + x[2] * x[3]
    end
    res = myalloc(1, 3)
    derivative(f, 1, 3, [1.0, 1.0, 2.0], :jac, res)

    @test res[1, 1] = 2.0
    @test res[1, 2] = 2.0
    @test res[1, 3] = 1.0
end

function test_first_order_jac()
    # m > 1, n / 2 < m
    function f(x)
        return [x[1]^2 + x[2], x[3]^3]
    end
    res = myalloc(2, 3)
    derivative(f, 2, 3, [1.0, 1.0, 2.0], :jac, res)

    @test res[1, 1] = 2.0
    @test res[1, 2] = 1.0
    @test res[1, 3] = 0.0

    @test res[2, 1] = 0.0
    @test res[2, 2] = 0.0
    @test res[2, 3] = 12.0
end


function test_first_order_jac()
    # m > 1, n / 2 >= m
    function f(x)
        return [x[1]^2 + x[2], x[3]^2 * x[4]]
    end
    res = myalloc(2, 4)
    derivative(f, 2, 4, [1.0, 1.0, 2.0, -1.0], :jac, res)

    @test res[1, 1] = 2.0
    @test res[1, 2] = 1.0
    @test res[1, 3] = 0.0
    @test res[1, 4] = 0.0

    @test res[2, 1] = 0.0
    @test res[2, 2] = 0.0
    @test res[2, 3] = -4.0
    @test res[2, 4] = 4.0

end

function test_first_order_jac_vec()
    function f(x)
        return [x[1]^2 + x[2], x[3]^3]
    end
    res = myalloc(1, 2)
    derivative(f, 2, 3, [1.0, 1.0, 2.0], :jac_vec, [-1.0, 1.0, 0.0], res)

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

    derivative(f, 2, 3, [1.0, 1.0, 2.0], :jac_mat, dir, res)

    @test res[1, 1] = 2.0
    @test res[1, 2] = -1.0
    @test res[1, 3] = 0.0

    @test res[2, 1] = 0.0
    @test res[2, 2] = 0.0
    @test res[2, 3] = 12.0
end




function test_first_order_vec_jac()
    function f(x)
        return [x[1]^2 + x[2], x[3]^3]
    end
    res = myalloc(1, 3)
    derivative(f, 2, 3, [1.0, 1.0, 2.0], :vec_jac, [-1.0, 1.0], res)

    @test res[1, 1] = -2
    @test res[1, 2] = -1
    @test res[1, 3] = 12
end


function test_first_order_mat_jac()
    function f(x)
        return [x[1]^2 + x[2], x[3]^3]
    end

    weights = Matrix{Float64}(undef, 3, 2)
    for i in 1:3
        for j in 1:2
            dir[i, j] = 0.0
            if i == j
                weights[i, i] = 1.0
            end
        end
    end
    weigthts[1, 2] = -1.0

    res = myalloc(3, 3)
    derivative(f, 2, 3, [1.0, 1.0, 2.0], :mat_jac, weights, res)

    @test res[1, 1] = 2.0
    @test res[1, 2] = 1.0
    @test res[1, 3] = -12.0

    @test res[2, 1] = 0.0
    @test res[2, 2] = 0.0
    @test res[2, 3] = 12.0

    @test res[3, 1] = 0.0
    @test res[3, 2] = 0.0
    @test res[3, 3] = 0.0
end




function test_first_order_abs_normal()

    function func_eval(x)
        return (max(-x[1]-x[2], -x[1]-x[2]+x[1]^2+x[2]^2-1) + max(-x[2]-x[3], -x[2]-x[3]+x[2]^2+x[3]^2-1))
    end 
    
    x = [-0.5, -0.5, -0.5]

    
    abs_normal_problem = derivative(f, 1, 3, x, :abs_normal)

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
end





######### second_order ########


function test_second_order_hess()
    derivative(f, 2, 3, [1.0, 1.0, 2.0], :hess, [-1.0, 1.0], res)
end

function test_second_order_hess_vec()
    derivative(f, 2, 3, [1.0, 1.0, 2.0], :hess_vec, [-1.0, 1.0], res)
end
function test_second_order_hess_mat()
    derivative(f, 2, 3, [1.0, 1.0, 2.0], :hess_mat, [-1.0, 1.0], res)
end
function test_second_order_vec_hess()
    derivative(f, 2, 3, [1.0, 1.0, 2.0], :vec_hess, [-1.0, 1.0], res)
end
function test_second_order_mat_hess()
    derivative(f, 2, 3, [1.0, 1.0, 2.0], :mat_hess, [-1.0, 1.0], res)
end
function test_second_order_vec_hess_vec()
    derivative(f, 2, 3, [1.0, 1.0, 2.0], :vec_hess_vec, [-1.0, 1.0], res)
end
function test_second_order_vec_hess_mat()
    derivative(f, 2, 3, [1.0, 1.0, 2.0], :vec_hess_mat, [-1.0, 1.0], res)
end
function test_second_order_mat_hess_mat()
    derivative(f, 2, 3, [1.0, 1.0, 2.0], :mat_hess_mat, [-1.0, 1.0], res)
end
function test_second_order_mat_hess_vec()
    derivative(f, 2, 3, [1.0, 1.0, 2.0], :mat_hess_vec, [-1.0, 1.0], res)
end



##### higher_order #######

function test_higher_order()
    derivative(f, 2, 3, [1.0, 1.0, 2.0], [-1.0, 1.0], res)
end
