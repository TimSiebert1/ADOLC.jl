
@testset "jac" begin
    # m = 1
    function f(x)
        return x[1]^2 + x[2] * x[3]
    end
    grad_res = CxxVector(3)
    func_res = [0.0]
    res = [func_res, grad_res]
    x = [1.0, 1.0, 2.0]
    function_and_derivative_value!(res, f, 1, 3, x, :jac)


    @test res[1][1] == f(x)
    @test res[2][1] == 2.0
    @test res[2][2] == 2.0
    @test res[2][3] == 1.0
end


@testset "reuse_jac" begin
    # m = 1
    function f(x)
        return x[1]^2 + x[2] * x[3]
    end
    grad_res = CxxVector(3)
    func_res = [0.0]
    res = [func_res, grad_res]
    x = [1.0, 1.0, 2.0]
    function_and_derivative_value!(res, f, 1, 3, x, :jac)

    @test res[1][1] == f(x)
    @test res[2][1] == 2.0
    @test res[2][2] == 2.0
    @test res[2][3] == 1.0

    x = [1.0, 1.0, 0.0]
    function_and_derivative_value!(res, f, 1, 3, x, :jac; reuse_tape=true)
    @test res[1][1] == f(x)
    @test res[2][1] == 2.0
    @test res[2][2] == 0.0
    @test res[2][3] == 1.0
end
