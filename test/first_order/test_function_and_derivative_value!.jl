
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

@testset "jac_tl" begin
    # m > 1, n / 2 < m
    function f(x)
        return [x[1]^2 + x[2], x[3]^3]
    end
    grad_res = CxxMatrix(2, 3)
    func_res = Vector{Cdouble}(undef, 2)
    res = [func_res, grad_res]
    x = [1.0, 1.0, 2.0]
    function_and_derivative_value!(res, f, 2, 3, x, :jac)

    @test res[1][1] == f(x)[1]
    @test res[1][2] == f(x)[2]
    @test res[2][1, 1] == 2.0
    @test res[2][1, 2] == 1.0
    @test res[2][1, 3] == 0.0

    @test res[2][2, 1] == 0.0
    @test res[2][2, 2] == 0.0
    @test res[2][2, 3] == 12.0
end

@testset "jac" begin
    # m > 1, n / 2 >= m
    function f(x)
        return [x[1]^2 + x[2], x[3]^2 * x[4]]
    end
    grad_res = CxxMatrix(2, 4)
    func_res = Vector{Cdouble}(undef, 2)
    res = [func_res, grad_res]
    x = [1.0, 1.0, 2.0, -1.0]
    function_and_derivative_value!(res, f, 2, 4, x, :jac)

    @test res[1][1] == f(x)[1]
    @test res[1][2] == f(x)[2]
    @test res[2][1, 1] == 2.0
    @test res[2][1, 2] == 1.0
    @test res[2][1, 3] == 0.0
    @test res[2][1, 4] == 0.0

    @test res[2][2, 1] == 0.0
    @test res[2][2, 2] == 0.0
    @test res[2][2, 3] == -4.0
    @test res[2][2, 4] == 4.0
end

@testset "reuse_jac" begin
    # m > 1, n / 2 >= m
    function f(x)
        return [x[1]^2 + x[2], x[3]^2 * x[4]]
    end
    grad_res = CxxMatrix(2, 4)
    func_res = Vector{Cdouble}(undef, 2)
    res = [func_res, grad_res]
    x = [1.0, 1.0, 2.0, -1.0]
    function_and_derivative_value!(res, f, 2, 4, x, :jac)

    @test res[1][1] == f(x)[1]
    @test res[1][2] == f(x)[2]
    @test res[2][1, 1] == 2.0
    @test res[2][1, 2] == 1.0
    @test res[2][1, 3] == 0.0
    @test res[2][1, 4] == 0.0

    @test res[2][2, 1] == 0.0
    @test res[2][2, 2] == 0.0
    @test res[2][2, 3] == -4.0
    @test res[2][2, 4] == 4.0

    x = [0.0, 1.0, 2.0, -1.0]
    function_and_derivative_value!(res, f, 2, 4, x, :jac; reuse_tape=true)

    @test res[1][1] == f(x)[1]
    @test res[1][2] == f(x)[2]
    @test res[2][1, 1] == 0.0
    @test res[2][1, 2] == 1.0
    @test res[2][1, 3] == 0.0
    @test res[2][1, 4] == 0.0

    @test res[2][2, 1] == 0.0
    @test res[2][2, 2] == 0.0
    @test res[2][2, 3] == -4.0
    @test res[2][2, 4] == 4.0
end

@testset "jac_vec" begin
    function f(x)
        return [x[1]^2 + x[2], x[3]^3]
    end
    grad_res = CxxVector(2)
    func_res = Vector{Cdouble}(undef, 2)
    res = [func_res, grad_res]
    x = [1.0, 1.0, 2.0]
    function_and_derivative_value!(res, f, 2, 3, x, :jac_vec; dir=[-1.0, 1.0, 0.0])

    @test res[1][1] == f(x)[1]
    @test res[1][2] == f(x)[2]
    @test res[2][1] == -1.0
    @test res[2][2] == 0.0
end

@testset "reuse_jac_vec" begin
    function f(x)
        return [x[1]^2 + x[2], x[3]^3]
    end
    grad_res = CxxVector(2)
    func_res = Vector{Cdouble}(undef, 2)
    res = [func_res, grad_res]
    x = [1.0, 1.0, 2.0]
    function_and_derivative_value!(res, f, 2, 3, x, :jac_vec; dir=[-1.0, 1.0, 0.0])

    @test res[1][1] == f(x)[1]
    @test res[1][2] == f(x)[2]
    @test res[2][1] == -1.0
    @test res[2][2] == 0.0

    x = [2.0, 1.0, 2.0]
    function_and_derivative_value!(
        res, f, 2, 3, x, :jac_vec; dir=[-1.0, 1.0, 0.0], reuse_tape=true
    )

    @test res[1][1] == f(x)[1]
    @test res[1][2] == f(x)[2]
    @test res[2][1] == -3.0
    @test res[2][2] == 0.0

    x = [2.0, 1.0, 2.0]
    function_and_derivative_value!(
        res, f, 2, 3, x, :jac_vec; dir=[0.0, 1.0, 0.0], reuse_tape=true
    )

    @test res[1][1] == f(x)[1]
    @test res[1][2] == f(x)[2]
    @test res[2][1] == 1.0
    @test res[2][2] == 0.0
end

@testset "jac_mat" begin
    function f(x)
        return [x[1]^2 + x[2], x[3]^3]
    end
    grad_res = CxxMatrix(2, 3)
    func_res = Vector{Cdouble}(undef, 2)
    res = [func_res, grad_res]

    dir = CxxMatrix(create_cxx_identity(3, 3), 3, 3)
    dir[1, 2] = -1.0
    x = [1.0, 1.0, 2.0]
    function_and_derivative_value!(res, f, 2, 3, x, :jac_mat; dir=dir)

    @test res[1][1] == f(x)[1]
    @test res[1][2] == f(x)[2]
    @test res[2][1, 1] == 2.0
    @test res[2][1, 2] == -1.0
    @test res[2][1, 3] == 0.0

    @test res[2][2, 1] == 0.0
    @test res[2][2, 2] == 0.0
    @test res[2][2, 3] == 12.0
end

@testset "reuse_jac_mat" begin
    function f(x)
        return [x[1]^2 + x[2], x[3]^3]
    end
    grad_res = CxxMatrix(2, 3)
    func_res = Vector{Cdouble}(undef, 2)
    res = [func_res, grad_res]

    dir = CxxMatrix(create_cxx_identity(3, 3), 3, 3)
    dir[1, 2] = -1.0
    x = [1.0, 1.0, 2.0]
    function_and_derivative_value!(res, f, 2, 3, x, :jac_mat; dir=dir)

    @test res[1][1] == f(x)[1]
    @test res[1][2] == f(x)[2]
    @test res[2][1, 1] == 2.0
    @test res[2][1, 2] == -1.0
    @test res[2][1, 3] == 0.0

    @test res[2][2, 1] == 0.0
    @test res[2][2, 2] == 0.0
    @test res[2][2, 3] == 12.0

    x = [2.0, 1.0, 2.0]
    function_and_derivative_value!(res, f, 2, 3, x, :jac_mat; dir=dir, reuse_tape=true)

    @test res[1][1] == f(x)[1]
    @test res[1][2] == f(x)[2]
    @test res[2][1, 1] == 4.0
    @test res[2][1, 2] == -3.0
    @test res[2][1, 3] == 0.0

    @test res[2][2, 1] == 0.0
    @test res[2][2, 2] == 0.0
    @test res[2][2, 3] == 12.0

    dir[1, 1] = 0.0
    function_and_derivative_value!(res, f, 2, 3, x, :jac_mat; dir=dir, reuse_tape=true)

    @test res[1][1] == f(x)[1]
    @test res[1][2] == f(x)[2]
    @test res[2][1, 1] == 0.0
    @test res[2][1, 2] == -3.0
    @test res[2][1, 3] == 0.0

    @test res[2][2, 1] == 0.0
    @test res[2][2, 2] == 0.0
    @test res[2][2, 3] == 12.0
end

@testset "vec_jac" begin
    function f(x)
        return [x[1]^2 + x[2], x[3]^3]
    end

    grad_res = CxxVector(3)
    func_res = Vector{Cdouble}(undef, 2)
    res = [func_res, grad_res]
    x = [1.0, 1.0, 2.0]
    function_and_derivative_value!(res, f, 2, 3, x, :vec_jac; weights=[-1.0, 1.0])

    @test res[1][1] == f(x)[1]
    @test res[1][2] == f(x)[2]
    @test res[2][1] == -2
    @test res[2][2] == -1
    @test res[2][3] == 12
end

@testset "reuse_vec_jac" begin
    function f(x)
        return [x[1]^2 + x[2], x[3]^3]
    end
    grad_res = CxxVector(3)
    func_res = Vector{Cdouble}(undef, 2)
    res = [func_res, grad_res]
    x = [1.0, 1.0, 2.0]
    function_and_derivative_value!(res, f, 2, 3, x, :vec_jac; weights=[-1.0, 1.0])

    @test res[1][1] == f(x)[1]
    @test res[1][2] == f(x)[2]
    @test res[2][1] == -2
    @test res[2][2] == -1
    @test res[2][3] == 12

    x = [2.0, 1.0, 2.0]
    function_and_derivative_value!(
        res, f, 2, 3, x, :vec_jac; weights=[-1.0, 1.0], reuse_tape=true
    )

    @test res[1][1] == f(x)[1]
    @test res[1][2] == f(x)[2]
    @test res[2][1] == -4
    @test res[2][2] == -1
    @test res[2][3] == 12

    function_and_derivative_value!(
        res, f, 2, 3, x, :vec_jac; weights=[0.0, 1.0], reuse_tape=true
    )

    @test res[1][1] == f(x)[1]
    @test res[1][2] == f(x)[2]
    @test res[2][1] == 0
    @test res[2][2] == 0
    @test res[2][3] == 12
end

@testset "mat_jac" begin
    function f(x)
        return [x[1]^2 + x[2], x[3]^3]
    end

    weights = Matrix{Float64}(undef, 3, 2)
    for i in 1:3
        for j in 1:2
            weights[i, j] = 0.0
            if i == j
                weights[i, i] = 1.0
            end
        end
    end
    weights[1, 2] = -1.0

    grad_res = CxxMatrix(3, 3)
    func_res = Vector{Cdouble}(undef, 2)
    res = [func_res, grad_res]
    x = [1.0, 1.0, 2.0]

    function_and_derivative_value!(res, f, 2, 3, x, :mat_jac; weights=weights)

    @test res[1][1] == f(x)[1]
    @test res[1][2] == f(x)[2]
    @test res[2][1, 1] == 2.0
    @test res[2][1, 2] == 1.0
    @test res[2][1, 3] == -12.0

    @test res[2][2, 1] == 0.0
    @test res[2][2, 2] == 0.0
    @test res[2][2, 3] == 12.0

    @test res[2][3, 1] == 0.0
    @test res[2][3, 2] == 0.0
    @test res[2][3, 3] == 0.0
end

@testset "reuse_mat_jac" begin
    function f(x)
        return [x[1]^2 + x[2], x[3]^3]
    end

    weights = Matrix{Float64}(undef, 3, 2)
    for i in 1:3
        for j in 1:2
            weights[i, j] = 0.0
            if i == j
                weights[i, i] = 1.0
            end
        end
    end
    weights[1, 2] = -1.0

    grad_res = CxxMatrix(3, 3)
    func_res = Vector{Cdouble}(undef, 2)
    res = [func_res, grad_res]
    x = [1.0, 1.0, 2.0]
    function_and_derivative_value!(res, f, 2, 3, x, :mat_jac; weights=weights)

    @test res[1][1] == f(x)[1]
    @test res[1][2] == f(x)[2]

    @test res[2][1, 1] == 2.0
    @test res[2][1, 2] == 1.0
    @test res[2][1, 3] == -12.0

    @test res[2][2, 1] == 0.0
    @test res[2][2, 2] == 0.0
    @test res[2][2, 3] == 12.0

    @test res[2][3, 1] == 0.0
    @test res[2][3, 2] == 0.0
    @test res[2][3, 3] == 0.0

    x = [2.0, 1.0, 2.0]
    function_and_derivative_value!(
        res, f, 2, 3, x, :mat_jac; weights=weights, reuse_tape=true
    )

    @test res[1][1] == f(x)[1]
    @test res[1][2] == f(x)[2]

    @test res[2][1, 1] == 4.0
    @test res[2][1, 2] == 1.0
    @test res[2][1, 3] == -12.0

    @test res[2][2, 1] == 0.0
    @test res[2][2, 2] == 0.0
    @test res[2][2, 3] == 12.0

    @test res[2][3, 1] == 0.0
    @test res[2][3, 2] == 0.0
    @test res[2][3, 3] == 0.0

    weights[1, 1] = 0.0

    x = [2.0, 1.0, 2.0]
    function_and_derivative_value!(
        res, f, 2, 3, x, :mat_jac; weights=weights, reuse_tape=true
    )

    @test res[1][1] == f(x)[1]
    @test res[1][2] == f(x)[2]

    @test res[2][1, 1] == 0.0
    @test res[2][1, 2] == 0.0
    @test res[2][1, 3] == -12.0

    @test res[2][2, 1] == 0.0
    @test res[2][2, 2] == 0.0
    @test res[2][2, 3] == 12.0

    @test res[2][3, 1] == 0.0
    @test res[2][3, 2] == 0.0
    @test res[2][3, 3] == 0.0
end
