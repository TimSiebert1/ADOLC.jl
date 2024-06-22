@testset "vec_hess_vec" begin
    function f(x)
        return [x[1]^2 * x[2], x[1] * x[3]^3]
    end

    x = [1.0, 2.0, 2.0]
    dir = [-1.0, 1.0, 3.0]
    weights = [2.0, 1.0]
    res = CxxVector(3)
    derivative!(res, f, 2, 3, x, :vec_hess_vec; dir=dir, weights=weights)

    @test res[1] == 32.0
    @test res[2] == -4.0
    @test res[3] == 24.0
end

@testset "vec_hess_mat" begin
    function f(x)
        return [x[1]^2 * x[2], x[1] * x[3]^3]
    end

    # 4 2 0 
    # 2 0 0 
    # 0 0 0

    # 0  0   12
    # 0  0   0 
    # 12 0   12

    x = [1.0, 2.0, 2.0]
    dir = [[1.0, 2.0, 3.0] [-1.0, -2.0, -3.0]]
    weights = [2.0, 1.0]
    res = CxxMatrix(3, 2)
    derivative!(res, f, 2, 3, x, :vec_hess_mat; dir=dir, weights=weights)

    @test res[1, 1] == 52.0
    @test res[1, 2] == -52.0

    @test res[2, 1] == 4.0
    @test res[2, 2] == -4.0

    @test res[3, 1] == 48.0
    @test res[3, 2] == -48.0
end

@testset "1D_vec_hess" begin
    ()
    function f(x)
        return x[1] * x[3]^3
    end

    # 0.0 0.0 12.0
    # 0.0 0.0 0.0
    # 12.0 0.0 12.0

    # 
    x = [1.0, 2.0, 2.0]
    weights = [-1.0]
    res = CxxMatrix(3, 3)
    derivative!(res, f, 1, 3, x, :vec_hess; weights=weights)

    @test res[1, 1] == 0.0
    @test res[1, 2] == 0.0
    @test res[1, 3] == -12.0

    @test res[2, 1] == 0.0
    @test res[2, 2] == 0.0
    @test res[2, 3] == 0.0

    @test res[3, 1] == -12.0
    @test res[3, 2] == 0.0
    @test res[3, 3] == -12.0
end

@testset "2D_vec_hess" begin
    ()
    function f(x)
        return [x[1]^2 * x[2], x[1] * x[3]^3]
    end

    # 4 2 0 
    # 2 0 0 
    # 0 0 0

    # 0  0   12
    # 0  0   0 
    # 12 0   12

    x = [1.0, 2.0, 2.0]
    weights = [-1.0, 1.0]

    res = CxxMatrix(3, 3)
    derivative!(res, f, 2, 3, x, :vec_hess; weights=weights)

    @test res[1, 1] == -4.0
    @test res[1, 2] == -2.0
    @test res[1, 3] == 12.0

    @test res[2, 1] == -2.0
    @test res[2, 2] == 0.0
    @test res[2, 3] == 0.0

    @test res[3, 1] == 12.0
    @test res[3, 2] == 0.0
    @test res[3, 3] == 12.0
end

@testset "mat_hess_vec" begin
    function f(x)
        return [x[1]^2 * x[2], x[1] * x[3]^3]
    end

    # 4 2 0 
    # 2 0 0 
    # 0 0 0

    # 0  0   12
    # 0  0   0 
    # 12 0   12

    x = [1.0, 2.0, 2.0]
    dir = [-1.0, 1.0, 3.0]
    weights = Matrix{Float64}(undef, 3, 2)
    for i in 1:3
        for j in 1:2
            weights[i, j] = 0.0
            if i == j
                weights[i, i] = 1.0
            end
        end
    end
    weights[3, 2] = -1.0
    res = CxxMatrix(3, 3)
    derivative!(res, f, 2, 3, x, :mat_hess_vec; dir=dir, weights=weights)

    @test res[1, 1] == -2.0
    @test res[1, 2] == -2.0
    @test res[1, 3] == 0.0

    @test res[2, 1] == 36.0
    @test res[2, 2] == 0.0
    @test res[2, 3] == 24.0

    @test res[3, 1] == -36.0
    @test res[3, 2] == 0.0
    @test res[3, 3] == -24.0
end

@testset "hess_1D" begin
    function f(x)
        return x[1]^3 + x[2]^2 * x[3]
    end
    res = CxxTensor(1, 3, 3)
    derivative!(res, f, 1, 3, [-1.0, 1.0, 2.0], :hess)

    @test res[1, 1, 1] == -6.0
    @test res[1, 2, 1] == 0.0
    @test res[1, 3, 1] == 0.0

    @test res[1, 1, 2] == 0.0
    @test res[1, 2, 2] == 4.0
    @test res[1, 3, 2] == 2.0

    @test res[1, 1, 3] == 0.0
    @test res[1, 2, 3] == 0.0
    @test res[1, 3, 3] == 0.0
end

@testset "hess_2D" begin
    function f(x)
        return [x[1]^3 + x[2]^2, 3 * x[2] * x[3]^3]
    end

    x = [-1.0, 2.0, 2.0]
    res = CxxTensor(2, 3, 3)
    derivative!(res, f, 2, 3, x, :hess; tape_id=1)

    @test res[1, 1, 1] == -6.0
    @test res[1, 2, 1] == 0.0
    @test res[1, 3, 1] == 0.0

    @test res[2, 1, 1] == 0.0
    @test res[2, 2, 1] == 0.0
    @test res[2, 3, 1] == 0.0

    @test res[1, 1, 2] == 0.0
    @test res[1, 2, 2] == 2.0
    @test res[1, 3, 2] == 0.0

    @test res[2, 1, 2] == 0.0
    @test res[2, 2, 2] == 0.0
    @test res[2, 3, 2] == 36.0

    @test res[1, 1, 3] == 0.0
    @test res[1, 2, 3] == 0.0
    @test res[1, 3, 3] == 0.0

    @test res[2, 1, 3] == 0.0
    @test res[2, 2, 3] == 0.0
    @test res[2, 3, 3] == 72.0
end

@testset "mat_hess_mat" begin
    ()
    function f(x)
        return [x[1]^2 * x[2], x[1] * x[3]^3]
    end

    x = [1.0, 2.0, 2.0]
    dir = [[1.0, 2.0, 3.0] [-1.0, -2.0, -3.0]]
    weights = [[1.0, 0.0, 0.0] [0.0, 1.0, -1.0]]

    res = CxxTensor(3, 3, 2)

    derivative!(res, f, 2, 3, x, :mat_hess_mat; dir=dir, weights=weights)

    @test res[1, 1, 1] == 8.0
    @test res[1, 1, 2] == -8.0

    @test res[1, 2, 1] == 2.0
    @test res[1, 2, 2] == -2.0

    @test res[1, 3, 1] == 0.0
    @test res[1, 3, 2] == 0.0

    @test res[2, 1, 1] == 36.0
    @test res[2, 1, 2] == -36.0

    @test res[2, 2, 1] == 0.0
    @test res[2, 2, 2] == 0.0

    @test res[2, 3, 1] == 48.0
    @test res[2, 3, 2] == -48.0

    @test res[3, 1, 1] == -36.0
    @test res[3, 1, 2] == 36.0

    @test res[3, 2, 1] == 0.0
    @test res[3, 2, 2] == 0.0

    @test res[3, 3, 1] == -48.0
    @test res[3, 3, 2] == 48.0
end

@testset "hess_vec" begin
    function f(x)
        return [x[1]^2 * x[2], x[1] * x[3]^3]
    end

    x = [1.0, 2.0, 2.0]
    dir = [-1.0, 1.0, 3.0]

    res = CxxMatrix(2, 3)
    derivative!(res, f, 2, 3, x, :hess_vec; dir=dir)

    @test res[1, 1] == -2.0
    @test res[1, 2] == -2.0
    @test res[1, 3] == 0.0

    @test res[2, 1] == 36.0
    @test res[2, 2] == 0.0
    @test res[2, 3] == 24.0
end

@testset "mat_hess" begin
    ()
    function f(x)
        return [x[1]^2 * x[2], x[1] * x[3]^3]
    end

    x = [1.0, 2.0, 2.0]
    weights = [[1.0, 0.0, 0.0] [0.0, 1.0, -1.0]]

    res = CxxTensor(3, 3, 3)

    derivative!(res, f, 2, 3, x, :mat_hess; weights=weights)

    @test res[1, 1, 1] == 4.0
    @test res[1, 1, 2] == 2.0
    @test res[1, 1, 3] == 0.0

    @test res[1, 2, 1] == 2.0
    @test res[1, 2, 2] == 0.0
    @test res[1, 2, 3] == 0.0

    @test res[1, 3, 1] == 0.0
    @test res[1, 3, 2] == 0.0
    @test res[1, 3, 3] == 0.0

    @test res[2, 1, 1] == 0.0
    @test res[2, 1, 2] == 0.0
    @test res[2, 1, 3] == 12.0

    @test res[2, 2, 1] == 0.0
    @test res[2, 2, 2] == 0.0
    @test res[2, 2, 3] == 0.0

    @test res[2, 3, 1] == 12.0
    @test res[2, 3, 2] == 0.0
    @test res[2, 3, 3] == 12.0

    @test res[3, 1, 1] == 0.0
    @test res[3, 1, 2] == 0.0
    @test res[3, 1, 3] == -12.0

    @test res[3, 2, 1] == 0.0
    @test res[3, 2, 2] == 0.0
    @test res[3, 2, 3] == 0.0

    @test res[3, 3, 1] == -12.0
    @test res[3, 3, 2] == 0.0
    @test res[3, 3, 3] == -12.0
end

@testset "1D_hess_mat" begin
    ()
    function f(x)
        return x[1] * x[3]^3
    end

    x = [1.0, 2.0, 2.0]
    dir = Matrix{Float64}(undef, 3, 2)
    for i in 1:3
        for j in 1:2
            dir[i, j] = 0.0
            if i == j
                dir[i, i] = 1.0
            end
        end
    end
    dir[3, 2] = -1.0
    res = CxxTensor(1, 3, 2)
    derivative!(res, f, 1, 3, x, :hess_mat; dir=dir)

    @test res[1, 1, 1] == 0.0
    @test res[1, 1, 2] == -12.0

    @test res[1, 2, 1] == 0.0
    @test res[1, 2, 2] == 0.0

    @test res[1, 3, 1] == 12.0
    @test res[1, 3, 2] == -12.0
end

@testset "mat_hess_vec" begin
    ()
    function f(x)
        return [x[1]^2 * x[2], x[1] * x[3]^3]
    end

    # 4 2 0 
    # 2 0 0 
    # 0 0 0

    # 0  0   12
    # 0  0   0 
    # 12 0   12

    x = [1.0, 2.0, 2.0]
    dir = [-1.0, 1.0, 3.0]
    weights = Matrix{Float64}(undef, 3, 2)
    for i in 1:3
        for j in 1:2
            weights[i, j] = 0.0
            if i == j
                weights[i, i] = 1.0
            end
        end
    end
    weights[3, 2] = -1.0
    res = CxxMatrix(3, 3)
    derivative!(res, f, 2, 3, x, :mat_hess_vec; dir=dir, weights=weights)

    @test res[1, 1] == -2.0
    @test res[1, 2] == -2.0
    @test res[1, 3] == 0.0

    @test res[2, 1] == 36.0
    @test res[2, 2] == 0.0
    @test res[2, 3] == 24.0

    @test res[3, 1] == -36.0
    @test res[3, 2] == 0.0
    @test res[3, 3] == -24.0
end
