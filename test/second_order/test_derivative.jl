
@testset "vec_hess_vec" begin
    function f(x)
        return [x[1]^2 * x[2], x[1] * x[3]^3]
    end

    x = [1.0, 2.0, 2.0]
    dir = [-1.0, 1.0, 3.0]
    weights = [2.0, 1.0]
    res = derivative(f, x, :vec_hess_vec; dir=dir, weights=weights)

    @test res[1] == 32.0
    @test res[2] == -4.0
    @test res[3] == 24.0

    res = derivative(f, x, :vec_hess_vec; dir=CxxVector(dir), weights=CxxVector(weights))

    @test res[1] == 32.0
    @test res[2] == -4.0
    @test res[3] == 24.0
end

@testset "vec_hess_mat" begin
    function f(x)
        return [x[1]^2 * x[2], x[1] * x[3]^3]
    end

    x = [1.0, 2.0, 2.0]
    dir = [[1.0, 2.0, 3.0] [-1.0, -2.0, -3.0]]
    weights = [2.0, 1.0]
    res = derivative(f, x, :vec_hess_mat; dir=dir, weights=weights)

    @test res[1, 1] == 52.0
    @test res[1, 2] == -52.0

    @test res[2, 1] == 4.0
    @test res[2, 2] == -4.0

    @test res[3, 1] == 48.0
    @test res[3, 2] == -48.0

    res = derivative(f, x, :vec_hess_mat; dir=CxxMatrix(dir), weights=CxxVector(weights))

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

    x = [1.0, 2.0, 2.0]
    weights = [-1.0]
    res = derivative(f, x, :vec_hess; weights=weights)

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

    x = [1.0, 2.0, 2.0]
    weights = [-1.0, 1.0]

    res = derivative(f, x, :vec_hess; weights=weights)

    @test res[1, 1] == -4.0
    @test res[1, 2] == -2.0
    @test res[1, 3] == 12.0

    @test res[2, 1] == -2.0
    @test res[2, 2] == 0.0
    @test res[2, 3] == 0.0

    @test res[3, 1] == 12.0
    @test res[3, 2] == 0.0
    @test res[3, 3] == 12.0

    res = derivative(f, x, :vec_hess; weights=CxxVector(weights))

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

    x = [1.0, 2.0, 2.0]
    dir = [-1.0, 1.0, 3.0]
    weights = [[1.0, 0.0, 0.0] [0.0, 1.0, -1.0]]
    res = derivative(f, x, :mat_hess_vec; dir=dir, weights=weights)

    @test res[1, 1] == -2.0
    @test res[1, 2] == -2.0
    @test res[1, 3] == 0.0

    @test res[2, 1] == 36.0
    @test res[2, 2] == 0.0
    @test res[2, 3] == 24.0

    @test res[3, 1] == -36.0
    @test res[3, 2] == 0.0
    @test res[3, 3] == -24.0

    res = derivative(f, x, :mat_hess_vec; dir=CxxVector(dir), weights=CxxMatrix(weights))

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

    res = derivative(f, [-1.0, 1.0, 2.0], :hess)

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
    res = derivative(f, x, :hess; tape_id=1)

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

    res = derivative(f, x, :mat_hess_mat; dir=dir, weights=weights)

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

    res = derivative(f, x, :mat_hess_mat; dir=CxxMatrix(dir), weights=CxxMatrix(weights))

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

    res = derivative(f, x, :hess_vec; dir=dir)

    @test res[1, 1] == -2.0
    @test res[1, 2] == -2.0
    @test res[1, 3] == 0.0

    @test res[2, 1] == 36.0
    @test res[2, 2] == 0.0
    @test res[2, 3] == 24.0

    res = derivative(f, x, :hess_vec; dir=CxxVector(dir))

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

    res = derivative(f, x, :mat_hess; weights=weights)

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


    res = derivative(f, x, :mat_hess; weights=CxxMatrix(weights))

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
    dir = [[1.0, 0.0, 0.0] [0.0, 1.0, -1.0]]

    res = derivative(f, x, :hess_mat; dir=dir)

    @test res[1, 1, 1] == 0.0
    @test res[1, 1, 2] == -12.0

    @test res[1, 2, 1] == 0.0
    @test res[1, 2, 2] == 0.0

    @test res[1, 3, 1] == 12.0
    @test res[1, 3, 2] == -12.0

    res = derivative(f, x, :hess_mat; dir=CxxMatrix(dir))

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

    x = [1.0, 2.0, 2.0]
    dir = [-1.0, 1.0, 3.0]
    weights = [[1.0, 0.0, 0.0] [0.0, 1.0, -1.0]]

    res = derivative(f, x, :mat_hess_vec; dir=dir, weights=weights)

    @test res[1, 1] == -2.0
    @test res[1, 2] == -2.0
    @test res[1, 3] == 0.0

    @test res[2, 1] == 36.0
    @test res[2, 2] == 0.0
    @test res[2, 3] == 24.0

    @test res[3, 1] == -36.0
    @test res[3, 2] == 0.0
    @test res[3, 3] == -24.0

    res = derivative(f, x, :mat_hess_vec; dir=CxxVector(dir), weights=CxxMatrix(weights))

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


