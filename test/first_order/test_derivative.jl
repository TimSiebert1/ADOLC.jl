@testset "jac" begin
    # m = 1
    function f(x)
        return x[1]^2 + x[2] * x[3]
    end
    res = derivative(f, 1, 3, [1.0, 1.0, 2.0], :jac)

    @test res[1] == 2.0
    @test res[2] == 2.0
    @test res[3] == 1.0
end


@testset "jac_tl" begin
    # m > 1, n / 2 < m
    function f(x)
        return [x[1]^2 + x[2], x[3]^3]
    end

    res = derivative(f, 2, 3, [1.0, 1.0, 2.0], :jac)

    @test res[1, 1] == 2.0
    @test res[1, 2] == 1.0
    @test res[1, 3] == 0.0

    @test res[2, 1] == 0.0
    @test res[2, 2] == 0.0
    @test res[2, 3] == 12.0
end


@testset "jac" begin
    # m > 1, n / 2 >= m
    function f(x)
        return [x[1]^2 + x[2], x[3]^2 * x[4]]
    end

    res = derivative(f, 2, 4, [1.0, 1.0, 2.0, -1.0], :jac)

    @test res[1, 1] == 2.0
    @test res[1, 2] == 1.0
    @test res[1, 3] == 0.0
    @test res[1, 4] == 0.0

    @test res[2, 1] == 0.0
    @test res[2, 2] == 0.0
    @test res[2, 3] == -4.0
    @test res[2, 4] == 4.0
end


@testset "jac_vec" begin
    function f(x)
        return [x[1]^2 + x[2], x[3]^3]
    end
    res = derivative(f, 2, 3, [1.0, 1.0, 2.0], :jac_vec; dir=[-1.0, 1.0, 0.0])

    @test res[1] == -1.0
    @test res[2] == 0.0
end

@testset "jac_mat" begin
    function f(x)
        return [x[1]^2 + x[2], x[3]^3]
    end
    dir = [[1.0, 0.0, 0.0] [-1.0, 1.0, 0.0] [0.0, 0.0, 1.0]]

    res = derivative(f, 2, 3, [1.0, 1.0, 2.0], :jac_mat; dir=dir)

    @test res[1, 1] == 2.0
    @test res[1, 2] == -1.0
    @test res[1, 3] == 0.0

    @test res[2, 1] == 0.0
    @test res[2, 2] == 0.0
    @test res[2, 3] == 12.0
end

@testset "vec_jac" begin
    function f(x)
        return [x[1]^2 + x[2], x[3]^3]
    end

    res = derivative(f, 2, 3, [1.0, 1.0, 2.0], :vec_jac; weights=[-1.0, 1.0])

    @test res[1] == -2
    @test res[2] == -1
    @test res[3] == 12
end

@testset "mat_jac" begin
    function f(x)
        return [x[1]^2 + x[2], x[3]^3]
    end

    weights = [[1.0, 0.0, 0.0] [-1.0, 1.0, 0.0]]

    res = derivative(f, 2, 3, [1.0, 1.0, 2.0], :mat_jac; weights=weights)

    @test res[1, 1] == 2.0
    @test res[1, 2] == 1.0
    @test res[1, 3] == -12.0

    @test res[2, 1] == 0.0
    @test res[2, 2] == 0.0
    @test res[2, 3] == 12.0

    @test res[3, 1] == 0.0
    @test res[3, 2] == 0.0
    @test res[3, 3] == 0.0
end