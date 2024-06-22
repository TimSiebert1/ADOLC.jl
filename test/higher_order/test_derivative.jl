
@testset "higher_order_1D" begin
    ()
    function f(x)
        return x[1] * x[2] * x[3] * x[4]
    end
    x = [1.0, 2.0, 3.0, 4.0]
    partials = [
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [1, 1, 0, 0],
        [0, 0, 1, 1],
        [1, 1, 1, 0],
        [1, 1, 1, 1],
        [2, 0, 0, 0],
    ]
    res = derivative(f, x, partials)

    @test res[1] ≈ 24.0
    @test res[2] ≈ 8.0
    @test res[3] ≈ 12.0
    @test res[4] ≈ 2.0
    @test res[5] ≈ 4.0
    @test res[6] ≈ 1.0
    @test res[7] ≈ 0.0
end

@testset "higher_order_2D" begin
    ()
    function f(x)
        return [x[1]^2 * x[2]^2, x[3]^2 * x[4]^2]
    end
    x = [1.0, 2.0, 3.0, 4.0]
    partials = [
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [1, 1, 0, 0],
        [0, 0, 1, 1],
        [1, 1, 1, 0],
        [0, 0, 2, 1],
        [2, 0, 0, 0],
    ]
    res = derivative(f, x, partials)

    @test res[1, 1] ≈ 8.0
    @test res[2, 1] ≈ 0.0

    @test res[1, 2] ≈ 0.0
    @test res[2, 2] ≈ 96.0

    @test res[1, 3] ≈ 8.0
    @test res[2, 3] ≈ 0.0

    @test res[1, 4] ≈ 0.0
    @test res[2, 4] ≈ 48.0

    @test res[1, 5] ≈ 0.0
    @test res[2, 5] ≈ 0.0

    @test res[1, 6] ≈ 0.0
    @test res[2, 6] ≈ 16.0

    @test res[1, 7] ≈ 8.0
    @test res[2, 7] ≈ 0.0
end

@testset "higher_order_adolc_format" begin
    ()
    function f(x)
        return [x[1]^2 * x[2]^2, x[3]^2 * x[4]^2]
    end
    x = [1.0, 2.0, 3.0, 4.0]
    partials = [[1, 0, 0], [3, 0, 0], [2, 1, 0], [4, 3, 0], [3, 2, 1], [4, 3, 3], [1, 1, 0]]
    res = derivative(f, x, partials; adolc_format=true)

    @test res[1, 1] ≈ 8.0
    @test res[2, 1] ≈ 0.0

    @test res[1, 2] ≈ 0.0
    @test res[2, 2] ≈ 96.0

    @test res[1, 3] ≈ 8.0
    @test res[2, 3] ≈ 0.0

    @test res[1, 4] ≈ 0.0
    @test res[2, 4] ≈ 48.0

    @test res[1, 5] ≈ 0.0
    @test res[2, 5] ≈ 0.0

    @test res[1, 6] ≈ 0.0
    @test res[2, 6] ≈ 16.0

    @test res[1, 7] ≈ 8.0
    @test res[2, 7] ≈ 0.0
end

@testset "higher_order_not_full_seed" begin
    ()
    function f(x)
        return [x[1]^2 * x[2]^2, x[3]^2 * x[4]^2]
    end
    x = [1.0, 2.0, 3.0, 4.0]
    partials = [[1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 2, 0], [2, 0, 0, 0]]
    res = derivative(f, x, partials)

    @test res[1, 1] ≈ 8.0
    @test res[2, 1] ≈ 0.0

    @test res[1, 2] ≈ 0.0
    @test res[2, 2] ≈ 96.0

    @test res[1, 3] ≈ 0.0
    @test res[2, 3] ≈ 32.0

    @test res[1, 4] ≈ 8.0
    @test res[2, 4] ≈ 0.0
end

@testset "higher_order_not_full_seed_adolc_format" begin
    ()
    function f(x)
        return [x[1]^2 * x[2]^2, x[3]^2 * x[4]^2]
    end
    x = [1.0, 2.0, 3.0, 4.0]
    partials = [[1, 0], [3, 0], [3, 3], [1, 1]]
    res = derivative(f, x, partials; adolc_format=true)

    @test res[1, 1] ≈ 8.0
    @test res[2, 1] ≈ 0.0

    @test res[1, 2] ≈ 0.0
    @test res[2, 2] ≈ 96.0

    @test res[1, 3] ≈ 0.0
    @test res[2, 3] ≈ 32.0

    @test res[1, 4] ≈ 8.0
    @test res[2, 4] ≈ 0.0
end

@testset "higher_order_seed" begin
    ()
    function f(x)
        return x[1]^2 * x[2]^2
    end

    x = [1.0, 2.0]

    partials = [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 0],
        [0, 0, 2],
        [1, 2, 0],
        [0, 1, 2],
        [4, 0, 0],
        [0, 1, 3],
    ]

    seed = [[1.0, 0.0] [0.0, 2.0] [1.0, 1.0]]

    res = derivative(f, x, partials, seed)

    @test res[1] ≈ 8.0
    @test res[2] ≈ 8.0
    @test res[3] ≈ 12.0
    @test res[4] ≈ 16.0
    @test res[5] ≈ 26.0
    @test res[6] ≈ 16.0
    @test res[7] ≈ 32.0
    @test res[8] ≈ 0.0
    @test res[9] ≈ 24.0
end
