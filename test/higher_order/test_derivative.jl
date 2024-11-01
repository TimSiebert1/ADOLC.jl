
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

@testset "higher_order_1D" begin
    ()
    function f(x)
        return x[1] * x[2] * x[3] * x[4]
    end
    x = [1.0, 2.0, 3.0, 4.0]
    partials = [[0, 1, 1, 1], [1, 1, 1, 0]]
    res = derivative(f, x, partials)

    @test res[1] ≈ 1.0
    @test res[2] ≈ 4.0

    x = [10.0, 2.0, 3.0, 4.4]
    partials = [[0, 1, 1, 1], [1, 1, 1, 0]]
    res = derivative(f, x, partials; tape_id=0, reuse_tape=true)

    @test res[1] ≈ 10.0
    @test res[2] ≈ 4.4
end

@testset "higher_order_id_seed" begin
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
    res = derivative(f, x, partials; id_seed=true)

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

    seed = CxxMatrix([[1.0, 0.0] [0.0, 2.0] [1.0, 1.0]])

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
@testset "higher_order_seed_reuse" begin
    ()
    function f(x)
        return x[1]^2 * x[2]^2
    end

    x = [1.0, 2.0]

    partials = [[1], [2]]

    seed = CxxMatrix([[1.0, 1.0];;])
    res = derivative(f, x, partials, seed)
    @test res[1] ≈ 12.0
    @test res[2] ≈ 26.0

    seed = CxxMatrix([[1.0, 2.0];;])
    res = derivative(f, x, partials, seed; reuse_tape=true, tape_id=0)

    @test res[1] ≈ 16.0
    @test res[2] ≈ 48.0
end

@testset "higher_order_full_tensor" begin
    ()

    function f(x)
        return [x[1]^2 * x[2]^2, x[3]^2 * x[4]^2]
    end

    x = [1.0, 2.0, 3.0, 4.0]
    m = 2
    n = 4
    degree = 2
    res = derivative(f, x, degree, CxxMatrix(create_cxx_identity(n, n), n, n))

    @test res[1, tensor_address(degree, [1, 0])] == 8.0
    @test res[1, tensor_address(degree, [2, 0])] == 4.0
    @test res[1, tensor_address(degree, [3, 0])] == 0.0
    @test res[1, tensor_address(degree, [4, 0])] == 0.0

    @test res[2, tensor_address(degree, [1, 0])] == 0.0
    @test res[2, tensor_address(degree, [2, 0])] == 0.0
    @test res[2, tensor_address(degree, [3, 0])] == 96.0
    @test res[2, tensor_address(degree, [4, 0])] == 72.0

    @test res[1, tensor_address(degree, [1, 1])] == 8.0
    @test res[1, tensor_address(degree, [2, 1])] == 8.0
    @test res[1, tensor_address(degree, [3, 1])] == 0.0
    @test res[1, tensor_address(degree, [4, 1])] == 0.0
    @test res[1, tensor_address(degree, [2, 2])] == 2.0
    @test res[1, tensor_address(degree, [3, 2])] == 0.0
    @test res[1, tensor_address(degree, [4, 3])] == 0.0
    @test res[1, tensor_address(degree, [3, 3])] == 0.0
    @test res[1, tensor_address(degree, [4, 3])] == 0.0
    @test res[1, tensor_address(degree, [4, 4])] == 0.0
    @test res[1, tensor_address(degree, [4, 2])] == 0.0

    @test res[2, tensor_address(degree, [1, 1])] == 0.0
    @test res[2, tensor_address(degree, [2, 1])] == 0.0
    @test res[2, tensor_address(degree, [3, 1])] == 0.0
    @test res[2, tensor_address(degree, [4, 1])] == 0.0
    @test res[2, tensor_address(degree, [2, 2])] == 0.0
    @test res[2, tensor_address(degree, [3, 2])] == 0.0
    @test res[2, tensor_address(degree, [4, 2])] == 0.0
    @test res[2, tensor_address(degree, [4, 3])] == 48.0
    @test res[2, tensor_address(degree, [3, 3])] == 32.0
    @test res[2, tensor_address(degree, [4, 4])] == 18.0

    res = derivative(f, x, degree)

    @test res[1, tensor_address(degree, [1, 0])] == 8.0
    @test res[1, tensor_address(degree, [2, 0])] == 4.0
    @test res[1, tensor_address(degree, [3, 0])] == 0.0
    @test res[1, tensor_address(degree, [4, 0])] == 0.0

    @test res[2, tensor_address(degree, [1, 0])] == 0.0
    @test res[2, tensor_address(degree, [2, 0])] == 0.0
    @test res[2, tensor_address(degree, [3, 0])] == 96.0
    @test res[2, tensor_address(degree, [4, 0])] == 72.0

    @test res[1, tensor_address(degree, [1, 1])] == 8.0
    @test res[1, tensor_address(degree, [2, 1])] == 8.0
    @test res[1, tensor_address(degree, [3, 1])] == 0.0
    @test res[1, tensor_address(degree, [4, 1])] == 0.0
    @test res[1, tensor_address(degree, [2, 2])] == 2.0
    @test res[1, tensor_address(degree, [3, 2])] == 0.0
    @test res[1, tensor_address(degree, [4, 3])] == 0.0
    @test res[1, tensor_address(degree, [3, 3])] == 0.0
    @test res[1, tensor_address(degree, [4, 3])] == 0.0
    @test res[1, tensor_address(degree, [4, 4])] == 0.0
    @test res[1, tensor_address(degree, [4, 2])] == 0.0

    @test res[2, tensor_address(degree, [1, 1])] == 0.0
    @test res[2, tensor_address(degree, [2, 1])] == 0.0
    @test res[2, tensor_address(degree, [3, 1])] == 0.0
    @test res[2, tensor_address(degree, [4, 1])] == 0.0
    @test res[2, tensor_address(degree, [2, 2])] == 0.0
    @test res[2, tensor_address(degree, [3, 2])] == 0.0
    @test res[2, tensor_address(degree, [4, 2])] == 0.0
    @test res[2, tensor_address(degree, [4, 3])] == 48.0
    @test res[2, tensor_address(degree, [3, 3])] == 32.0
    @test res[2, tensor_address(degree, [4, 4])] == 18.0
end
