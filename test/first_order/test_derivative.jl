@testset "jac" begin
    # m = 1
    function f(x)
        return x[1]^2 + x[2] * x[3]
    end
    x = [1.0, 1.0, 2.0]
    res = derivative(f, x, :jac)

    @test res[1] == 2.0
    @test res[2] == 2.0
    @test res[3] == 1.0
end

@testset "reuse_jac" begin
    # m = 1
    function f(x)
        return x[1]^2 + x[2] * x[3]
    end
    x = [1.0, 1.0, 2.0]
    tape_id = 1
    res = derivative(f, x, :jac, tape_id=tape_id)

    @test res[1] == 2.0
    @test res[2] == 2.0
    @test res[3] == 1.0

    x = [1.0, 1.0, 0.0]
    res = derivative(f, x, :jac, tape_id=tape_id, reuse_tape=true)

    @test res[1] == 2.0
    @test res[2] == 0.0
    @test res[3] == 1.0
end

@testset "jac" begin
    # m > 1, n / 2 < m
    function f(x)
        return [x[1]^2 + x[2], x[3]^3]
    end
    x = [1.0, 1.0, 2.0]
    res = derivative(f, x, :jac)

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
    x = [1.0, 1.0, 2.0, -1.0]
    res = derivative(f, x, :jac)

    @test res[1, 1] == 2.0
    @test res[1, 2] == 1.0
    @test res[1, 3] == 0.0
    @test res[1, 4] == 0.0

    @test res[2, 1] == 0.0
    @test res[2, 2] == 0.0
    @test res[2, 3] == -4.0
    @test res[2, 4] == 4.0
end

@testset "reuse_jac" begin
    # m > 1, n / 2 >= m
    function f(x)
        return [x[1]^2 + x[2], x[3]^2 * x[4]]
    end
    x = [1.0, 1.0, 2.0, -1.0]
    tape_id = 3
    res = derivative(f, x, :jac, tape_id=tape_id)

    @test res[1, 1] == 2.0
    @test res[1, 2] == 1.0
    @test res[1, 3] == 0.0
    @test res[1, 4] == 0.0

    @test res[2, 1] == 0.0
    @test res[2, 2] == 0.0
    @test res[2, 3] == -4.0
    @test res[2, 4] == 4.0

    x = [0.0, 1.0, 2.0, -1.0]
    res = derivative(f, x, :jac, tape_id=tape_id, reuse_tape=true)
    @test res[1, 1] == 0.0
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
    x = [1.0, 1.0, 2.0]
    res = derivative(f, x, :jac_vec; dir=[-1.0, 1.0, 0.0])

    @test res[1] == -1.0
    @test res[2] == 0.0
end

@testset "reuse_jac_vec" begin
    function f(x)
        return [x[1]^2 + x[2], x[3]^3]
    end
    x = [1.0, 1.0, 2.0]
    tape_id = 3
    res = derivative(f, x, :jac_vec; dir=[-1.0, 1.0, 0.0], tape_id=tape_id)

    @test res[1] == -1.0
    @test res[2] == 0.0

    res = derivative(f, [2.0, 1.0, 2.0], :jac_vec; dir=[-1.0, 1.0, 0.0], tape_id=tape_id, reuse_tape=true)

    @test res[1] == -3.0
    @test res[2] == 0.0

    res = derivative(f, [2.0, 1.0, 2.0], :jac_vec; dir=[0.0, 1.0, 0.0], tape_id=tape_id, reuse_tape=true)

    @test res[1] == 1.0
    @test res[2] == 0.0
end

@testset "jac_mat" begin
    function f(x)
        return [x[1]^2 + x[2], x[3]^3]
    end
    x = [1.0, 1.0, 2.0]
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
    res = derivative(f, x, :jac_mat; dir=dir)

    @test res[1, 1] == 2.0
    @test res[1, 2] == -1.0
    @test res[1, 3] == 0.0

    @test res[2, 1] == 0.0
    @test res[2, 2] == 0.0
    @test res[2, 3] == 12.0
end

@testset "reuse_jac_mat" begin
    function f(x)
        return [x[1]^2 + x[2], x[3]^3]
    end
    x = [1.0, 1.0, 2.0]
    tape_id = 1
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

    res = derivative(f, x, :jac_mat; dir=dir, tape_id=tape_id)

    @test res[1, 1] == 2.0
    @test res[1, 2] == -1.0
    @test res[1, 3] == 0.0

    @test res[2, 1] == 0.0
    @test res[2, 2] == 0.0
    @test res[2, 3] == 12.0

    res = derivative(f, [2.0, 1.0, 2.0], :jac_mat; dir=dir, tape_id=tape_id, reuse_tape=true)

    @test res[1, 1] == 4.0
    @test res[1, 2] == -3.0
    @test res[1, 3] == 0.0

    @test res[2, 1] == 0.0
    @test res[2, 2] == 0.0
    @test res[2, 3] == 12.0

    dir[1, 1] = 0.0
    res = derivative(f, [2.0, 1.0, 2.0], :jac_mat; dir=dir, tape_id=tape_id, reuse_tape=true)

    @test res[1, 1] == 0.0
    @test res[1, 2] == -3.0
    @test res[1, 3] == 0.0

    @test res[2, 1] == 0.0
    @test res[2, 2] == 0.0
    @test res[2, 3] == 12.0
end

@testset "vec_jac" begin
    function f(x)
        return [x[1]^2 + x[2], x[3]^3]
    end
    x = [1.0, 1.0, 2.0]
    res = derivative(f, x, :vec_jac; weights=[-1.0, 1.0])

    @test res[1] == -2
    @test res[2] == -1
    @test res[3] == 12
end

@testset "reuse_vec_jac" begin
    function f(x)
        return [x[1]^2 + x[2], x[3]^3]
    end
    x = [1.0, 1.0, 2.0]
    tape_id = 1
    res = derivative(f, x, :vec_jac; weights=[-1.0, 1.0], tape_id=tape_id)

    @test res[1] == -2
    @test res[2] == -1
    @test res[3] == 12

    res = derivative(f, [2.0, 1.0, 2.0], :vec_jac; weights=[-1.0, 1.0], tape_id=tape_id, reuse_tape=true)

    @test res[1] == -4
    @test res[2] == -1
    @test res[3] == 12

    res = derivative(f, [2.0, 1.0, 2.0], :vec_jac; weights=[0.0, 1.0], tape_id=tape_id, reuse_tape=true)

    @test res[1] == 0.0
    @test res[2] == 0.0
    @test res[3] == 12
end

@testset "mat_jac" begin
    function f(x)
        return [x[1]^2 + x[2], x[3]^3]
    end

    x = [1.0, 1.0, 2.0]
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

    res = derivative(f, x, :mat_jac; weights=weights)

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

@testset "reuse_mat_jac" begin
    function f(x)
        return [x[1]^2 + x[2], x[3]^3]
    end
    x = [1.0, 1.0, 2.0]
    tape_id = 1
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

    res = derivative(f, x, :mat_jac; weights=weights, tape_id=tape_id)

    @test res[1, 1] == 2.0
    @test res[1, 2] == 1.0
    @test res[1, 3] == -12.0

    @test res[2, 1] == 0.0
    @test res[2, 2] == 0.0
    @test res[2, 3] == 12.0

    @test res[3, 1] == 0.0
    @test res[3, 2] == 0.0
    @test res[3, 3] == 0.0

    res = derivative(f, [2.0, 1.0, 2.0], :mat_jac; weights=weights, tape_id=tape_id, reuse_tape=true)
    @test res[1, 1] == 4.0
    @test res[1, 2] == 1.0
    @test res[1, 3] == -12.0

    @test res[2, 1] == 0.0
    @test res[2, 2] == 0.0
    @test res[2, 3] == 12.0

    @test res[3, 1] == 0.0
    @test res[3, 2] == 0.0
    @test res[3, 3] == 0.0

    weights[1, 1] = 0.0

    res = derivative(f, [2.0, 1.0, 2.0], :mat_jac; weights=weights, tape_id=tape_id, reuse_tape=true)
    @test res[1, 1] == 0.0
    @test res[1, 2] == 0.0
    @test res[1, 3] == -12.0

    @test res[2, 1] == 0.0
    @test res[2, 2] == 0.0
    @test res[2, 3] == 12.0

    @test res[3, 1] == 0.0
    @test res[3, 2] == 0.0
    @test res[3, 3] == 0.0
end
