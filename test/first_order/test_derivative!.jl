@testset "not_implemented" begin
    ()
    # m = 1
    function f(x)
        return x[1]^2 + x[2] * x[3]
    end
    res = CxxVector(3)
    @test_throws ArgumentError derivative!(res, f, 1, 3, [1.0, 1.0, 2.0], :ja)
end
@testset "jac" begin
    # m = 1
    function f(x)
        return x[1]^2 + x[2] * x[3]
    end
    res = CxxVector(3)
    derivative!(res, f, 1, 3, [1.0, 1.0, 2.0], :jac)

    @test res[1] == 2.0
    @test res[2] == 2.0
    @test res[3] == 1.0
end

@testset "reuse_jac" begin
    # m = 1
    function f(x)
        return x[1]^2 + x[2] * x[3]
    end
    res = CxxVector(3)
    derivative!(res, f, 1, 3, [1.0, 1.0, 2.0], :jac)

    @test res[1] == 2.0
    @test res[2] == 2.0
    @test res[3] == 1.0

    derivative!(res, f, 1, 3, [1.0, 1.0, 0.0], :jac; reuse_tape=true)

    @test res[1] == 2.0
    @test res[2] == 0.0
    @test res[3] == 1.0
end

@testset "jac_tl" begin
    # m > 1, n / 2 < m
    function f(x)
        return [x[1]^2 + x[2], x[3]^3]
    end
    res = CxxMatrix(2, 3)
    derivative!(res, f, 2, 3, [1.0, 1.0, 2.0], :jac)

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
    res = CxxMatrix(2, 4)

    derivative!(res, f, 2, 4, [1.0, 1.0, 2.0, -1.0], :jac)

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
    res = CxxMatrix(2, 4)

    derivative!(res, f, 2, 4, [1.0, 1.0, 2.0, -1.0], :jac)

    @test res[1, 1] == 2.0
    @test res[1, 2] == 1.0
    @test res[1, 3] == 0.0
    @test res[1, 4] == 0.0

    @test res[2, 1] == 0.0
    @test res[2, 2] == 0.0
    @test res[2, 3] == -4.0
    @test res[2, 4] == 4.0

    derivative!(res, f, 2, 4, [0.0, 1.0, 2.0, -1.0], :jac; reuse_tape=true)

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
    res = CxxVector(2)
    derivative!(res, f, 2, 3, [1.0, 1.0, 2.0], :jac_vec; dir=[-1.0, 1.0, 0.0])

    @test res[1] == -1.0
    @test res[2] == 0.0
end

@testset "reuse_jac_vec" begin
    function f(x)
        return [x[1]^2 + x[2], x[3]^3]
    end
    res = CxxVector(2)
    derivative!(res, f, 2, 3, [1.0, 1.0, 2.0], :jac_vec; dir=[-1.0, 1.0, 0.0])

    @test res[1] == -1.0
    @test res[2] == 0.0

    derivative!(
        res, f, 2, 3, [2.0, 1.0, 2.0], :jac_vec; dir=[-1.0, 1.0, 0.0], reuse_tape=true
    )

    @test res[1] == -3.0
    @test res[2] == 0.0

    derivative!(
        res, f, 2, 3, [2.0, 1.0, 2.0], :jac_vec; dir=[0.0, 1.0, 0.0], reuse_tape=true
    )

    @test res[1] == 1.0
    @test res[2] == 0.0
end

@testset "jac_mat" begin
    function f(x)
        return [x[1]^2 + x[2], x[3]^3]
    end
    res = CxxMatrix(2, 3)
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

    derivative!(res, f, 2, 3, [1.0, 1.0, 2.0], :jac_mat; dir=dir)

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
    res = CxxMatrix(2, 3)
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

    derivative!(res, f, 2, 3, [1.0, 1.0, 2.0], :jac_mat; dir=dir)

    @test res[1, 1] == 2.0
    @test res[1, 2] == -1.0
    @test res[1, 3] == 0.0

    @test res[2, 1] == 0.0
    @test res[2, 2] == 0.0
    @test res[2, 3] == 12.0

    derivative!(res, f, 2, 3, [2.0, 1.0, 2.0], :jac_mat; dir=dir, reuse_tape=true)

    @test res[1, 1] == 4.0
    @test res[1, 2] == -3.0
    @test res[1, 3] == 0.0

    @test res[2, 1] == 0.0
    @test res[2, 2] == 0.0
    @test res[2, 3] == 12.0

    dir[1, 1] = 0.0
    derivative!(res, f, 2, 3, [2.0, 1.0, 2.0], :jac_mat; dir=dir, reuse_tape=true)

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
    res = CxxVector(3)
    derivative!(res, f, 2, 3, [1.0, 1.0, 2.0], :vec_jac; weights=[-1.0, 1.0])

    @test res[1] == -2
    @test res[2] == -1
    @test res[3] == 12
end

@testset "reuse_vec_jac" begin
    function f(x)
        return [x[1]^2 + x[2], x[3]^3]
    end
    res = CxxVector(3)
    derivative!(res, f, 2, 3, [1.0, 1.0, 2.0], :vec_jac; weights=[-1.0, 1.0])

    @test res[1] == -2
    @test res[2] == -1
    @test res[3] == 12

    derivative!(
        res, f, 2, 3, [2.0, 1.0, 2.0], :vec_jac; weights=[-1.0, 1.0], reuse_tape=true
    )

    @test res[1] == -4
    @test res[2] == -1
    @test res[3] == 12

    derivative!(
        res, f, 2, 3, [2.0, 1.0, 2.0], :vec_jac; weights=[0.0, 1.0], reuse_tape=true
    )

    @test res[1] == 0.0
    @test res[2] == 0.0
    @test res[3] == 12
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

    res = CxxMatrix(3, 3)
    derivative!(res, f, 2, 3, [1.0, 1.0, 2.0], :mat_jac; weights=weights)

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

    #res = myalloc2(3, 3)
    res = CxxMatrix(3, 3)
    derivative!(res, f, 2, 3, [1.0, 1.0, 2.0], :mat_jac; weights=weights)

    @test res[1, 1] == 2.0
    @test res[1, 2] == 1.0
    @test res[1, 3] == -12.0

    @test res[2, 1] == 0.0
    @test res[2, 2] == 0.0
    @test res[2, 3] == 12.0

    @test res[3, 1] == 0.0
    @test res[3, 2] == 0.0
    @test res[3, 3] == 0.0

    derivative!(res, f, 2, 3, [2.0, 1.0, 2.0], :mat_jac; weights=weights, reuse_tape=true)
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

    derivative!(res, f, 2, 3, [2.0, 1.0, 2.0], :mat_jac; weights=weights, reuse_tape=true)
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

@testset "tape_less_forward!" begin()
    f(x) = x[1]^2
    x = 2.0
    res = allocator(1, 1, :jac, 0, 0)
    ADOLC.tape_less_forward!(res, f, 1, x)
    @test res[1] == 4.0
end