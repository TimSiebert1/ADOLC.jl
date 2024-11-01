@testset "univariate_tpp_wo_seed" begin
    ()
    f(x) = x[1]^2 * x[2]^2
    x = [1.0, 2.0]
    m = 1
    n = 2
    degree = 3
    res = univariate_tpp(f, x, degree)

    @test res[1, 1] ≈ 4.0
    @test res[1, 2] ≈ 12.0
    @test res[1, 3] ≈ 13.0
    @test res[1, 4] ≈ 6.0
    @test isapprox(res[1, 5], 0.0, atol=1e-16)
end
@testset "univariate_tpp" begin
    ()
    f(x) = x[1]^2 * x[2]^2
    x = [1.0, 2.0]
    m = 1
    n = 2
    degree = 3
    init_tp = CxxMatrix(zeros(Cdouble, n, degree + 1))
    for j in 1:n
        for i in 1:n
            if j == 1
                init_tp[i, j] = x[i]
            elseif j == 2
                init_tp[i, j] = 1.0
            end
        end
    end
    res = CxxMatrix(m, degree + 1)
    tape_id = 1
    _, m, n = create_tape(f, x, tape_id)
    univariate_tpp!(res, tape_id, m, n, degree, init_tp)

    @test res[1, 1] ≈ 4.0
    @test res[1, 2] ≈ 12.0
    @test res[1, 3] ≈ 13.0
    @test res[1, 4] ≈ 6.0
    @test isapprox(res[1, 5], 0.0, atol=1e-16)
end

@testset "univariate_tpp" begin
    ()
    f(x) = x[1]^2 * x[2]^2
    x = [1.0, 2.0]
    m = 1
    n = 2
    degree = 3
    init_tp = CxxMatrix(zeros(Cdouble, n, degree + 1))
    for j in 1:n
        for i in 1:n
            if j == 1
                init_tp[i, j] = x[i]
            elseif j == 2
                init_tp[i, j] = 1.0
            end
        end
    end
    res = univariate_tpp(f, x, degree, init_tp; tape_id=1)

    @test res[1, 1] ≈ 4.0
    @test res[1, 2] ≈ 12.0
    @test res[1, 3] ≈ 13.0
    @test res[1, 4] ≈ 6.0
    @test isapprox(res[1, 5], 0.0, atol=1e-16)

    res = univariate_tpp(f, x, degree, init_tp; tape_id=1)

    @test res[1, 1] == 4.0
    @test res[1, 2] == 12.0
    @test res[1, 3] == 13.0
    @test res[1, 4] == 6.0
    @test isapprox(res[1, 5], 0.0, atol=1e-16)
end
