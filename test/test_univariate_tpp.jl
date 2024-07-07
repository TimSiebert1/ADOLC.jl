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
    univariate_tpp!(res, f, degree, x, init_tp)

    @test res[1, 1] == 4.0
    @test res[1, 2] == 12.0
    @test res[1, 3] == 13.0
    @test res[1, 4] == 6.0
    @test res[1, 5] == 0.0
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
    res = univariate_tpp(f, degree, x, init_tp; tape_id=1)

    @test res[1, 1] == 4.0
    @test res[1, 2] == 12.0
    @test res[1, 3] == 13.0
    @test res[1, 4] == 6.0
    @test res[1, 5] == 0.0

    res = univariate_tpp(f, degree, x, init_tp; tape_id=1)

    @test res[1, 1] == 4.0
    @test res[1, 2] == 12.0
    @test res[1, 3] == 13.0
    @test res[1, 4] == 6.0
    @test res[1, 5] == 0.0
end
