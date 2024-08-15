
@testset "test1" begin()
function f(x, param)
    a = x*param
    return a^2
end
    x = 3.0
    param = -1.0
    tape_id = 0
    res = derivative(f, x, param, :jac, tape_id=tape_id)
    @test res[1] == 2*x*param^2


    param = -4.5
    res = derivative(f, x, param, :jac, reuse_tape=true, tape_id=tape_id)
    @test res[1] == 2*x*param^2
end

@testset "test2" begin()
    f(x, param) = x[1] * x[2] * param[1] * param[2]
    x = [1.0, 4.0]
    param = [3.0, 2.0]
    res = derivative(f, x, param, :jac, tape_id=1)
    @test res[1] == 24.0
    @test res[2] == 6.0

    param_new = [-3.0, 1/2]
    res = derivative(f, x, param_new, :jac, tape_id=1, reuse_tape=true)
    @test res[1] == -6.0
    @test res[2] == -3/ 2
end

@testset "test3" begin()

    function f(x, param)
        x1 = x[1] * param[1]
        return [x1*x[2], x[2]] 
    end
    x = [-1.0, 1/2]
    param = 3.0
    dir = [2.0, -2.0]
    res = derivative(f, x, param, :jac_vec, dir=dir, tape_id=1)

    @test res[1] == 9.0
    @test res[2] == -2.0

    param = -3.0
    x = [1.0, 1.0]
    res = derivative(f, x, param, :jac_vec, dir=dir, tape_id=1, reuse_tape=true)

    @test res[1] == 0.0
    @test res[2] == -2.0
end


@testset "test1_mutating" begin()
    function f(x, param)
        a = x*param
        return a^2
    end
    x = 3.0
    param = -1.0
    tape_id = 0
    m = n = 1
    res = CxxVector(1)
    derivative!(res, f, m, n, x, param, :jac, tape_id=tape_id)
    @test res[1] == 2*x*param^2


    param = -4.5
    derivative!(res, f, m, n, x, param, :jac, reuse_tape=true, tape_id=tape_id)
    @test res[1] == 2*x*param^2
end

@testset "test2_mutating" begin()
    f(x, param) = x[1] * x[2] * param[1] * param[2]
    x = [1.0, 4.0]
    param = [3.0, 2.0]
    m = 1
    n = 2
    res = CxxVector(2)
    derivative!(res, f, m, n, x, param, :jac, tape_id=1)
    @test res[1] == 24.0
    @test res[2] == 6.0

    param_new = [-3.0, 1/2]
    derivative!(res, f, m, n, x, param_new, :jac, tape_id=1, reuse_tape=true)
    @test res[1] == -6.0
    @test res[2] == -3/ 2
end

@testset "test3_mutating" begin()

    function f(x, param)
        x1 = x[1] * param[1]
        return [x1*x[2], x[2]] 
    end
    x = [-1.0, 1/2]
    param = 3.0
    dir = [2.0, -2.0]
    m = n = 2
    res = CxxVector(2)
    derivative!(res, f, m, n, x, param, :jac_vec, dir=dir, tape_id=1)

    @test res[1] == 9.0
    @test res[2] == -2.0

    param = -3.0
    x = [1.0, 1.0]
    derivative!(res, f, m, n, x, param, :jac_vec, dir=dir, tape_id=1, reuse_tape=true)

    @test res[1] == 0.0
    @test res[2] == -2.0
end