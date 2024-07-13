@testset "test_active_variables" begin()
    f(x) = x[1] * x[2] * x[3] * x[4]
    xx = [1.0, 3.0, 4.0, 2.0]
    active = [1, 3]
    mode = :jac
    res = derivative(f, xx, mode, active)
    @test res[1] == 24.0
    @test res[2] == 6.0
    # active is not used again.... 
    @test_throws AssertionError derivative(f, xx, :jac, active, reuse_tape=true, tape_id=0)
    res = derivative(f, [-3.0, 1/2], :jac, active, tape_id=0, reuse_tape=true)
    @test res[1] == 3.0
    @test res[2] == -18.0

    f1(x) = [x[1]*x[2], x[2]] 
    x = [4.0, 1.0]
    dir = [2.0]
    active = [2]
    res = derivative(f1, x, :jac_vec, active, dir=dir)

    @test res[1] == 8.0
    @test res[2] == 2.0
end