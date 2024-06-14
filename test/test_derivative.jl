
####### first_order ########

@testset "jac" begin
    # m = 1
    function f(x)
        return x[1]^2 + x[2] * x[3]
    end
    derivative(f, 1, 3, [1.0, 1.0, 2.0], :jac)

    @test res[1] == 2.0
    @test res[2] == 2.0
    @test res[3] == 1.0
end
