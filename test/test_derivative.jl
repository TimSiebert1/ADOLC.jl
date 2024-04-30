

####### first_order ########

@testset "jac" begin
    # m = 1
    function f(x)
        return x[1]^2 + x[2] * x[3]
    end
    res = alloc_vec_double(3)
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
    res = alloc_vec_double(3)
    derivative!(res, f, 1, 3, [1.0, 1.0, 2.0], :jac)

    @test res[1] == 2.0
    @test res[2] == 2.0
    @test res[3] == 1.0

    derivative!(res, f, 1, 3, [1.0, 1.0, 0.0], :jac, reuse_tape=true)

    @test res[1] == 2.0
    @test res[2] == 0.0
    @test res[3] == 1.0
end

@testset "jac" begin
    # m > 1, n / 2 < m
    function f(x)
        return [x[1]^2 + x[2], x[3]^3]
    end
    res = myalloc2(2, 3)
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
    res = myalloc2(2, 4)

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
    res = myalloc2(2, 4)

    derivative!(res, f, 2, 4, [1.0, 1.0, 2.0, -1.0], :jac)

    @test res[1, 1] == 2.0
    @test res[1, 2] == 1.0
    @test res[1, 3] == 0.0
    @test res[1, 4] == 0.0

    @test res[2, 1] == 0.0
    @test res[2, 2] == 0.0
    @test res[2, 3] == -4.0
    @test res[2, 4] == 4.0

    derivative!(res, f, 2, 4, [0.0, 1.0, 2.0, -1.0], :jac, reuse_tape=true)

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
    res = alloc_vec_double(2)
    derivative!(res, f, 2, 3, [1.0, 1.0, 2.0], :jac_vec, dir=[-1.0, 1.0, 0.0])

    @test res[1] == -1.0
    @test res[2] == 0.0
end


@testset "reuse_jac_vec" begin
    function f(x)
        return [x[1]^2 + x[2], x[3]^3]
    end
    res = alloc_vec_double(2)
    derivative!(res, f, 2, 3, [1.0, 1.0, 2.0], :jac_vec, dir=[-1.0, 1.0, 0.0])

    @test res[1] == -1.0
    @test res[2] == 0.0

    derivative!(res, f, 2, 3, [2.0, 1.0, 2.0], :jac_vec, dir=[-1.0, 1.0, 0.0], reuse_tape=true)

    @test res[1] == -3.0
    @test res[2] == 0.0

    derivative!(res, f, 2, 3, [2.0, 1.0, 2.0], :jac_vec, dir=[0.0, 1.0, 0.0], reuse_tape=true)

    @test res[1] == 1.0
    @test res[2] == 0.0
end


@testset "jac_mat" begin
    function f(x)
        return [x[1]^2 + x[2], x[3]^3]
    end
    res = myalloc2(2, 3)
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

    derivative!(res, f, 2, 3, [1.0, 1.0, 2.0], :jac_mat, dir=dir)

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
    res = myalloc2(2, 3)
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

    derivative!(res, f, 2, 3, [1.0, 1.0, 2.0], :jac_mat, dir=dir)

    @test res[1, 1] == 2.0
    @test res[1, 2] == -1.0
    @test res[1, 3] == 0.0

    @test res[2, 1] == 0.0
    @test res[2, 2] == 0.0
    @test res[2, 3] == 12.0


    derivative!(res, f, 2, 3, [2.0, 1.0, 2.0], :jac_mat, dir=dir, reuse_tape=true)

    @test res[1, 1] == 4.0
    @test res[1, 2] == -3.0
    @test res[1, 3] == 0.0

    @test res[2, 1] == 0.0
    @test res[2, 2] == 0.0
    @test res[2, 3] == 12.0



    dir[1, 1] = 0.0
    derivative!(res, f, 2, 3, [2.0, 1.0, 2.0], :jac_mat, dir=dir, reuse_tape=true)

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
    res = alloc_vec_double(3)
    derivative!(res, f, 2, 3, [1.0, 1.0, 2.0], :vec_jac, weights=[-1.0, 1.0])

    @test res[1] == -2
    @test res[2] == -1
    @test res[3] == 12
end

@testset "reuse_vec_jac" begin
    function f(x)
        return [x[1]^2 + x[2], x[3]^3]
    end
    res = alloc_vec_double(3)
    derivative!(res, f, 2, 3, [1.0, 1.0, 2.0], :vec_jac, weights=[-1.0, 1.0])

    @test res[1] == -2
    @test res[2] == -1
    @test res[3] == 12

    derivative!(res, f, 2, 3, [2.0, 1.0, 2.0], :vec_jac, weights=[-1.0, 1.0], reuse_tape=true)

    @test res[1] == -4
    @test res[2] == -1
    @test res[3] == 12

    derivative!(res, f, 2, 3, [2.0, 1.0, 2.0], :vec_jac, weights=[0.0, 1.0], reuse_tape=true)

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


    res = myalloc2(3, 3)
    derivative!(res, f, 2, 3, [1.0, 1.0, 2.0], :mat_jac, weights=weights)

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


    res = myalloc2(3, 3)
    derivative!(res, f, 2, 3, [1.0, 1.0, 2.0], :mat_jac, weights=weights)

    @test res[1, 1] == 2.0
    @test res[1, 2] == 1.0
    @test res[1, 3] == -12.0

    @test res[2, 1] == 0.0
    @test res[2, 2] == 0.0
    @test res[2, 3] == 12.0

    @test res[3, 1] == 0.0
    @test res[3, 2] == 0.0
    @test res[3, 3] == 0.0


    derivative!(res, f, 2, 3, [2.0, 1.0, 2.0], :mat_jac, weights=weights, reuse_tape=true)
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

    derivative!(res, f, 2, 3, [2.0, 1.0, 2.0], :mat_jac, weights=weights, reuse_tape=true)
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

@testset "abs_normal" begin

    function f(x)
        return (max(-x[1]-x[2], -x[1]-x[2]+x[1]^2+x[2]^2-1) + max(-x[2]-x[3], -x[2]-x[3]+x[2]^2+x[3]^2-1))
    end 
    
    x = [-0.5, -0.5, -0.5]

    abs_normal_form = ADOLC.init_abs_normal_form(f, 1, 3, x)
    derivative!(abs_normal_form, f, 1, 3, x, :abs_normal, tape_id=abs_normal_form.tape_id, reuse_tape=true)

    @test abs_normal_form.Y[1, 1] == -1.5
    @test abs_normal_form.Y[1, 2] == -3.0
    @test abs_normal_form.Y[1, 3] == -1.5

    @test abs_normal_form.J[1, 1] == 0.5
    @test abs_normal_form.J[1, 2] == 0.5

    @test abs_normal_form.Z[1, 1] == -1.0
    @test abs_normal_form.Z[1, 2] == -1.0
    @test abs_normal_form.Z[1, 3] == 0.0
    @test abs_normal_form.Z[2, 1] == 0.0
    @test abs_normal_form.Z[2, 2] == -1.0
    @test abs_normal_form.Z[2, 3] == -1.0

    @test abs_normal_form.L[1, 1] == 0.0
    @test abs_normal_form.L[1, 2] == 0.0
    @test abs_normal_form.L[2, 1] == 0.0
    @test abs_normal_form.L[2, 2] == 0.0
end


@testset "resuse_abs_normal" begin

    function f(x)
        return (max(-x[1]-x[2], -x[1]-x[2]+x[1]^2+x[2]^2-1) + max(-x[2]-x[3], -x[2]-x[3]+x[2]^2+x[3]^2-1))
    end 
    
    x = [-1.5, -1.5, -1.5]

    abs_normal_form = ADOLC.init_abs_normal_form(f, 1, 3, x)
    derivative!(abs_normal_form, f, 1, 3, x, :abs_normal, tape_id=abs_normal_form.tape_id, reuse_tape=true)
    y = f(x)

    @test abs_normal_form.y[1] == y

    x = [-0.5, -0.5, -0.5]
    # reuse abs_normal_form with same id and without retaping
    derivative!(abs_normal_form, f, 1, 3, x, :abs_normal, tape_id=abs_normal_form.tape_id, reuse_tape=true)
    y = f(x)

    @test abs_normal_form.y[1] == y

    @test abs_normal_form.Y[1, 1] == -1.5
    @test abs_normal_form.Y[1, 2] == -3.0
    @test abs_normal_form.Y[1, 3] == -1.5

    @test abs_normal_form.J[1, 1] == 0.5
    @test abs_normal_form.J[1, 2] == 0.5

    @test abs_normal_form.Z[1, 1] == -1.0
    @test abs_normal_form.Z[1, 2] == -1.0
    @test abs_normal_form.Z[1, 3] == 0.0
    @test abs_normal_form.Z[2, 1] == 0.0
    @test abs_normal_form.Z[2, 2] == -1.0
    @test abs_normal_form.Z[2, 3] == -1.0

    @test abs_normal_form.L[1, 1] == 0.0
    @test abs_normal_form.L[1, 2] == 0.0
    @test abs_normal_form.L[2, 1] == 0.0
    @test abs_normal_form.L[2, 2] == 0.0
end




######### second_order ########



@testset "hess_1D" begin
    function f(x) 
        return x[1]^3 + x[2]^2 * x[3]
    end
    res = myalloc2(3, 3)
    derivative!(res, f, 1, 3, [-1.0, 1.0, 2.0], :hess)

    @test res[1, 1] == -6.0
    @test res[2, 1] == 0.0
    @test res[3, 1] == 0.0

    @test res[1, 2] == 0.0
    @test res[2, 2] == 4.0
    @test res[3, 2] == 2.0 

    @test res[1, 3] == 0.0
    @test res[2, 3] == 0.0
    @test res[3, 3] == 0.0


end

@testset "hess_2D" begin



    function f(x) 
        return [x[1]^3 + x[2]^2, 3*x[3]^3]
    end

    x = [-1.0, 2.0, 2.0]
    res = myalloc3(2, 3, 3)
    derivative!(res, f, 2, 3, x, :hess, tape_id=1)

    @test res[1, 1, 1] == -6.0
    @test res[1, 2, 1] == 0.0
    @test res[1, 3, 1] == 0.0


    @test res[2, 1, 1] == 0.0
    @test res[2, 2, 1] == 0.0
    @test res[2, 3, 1] == 0.0


    @test res[1, 1, 2] == 0.0
    @test res[1, 2, 2] == 2.0
    @test res[1, 3, 2] == 0.0


    @test res[2, 1, 2] == 0.0
    @test res[2, 2, 2] == 0.0
    @test res[2, 3, 2] == 0.0


    @test res[1, 1, 3] == 0.0
    @test res[1, 2, 3] == 0.0
    @test res[1, 3, 3] == 0.0


    @test res[2, 1, 3] == 0.0
    @test res[2, 2, 3] == 0.0
    @test res[2, 3, 3] == 36.0
end




@testset "hess_vec" begin
    function f(x)
        return [x[1]^2*x[2], x[1]*x[3]^3]
    end

    x = [1.0, 2.0, 2.0]
    dir = [-1.0, 1.0, 3.0]

    res = myalloc2(2, 3)
    derivative!(res, f, 2, 3, x, :hess_vec, dir=dir)

    @test res[1, 1] == -2.0
    @test res[1, 2] == -2.0
    @test res[1, 3] == 0.0

    @test res[2, 1] == 36.0
    @test res[2, 2] == 0.0
    @test res[2, 3] == 24.0
end


@testset "vec_hess_vec" begin
    function f(x)
        return [x[1]^2*x[2], x[1]*x[3]^3]
    end

    x = [1.0, 2.0, 2.0]
    dir = [-1.0, 1.0, 3.0]
    weights = [2.0, 1.0]
    res = alloc_vec_double(3)
    derivative!(res, f, 2, 3, x, :vec_hess_vec, dir=dir, weights=weights)

    @test res[1] == 32.0
    @test res[2] == -4.0
    @test res[3] == 24.0

end



@testset "mat_hess_vec" begin
    function f(x)
        return [x[1]^2*x[2], x[1]*x[3]^3]
    end

    # 4 2 0 
    # 2 0 0 
    # 0 0 0
    
    # 0  0   12
    # 0  0   0 
    # 12 0   12

    x = [1.0, 2.0, 2.0]
    dir = [-1.0, 1.0, 3.0]
    weights = Matrix{Float64}(undef, 3, 2)
    for i in 1:3
        for j in 1:2
            weights[i, j] = 0.0
            if i == j
                weights[i, i] = 1.0
            end
        end
    end
    weights[3, 2] = -1.0
    res = myalloc2(3, 3)
    derivative!(res, f, 2, 3, x, :mat_hess_vec, dir=dir, weights=weights)

    @test res[1, 1] == -2.0
    @test res[1, 2] == -2.0
    @test res[1, 3] == 0.0

    @test res[2, 1] == 36.0
    @test res[2, 2] == 0.0
    @test res[2, 3] == 24.0

    @test res[3, 1] == -36.0
    @test res[3, 2] == 0.0
    @test res[3, 3] == -24.0
end



@testset "1D_hess_mat" begin()
    function f(x)
        return x[1]*x[3]^3
    end

    x = [1.0, 2.0, 2.0]
    dir = Matrix{Float64}(undef, 3, 2)
    for i in 1:3
        for j in 1:2
            dir[i, j] = 0.0
            if i == j
                dir[i, i] = 1.0
            end
        end
    end
    dir[3, 2] = -1.0
    res = myalloc2(3, 2)
    derivative!(res, f, 1, 3, x, :hess_mat, dir=dir)

    @test res[1, 1] == 0.0
    @test res[1, 2] == -12.0

    @test res[2, 1] == 0.0
    @test res[2, 2] == 0.0

    @test res[3, 1] == 12.0
    @test res[3, 2] == -12.0

end




@testset "2D_hess_mat" begin()
    function f(x)
        return [x[1]^2*x[2], x[1]*x[3]^3]
    end

    # 4 2 0 
    # 2 0 0 
    # 0 0 0
    
    # 0  0   12
    # 0  0   0 
    # 12 0   12

    x = [1.0, 2.0, 2.0]
    dir = [-1.0, 1.0, 3.0]
    weights = Matrix{Float64}(undef, 3, 2)
    for i in 1:3
        for j in 1:2
            weights[i, j] = 0.0
            if i == j
                weights[i, i] = 1.0
            end
        end
    end
    weights[3, 2] = -1.0
    res = myalloc2(3, 3)
    derivative!(res, f, 2, 3, x, :mat_hess_vec, dir=dir, weights=weights)

    @test res[1, 1] == -2.0
    @test res[1, 2] == -2.0
    @test res[1, 3] == 0.0

    @test res[2, 1] == 36.0
    @test res[2, 2] == 0.0
    @test res[2, 3] == 24.0

    @test res[3, 1] == -36.0
    @test res[3, 2] == 0.0
    @test res[3, 3] == -24.0
end



@testset "1D_vec_hess" begin()
    function f(x)
        return x[1]*x[3]^3
    end

    # 0.0 0.0 12.0
    # 0.0 0.0 0.0
    # 12.0 0.0 12.0

    # 
    x = [1.0, 2.0, 2.0]
    weights = [-1.0]
    res = myalloc2(3, 3)
    derivative!(res, f, 1, 3, x, :vec_hess, weights=weights)

    @test res[1, 1] == 0.0
    @test res[1, 2] == 0.0
    @test res[1, 3] == -12.0

    @test res[2, 1] == 0.0
    @test res[2, 2] == 0.0
    @test res[2, 3] == 0.0

    @test res[3, 1] == -12.0
    @test res[3, 2] == 0.0
    @test res[3, 3] == -12.0

end




@testset "2D_vec_hess" begin()
    function f(x)
        return [x[1]^2*x[2], x[1]*x[3]^3]
    end

    # 4 2 0 
    # 2 0 0 
    # 0 0 0
    
    # 0  0   12
    # 0  0   0 
    # 12 0   12

    x = [1.0, 2.0, 2.0]
    weights = [-1.0, 1.0]

    res = myalloc2(3, 3)
    derivative!(res, f, 2, 3, x, :vec_hess, weights=weights)

    @test res[1, 1] == -4.0
    @test res[1, 2] == -2.0
    @test res[1, 3] == 12.0

    @test res[2, 1] == -2.0
    @test res[2, 2] == 0.0
    @test res[2, 3] == 0.0

    @test res[3, 1] == 12.0
    @test res[3, 2] == 0.0
    @test res[3, 3] == 12.0
end

"""
function test_second_order_mat_hess()
    derivative(f, 2, 3, [1.0, 1.0, 2.0], :mat_hess, [-1.0, 1.0], res)
end

function test_second_order_vec_hess_mat()
    derivative(f, 2, 3, [1.0, 1.0, 2.0], :vec_hess_mat, [-1.0, 1.0], res)
end
function test_second_order_mat_hess_mat()
    derivative(f, 2, 3, [1.0, 1.0, 2.0], :mat_hess_mat, [-1.0, 1.0], res)
end




##### higher_order #######

function test_higher_order()
    derivative(f, 2, 3, [1.0, 1.0, 2.0], [-1.0, 1.0], res)
end


"""