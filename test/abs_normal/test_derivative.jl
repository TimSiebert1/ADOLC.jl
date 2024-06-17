

@testset "abs_normal" begin
    function f(x)
        return (
            max(-x[1] - x[2], -x[1] - x[2] + x[1]^2 + x[2]^2 - 1) +
            max(-x[2] - x[3], -x[2] - x[3] + x[2]^2 + x[3]^2 - 1)
        )
    end

    x = [-0.5, -0.5, -0.5]

    abs_normal_form = derivative(
        f,
        x,
        :abs_normal
    )

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
        return (
            max(-x[1] - x[2], -x[1] - x[2] + x[1]^2 + x[2]^2 - 1) +
            max(-x[2] - x[3], -x[2] - x[3] + x[2]^2 + x[3]^2 - 1)
        )
    end

    x = [-1.5, -1.5, -1.5]

    abs_normal_form = derivative(
        f,
        x,
        :abs_normal
    )
    y = f(x)

    @test abs_normal_form.y[1] == y

    x = [-0.5, -0.5, -0.5]
    # reuse abs_normal_form with same id and without retaping
    abs_normal_form = derivative(f,
        x,
        :abs_normal;
        tape_id=abs_normal_form.tape_id,
        reuse_tape=true,
    )
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

