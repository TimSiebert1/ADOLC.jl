
@testset "adouble" begin
    ()
    a = ccall((:create_tb_adouble, adolc_interface_lib), TapeBasedAD, (Cdouble,), 3.0)
    @test get_value(Adouble{TapeBasedAD}(a)) == 3.0
    @test get_value(Adouble{TapeBasedAD}(3.0)) == 3.0
    @test get_value(Adouble{TapeBasedAD}(3.0; is_diff=true)) == 3.0

    @test get_value(Adouble{TapeBasedAD}()) == 0.0

    @test get_value(Adouble{TapeLessAD}(3.0)) == 3.0
    @test get_value(Adouble{TapeLessAD}(3.0; is_diff=true)) == 3.0
    @test get_value(Adouble{TapeLessAD}(3.0, 4.0)) == 3.0
    @test get_ad_value(Adouble{TapeLessAD}(3.0, 4.0)) == 4.0
    a = Adouble{TapeLessAD}(0.0; is_diff=true)
    set_value(a, 3.0)
    @test get_value(a) == 3.0
    set_ad_value(a, 4.0)
    @test get_ad_value(a) == 4.0

    set_num_dir(10)
    a = Adouble{TapeLessAD}(-1.0; is_diff=true)
    @test all(
        get_ad_values(Adouble{TapeLessAD}(3.0, ones(Cdouble, 10)), 10) == ones(Cdouble, 10)
    )
    set_ad_value(a, ones(Cdouble, 10))
    @test all(get_ad_values(a, 10) == ones(Cdouble, 10))
    set_ad_value(a, zeros(Cdouble, 10))
    set_ad_value(a, 3, -4.0)
    @test get_ad_value(a, 3) == -4.0
    set_ad_value(a, 10, -2.0)
    @test get_ad_value(a, 10) == -2.0
    set_ad_value(a, 1, -2.1)
    @test get_ad_value(a, 1) == -2.1
end

@testset "type handling" begin
    ()

    for t in [Adouble{TapeLessAD}, Adouble{TapeBasedAD}]
        a = t(3.0; is_diff=true)
        @test typeof(promote(1, a).val) == Cdouble
        @test typeof(promote(1, a)) == t

        b = t(3.0; is_diff=false)
        @test typeof(promote(1, b).val) == Cdouble
        @test typeof(promote(1, b)) == t

        @test typeof(promote(a, b)) == t
        @test typeof(promote(a, b).val) == typeof(a.val)

        @test Base.promote_rule(t, Real) == t
        @test Base.promote_op(x -> x, t, t) == t
        @test Base.promote_op(x -> x, Int64, t) == t
        @test Base.promote_op(x -> x, t, Int64) == t
        @test typeof(convert(t, 1).val) == Cdouble
        @test typeof(convert(t, 1)) == t
        @test convert(t, b) === b
    end
end
