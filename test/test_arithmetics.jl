
bin_ops = [*, +, -, /, max, min]
un_ops = [
    abs,
    sqrt,
    log,
    log10,
    sin,
    cos,
    exp,
    tan,
    asin,
    acos,
    atan,
    sinh,
    cosh,
    tanh,
    asinh,
    #acosh,
    atanh,
    ceil,
    floor,
    #frexp, frexp(a, Ref(Cint(3)))
]

comps = [>=, >, <=, <, ==]

@testset "binary operations" begin
    ()
    for t in [Adouble{TapeBasedAD}, Adouble{TapeLessAD}]
        for op in bin_ops
            a = t(2.0; is_diff=true)
            @test get_value(op(a, -2.0)) == op(2.0, -2.0)
            @test get_value(op(-2.0, a)) == op(-2.0, 2.0)
            @test get_value(op(a, a)) == op(get_value(a), get_value(a))
            a = t(2.0; is_diff=false)
            @test get_value(op(a, -2.0)) == op(2.0, -2.0)
            @test get_value(op(-2.0, a)) == op(-2.0, 2.0)
            @test get_value(op(a, a)) == op(get_value(a), get_value(a))
        end
        a = t(2.0; is_diff=true)
        @test get_value(-a) == -2.0
        a = t(2.0; is_diff=false)
        @test get_value(-a) == -2.0

        op = ldexp
        a = t(2.0; is_diff=true)
        @test get_value(op(a, 3)) == op(2.0, 3)
        a = t(2.0; is_diff=false)
        @test get_value(op(a, 3)) == op(2.0, 3)
        a = t(2.0; is_diff=true)

        op = ^
        @test get_value(op(a, -2.0)) == op(2.0, -2.0)
        @test get_value(op(a, a)) == op(get_value(a), get_value(a))
        a = t(2.0; is_diff=false)
        @test get_value(op(a, -2.0)) == op(2.0, -2.0)
        @test get_value(op(a, a)) == op(get_value(a), get_value(a))
    end
end

@testset "unary operations" begin
    ()
    for t in [Adouble{TapeBasedAD}, Adouble{TapeLessAD}]
        for op in un_ops
            a = t(0.5; is_diff=true)
            @test get_value(op(a)) ≈ op(0.5)
            a = t(0.5; is_diff=false)
            @test get_value(op(a)) ≈ op(0.5)
        end
        a = t(1.5; is_diff=true)
        @test get_value(acosh(a)) ≈ acosh(1.5)
        a = t(1.5; is_diff=false)
        @test get_value(acosh(a)) ≈ acosh(1.5)
        @test eps(t) == eps(Cdouble)
    end
end

@testset "comps" begin
    ()
    for t in [Adouble{TapeBasedAD}, Adouble{TapeLessAD}]
        for op in comps
            a = t(0.5; is_diff=true)
            @test op(a, 2) == op(0.5, 2)
            @test op(2, a) == op(2, 0.5)
            @test op(a, a) == op(0.5, 0.5)

            a = t(0.5; is_diff=false)
            @test op(a, 2) == op(0.5, 2)
            @test op(2, a) == op(2, 0.5)
            @test op(a, a) == op(0.5, 0.5)
        end
    end
end

special_funcs = [SpecialFunctions.erfc, SpecialFunctions.erf]
@testset "special_funcs" begin
    ()
    for t in [Adouble{TapeBasedAD}, Adouble{TapeLessAD}]
        for op in special_funcs
            a = t(0.5; is_diff=true)
            @test get_value(op(a)) ≈ op(0.5)
            a = t(0.5; is_diff=false)
            @test get_value(op(a)) ≈ op(0.5)
        end
    end
end
