
bin_ops = [+, -, *, /, max, min]
un_ops = [
    abs,
    sqrt,
    log,
    log10,
    sin,
    cos,
    exp,
    tan,
    #asin,
    acos,
    atan,
    sinh,
    cosh,
    tanh,
    sinh,
    asinh,
    atanh,
    ceil,
    floor,
    #frexp, frexp(a, Ref(Cint(3)))
]

comps = [>=, >, <=, <, ==]

@testset "binary operations" begin()
    for t in [Adouble{TlAlloc}, Adouble{TbAlloc}]
        for op in bin_ops
            a = t(2.0, adouble=true)
            @test getValue(op(a, -2.0)) == op(2.0, -2.0)
            @test getValue(op(-2.0, a)) == op(-2.0, 2.0)
            @test getValue(op(a, a)) == op(getValue(a), getValue(a))
            a = t(2.0, adouble=false)
            @test getValue(op(a, -2.0)) == op(2.0, -2.0)
            @test getValue(op(-2.0, a)) == op(-2.0, 2.0)
            @test getValue(op(a, a)) == op(getValue(a), getValue(a))
        end
        op = ldexp
        a = t(2.0, adouble=true)
        @test getValue(op(a, 3)) == op(2.0, 3)
        a = t(2.0, adouble=false)
        @test getValue(op(a, 3)) == op(2.0, 3)

        op = ^
        a = t(2.0, adouble=true)
        @test getValue(op(a, 3)) == op(2.0, 3)
        a = t(2.0, adouble=false)
        @test getValue(op(a, 3)) == op(2.0, 3)  
    end
end

@testset "unary operations" begin()
    for t in [Adouble{TlAlloc}, Adouble{TbAlloc}]
        for op in un_ops
            a = t(0.5, adouble=true)
            @test getValue(op(a)) ≈ op(0.5)
            a = t(0.5, adouble=false)
            @test getValue(op(a)) ≈ op(0.5)
        end
        a = t(1.5, adouble=true)
        @test getValue(acosh(a)) ≈ acosh(1.5)
        a = t(1.5, adouble=false)
        @test getValue(acosh(a)) ≈ acosh(1.5)
        @test eps(t) == eps(Cdouble)
    end
end

@testset "comps" begin()
    for t in [Adouble{TlAlloc}, Adouble{TbAlloc}]
        for op in comps
            a = t(0.5, adouble=true)
            @test op(a, 2) == op(0.5, 2)
            @test op(2, a) == op(2, 0.5)
            @test op(a, a) == op(0.5, 0.5)

            a = t(0.5, adouble=false)
            @test op(a, 2) == op(0.5, 2)
            @test op(2, a) == op(2, 0.5)
            @test op(a, a) == op(0.5, 0.5)
        end
    end
end


special_funcs = [SpecialFunctions.erfc, SpecialFunctions.erf]
@testset "special_funcs" begin()
    for t in [Adouble{TlAlloc}, Adouble{TbAlloc}]
        for op in un_ops
            a = t(0.5, adouble=true)
            @test getValue(op(a)) ≈ op(0.5)
            a = t(0.5, adouble=false)
            @test getValue(op(a)) ≈ op(0.5)
        end
    end
end 

