
@testset "Adouble" begin
    for t in [TbAlloc, TlAlloc]
        A = Adouble{t}()
        @test typeof(A.val) == t
        @test typeof(A) == Adouble{t}

        a = t == TbAlloc ? ADOLC.TbadoubleCxx(3.0) : ADOLC.TladoubleCxx(3.0)
        A = Adouble{t}(a)

        @test a === A.val
        @test typeof(A.val) == t
        @test typeof(A) == Adouble{t}

        a = Adouble{t}(1)
        @test typeof(a.val) == Float64
        @test typeof(a) == Adouble{t}
        @test a.val == 1.0

        A = Adouble{t}(3.0)
        B = Adouble{t}(A)
        @test A.val == B.val

        a = Adouble{t}(15.3; adouble=true)
        @test getValue(a) == 15.3
        @test typeof(a.val) == t

        a = Adouble{t}(true; adouble=true)
        @test getValue(a) == 1.0
        @test typeof(a.val) == t

        a = Adouble{t}(15.3; adouble=false)
        @test getValue(a) == 15.3
        @test typeof(a.val) == Float64

        A = Adouble{t}([i for i in 1:10]; adouble=true)
        @test [float(i) for i in 1:10] == getValue(A)
        @test typeof(A) == Vector{Adouble{t}}

        A = Adouble{t}([i for i in 1:10])
        @test [float(i) for i in 1:10] == A
        @test typeof(A) == Vector{Adouble{t}}
    end
end
