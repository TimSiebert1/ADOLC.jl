
@testset "Adouble" begin
    # test Tballoc empty constructor

    A = Adouble{TbAlloc}()

    @test typeof(A.val) == TbAlloc
    @test typeof(A) == Adouble{TbAlloc}

    # test Tlalloc empty constructor

    A = Adouble{TlAlloc}()

    @test typeof(A.val) == TlAlloc
    @test typeof(A) == Adouble{TlAlloc}

    # test tballoc constructor
    a = ADOLC.TbadoubleCxx(3.0)
    A = Adouble{TbAlloc}(a)

    @test a === A.val
    @test typeof(A) == Adouble{TbAlloc}

    # test tlalloc constructor
    a = ADOLC.TladoubleCxx(3.0)
    A = Adouble{TlAlloc}(a)

    @test a === A.val
    @test typeof(A) == Adouble{TlAlloc}

    # test int constructor Tballoc
    a = Adouble{TbAlloc}(1)

    @test typeof(a.val) == Float64
    @test typeof(a) == Adouble{TbAlloc}
    @test a.val == 1

    # test float constructor Tballoc
    a = Adouble{TbAlloc}(-1.2)

    @test typeof(a.val) == Float64
    @test typeof(a) == Adouble{TbAlloc}
    @test a.val == -1.2

    # test int constructor Tlalloc
    a = Adouble{TlAlloc}(1)

    @test typeof(a.val) == Float64
    @test typeof(a) == Adouble{TlAlloc}
    @test a.val == 1

    # test float constructor Tlalloc
    a = Adouble{TlAlloc}(-1.2)

    @test typeof(a.val) == Float64
    @test typeof(a) == Adouble{TlAlloc}
    @test a.val == -1.2

    # test Tballoc constructor from float
    a = Adouble{TbAlloc}(15.3, true)

    @test getValue(a) == 15.3
    @test typeof(a.val) == TbAlloc

    # test Tballoc constructor from bool
    a = Adouble{TbAlloc}(true, true)

    @test getValue(a) == 1.0
    @test typeof(a.val) == TbAlloc

    # test Tlalloc constructor from float
    a = Adouble{TlAlloc}(15.3, true)

    @test getValue(a) == 15.3
    @test typeof(a.val) == TlAlloc

    # test Tballoc constructor from bool
    a = Adouble{TlAlloc}(false, true)

    @test getValue(a) == 0.0
    @test typeof(a.val) == TlAlloc

    # test Adouble{TbAlloc} constructor from float
    a = Adouble{TbAlloc}(15.3, false)

    @test getValue(a) == 15.3
    @test typeof(a.val) == Float64

    # test Adouble{TbAlloc} constructor from bool
    a = Adouble{TbAlloc}(true, false)

    @test getValue(a) == 1.0
    @test typeof(a.val) == Float64

    # test Adouble{TlAlloc} constructor from float
    a = Adouble{TlAlloc}(15.3, false)

    @test getValue(a) == 15.3
    @test typeof(a.val) == Float64

    # test Adouble{TlAlloc} constructor from bool
    a = Adouble{TlAlloc}(false, false)

    @test getValue(a) == 0.0
    @test typeof(a.val) == Float64

    # test getValue vector Adouble{TbAlloc}
    A = [Adouble{TbAlloc}(i, true) for i in 1:10]

    @test [float(i) for i in 1:10] == getValue(A)
    @test typeof(A) == Vector{Adouble{TbAlloc}}

    # test getValue vector Adouble{TlAlloc}
    A = [Adouble{TlAlloc}(i, true) for i in 1:10]

    @test [float(i) for i in 1:10] == getValue(A)
    @test typeof(A) == Vector{Adouble{TlAlloc}}

    # test operation: * 

    # ---- Tballoc
    a = Adouble{TbAlloc}(25.0, true)
    c = 3 * a
    @test typeof(c) == Adouble{TbAlloc}
    @test getValue(c) == 75.0

    a = Adouble{TbAlloc}(25.0, true)
    c = true * a
    @test typeof(c) == Adouble{TbAlloc}
    @test getValue(c) == 25.0

    a = Adouble{TbAlloc}(25.0, true)
    c = a * a
    @test typeof(c) == Adouble{TbAlloc}
    @test getValue(c) == 625.0

    # ---- Tlalloc
    a = Adouble{TlAlloc}(25.0, true)
    c = 3 * a
    @test typeof(c) == Adouble{TlAlloc}
    @test getValue(c) == 75.0

    a = Adouble{TlAlloc}(25.0, true)
    c = a * a
    @test typeof(c) == Adouble{TlAlloc}
    @test getValue(c) == 625.0

    a = Adouble{TlAlloc}(25.0, true)
    c = false * a
    @test typeof(c) == Adouble{TlAlloc}
    @test getValue(c) == 0.0

    # test operation: * 
    a = Adouble{TlAlloc}(5.0, true)
    b = Adouble{TlAlloc}(5.0, false)
end
