@testset "cxx_ptr_ptr_ptr" begin
    ()
    a = myalloc3(1, 2, 3)
    @test typeof(a[1]) == CxxPtr{CxxPtr{Cdouble}}
    @test typeof(a[Int32(1)]) == CxxPtr{CxxPtr{Cdouble}}
    @test typeof(a) == CxxPtr{CxxPtr{CxxPtr{Cdouble}}}
    @test (a[1, 1, 1] = 1.0) == 1.0
    @test a[1, 1, 1] == 1.0
    @test typeof(a[1, 1, 1]) == Cdouble
    @test (a[Int32(1), 1, 1] = 1.0) == 1.0
    @test a[1, 1, Cshort(1)] == 1.0
    myfree3(a)
end

@testset "cxx_ptr_ptr" begin
    ()
    a = myalloc2(2, 3)
    @test typeof(a) == CxxPtr{CxxPtr{Cdouble}}
    @test (a[1, 1] = 1.0) == 1.0
    @test a[1, 1] == 1.0
    @test typeof(a[1, 1]) == Cdouble
    @test (a[Int32(1), 1] = 1.0) == 1.0
    @test a[1, Cshort(1)] == 1.0
    myfree2(a)
end

@testset "cxx_ptr" begin
    ()
    a = alloc_vec_double(4)
    @test typeof(a) == CxxPtr{Cdouble}
    @test (a[1] = 1.0) == 1.0
    @test typeof(a[1]) == Cdouble
    @test (a[Integer(1)] = 1.0) == 1.0
    @test a[1] == 1.0
    @test a[Int32(1)] == 1.0
    free_vec_double(a)

    a = alloc_vec_short(4)
    @test typeof(a) == CxxPtr{Cshort}
    @test (a[1] = Cshort(1)) == 1.0
    @test typeof(a[1]) == Cshort
    @test (a[Cshort(1)]= Cdouble(1)) == 1.0
    @test a[1] == 1.0
    @test a[Int32(1)] == 1.0
    free_vec_short(a)
end

@testset "CxxTensor" begin
    ()
    a = CxxTensor(1, 2, 3)
    @test typeof(a.data) == CxxPtr{CxxPtr{CxxPtr{Cdouble}}}
    @test a.dim1 == 1
    @test a.dim2 == 2
    @test a.dim3 == 3
    finalize(a)

    a = myalloc3(2, 3, 4)
    b = CxxTensor(a, 2, 3, 4)
    @test typeof(b.data) == CxxPtr{CxxPtr{CxxPtr{Cdouble}}}
    @test b.data === a
    @test b.dim1 == 2
    @test b.dim2 == 3
    @test b.dim3 == 4

    a = zeros(Cdouble, 2, 3, 4)
    a[1, 1, 3] = -1.0
    b = CxxTensor(a)
    @test b.dim1 == 2
    @test b.dim2 == 3
    @test b.dim3 == 4
    @test all(a .== b)
    @test size(b) == (2, 3, 4)
    @test axes(b) == (Base.OneTo(2), Base.OneTo(3), Base.OneTo(4))
    @test axes(b, 1) == Base.OneTo(2)
    @test axes(b, 2) == Base.OneTo(3)
    @test axes(b, 3) == Base.OneTo(4)
    @test axes(b, 4) == Base.OneTo(1)
    @test b[1, 1, Int32(3)] == -1.0
    @test typeof(b[1, 1, 3]) == Cdouble
    @test (b[1, 2, Int32(3)] = -5.0) == -5.0
    @test b[1, 2, 3] == -5.0
end

@testset "CxxMatrix" begin
    ()
    a = CxxMatrix(2, 3)
    @test typeof(a.data) == CxxPtr{CxxPtr{Cdouble}}
    @test a.dim1 == 2
    @test a.dim2 == 3
    finalize(a)

    a = myalloc2(3, 4)
    b = CxxMatrix(a, 3, 4)
    @test typeof(b.data) == CxxPtr{CxxPtr{Cdouble}}
    @test b.data === a
    @test b.dim1 == 3
    @test b.dim2 == 4

    a = zeros(Cdouble, 3, 4)
    a[1, 3] = -1.0
    b = CxxMatrix(a)
    @test b.dim1 == 3
    @test b.dim2 == 4
    @test all(a .== b)
    @test size(b) == (3, 4)
    @test axes(b) == (Base.OneTo(3), Base.OneTo(4))
    @test axes(b, 1) == Base.OneTo(3)
    @test axes(b, 2) == Base.OneTo(4)
    @test axes(b, 4) == Base.OneTo(1)
    @test b[1, Int32(3)] == -1.0
    @test typeof(b[1, 3]) == Cdouble
    @test (b[2, Int32(3)] = -5.0) == -5.0
    @test b[2, 3] == -5.0
end

@testset "CxxVector" begin
    ()
    a = CxxVector(3)
    @test typeof(a.data) == CxxPtr{Cdouble}
    @test a.dim == 3
    finalize(a)

    a = alloc_vec_double(4)
    b = CxxVector(a, 4)
    @test typeof(b.data) == CxxPtr{Cdouble}
    @test b.data === a
    @test b.dim == 4

    a = zeros(Cdouble, 4)
    a[3] = -1.0
    b = CxxVector(a)
    @test b.dim == 4
    @test all(a .== b)
    @test size(b) == (4,)
    @test axes(b) == (Base.OneTo(4),)
    @test axes(b, 1) == Base.OneTo(4)
    @test axes(b, 4) == Base.OneTo(1)
    @test b[Int32(3)] == -1.0
    @test typeof(b[3]) == Cdouble
    @test (b[Int32(2)] = -5.0) == -5.0
    @test b[2] == -5.0
end

@testset "jl_to_cxx" begin
    ()
    a = [[1.0, 2.0, 3.0] [4.0, 5.5, 7.7]]
    b = CxxMatrix(size(a)...)

    jl_res_to_cxx_res!(b, a)
    @test all(a .== b)
    @test_throws DimensionMismatch jl_res_to_cxx_res!(CxxMatrix(1, 1), a)

    a = [1.0, 2.0, 3.0]
    b = CxxVector(size(a)...)

    jl_res_to_cxx_res!(b, a)
    @test all(a .== b)
    @test_throws DimensionMismatch jl_res_to_cxx_res!(CxxVector(1), a)

    a = ones(Cdouble, 3, 3, 4)
    b = CxxTensor(size(a)...)

    jl_res_to_cxx_res!(b, a)
    @test all(a .== b)
    @test_throws DimensionMismatch jl_res_to_cxx_res!(CxxTensor(1, 4, 5), a)

    a = [[1.0, 2.0, 3.0] [4.0, 5.5, 7.7]]
    b = jl_res_to_cxx_res(a)
    @test all(a .== b)

    a = [1.0, 2.0, 3.0]
    b = jl_res_to_cxx_res(a)
    @test all(a .== b)

    a = ones(Cdouble, 3, 3, 4)
    b = jl_res_to_cxx_res(a)
    @test all(a .== b)
end

@testset "cxx_to_jl" begin
    ()
    a = [[1.0, 2.0, 3.0] [4.0, 5.5, 7.7]]
    b = CxxMatrix(a)
    c = Matrix{Cdouble}(undef, size(a)...)

    cxx_res_to_jl_res!(c, b)
    @test all(c .== b)
    @test_throws DimensionMismatch cxx_res_to_jl_res!(Matrix{Cdouble}(undef, 1, 1), b)

    a = [1.0, 2.0, 3.0]
    b = CxxVector(a)
    c = Vector{Cdouble}(undef, size(a)...)
    cxx_res_to_jl_res!(c, b)
    @test all(c .== b)
    @test_throws DimensionMismatch cxx_res_to_jl_res!(Vector{Cdouble}(undef, 1), b)

    a = ones(Cdouble, 3, 3, 4)
    b = CxxTensor(a)
    c = Array{Cdouble,3}(undef, size(a)...)
    cxx_res_to_jl_res!(c, b)
    @test all(c .== b)
    @test_throws DimensionMismatch cxx_res_to_jl_res!(Array{Cdouble}(undef, 2, 6, 4), b)

    a = [[1.0, 2.0, 3.0] [4.0, 5.5, 7.7]]
    b = CxxMatrix(a)
    c = cxx_res_to_jl_res(b)
    @test all(c .== b)

    a = [1.0, 2.0, 3.0]
    b = CxxVector(a)
    c = cxx_res_to_jl_res(b)
    @test all(c .== b)

    a = ones(Cdouble, 3, 3, 4)
    b = CxxTensor(a)
    c = cxx_res_to_jl_res(b)
    @test all(c .== b)
end
