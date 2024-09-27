using ADOLC
using BenchmarkTools
using ForwardDiff
using ReverseDiff
using Zygote
using DifferentiationInterface
using Tapir: Tapir
using Enzyme: Enzyme
using LinearAlgebra

function bench1()
    function f(x::AbstractVector{T}) where {T}
        y = zero(T)
        for i in eachindex(x)
            y += abs2(x[i])
        end
        return y
    end

    x = collect(1.0:1000.0)
    println(length(x))
    n = length(x)
    tape_id = 1

    out = similar(x)
    cfg10 = ForwardDiff.GradientConfig(f, x, ForwardDiff.Chunk{5}())
    println("benchmarking ForwardDiff...")
    t3 = @benchmark ForwardDiff.gradient!($out, $f, $x, $cfg10)

    res = CxxMatrix(1, n)
    y, m, n = ADOLC.create_tape(f, x, tape_id)

    dir = CxxMatrix(create_cxx_identity(n, n), n, n)
    println("Benchmarking ADOLC Forward...")
    t1 = @benchmark ADOLC.fov_forward!($res, $tape_id, $m, $n, $x, $dir)

    res = CxxVector(n)
    println("Benchmarking ADOLC Reverse ...")
    tmp_out = Vector{Cdouble}(undef, m)

    t2 = @benchmark zos_rev($tmp_out, $tape_id, $m, $n, $x, $res)

    f_tape = ReverseDiff.GradientTape(f, x)
    compiled_f_tape = ReverseDiff.compile(f_tape)
    inputs = x
    results = similar(x)
    println("benchmarking ReverseDiff...")
    t4 = @benchmark ReverseDiff.gradient!($results, $compiled_f_tape, $inputs)

    println("benchmarking Zygote...")
    t5 = @benchmark Zygote.gradient($f, $x)

    backend = AutoTapir()
    extras = prepare_gradient(f, backend, x)
    println("benchmarking Tapir...")
    t6 = @benchmark DifferentiationInterface.gradient($f, $backend, $x, $extras)

    #println("benchmarking Enzyme...")
    #backend = AutoEnzyme()
    #extras = prepare_gradient(f, backend, x)
    #t7 = @benchmark DifferentiationInterface.gradient($f, $backend, $x, $extras)

    return t1, t2, t3, t4, t5, t6#, t7
end

function zos_rev(tmp_out, tape_id, m, n, x, res)
    ADOLC.zos_forward!(tmp_out, tape_id, m, n, x)
    return ADOLC.fos_reverse!(res, tape_id, m, n, [1.0])
end
function bench_rosenbrock()
    n = 1000
    x = rand(n)
    function rosenbrock(x)
        a = one(eltype(x))
        b = 100 * a
        result = zero(eltype(x))
        for i in 1:(length(x) - 1)
            result += (a - x[i])^2 + b * (x[i + 1] - x[i]^2)^2
        end
        return result
    end
    out = similar(x)
    cfg10 = ForwardDiff.GradientConfig(rosenbrock, x, ForwardDiff.Chunk{10}())
    println("benchmarking ForwardDiff...")
    t1 = @benchmark ForwardDiff.gradient!($out, $rosenbrock, $x, $cfg10)
    grad = out
    tape_id = 1

    res = CxxMatrix(1, n)
    m = 1
    tape_id = 1
    y, m, n = ADOLC.create_tape(rosenbrock, x, tape_id)
    dir = CxxMatrix(create_cxx_identity(n, n), n, n)
    println("Benchmarking ADOLC Forward...")
    t2 = @benchmark ADOLC.fov_forward!($res, $tape_id, $m, $n, $x, $dir)
    @assert(all([isapprox(res[i], grad[i]; atol=1e-10) for i in 1:n]))

    res = CxxVector(n)
    tmp_out = Vector{Cdouble}(undef, m)

    println("Benchmarking ADOLC Reverse ...")
    t3 = @benchmark zos_rev($tmp_out, $tape_id, $m, $n, $x, $res)

    @assert(all([isapprox(res[i], grad[i]; atol=1e-10) for i in 1:n]))

    f_tape = ReverseDiff.GradientTape(rosenbrock, x)
    compiled_f_tape = ReverseDiff.compile(f_tape)
    inputs = x
    results = similar(x)
    println("benchmarking ReverseDiff...")
    t4 = @benchmark ReverseDiff.gradient!($results, $compiled_f_tape, $inputs)
    @assert(all(isapprox(results, grad; atol=1e-10)))

    println("benchmarking Zygote...")
    t5 = @benchmark Zygote.gradient($rosenbrock, $x)

    backend = AutoTapir()
    extras = prepare_gradient(rosenbrock, backend, x)
    println("benchmarking Tapir...")
    t6 = @benchmark DifferentiationInterface.gradient($rosenbrock, $backend, $x, $extras)

    """
    println("benchmarking Enzyme...")
    backend = AutoEnzyme()
    extras = prepare_gradient(rosenbrock, backend, x)
    t8 = @benchmark DifferentiationInterface.gradient($rosenbrock, $backend, $x, $extras)
    """
    return t1, t2, t3, t4, t5, t6
end

function bench_rosenbrock2()
    n = 1000
    x = rand(n)
    function rosenbrock(x)
        a = one(eltype(x))
        b = 100 * a
        result = zero(eltype(x))
        for i in 1:(length(x) - 1)
            result += (a - x[i])^2 + b * (x[i + 1] - x[i]^2)^2
        end
        return result * x
    end
    out = Matrix{Cdouble}(undef, n, n)
    cfg10 = ForwardDiff.JacobianConfig(rosenbrock, x, ForwardDiff.Chunk{10}())
    println("benchmarking ForwardDiff...")
    t1 = @benchmark ForwardDiff.jacobian!($out, $rosenbrock, $x, $cfg10)

    tape_id = 1
    res = CxxMatrix(n, n)
    y, m, n = ADOLC.create_tape(rosenbrock, x, tape_id)
    cxx_weights = CxxMatrix(create_cxx_identity(m, m), m, m)

    println("benchmarking ADOLC...")
    t2 = @benchmark ADOLC.fov_reverse!($res, $tape_id, $m, $n, $x, $cxx_weights)

    f_tape = ReverseDiff.JacobianTape(rosenbrock, x)
    compiled_f_tape = ReverseDiff.compile(f_tape)
    inputs = x
    results = Matrix{Float64}(undef, n, n)
    println("benchmarking ReverseDiff...")
    t3 = @benchmark ReverseDiff.jacobian!($results, $compiled_f_tape, $inputs)

    println("benchmarking Zygote...")
    t4 = @benchmark Zygote.jacobian($rosenbrock, $x)

    backend = AutoTapir()
    extras = prepare_jacobian(rosenbrock, backend, x)
    println("benchmarking Tapir...")
    t5 = @benchmark DifferentiationInterface.jacobian($rosenbrock, $backend, $x, $extras)

    println("benchmarking Enzyme...")
    backend = AutoEnzyme()
    extras = prepare_jacobian(rosenbrock, backend, x)
    t6 = @benchmark DifferentiationInterface.jacobian($rosenbrock, $backend, $x, $extras)
    return t1, t2, t3, t4, t5, t6
end

include("examples.jl")
function bench_forward()
    n = 1500
    x = rand(n)
    A = rand(n, n)
    f(b) = A \ b

    out = Matrix{Cdouble}(undef, n, n)
    cfg10 = ForwardDiff.JacobianConfig(f, x, ForwardDiff.Chunk{10}())
    println("Benchmarking ForwardDiff...")
    t1 = @benchmark ForwardDiff.jacobian!($out, $f, $x, $cfg10)

    tape_id = 1
    ADOLC.create_tape(f, x, tape_id)
    res = CxxMatrix(n, n)
    dir = CxxMatrix(create_cxx_identity(n, n), n, n)
    println("Benchmarking ADOLC Forward...")
    t2 = @benchmark ADOLC.fov_forward!($res, $tape_id, $n, $n, $x, $dir)

    tape_id = 1
    res = CxxMatrix(n, n)
    y, m, n = ADOLC.create_tape(f, x, tape_id)
    cxx_weights = CxxMatrix(create_cxx_identity(n, n), n, n)
    tmp_out = Vector{Cdouble}(undef, n)
    println("benchmarking ADOLC Reverse...")
    t3 = @benchmark zos_rev($tmp_out, $tape_id, $m, $n, $x, $cxx_weights)

    #f_tape = ReverseDiff.JacobianTape(f, x)
    #compiled_f_tape = ReverseDiff.compile(f_tape)
    #inputs = x
    #results = Matrix{Float64}(undef, n, n)
    #println("benchmarking ReverseDiff...")
    #t4 = @benchmark ReverseDiff.jacobian!($results, $compiled_f_tape, $inputs)

    #println("benchmarking Zygote...")
    #t3 = @benchmark Zygote.jacobian($f, $x)

    #backend = AutoTapir()
    #extras = prepare_jacobian(f, backend, x)
    #println("benchmarking Tapir...")
    #t6 = @benchmark DifferentiationInterface.jacobian($f, $backend, $x, $extras)

    #println("benchmarking Enzyme...")
    #backend = AutoEnzyme()
    #extras = prepare_jacobian(f, backend, x)
    #t7 = @benchmark DifferentiationInterface.jacobian($f, $backend, $x, $extras)

    return t1, t2, t3
end
