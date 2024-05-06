module sppk

using Test
using LinearAlgebra
using Sparspak
using SparseArrays
using ForwardDiff
using ADOLC

# This should have the M property
function tridiagonal(p, n)
    T = typeof(p)
    b = T[(p^2 * i + p) for i = 1:n]
    a = T[(-0.1 * p) for i = 1:n-1]
    c = T[(-0.1 * p) for i = 1:n-1]
    Tridiagonal(a, b, c)
end

# Use the tridiagonal solve from Julia
function ftrid(x)
    p = x[1]
    n = 100
    M = tridiagonal(p, n)
    f = ones(n)
    sum(M \ f)
end


dftrid_forwarddiff(x) = ForwardDiff.gradient(ftrid, [x])[1]

function dftrid_adolc(x)
    res = zeros(Float64, length(x))
    derivative!(res, ftrid, 1, length(x), x, :jac)
    res[1]
end

# Sparse version
function fsppk(x)
    p = x[1]
    n = 100
    M = sparse(tridiagonal(p, n))
    f = ones(typeof(p), n)
    sum(sparspaklu(M) \ f)
end

dfsppk_forwarddiff(x) = ForwardDiff.gradient(fsppk, [x])[1]

function dfsppk_adolc(x)
    res = zeros(Float64, length(x))
    derivative!(res, fsppk, 1, length(x), x, :jac)
    res[1]
end

function runtests()
    X = 1:0.1:10
    @test all(x -> (dftrid_forwarddiff(x) ≈ dftrid_adolc(x)), 1:0.1:10)
    @test all(x -> (dftrid_forwarddiff(x) ≈ dfsppk_forwarddiff(x)), 1:0.1:10)
    @test all(x -> (dfsppk_forwarddiff(x) ≈ dfsppk_adolc(x)), 1:0.1:10)
end

end
