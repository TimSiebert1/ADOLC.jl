module BenchmarkExamples

using BandedMatrices
function speelpenning(x)
    y = x[1]
    for x_i in x[2:end]
        y *= x_i
    end
    return y
end

function build_banded_matrix(dim)
    h = 1 / dim
    A = BandedMatrix{Float64}(undef, (dim, dim), (1, 1))
    A[band(0)] .= -2 / h^2
    A[band(1)] .= A[band(-1)] .= 1 / h^2
    return A
end

function lin_solve(A)
    return x -> A \ x
end

function rosenbrock(x)
    a = one(eltype(x))
    b = 100 * a
    result = zero(eltype(x))
    for i in 1:(length(x) - 1)
        result += (a - x[i])^2 + b * (x[i + 1] - x[i]^2)^2
    end
    return result
end
end
