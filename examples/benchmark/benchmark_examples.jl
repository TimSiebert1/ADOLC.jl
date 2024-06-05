module BenchmarkExamples

using BandedMatrices
function speelpenning(x)
    for x_i in x[2:end]
        x[1] *= x_i
    end
    return x[1]
end

function build_banded_matrix(dim)
    h = 1 / dim
    A = BandedMatrix{Float64}(undef, (dim, dim), (1, 1))
    A[band(0)] .= -2 / h^2
    A[band(1)] .= A[band(-1)] .= 1 / h^2
    return Matrix(A)
end

function lin_solve(A, x)
    return A \ x
end
end
