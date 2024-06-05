using ADOLC
function demo_higher_order()
    function f(x)
        return [x[1]^2 * x[2], x[2]^2 * x[3]^3]
    end
    x = [1.0, 1.0, 2.0]
    partials = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [0, 1, 3]]
    res = Matrix{Float64}(undef, 2, length(partials))
    derivative!(res, f, 2, 3, x, partials)
    return res
end
res = demo_higher_order()
