using ADOLC
function demo_higher_order()
    function f(x)
        return [x[1]^2 * x[2]^2, x[2]^2]
    end

    x = [1.0, 2.0]

    partials = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [0, 1, 3]]
    seed = [[1.0, 0.0] [0.0, 2.0] [1.0, 1.0]]
    res = Matrix{Float64}(undef, 2, length(partials))
    derivative!(res, f, 2, length(x), x, partials, seed)

    for i in axes(res, 2)
        println("partial $(partials[i]): ", map(round, res[:, i]))
    end
end

demo_higher_order()
