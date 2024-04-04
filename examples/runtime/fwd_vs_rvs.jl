using BenchmarkTools
using ADOLC.TbadoubleModule
using ADOLC.array_types
using Plots


function setup_finalizer_fw(su)
    myfree2(su.derivative_init)
    myfree2(su.derivative_vec)
end

function setup_finalizer_rv(su)
    su.derivative_init = nothing
    su.derivative_vec = nothing
end
mutable struct SetUp
    n
    m
    x
    y
    a
    b
    derivative_init
    derivative_vec
    function SetUp(n, m, x, y, a,  b, derivative_init, derivative_vec, mode)
        x = new(n, m, x, y, a, b, derivative_init, derivative_vec)
        if mode == "forward"
            finalizer(setup_finalizer_fw, x)
        elseif mode == "reverse"
            finalizer(setup_finalizer_rv, x)
        end
    end

end

function speelpenning(x::Vector{Float64})
    y = 1.0
    for i in eachindex(x)
        y *= x[i]
    end 
    return y
end

function speelpenning(x)
    y = TbadoubleCxx(1.0)
    for i in eachindex(x)
        y *= x[i]
    end
    return y
end
function speelpenning_vec(x)
    y = [TbadoubleCxx(x[i]) for i in eachindex(x)]
    y[1] = TbadoubleCxx(1.0)
    for i in eachindex(x)
        y[1] *= x[i]
    end
    return y
end

function plot(base_group::BenchmarkTools.BenchmarkGroup, fwd_group::BenchmarkTools.BenchmarkGroup, rvs_group::BenchmarkTools.BenchmarkGroup)
    
    sorted_keys = sort!((collect(Int64, keys(base_group))))
    base_times = [base_group[key].time for key in sorted_keys]
    
    sorted_keys = sort!((collect(Int64, keys(fwd_group))))
    fwd_times = [fwd_group[key].time for key in sorted_keys]
    
    sorted_keys = sort!((collect(Int64, keys(rvs_group))))
    rvs_times = [rvs_group[key].time for key in sorted_keys]

    labels = ["forward" "reverse"]
    Plots.plot(sorted_keys, [fwd_times ./ base_times, rvs_times ./ base_times], legend = :topleft, titlefontsize=10, xformatter=:none, yformatter=:none, labels=labels)
    #Plots.plot(sorted_keys, [fwd_times, rvs_times], legend = :topleft, titlefontsize=10, xformatter=:none, yformatter=:none, labels=labels)
    xlabel!("num independant")
    ylabel!("time")
    title!("Forward vs. Reverse for Speelpenning with increasing dimension")
end


function setup_tape(func, n::Int64, mode)

    x = [(i+1.0)/(2.0+i) for i = 1:n]
    a = [TbadoubleCxx() for _ in eachindex(x)]
    b = TbadoubleCxx()
    y = 0.0

    if mode == "reverse"
        trace_on(0, 1)
    else
        trace_on(0)
    end

    a << x
    b = func(a)
    y = b >> y
    trace_off(1)

    if mode == "forward"
        x_tangent = myalloc2(length(x), length(x))
        for i in 1:n
            for j in 1:n
                x_tangent[i, j] = 0.0
                if i == j 
                    x_tangent[i, j] = 1.0
                end
            end
        end

        y_tangent = myalloc2(length(y), length(x))
        return SetUp(n, length(y), x, y, a, b, x_tangent, y_tangent, mode)

    elseif mode == "reverse"
        weight_vector = [1.0]
        derivative_vec = Vector{Float64}(undef, length(x))
        return SetUp(n, length(y), x, y, a, b, weight_vector, derivative_vec, mode)

    else throw("$mode is not implemented!")
    end
end

function diff(su::SetUp, mode)
    if mode == "forward"
        fov_forward(
            0,
            su.m,
            su.n,
            0,
            su.x,
            su.derivative_init,
            su.y,
            su.derivative_vec,
        )
        return su.derivative_vec

    elseif mode == "reverse"
        fos_reverse(0, su.m, su.n, su.derivative_init, su.derivative_vec)
        return su.derivative_vec
    else
        throw("$mode not implemented!")
    end

end

function runner(max_dim, steps)
    suite = BenchmarkGroup()
    suite["runtime_f"] = BenchmarkGroup()
    modes = ("forward", "reverse")
    experiments = [speelpenning]
    dims = 1000:steps:max_dim

    for mode in modes
        suite[mode] => BenchmarkGroup()
        for experiment in experiments
            suite[mode][experiment] => BenchmarkGroup()
            for dim in dims
                suite["runtime_f"][dim] = @benchmarkable speelpenning(x) setup=(x = [(i+1.0)/(2.0+i) for i = 1:$dim])
                suite[mode][experiment][dim] = @benchmarkable diff(su, $mode) setup=(su = setup_tape($experiment, $dim, $mode))
            end
        end
    end

    tune!(suite)
    results = BenchmarkTools.run(suite, verbose = true)
    plot(mean(results["runtime_f"]) ,mean(results["forward"]["speelpenning"]), mean(results["reverse"]["speelpenning"]))
end



function runner2()
    su = setup_tape(speelpenning, 10, "reverse")
    fos_reverse(0, su.m, su.n, su.derivative_init, su.derivative_vec)
    return su
end