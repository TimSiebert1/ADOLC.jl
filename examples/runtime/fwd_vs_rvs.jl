using BenchmarkTools
using ADOLC
using ADOLC.TbadoubleModule
using ADOLC.array_types
using Plots
using LinearAlgebra
using BandedMatrices
using ForwardDiff


function speelpenning(x)
    y = x[1]
    for i in eachindex(x)
        y *= x[i]
    end
    return y
end

function speelpenning(x::Vector{Adouble{T}}) where T<:Union{TbAlloc, TlAlloc}
    y = Adouble{T}(1.0, true)
    for i in eachindex(x)
        y *= x[i]
    end
    return y
end

function build_banded_matrix(dim)
    h = 1/dim
    A = BandedMatrix{Float64}(undef, (dim, dim), (1,1))
    A[band(0)] .= -2/h^2
    A[band(1)] .= A[band(-1)] .= 1/h^2
    return Matrix(A)
end

function lin_solve(A, x)
    return A \ x
end



function plot(suite::Dict)

    
    experiments = keys(suite)
    for experiment in experiments
        base_times = values(suite[experiment]["base_time"])
        modes = keys(suite[experiment])
        p = Plots.plot()
        for mode in modes
            if mode != "base_time"
                dims_sorted = sort(collect(keys(suite[experiment][mode])))
                mode_vals = [suite[experiment][mode][key] for key in dims_sorted]
                #Plots.plot!(p, dims_sorted, mode_vals ./ base_times, label="$mode")
                Plots.plot!(p, dims_sorted, mode_vals, label="$mode")
            end
        end
        Plots.plot!(p, legend = :topleft, titlefontsize=10, xformatter=:none, yformatter=:none)
        title!(p, "Forward vs. Reverse for $experiment with increasing dimension")
        xlabel!(p, "num independant")
        ylabel!(p, "time")
        Plots.savefig(p, "$experiment.pdf")
    end
    
end


function run_base_time(experiment, dim, num_iter)

    if experiment === speelpenning
        x = [(i+1.0)/(2.0+i) for i = 1:dim]
        experiment(x)
        t1 = time()
        for _ in 1:num_iter
            experiment(x)
        end
        t2 = time()
        return (t2 - t1) / num_iter * 10^6

    elseif experiment === lin_solve
        x = [(i+1.0)/(2.0+i) for i = 1:dim]
        A = build_banded_matrix(dim)
        experiment(A, x)
        t1 = time()
        for _ in 1:num_iter
            experiment(A, x)
        end
        t2 = time()
        return (t2 - t1) / num_iter * 10^6
    else throw("$expriment not implemented!")
    end
end

function run_mode(experiment, mode, dim, num_iter)

    if experiment === speelpenning
        x = [(i+1.0)/(2.0+i) for i = 1:dim]
        a = [Adouble{TbAlloc}() for _ in eachindex(x)]
        b = [Adouble{TbAlloc}()]
        y = 0.0

        trace_on(0)

        a << x
        b = experiment(a)
        y = b >> y
        trace_off()


    elseif experiment === lin_solve
        x = [(i+1.0)/(2.0+i) for i = 1:dim]
        A = build_banded_matrix(dim)
        a = [Adouble{TbAlloc}() for _ in 1:dim]
        b = [Adouble{TbAlloc}() for _ in 1:dim]
        y = Vector{Float64}(undef, dim)

        trace_on(0)
        a << x
        b = experiment(A, a)
        y = b >> y
        trace_off()
    else throw("$experiment is not implemented!")
    end

    if mode == "forward"
        a = Adouble{TlAlloc}(x)
        if experiment == speelpenning
            t1 = time()
            for _ in 1:num_iter
                b = experiment(a)
            end
            t2 = time() 
             
        else
            t1 = time()
            for _ in 1:num_iter
                b = experiment(A, a)
            end
            t2 = time()  
        end
        return (t2 - t1) / num_iter * 10^6
    elseif mode == "forwarddiff"
        if experiment == speelpenning
            t1 = time()
            for _ in 1:num_iter
                ForwardDiff.gradient(speelpenning, x)
            end
            t2 = time()
        else
            throw("Frowarddiff not implement for $experiment")
        end
    elseif mode == "forward_tb"
        x_tangent = myalloc2(length(x), length(x))
        for i in 1:dim
            for j in 1:dim
                x_tangent[i, j] = 0.0
                if i == j 
                    x_tangent[i, j] = 1.0
                end
            end
        end
        y_tangent = myalloc2(length(y), length(x))
        t1 = time()
        for _ in 1:num_iter
            fov_forward(
                0,
                length(y),
                dim,
                0,
                x,
                x_tangent,
                y,
                y_tangent,
        )
        end
        t2 = time()
        myfree2(x_tangent)
        myfree2(y_tangent)
        return (t2 - t1) / num_iter * 10^6

    elseif mode == "reverse"
        weights = myalloc2(length(y), length(y))
        jacobian = myalloc2(length(y), dim)
        zos_forward(0,length(y), dim, 1, x ,y)
        t1 = time()
        for _ in 1:num_iter
            fov_reverse(0, length(y), dim, length(y), weights, jacobian)
        end
        t2 = time()
        myfree2(weights)
        myfree2(jacobian)
        return (t2 - t1) / num_iter * 10^6

    else throw("$mode is not implemented!")
    end
end

function runner(max_dim, steps) 
    suite = Dict()
    modes = ("forward", "reverse", "forwarddiff")
    #experiments = [speelpenning, lin_solve]
    experiments = [speelpenning]
    dims = Dict(speelpenning => 100:steps:max_dim, lin_solve => 1:50:300)

    for experiment in experiments
        println("Running $experiment ...")
        suite[experiment] = Dict()
        suite[experiment]["base_time"] = Dict()
        for mode in modes
            suite[experiment][mode] = Dict()
            for dim in dims[experiment]
                println("Running $mode - $(indexin(dim, dims[experiment])[1])/$(length(dims[experiment])) ... ")
                suite[experiment]["base_time"][dim] = run_base_time(experiment, dim, 1000)
                suite[experiment][mode][dim] = run_mode(experiment, mode, dim , 1000)
            end
        end
    end
    plot(suite)
    return suite
end


runner(1000, 100)

