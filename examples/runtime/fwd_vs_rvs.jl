using BenchmarkTools
using ADOLC
using ADOLC.TbadoubleModule
using ADOLC.array_types
using Plots
using LinearAlgebra
using BandedMatrices
using ForwardDiff
using ReverseDiff


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
        modes = keys(suite[experiment])
        p = Plots.plot()
        for mode in modes
            if mode != "base_time"
                dims_sorted = sort(collect(keys(suite[experiment][mode])))
                mode_vals = [suite[experiment][mode][key] for key in dims_sorted]
                base_times = [suite[experiment]["base_time"][key] for key in dims_sorted]
                Plots.plot!(p, dims_sorted, mode_vals ./ base_times, label="$mode")
            end
        end
        Plots.plot!(p, legend = :topleft)
        xlabel!(p, "dimension")
        ylabel!(p, "time")
        Plots.savefig(p, "fwd_vs_rvs_$experiment.pdf")
    end
end


function run_base_time(experiment, dim)
    if experiment === speelpenning
        x = [(i+1.0)/(2.0+i) for i = 1:dim]
        time = @benchmark speelpenning($x)

    elseif experiment === lin_solve
        x = [(i+1.0)/(2.0+i) for i = 1:dim]
        A = build_banded_matrix(dim)
        time = @benchmark lin_solve($A, $x)

    else throw("$expriment not implemented!")
    end
    time = minimum(time.times)
    return time
end

function run_mode(experiment, mode, dim)
    x = [(i+1.0)/(2.0+i) for i = 1:dim]
    if mode != "ReverseDiff" && mode != "Forward" && mode != "ForwardDiff" && experiment === speelpenning
        a = [Adouble{TbAlloc}() for _ in eachindex(x)]
        b = [Adouble{TbAlloc}()]
        y = 0.0
        trace_on(0)
        a << x
        b = speelpenning(a)
        y = b >> y
        trace_off()

    end
    if mode != "ReverseDiff" && mode != "Forward" && mode != "ForwardDiff" && experiment === lin_solve
        A = build_banded_matrix(dim)
        a = [Adouble{TbAlloc}() for _ in 1:dim]
        b = [Adouble{TbAlloc}() for _ in 1:dim]
        y = Vector{Float64}(undef, dim)

        trace_on(0)
        a << x
        b = lin_solve(A, a)
        y = b >> y
        trace_off()
    end

    if mode == "Forward"
        a = Adouble{TlAlloc}(x)
        if experiment == speelpenning
            time = @benchmark speelpenning($a)        
        else
            A = build_banded_matrix(dim)
            time = @benchmark lin_solve($A, $a)
        end

    elseif mode == "ForwardDiff"
        if experiment == speelpenning
            time = @benchmark ForwardDiff.gradient(speelpenning, $x)
        else
            A = build_banded_matrix(dim)
            time = @benchmark ForwardDiff.jacobian(Base.Fix1(lin_solve, $A), $x)
        end
    
    elseif mode == "ReverseDiff"
        if experiment == speelpenning
            time = @benchmark ReverseDiff.gradient(speelpenning, $x)
        else
            A = build_banded_matrix(dim)
            time = @benchmark ReverseDiff.jacobian(Base.Fix1(lin_solve, $A), $x)
        end  
    elseif mode == "Forward_TB"
        m = length(y)
        x_tangent = myalloc2(dim, dim)
        for i in 1:dim
            for j in 1:dim
                x_tangent[i, j] = 0.0
                if i == j 
                    x_tangent[i, i] = 1.0
                end
            end
        end
        y_tangent = myalloc2(m, dim)

        time = @benchmark fov_forward(
                0,
                $m,
                $dim,
                $dim,
                $x,
                $x_tangent,
                $y,
                $y_tangent,
        )

        myfree2(x_tangent)
        myfree2(y_tangent)

    elseif mode == "Reverse"
        m = length(y)
        weights = myalloc2(m, m)
        for i in 1:m
            for j in 1:m
                weights[i, j] = 0.0
                if i == j 
                    weights[i, i] = 1.0
                end
            end
        end
        jacobian = myalloc2(m, dim)
        zos_forward(0,m, dim, 1, x ,y)
        time = @benchmark fov_reverse(0, $m, $dim, $m, $weights, $jacobian)
        myfree2(weights)
        myfree2(jacobian)

    else throw("$mode is not implemented!")
    end
    time = minimum(time.times)
    return time
end

function runner(max_dim, steps) 
    suite = Dict()
    #modes = ("Forward", "Reverse", "ForwardDiff", "Forward_TB", "ReverseDiff")
    modes = ("Forward", "Reverse")
    experiments = [speelpenning, lin_solve]
    dims = Dict(speelpenning => 100:steps:max_dim, lin_solve => 100:steps:max_dim)

    for experiment in experiments
        println("Running $experiment ...")
        suite[experiment] = Dict()
        suite[experiment]["base_time"] = Dict(dim => run_base_time(experiment, dim) for dim in dims[experiment])
        for mode in modes
            suite[experiment][mode] = Dict()
            for dim in dims[experiment]
                println("Running $mode - $(indexin(dim, dims[experiment])[1])/$(length(dims[experiment])) ... ")
                suite[experiment][mode][dim] = run_mode(experiment, mode, dim)
            end
        end
    end
    plot(suite)
    return suite
end


result = runner(1000, 100)

