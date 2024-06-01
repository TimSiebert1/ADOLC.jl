using BenchmarkTools
using ADOLC
using ADOLC.TbadoubleModule
using ADOLC.array_types
using Plots
using Measures
include("benchmark_examples.jl")



function plot_dims_fixed_d(suite::Dict, dims, fixed_ds)
    for experiment in keys(suite)
        p = Plots.plot()
        for fixed_d in fixed_ds
            base_times = [suite[experiment]["base_time"][dim] for dim in dims]
            time_vals = [suite[experiment][dim][fixed_d] for dim in dims]
            Plots.plot!(
                p,
                dims,
                time_vals ./ base_times,
                label = "d=$fixed_d",
            )
            Plots.plot!(p, legend = :topleft, yformatter = :scientific)
        end
        xlabel!(p, "number of dimensions")
        ylabel!(p, "runtime-ratio")
        Plots.savefig(p, "uni_tpp_$(experiment).pdf")
    end
end

function plot_d_fixed_dim(suite::Dict, fixed_dims, derivative_orders)
    for experiment in keys(suite)
        p = Plots.plot()
        for dim in fixed_dims
            base_time = suite[experiment]["base_time"][dim]
            time_vals = [suite[experiment][dim][d] for d in derivative_orders]
            Plots.plot!(
                p,
                derivative_orders,
                time_vals / base_time,
                label = "dim=$dim",
            )
            Plots.plot!(p, legend = :topleft, yformatter = :scientific)
        end
        xlabel!(p, "derivative order")
        ylabel!(p, "runtime-ratio")
        Plots.savefig(p, "uni_tpp_$(experiment)_2.pdf")
    end
end


function run_base_time(experiment, dim)
    if experiment === BenchmarkExamples.speelpenning
        x = [(i + 1.0) / (2.0 + i) for i = 1:dim]
        time = @benchmark BenchmarkExamples.speelpenning($x)

    elseif experiment === BenchmarkExamples.lin_solve
        x = [(i + 1.0) / (2.0 + i) for i = 1:dim]
        A = BenchmarkExamples.build_banded_matrix(dim)
        time = @benchmark BenchmarkExamples.lin_solve($A, $x)

    else
        throw("$expriment not implemented!")
    end
    time = median(time.times)
    return time
end

function run_mode(experiment, dim, derivative_order)
    x = [(i + 1.0) / (2.0 + i) for i = 1:dim]
    if experiment === BenchmarkExamples.speelpenning
        a = [Adouble{TbAlloc}() for _ in eachindex(x)]
        b = [Adouble{TbAlloc}()]
        y = 0.0
        trace_on(0)
        a << x
        b = BenchmarkExamples.speelpenning(a)
        y = b >> y
        trace_off()

    end
    if experiment === BenchmarkExamples.lin_solve
        A = BenchmarkExamples.build_banded_matrix(dim)
        a = [Adouble{TbAlloc}() for _ = 1:dim]
        b = [Adouble{TbAlloc}() for _ = 1:dim]
        y = Vector{Float64}(undef, dim)

        trace_on(0)
        a << x
        b = BenchmarkExamples.lin_solve(A, a)
        y = b >> y
        trace_off()
    end

    m = length(y)
    x_tangent = myalloc2(dim, derivative_order)
    for i = 1:dim
        for j = 1:derivative_order
            x_tangent[i, j] = 0.0
            if j == 1
                x_tangent[i, j] = 1.0
            end
        end
    end
    y_tangent = myalloc2(m, derivative_order)

    time = @benchmark hos_forward(0, $m, $dim, $derivative_order, $0, $x, $x_tangent, $y, $y_tangent)

    myfree2(y_tangent)
    myfree2(x_tangent)
    time = median(time.times)
    return time
end

function runner(dims, derivative_orders)
    suite = Dict()
    experiments = [BenchmarkExamples.speelpenning, BenchmarkExamples.lin_solve]
    for experiment in experiments
        println("Running $experiment ...")
        suite[experiment] = Dict()
        suite[experiment]["base_time"] =
            Dict(dim => run_base_time(experiment, dim) for dim in dims)
        for dim in dims
            suite[experiment][dim] = Dict()
            for d in derivative_orders
                println("""
                Running dim: $(indexin(dim, dims)[1])/$(length(dims)) 
                d: $(indexin(d, derivative_orders)[1])/$(length(derivative_orders)) 
                """)
                suite[experiment][dim][d] = run_mode(experiment, dim, d)
            end
        end
    end

    plot_dims_fixed_d(suite, dims, derivative_orders)
    plot_d_fixed_dim(suite, dims, derivative_orders)
    return suite
end


result = runner([100, 200, 300], 1:1:20)

