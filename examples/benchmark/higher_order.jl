using BenchmarkTools
using ADOLC
using ADOLC.TbadoubleModule
using ADOLC.array_types
using Plots
using Measures
include("benchmark_examples.jl")




function plot_fixed_dim_p(suite::Dict, fixed_dim, fixed_ps)
    for experiment in keys(suite)
        base_time = suite[experiment]["base_time"][fixed_dim]
        p = Plots.plot()
        for fixed_p in fixed_ps
            derviative_orders_sorted = sort(collect(keys(suite[experiment][fixed_dim])))
            time_vals =
                [suite[experiment][fixed_dim][d][fixed_p] for d in derviative_orders_sorted]
            Plots.plot!(
                p,
                derviative_orders_sorted,
                time_vals / base_time,
                label = "n=$fixed_dim, p=$fixed_p",
            )
        end
        Plots.plot!(p, legend = :topleft, yformatter = :scientific)
        xlabel!(p, "derivative order")
        ylabel!(p, "runtime-ratio")
        Plots.savefig(p, "higher_order_$(experiment)_dim=$(fixed_dim)_I.pdf")
    end
end


function plot_uni_vs_tensor_eval_dir(suite, fixed_dim, fixed_d, directions)
    for experiment in keys(suite)
        base_time = suite[experiment]["univariate"][fixed_dim][fixed_d]
        binomis = [binomial(p + fixed_d - 1, fixed_d) for p in directions]
        p = Plots.plot()
        time_vals =
                [suite[experiment][fixed_dim][fixed_d][p] for p in directions]
        Plots.plot!(
            p,
            directions,
            time_vals ./ (binomis * base_time),
            label = "dim=$fixed_dim, d=$fixed_d",
        )
        Plots.plot!(p, legend = :topleft, yformatter = :scientific)
        xlabel!(p, "num directions")
        ylabel!(p, "runtime-ratio")
        Plots.savefig(p, "higher_order_$(experiment)_uni_vs_inter_dir.pdf")
    end
end

function plot_uni_vs_tensor_eval_deriv(suite, fixed_dim, derivative_orders, fixed_p)
    for experiment in keys(suite)
        base_times = [suite[experiment]["univariate"][fixed_dim][d] for d in derivative_orders]
        binomis = [binomial(fixed_p + d - 1, d) for d in derivative_orders]
        time_vals =
            [suite[experiment][fixed_dim][d][fixed_p] for d in derivative_orders]
        p = Plots.plot()
        Plots.plot!(
            p,
            derivative_orders,
            time_vals ./ (binomis .* base_times),
            label = "dim=$fixed_dim, p=$fixed_p",
        )
        Plots.plot!(p, legend = :topleft, yformatter = :scientific)
        xlabel!(p, "derivative order")
        ylabel!(p, "runtime-ratio")
        Plots.savefig(p, "higher_order_$(experiment)_uni_vs_inter_deriv.pdf")
    end
end

function plot_fixed_dim_d(suite::Dict, fixed_dims, fixed_ds, directions)
    for experiment in keys(suite)
        plots = []
        for fixed_dim in fixed_dims
            for fixed_d in fixed_ds
                p = Plots.plot()
                base_time = suite[experiment]["base_time"][fixed_dim]
                time_vals =
                    [suite[experiment][fixed_dim][fixed_d][p] for p in directions]
                Plots.plot!(
                    p,
                    directions,
                    time_vals / base_time,
                    label = "dim=$fixed_dim, d=$fixed_d",
                    legendfontsize = 5,
                )
                Plots.plot!(
                    p,
                    yformatter = :scientific,
                    guidefontsize = 5,
                    xtickfontsize = 5,
                    ytickfontsize = 5,
                )
                xlabel!(p, "num directions")
                ylabel!(p, "runtime-ratio")
                push!(plots, p)
            end
        end
        p = Plots.plot(plots..., layout = (length(fixed_dims), length(fixed_ds)))
        Plots.savefig(p, "higher_order_$(experiment)_multi_plot.pdf")
    end
end


function print_result(suite, fixed_dim, fixed_d, directions,  derivative_orders, fixed_p)
    for experiment in keys(suite)
        println("experiment=$experiment, dim=$fixed_dim, d=$fixed_d:")
        base_time = suite[experiment]["univariate"][fixed_dim][fixed_d]
        binomis = [binomial(p + fixed_d - 1, fixed_d) for p in directions]
        time_vals =
                [suite[experiment][fixed_dim][fixed_d][p] for p in directions]
        println("directions: ", directions)
        println("ratio values ", time_vals ./ (binomis * base_time))

    end

    
    for experiment in keys(suite)
        println("experiment=$experiment, dim=$fixed_dim, p=$fixed_p:")
        base_times = [suite[experiment]["univariate"][fixed_dim][d] for d in derivative_orders]
        binomis = [binomial(fixed_p + d - 1, d) for d in derivative_orders]
        time_vals =
            [suite[experiment][fixed_dim][d][fixed_p] for d in derivative_orders]
        println("derivative_orders: ", derivative_orders)
        println("ratio values ", time_vals ./ (binomis .* base_times))
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

function run_mode(experiment, dim, derivative_order, directions; mode=:tensor_eval)
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

    if mode === :univariate
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

    elseif mode === :tensor_eval
        m = length(y)

        # allocate c++ memory for the tensor_eval function
        CxxTensor =
            ADOLC.myalloc2(m, binomial(directions + derivative_order, derivative_order))

        seed = ADOLC.create_cxx_identity(dim, directions)

        time = @benchmark tensor_eval(
            0,
            $m,
            $dim,
            $derivative_order,
            $directions,
            $x,
            $CxxTensor,
            $seed,
        )
        #build_tensor(derivative_order, num_dependents, num_independents, CxxTensor)
        myfree2(seed)
        myfree2(CxxTensor)
    end
    time = median(time.times)
    return time
end

function runner(dims, derivative_orders, directions)
    suite = Dict()
    experiments = [BenchmarkExamples.speelpenning, BenchmarkExamples.lin_solve]
    for experiment in experiments
        println("Running $experiment ...")
        suite[experiment] = Dict()
        suite[experiment]["base_time"] =
            Dict(dim => run_base_time(experiment, dim) for dim in dims)
        suite[experiment]["univariate"] = Dict()
        for dim in dims
            suite[experiment][dim] = Dict()
            suite[experiment]["univariate"][dim] = Dict()
            for d in derivative_orders
                suite[experiment][dim][d] = Dict()
                suite[experiment]["univariate"][dim][d] = run_mode(experiment, dim, d, 0, mode=:univariate)
                for p in directions
                    println("""
                    Running dim: $(indexin(dim, dims)[1])/$(length(dims)) 
                    d: $(indexin(d, derivative_orders)[1])/$(length(derivative_orders)) 
                    p: $(indexin(p, directions)[1])/$(length(directions)) ... 
                    """)
                    suite[experiment][dim][d][p] = run_mode(experiment, dim, d, p)
                end
            end
        end
    end

    plot_fixed_dim_p(suite, 10, [1, 5, 10])
    plot_fixed_dim_d(suite, [10, 30, 50], [1, 5, 10], directions)
    plot_uni_vs_tensor_eval_dir(suite, 10, 5, directions)
    plot_uni_vs_tensor_eval_deriv(suite, 10, derivative_orders, 5)
    print_result(suite, 10, 5, directions, derivative_orders, 5)
    return suite
end



result = runner([10, 30, 50], 1:1:10, 1:1:10)
