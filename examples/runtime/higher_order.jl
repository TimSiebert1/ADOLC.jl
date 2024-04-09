using BenchmarkTools
using ADOLC
using ADOLC.TbadoubleModule
using ADOLC.array_types
using Plots
using LinearAlgebra
using BandedMatrices
using Measures


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



function plot_fixed_dim_p(suite::Dict, fixed_dim, fixed_ps)
    for experiment in keys(suite)
        base_time = suite[experiment]["base_time"][fixed_dim]
        p = Plots.plot()
        for fixed_p in fixed_ps
            derviative_orders_sorted = sort(collect(keys(suite[experiment][fixed_dim])))
            time_vals = [suite[experiment][fixed_dim][d][fixed_p] for d in derviative_orders_sorted]
            Plots.plot!(p, derviative_orders_sorted, time_vals / base_time, label="n=$fixed_dim, p=$fixed_p")
        end
        Plots.plot!(p, legend = :topleft, yformatter=:scientific)
        xlabel!(p, "derivative-order")
        ylabel!(p, "runtime-ratio")
        Plots.savefig(p, "higher_order_$(experiment)_dim=$(fixed_dim)_I.pdf")
    end 
end

function plot_fixed_dim_d(suite::Dict, fixed_dims, fixed_ds)
    for experiment in keys(suite)
        plots = []
        for fixed_dim in fixed_dims
            for fixed_d in fixed_ds
                p = Plots.plot()
                directions_sorted = sort(collect(keys(suite[experiment][fixed_dim][fixed_d])))
                base_time = suite[experiment]["base_time"][fixed_dim]
                time_vals = [suite[experiment][fixed_dim][fixed_d][p] for p in directions_sorted]
                Plots.plot!(p, directions_sorted, time_vals / base_time, label="n=$fixed_dim, d=$fixed_d", legendfontsize=5)
                Plots.plot!(p, yformatter=:scientific, guidefontsize=5, xtickfontsize=5, ytickfontsize=5)
                xlabel!(p, "num directions")
                ylabel!(p, "time")
                push!(plots, p)
            end
        end
        p = Plots.plot(plots..., layout=(length(fixed_dims), length(fixed_ds)))
        Plots.savefig(p, "higher_order_$(experiment)_2.pdf")
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
    time = median(time.times)
    return time
end
    
    function run_mode(experiment, dim, derivative_order, num_directions)
        x = [(i+1.0)/(2.0+i) for i = 1:dim]
        if experiment === speelpenning
            a = [Adouble{TbAlloc}() for _ in eachindex(x)]
            b = [Adouble{TbAlloc}()]
            y = 0.0
            trace_on(0)
            a << x
            b = speelpenning(a)
            y = b >> y
            trace_off()
    
        end
        if experiment === lin_solve
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
        

        m = length(y)

        # allocate c++ memory for the tensor_eval function
        CxxTensor = ADOLC.myalloc2(m, binomial(num_directions + derivative_order, derivative_order))

        seed = ADOLC.create_cxx_identity(dim, num_directions)

        time = @benchmark tensor_eval(0, $m, $dim, $derivative_order, $num_directions, $x, $CxxTensor, $seed)
        #build_tensor(derivative_order, num_dependents, num_independents, CxxTensor)
        myfree2(seed)
        myfree2(CxxTensor)
        time = median(time.times)
        return time
    end
    
    function runner(dims, derivative_orders, num_directions) 
        suite = Dict()
        experiments = [speelpenning, lin_solve]
        for experiment in experiments
            println("Running $experiment ...")
            suite[experiment] = Dict()
            suite[experiment]["base_time"] = Dict(dim => run_base_time(experiment, dim) for dim in dims)
            for dim in dims
                suite[experiment][dim] = Dict()
                for d in derivative_orders
                    suite[experiment][dim][d] = Dict()
                    for p in num_directions
                        println("""
                        Running dim: $(indexin(dim, dims)[1])/$(length(dims)) 
                        d: $(indexin(d, derivative_orders)[1])/$(length(derivative_orders)) 
                        p: $(indexin(p, num_directions)[1])/$(length(num_directions)) ... 
                        """)
                        suite[experiment][dim][d][p] = run_mode(experiment, dim, d, p)
                    end
                end
            end
        end

        plot_fixed_dim_p(suite, 10, [1, 2, 3])
        plot_fixed_dim_d(suite, [10, 30, 50], [1, 5, 10])
        return suite
    end
    
    
    result = runner([10, 30, 50], [1, 2, 3, 4, 5, 10], 1:1:10)
    
    