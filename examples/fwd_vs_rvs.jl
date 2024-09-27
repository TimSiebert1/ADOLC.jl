using BenchmarkTools
using ADOLC
using ADOLC.array_types
using Plots
using ForwardDiff
using ReverseDiff
using Zygote
using DifferentiationInterface
using Tapir: Tapir

using Printf

include("examples.jl")

struct ForwardDiffTool end
struct ReverseDiffTool end
struct ADOLCForwardTool end
struct ADOLCReverseTool end
struct ADOLCTLForwardTool end
struct ZygoteTool end
struct TapirTool end

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
                Plots.plot!(
                    p, dims_sorted, mode_vals ./ base_times; label="$mode", marker=2
                )
            end
        end
        Plots.plot!(p; legend=:topleft)
        xlabel!(p, "dimension")
        ylabel!(p, "runtime-ratio")
        Plots.savefig(p, "fwd_vs_rvs_$experiment.pdf")
    end
end

function run_base_time(experiment, dim)
    if experiment === BenchmarkExamples.speelpenning
        x = [(i + 1.0) / (2.0 + i) for i in 1:dim]
        time = @benchmark BenchmarkExamples.speelpenning($x)

    elseif experiment === BenchmarkExamples.lin_solve
        x = collect(1:dim)
        A = BenchmarkExamples.build_banded_matrix(dim)
        time = @benchmark BenchmarkExamples.lin_solve($A)($x)
    elseif experiment == BenchmarkExamples.rosenbrock
        x = rand(dim)
        time = @benchmark BenchmarkExamples.rosenbrock($x)
    else
        throw("$experiment not implemented!")
    end
    time = median(time.times)
    return time
end

function setup(ADTool::ForwardDiffTool, experiment, x, dim, A)
    if experiment == BenchmarkExamples.lin_solve
        out = Matrix{Cdouble}(undef, dim, dim)
        cfg10 = ForwardDiff.JacobianConfig(experiment(A), x, ForwardDiff.Chunk{10}())
    else
        out = similar(x)
        cfg10 = ForwardDiff.GradientConfig(experiment, x, ForwardDiff.Chunk{10}())
    end

    return (out, cfg10)
end

function runAD(ADTool::ForwardDiffTool, cfg, experiment, x, dim, A)
    if experiment == BenchmarkExamples.lin_solve
        time = @benchmark ForwardDiff.jacobian!($cfg[1], $experiment($A), $x, $cfg[2])
    else
        time = @benchmark ForwardDiff.gradient!($cfg[1], $experiment, $x, $cfg[2])
    end
    return time
end

function setup(ADTool::ReverseDiffTool, experiment, x, dim, A)
    if experiment == BenchmarkExamples.lin_solve
        m = dim
        f_tape = ReverseDiff.JacobianTape(experiment(A), x)
        compiled_f_tape = ReverseDiff.compile(f_tape)
        results = Matrix{Float64}(undef, m, dim)
    else
        f_tape = ReverseDiff.GradientTape(experiment, x)
        compiled_f_tape = ReverseDiff.compile(f_tape)
        results = similar(x)
    end
    return (results, compiled_f_tape)
end

function runAD(ADTool::ReverseDiffTool, cfg, experiment, x, dim, A)
    if experiment == BenchmarkExamples.lin_solve
        time = @benchmark ReverseDiff.jacobian!($cfg[1], $cfg[2], $x)
    else
        time = @benchmark ReverseDiff.gradient!($cfg[1], $cfg[2], $x)
    end
    return time
end

function setup(ADTool::ADOLCForwardTool, experiment, x, dim, A)
    tape_id = 1
    if experiment == BenchmarkExamples.lin_solve
        y, m, n = ADOLC.create_tape(experiment(A), x, tape_id)
    else
        y, m, n = ADOLC.create_tape(experiment, x, tape_id)
    end
    res = CxxMatrix(m, n)
    dir = CxxMatrix(create_cxx_identity(n, n), n, n)
    return (res, dir, tape_id, m, n)
end

function runAD(ADTool::ADOLCForwardTool, cfg, experiment, x, dim, A)
    if experiment == BenchmarkExamples.lin_solve
        time = @benchmark ADOLC.fov_forward!(
            $cfg[1], $cfg[3], $cfg[4], $cfg[5], $x, $cfg[2]
        )
    else
        time = @benchmark ADOLC.fov_forward!(
            $cfg[1], $cfg[3], $cfg[4], $cfg[5], $x, $cfg[2]
        )
    end
    return time
end

function setup(ADTool::ADOLCTLForwardTool, experiment, x, dim, A)
    ADOLC.TladoubleModule.set_num_dir(dim)
    m = if experiment == BenchmarkExamples.lin_solve
        length(experiment(A)(x))
    else
        length(experiment(x))
    end
    a = Adouble{TlAlloc}(x; adouble=true)
    ADOLC.init_tl_gradient(a)
    res = dim == 1 ? CxxVector(n) : CxxMatrix(m, dim)
    return (res, a)
end

function runAD(ADTool::ADOLCTLForwardTool, cfg, experiment, x, dim, A)
    if experiment == BenchmarkExamples.lin_solve
        time = @benchmark ADOLC.tape_less_forward!($cfg[1], $experiment($A), $dim, $cfg[2])
    else
        time = @benchmark ADOLC.tape_less_forward!($cfg[1], $experiment, $dim, $cfg[2])
    end
    return time
end

function setup(ADTool::ADOLCReverseTool, experiment, x, dim, A)
    tape_id = 1
    if experiment == BenchmarkExamples.lin_solve
        y, m, n = ADOLC.create_tape(experiment(A), x, tape_id)
    else
        y, m, n = ADOLC.create_tape(experiment, x, tape_id)
    end
    res = m == 1 ? CxxVector(n) : CxxMatrix(m, n)
    tmp_out = Vector{Cdouble}(undef, m)
    weights = m == 1 ? [1.0] : CxxMatrix(create_cxx_identity(m, m), m, m)
    return (res, weights, tape_id, m, n, tmp_out)
end

function zos_rev(tmp_out, tape_id, m, n, x, res, weights)
    ADOLC.zos_forward!(tmp_out, tape_id, m, n, x)
    return ADOLC.fos_reverse!(res, tape_id, m, n, weights)
end

function runAD(ADTool::ADOLCReverseTool, cfg, experiment, x, dim, A)
    time = @benchmark zos_rev($cfg[6], $cfg[3], $cfg[4], $cfg[5], $x, $cfg[1], $cfg[2])
    return time
end

function setup(ADTool::ZygoteTool, experiment, x, dim, A) end

function runAD(ADTool::ZygoteTool, cfg, experiment, x, dim, A)
    if experiment == BenchmarkExamples.lin_solve
        time = @benchmark Zygote.jacobian($experiment($A), $x)
    else
        time = @benchmark Zygote.jacobian($experiment, $x)
    end
    return time
end

function setup(ADTool::TapirTool, experiment, x, dim, A)
    backend = AutoTapir()
    if experiment == BenchmarkExamples.lin_solve
        extras = prepare_jacobian(experiment(A), backend, x)
    else
        extras = prepare_gradient(experiment, backend, x)
    end
    return (backend, extras)
end

function runAD(ADTool::TapirTool, cfg, experiment, x, dim, A)
    if experiment == BenchmarkExamples.lin_solve
        time = @benchmark DifferentiationInterface.jacobian(
            $experiment($A), $cfg[1], $x, $cfg[2]
        )
    else
        time = @benchmark DifferentiationInterface.gradient(
            $experiment, $cfg[1], $x, $cfg[2]
        )
    end
    return time
end

function run_mode(experiment, ADTool, dim)
    A = nothing
    if experiment == BenchmarkExamples.speelpenning
        x = [(i + 1.0) / (2.0 + i) for i in 1:dim]
    elseif experiment == BenchmarkExamples.rosenbrock
        x = rand(dim)
    elseif experiment == BenchmarkExamples.lin_solve
        x = collect(Cdouble, 1:dim)
        A = BenchmarkExamples.build_banded_matrix(dim)
    else
        throw("$mode is not implemented!")
    end
    cfg = setup(ADTool, experiment, x, dim, A)
    time = runAD(ADTool, cfg, experiment, x, dim, A)
    return median(time.times)
end

function run_benchmark(max_dim, steps)
    suite = Dict()
    modes = (
        #ADOLCForwardTool(),
        ADOLCReverseTool(),
        #ForwardDiffTool(),
        ReverseDiffTool(),
        #ZygoteTool(),
        #TapirTool(),
    )
    #experiments = [BenchmarkExamples.rosenbrock, BenchmarkExamples.speelpenning]#, BenchmarkExamples.rosenbrock, BenchmarkExamples.lin_solve]
    experiments = [BenchmarkExamples.speelpenning]#, BenchmarkExamples.rosenbrock, BenchmarkExamples.lin_solve]
    #experiments = [BenchmarkExamples.lin_solve]
    dims = Dict(
        BenchmarkExamples.speelpenning => [10, 100:steps:max_dim...],
        BenchmarkExamples.rosenbrock => [10, 100:steps:max_dim...],
        BenchmarkExamples.lin_solve => [10, 100:steps:max_dim...],
    )

    for experiment in experiments
        println("Running $experiment ...")
        suite[experiment] = Dict()
        suite[experiment]["base_time"] = Dict(
            dim => run_base_time(experiment, dim) for dim in dims[experiment]
        )
        for mode in modes
            suite[experiment][mode] = Dict()
            for dim in dims[experiment]
                println(
                    "Running $mode - $(indexin(dim, dims[experiment])[1])/$(length(dims[experiment])) ... ",
                )
                suite[experiment][mode][dim] = run_mode(experiment, mode, dim)
            end
        end
    end
    return suite
end

function compute_slope(suite)
    slope = Dict()
    for experiment in keys(suite)
        slope[experiment] = Dict()
        base_times = suite[experiment]["base_time"]
        for mode in keys(suite[experiment])
            if mode != "base_time"
                dims_sorted = sort(collect(keys(suite[experiment][mode])))
                mode_vals = [
                    suite[experiment][mode][key] ./ base_times[key] for key in dims_sorted
                ]
                dim_diff = dims_sorted[2:end] - dims_sorted[1:(end - 1)]
                mode_val_diff = mode_vals[2:end] - mode_vals[1:(end - 1)]
                diff_quotient = mode_val_diff ./ dim_diff
                slope[experiment][mode] = mean(diff_quotient)
            end
        end
    end
    return slope
end

function print_slope(slope)
    for experiment in keys(slope)
        print("Slopes for $experiment: ")
        for mode in keys(slope[experiment])
            @printf "%s: %.2f   " mode slope[experiment][mode]
        end
        println(" ")
    end
end

function print_table(suite)
    for experiment in keys(suite)
        print_experiment(experiment, suite[experiment])
    end
end

function print_experiment(experiment, suite)
    println("experiment: $experiment")
    base_times = suite["base_time"]
    modes = [mode for mode in keys(suite) if mode != "base_time"]
    print("     |")
    for mode in modes
        print(mode, "   |")
    end
    println("")
    for i in [1, 100:100:1000...]
        print("$i     |")
        for mode in modes
            @printf "    %.2f|" suite[mode][i] ./ base_times[i]
        end
        println("")
    end
end

function run()
    suite = run_benchmark(2000, 500)
    return plot(suite)
    #slope = compute_slope(suite)
    #return print_slope(slope)
    #print_table(suite)
end

run()
