"""

    derivative(
        f::Function,
        x::Union{Cdouble,Vector{Cdouble}},
        mode::Symbol;
        dir::Union{Vector{Cdouble},Matrix{Cdouble}}=Vector{Cdouble}(),
        weights::Union{Vector{Cdouble},Matrix{Cdouble}}=Vector{Cdouble}(),
        tape_id::Integer=0,
        reuse_tape::Bool=false,
    )

A variant of the [`derivative`](@ref) driver, which can be used to compute
[first-order](@ref "First-Order") and [second-order](@ref "Second-Order") 
derivatives, as well as the [abs-normal-form](@ref "Abs-Normal-Form") 
of the given function `f` at the point `x`. The available modes are listed [here](@ref "Derivative Modes").
The formulas in the tables define `weights` (left multiplier) and `dir` (right multiplier).
Most modes leverage a tape, which has the identifier `tape_id`. If there is already a valid 
tape for the function `f` at the selected point `x` use `reuse_tape=true` and set the `tape_id`
accordingly to avoid the re-creation of the tape.

# Examples:

First-Order:
```jldoctest
f(x) = sin(x)
res = derivative(f, float(π), :jac)

# output

1-element CxxVector:
 -1.0
```

Second-Order:
```jldoctest
f(x) = [x[1]*x[2]^2, x[1]^2*x[3]^3]
x = [1.0, 2.0, -1.0]
dir = [1.0, 0.0, 0.0]
weights = [1.0, 1.0]
res = derivative(f, x, :vec_hess_vec, dir=dir, weights=weights)

# output

3-element CxxVector:
 -2.0
  4.0
  6.0
```

Abs-Normal-Form:
```jldoctest
f(x) = max(x[1]*x[2], x[1]^2)
x = [1.0, 1.0]
res = derivative(f, x, :abs_normal)

# output

AbsNormalForm(0, 1, 2, 1, [1.0, 1.0], [1.0], [0.0], [0.0], [1.0], [1.5 0.5], [0.5;;], [1.0 -1.0], [0.0;;])
```
"""
function derivative(
    f::Function,
    x::Union{Cdouble,Vector{Cdouble}},
    mode::Symbol;
    dir=Vector{Cdouble}(),
    weights=Vector{Cdouble}(),
    tape_id::Integer=0,
    reuse_tape::Bool=false,
)
    if !reuse_tape
        if mode == :abs_normal
            m, n = create_tape(f, x, tape_id; enableMinMaxUsingAbs=true)
        else
            m, n = create_tape(f, x, tape_id)
        end
        reuse_tape = true
    else
        m = TbadoubleModule.num_dependents(tape_id)
        n = TbadoubleModule.num_independents(tape_id)
    end

    res = if mode === :abs_normal
        init_abs_normal_form(f, x; tape_id=tape_id, reuse_tape=reuse_tape)
    else
        allocator(m, n, mode, size(dir, 2)[1], size(weights, 1)[1])
    end
    # allocator creates tape in case of :abs_normal
    derivative!(
        res,
        f,
        m,
        n,
        x,
        mode;
        dir=dir,
        weights=weights,
        tape_id=mode === :abs_normal ? res.tape_id : tape_id,
        reuse_tape=mode === :abs_normal ? true : reuse_tape,
    )

    return res
end

"""
    derivative(
        f::Function,
        x::Union{Cdouble,Vector{Cdouble}},
        partials::Vector{Vector{Int64}};
        tape_id::Integer=0,
        reuse_tape::Bool=false,
        id_seed::Bool=false,
        adolc_format::Bool=false,
    )
A variant of the [`derivative`](@ref) driver, which can be used to compute
[higher-order](@ref "Higher-Order") derivatives of the function `f` 
at the point `x`. The derivatives are
specified as mixed-partials in the `partials` vector. To define the partial-derivatives use either
the [Partial-Format](@ref) or the [ADOLC-Format](@ref) and set `adolc_format` accordingly.
The flag `id_seed` is used to specify the method for [seed-matrix generation](@ref "Seed-Matrix").
The underlying method leverages a tape, which has the identifier `tape_id`. If there is already a valid 
tape for the function `f` at the selected point `x` use `reuse_tape=true` and set the `tape_id`
accordingly to avoid the re-creation of the tape.


# Examples:


Partial-Format:
```jldoctest
f(x) = [x[1]^2*x[2]^2, x[3]^2*x[4]^2]
x = [1.0, 2.0, 3.0, 4.0]
partials = [[1, 1, 0, 0], [0, 0, 1, 1], [2, 2, 0, 0]]
res = derivative(f, x, partials)

# output

2×3 CxxMatrix:
 8.0   0.0  4.0
 0.0  48.0  0.0
```

ADOLC-Format:
```jldoctest
f(x) = [x[1]^2*x[2]^2, x[3]^2*x[4]^2]
x = [1.0, 2.0, 3.0, 4.0]
partials = [[2, 1, 0, 0], [4, 3, 0, 0], [2, 2, 1, 1]]
res = derivative(f, x, partials, adolc_format=true)

# output

2×3 CxxMatrix:
 8.0   0.0  4.0
 0.0  48.0  0.0
```
"""
function derivative(
    f,
    x::Union{Cdouble,Vector{Cdouble}},
    partials::Vector{Vector{Int64}};
    tape_id::Integer=0,
    reuse_tape::Bool=false,
    id_seed::Bool=false,
    adolc_format::Bool=false,
)
    if !reuse_tape
        m, n = create_tape(f, x, tape_id)
        reuse_tape = true
    else
        m = TbadoubleModule.num_dependents(tape_id)
        n = TbadoubleModule.num_independents(tape_id)
    end
    res = CxxMatrix(m, length(partials))
    derivative!(
        res,
        f,
        m,
        n,
        x,
        partials;
        tape_id=tape_id,
        reuse_tape=reuse_tape,
        id_seed=id_seed,
        adolc_format=adolc_format,
    )
    return res
end

"""
    derivative(
        f::Function,
        x::Union{Cdouble,Vector{Cdouble}},
        partials::Vector{Vector{Int64}},
        seed::Matrix{Cdouble};
        tape_id::Integer=0,
        reuse_tape::Bool=false,
        adolc_format::Bool=false,
    )
Variant of the [`derivative`](@ref) driver for the computation of [higher-order](@ref "Higher-Order")
derivatives, that requires a `seed`. Details on the idea behind `seed` can be found 
[here](@ref "Seed-Matrix").


Example:

```jldoctest
f(x) = [x[1]^4, x[2]^3*x[1]]
x = [1.0, 2.0]
partials = [[1], [2], [3]]
seed = CxxMatrix([[1.0, 1.0];;])
res = derivative(f, x, partials, seed)


# output

2×3 CxxMatrix:
  4.0  12.0  24.0
 20.0  36.0  42.0
```
"""
function derivative(
    f,
    x::Union{Cdouble,Vector{Cdouble}},
    partials::Vector{Vector{Int64}},
    seed;
    tape_id::Integer=0,
    reuse_tape::Bool=false,
    adolc_format::Bool=false,
)
    if !reuse_tape
        m, n = create_tape(f, x, tape_id)
        reuse_tape = true
    else
        m = TbadoubleModule.num_dependents(tape_id)
        n = TbadoubleModule.num_independents(tape_id)
    end

    res = CxxMatrix(m, length(partials))
    derivative!(
        res,
        f,
        m,
        n,
        x,
        partials,
        seed;
        tape_id=tape_id,
        reuse_tape=reuse_tape,
        adolc_format=adolc_format,
    )
    return res
end

"""
    derivative!(
        res,
        f::Function,
        m::Integer,
        n::Integer,
        x::Union{Cdouble,Vector{Cdouble}},
        mode::Symbol;
        dir::Union{Vector{Cdouble},Matrix{Cdouble}}=Vector{Cdouble}(),
        weights::Union{Vector{Cdouble},Matrix{Cdouble}}=Vector{Cdouble}(),
        tape_id::Integer=0,
        reuse_tape::Bool=false,
    )
A variant of the [`derivative`](@ref) driver for [first-](@ref "First-Order"),
[second-order](@ref "Second-Order") and [abs-normal-form](@ref "Abs-Normal-Form") 
computations that allows the user to provide a pre-allocated container for the result `res`. 
In addition to the arguments of [`derivative`](@ref), the output dimension `m` and 
input dimension `n` of the function `f` is required. If there is already a valid 
tape for the function `f` at the selected point `x` use `reuse_tape=true` and set the `tape_id`
accordingly to avoid the re-creation of the tape.

Example:
```jldoctest
f(x) = [cos(x[1]), x[2]*x[3]]
x = [0.0, 1.5, -1.0]
mode = :hess_mat
dir = [[1.0, -1.0, 1.0] [0.5, -0.5, 1.0]]
m = 2
n = 3
res =  allocator(m, n, mode, size(dir, 2)[1], 0)
derivative!(res, f, m, n, x, mode, dir=dir)
res
# output

2×3×2 CxxTensor:
[:, :, 1] =
 -1.0  0.0   0.0
  0.0  1.0  -1.0

[:, :, 2] =
 -0.5  0.0   0.0
  0.0  1.0  -0.5
```
"""
function derivative!(
    res,
    f::Function,
    m::Integer,
    n::Integer,
    x::Union{Cdouble,Vector{Cdouble}},
    mode::Symbol;
    dir=Vector{Cdouble}(),
    weights=Vector{Cdouble}(),
    tape_id::Integer=0,
    reuse_tape::Bool=false,
)
    if mode === :jac
        jac!(res, f, m, n, x, tape_id, reuse_tape)
    elseif mode === :hess
        hessian!(res, f, m, n, x, tape_id, reuse_tape)

    elseif mode === :jac_vec
        fos_forward!(res, f, m, n, x, dir, tape_id, reuse_tape)
    elseif mode === :jac_mat
        fov_forward!(res, f, m, n, x, dir, tape_id, reuse_tape)
    elseif mode === :hess_vec
        hess_vec!(res, f, m, n, x, dir, tape_id, reuse_tape)
    elseif mode === :hess_mat
        hess_mat!(res, f, m, n, x, dir, tape_id, reuse_tape)

    elseif mode === :vec_jac
        fos_reverse!(res, f, m, n, x, weights, tape_id, reuse_tape)
    elseif mode === :mat_jac
        fov_reverse!(res, f, m, n, x, weights, tape_id, reuse_tape)
    elseif mode === :vec_hess
        vec_hess!(res, f, m, n, x, weights, tape_id, reuse_tape)
    elseif mode === :mat_hess
        mat_hess!(res, f, m, n, x, weights, tape_id, reuse_tape)

    elseif mode === :vec_hess_vec
        vec_hess_vec!(res, f, m, n, x, dir, weights, tape_id, reuse_tape)
    elseif mode === :mat_hess_vec
        mat_hess_vec!(res, f, m, n, x, dir, weights, tape_id, reuse_tape)
    elseif mode === :vec_hess_mat
        vec_hess_mat!(res, f, m, n, x, dir, weights, tape_id, reuse_tape)
    elseif mode === :mat_hess_mat
        mat_hess_mat!(res, f, m, n, x, dir, weights, tape_id, reuse_tape)

    elseif mode === :abs_normal
        abs_normal!(res, f, x, tape_id, reuse_tape)

    else
        throw(ArgumentError("mode $mode not implemented!"))
    end
end

"""
    derivative!(
        res,
        f,
        m::Integer,
        n::Integer,
        x::Union{Cdouble,Vector{Cdouble}},
        partials::Vector{Vector{Int64}};
        tape_id::Integer=0,
        reuse_tape::Bool=false,
        id_seed::Bool=false,
        adolc_format::Bool=false,
    )

A variant of the [`derivative`](@ref) driver for the computation of 
[higher-order](@ref "Higher-Order") derivatives that allows the user to provide 
a pre-allocated container for the result `res`. In addition to the arguments of 
[`derivative`](@ref), the output dimension `m` and input dimension `n` of the function `f`
is required. If there is already a valid tape for the function `f` at the 
selected point `x` use `reuse_tape=true` and set the `tape_id` accordingly to 
avoid the re-creation of the tape.


Example: 
```jldoctest
f(x) = x[1]^4*x[2]*x[3]*x[4]^2
x = [3.0, -1.5, 1.5, -2.0]
partials = [[4, 0, 0, 0], [3, 0, 1, 2]]
m = 1
n = 4
res = CxxMatrix(m, length(partials))
derivative!(res, f, m, n, x, partials)
res

# output

1×2 CxxMatrix:
 -216.0  -216.0
```
"""
function derivative!(
    res,
    f,
    m::Integer,
    n::Integer,
    x::Union{Cdouble,Vector{Cdouble}},
    partials::Vector{Vector{Int64}};
    tape_id::Integer=0,
    reuse_tape::Bool=false,
    id_seed::Bool=false,
    adolc_format::Bool=false,
)
    if id_seed
        seed = CxxMatrix(create_cxx_identity(n, n), n, n)
    else
        seed_idxs = if adolc_format
            seed_idxs_adolc_format(partials)
        else
            seed_idxs_partial_format(partials)
        end
        partials = if adolc_format
            adolc_format_to_seed_space(partials, seed_idxs)
        else
            partial_format_to_seed_space(partials, seed_idxs)
        end
        seed = CxxMatrix(create_partial_cxx_identity(n, seed_idxs), n, length(seed_idxs))
    end
    return higher_order!(
        res, f, m, n, x, partials, seed, n, tape_id, reuse_tape, adolc_format
    )
end

"""
    derivative!(
        res,
        f,
        m::Integer,
        n::Integer,
        x::Union{Cdouble,Vector{Cdouble}},
        partials::Vector{Vector{Int64}},
        seed::CxxMatrix;
        tape_id::Integer=0,
        reuse_tape::Bool=false,
        adolc_format::Bool=false,
    )
Variant of the [`derivative!`](@ref) driver for the computation of [higher-order](@ref "Higher-Order")
derivatives, that requires a `seed`. Details on the idea behind `seed` can be found 
[here](@ref "Seed-Matrix").


Example:

```jldoctest
f(x) = [x[1]^4, x[2]^3*x[1]]
x = [1.0, 2.0]
partials = [[1], [2], [3]]
seed = CxxMatrix([[1.0, 1.0];;])
m = 2
n = 2
res = CxxMatrix(m, length(partials))
derivative!(res, f, m, n, x, partials, seed)
res

# output

2×3 CxxMatrix:
  4.0  12.0  24.0
 20.0  36.0  42.0
```
"""
function derivative!(
    res,
    f,
    m::Integer,
    n::Integer,
    x::Union{Cdouble,Vector{Cdouble}},
    partials::Vector{Vector{Int64}},
    seed::CxxMatrix;
    tape_id::Integer=0,
    reuse_tape::Bool=false,
    adolc_format::Bool=false,
)
    return higher_order!(
        res, f, m, n, x, partials, seed, size(seed, 2), tape_id, reuse_tape, adolc_format
    )
end


"""
    derivative!(
            res::CxxMatrix,
            f,
            m::Integer,
            n::Integer,
            x::Union{Cdouble, Vector{Cdouble}},
            degree::Integer,
            seed::CxxMatrix;
            tape_id::Integer=0,
            reuse_tape::Bool=false
    )
"""

function derivative!(
        res::CxxMatrix,
        f,
        m::Integer,
        n::Integer,
        x::Union{Cdouble, Vector{Cdouble}},
        degree::Integer,
        seed::CxxMatrix;
        tape_id::Integer=0,
        reuse_tape::Bool=false
    )
    higher_order!(res, f, m, n, x, degree, seed, tape_id, reuse_tape)
end

"""
    derivative!(
    res::CxxMatrix,
    f,
    m::Integer,
    n::Integer,
    x::Union{Cdouble, Vector{Cdouble}},
    degree::Integer;
    tape_id::Integer=0,
    reuse_tape::Bool=false
)
"""

function derivative!(
    res::CxxMatrix,
    f,
    m::Integer,
    n::Integer,
    x::Union{Cdouble, Vector{Cdouble}},
    degree::Integer;
    tape_id::Integer=0,
    reuse_tape::Bool=false
)
    higher_order!(res, f, m, n, x, degree, CxxMatrix(create_cxx_identity(n, n), n, n), tape_id, reuse_tape)
end

function jac!(
    res,
    f,
    m::Integer,
    n::Integer,
    x::Union{Cdouble,Vector{Cdouble}},
    tape_id::Integer,
    reuse_tape::Bool,
)
    if m == 1
        gradient!(res, f, n, x, tape_id, reuse_tape)
    else
        if n / 2 < m
            tape_less_forward!(res, f, n, x)
        else
            cxx_weights = CxxMatrix(create_cxx_identity(m, m), m, m)
            fov_reverse!(res, f, m, n, x, cxx_weights, tape_id, reuse_tape)
        end
    end
end

function gradient!(
    res,
    f,
    n::Integer,
    x::Union{Cdouble,Vector{Cdouble}},
    tape_id::Integer,
    reuse_tape::Bool,
)
    if !reuse_tape
        create_tape(f, x, tape_id)
    end
    return TbadoubleModule.gradient(tape_id, n, x, res.data)
end

function gradient!(res, n::Integer, a::Adouble{TlAlloc})
    for i in 1:n
        res[i] = getADValue(a, i)
    end
end

function gradient!(res, n::Integer, a::Vector{Adouble{TlAlloc}})
    for i in eachindex(a)
        for j in 1:n
            res[i, j] = getADValue(a[i], j)
        end
    end
end

function init_tl_gradient(a::Adouble{TlAlloc})
    TladoubleModule.setADValue(a.val, 1.0)
end

function init_tl_gradient(a::Vector{Adouble{TlAlloc}})
    for j in eachindex(a)
        for i in eachindex(a)
            TladoubleModule.setADValue(a[i].val, 0.0, j)
            if i == j
                TladoubleModule.setADValue(a[i].val, 1.0, i)
            end
        end
    end
end

function tape_less_forward!(res, f, n::Integer, x::Union{Cdouble,Vector{Cdouble}})
    TladoubleModule.set_num_dir(n)
    a = Adouble{TlAlloc}(x; adouble=true)
    init_tl_gradient(a)
    return tape_less_forward!(res, f, n, a)
end

function tape_less_forward!(
    res, f, n::Integer, a::Union{Adouble{TlAlloc},Vector{Adouble{TlAlloc}}}
)
    b = f(a)
    return gradient!(res, n, b)
end

function fos_reverse!(
    res,
    f,
    m::Integer,
    n::Integer,
    x::Union{Cdouble,Vector{Cdouble}},
    weights::Vector{Cdouble},
    tape_id::Integer,
    reuse_tape,
)
    if !reuse_tape
        create_tape(f, x, tape_id; keep=1)
    else
        TbadoubleModule.zos_forward(tape_id, m, n, 1, x, Vector{Cdouble}(undef, m))
    end
    return TbadoubleModule.fos_reverse(tape_id, m, n, weights, res.data)
end

function fos_reverse!(
    res,
    f,
    m::Integer,
    n::Integer,
    x::Union{Cdouble,Vector{Cdouble}},
    weights::CxxVector,
    tape_id::Integer,
    reuse_tape,
)
    if !reuse_tape
        create_tape(f, x, tape_id; keep=1)
    else
        TbadoubleModule.zos_forward(tape_id, m, n, 1, x, Vector{Cdouble}(undef, m))
    end
    return TbadoubleModule.fos_reverse(tape_id, m, n, weights.data, res.data)
end

function fov_reverse!(
    res,
    f,
    m::Integer,
    n::Integer,
    x::Union{Cdouble,Vector{Cdouble}},
    weights::Matrix{Cdouble},
    tape_id::Integer,
    reuse_tape::Bool,
)
    cxx_weights = CxxMatrix(weights)
    return fov_reverse!(res, f, m, n, x, cxx_weights, tape_id, reuse_tape)
end

function fov_reverse!(
    res,
    f,
    m::Integer,
    n::Integer,
    x::Union{Cdouble,Vector{Cdouble}},
    weights::CxxMatrix,
    tape_id::Integer,
    reuse_tape::Bool,
)
    if !reuse_tape
        create_tape(f, x, tape_id; keep=1)
    else
        TbadoubleModule.zos_forward(tape_id, m, n, 1, x, Vector{Cdouble}(undef, m))
    end

    return TbadoubleModule.fov_reverse(tape_id, m, n, weights.dim1, weights.data, res.data)
end

function fos_forward!(
    res,
    f,
    m::Integer,
    n::Integer,
    x::Union{Cdouble,Vector{Cdouble}},
    dir::Vector{Cdouble},
    tape_id::Integer,
    reuse_tape,
)
    if !reuse_tape
        create_tape(f, x, tape_id)
    end
    return TbadoubleModule.fos_forward(
        tape_id, m, n, 0, x, dir, Vector{Cdouble}(undef, m), res.data
    )
end

function fos_forward!(
    res,
    f,
    m::Integer,
    n::Integer,
    x::Union{Cdouble,Vector{Cdouble}},
    dir::CxxVector,
    tape_id::Integer,
    reuse_tape,
)
    if !reuse_tape
        create_tape(f, x, tape_id)
    end
    return TbadoubleModule.fos_forward(
        tape_id, m, n, 0, x, dir.data, Vector{Cdouble}(undef, m), res.data
    )
end

function fov_forward!(
    res,
    f,
    m::Integer,
    n::Integer,
    x::Union{Cdouble,Vector{Cdouble}},
    dir::Matrix{Cdouble},
    tape_id::Integer,
    reuse_tape::Bool,
)
    cxx_dir = CxxMatrix(dir)
    return fov_forward!(res, f, m, n, x, cxx_dir, tape_id, reuse_tape)
end

function fov_forward!(
    res,
    f,
    m::Integer,
    n::Integer,
    x::Union{Cdouble,Vector{Cdouble}},
    dir::CxxMatrix,
    tape_id::Integer,
    reuse_tape::Bool,
)
    if !reuse_tape
        create_tape(f, x, tape_id)
    end
    return TbadoubleModule.fov_forward(
        tape_id, m, n, dir.dim2, x, dir.data, Vector{Cdouble}(undef, m), res.data
    )
end

function check_resue_abs_normal_problem(
    tape_id::Integer, abs_normal_problem::AbsNormalForm
)
    m = TbadoubleModule.num_dependents(tape_id)
    n = TbadoubleModule.num_independents(tape_id)
    if abs_normal_problem.tape_id != tape_id
        throw(
            "Tape_id mismatch ($(abs_normal_problem.tape_id) vs. $tape_id)! The tape id has to be the same when reusing abs_normal_problem!",
        )
    end
    if abs_normal_problem.m != m
        throw(
            "Outputdimension mismatch ($(abs_normal_problem.m) vs. $m)! The dimensions has to remain the same when resuing abs_normal_problem!",
        )
    end
    if abs_normal_problem.n != n
        throw(
            "Inputdimension mismatch ($(abs_normal_problem.n) vs. $n)! The dimensions has to remain the same when resuing abs_normal_problem!",
        )
    end
    if get_num_switches(tape_id) != abs_normal_problem.num_switches
        throw(
            "NumSwitches mismatcht ($(abs_normal_problem.num_switches) vs. $(get_num_switches(tape_id)))! The number of switches has to remain the same when reusing abs_normal_problem!",
        )
    end
    return true
end

function abs_normal!(
    abs_normal_problem,
    f,
    x::Union{Cdouble,Vector{Cdouble}},
    tape_id::Integer,
    reuse_tape::Bool,
)
    if !reuse_tape
        _ = create_tape(f, x, tape_id; enableMinMaxUsingAbs=true)
    else
        @assert check_resue_abs_normal_problem(tape_id, abs_normal_problem)
        ADOLC.jl_res_to_cxx_res!(abs_normal_problem.x, x)
    end
    return abs_normal!(abs_normal_problem)
end

function vec_hess_vec!(
    res,
    f,
    m::Integer,
    n::Integer,
    x::Union{Cdouble,Vector{Cdouble}},
    dir::Vector{Cdouble},
    weights::Vector{Cdouble},
    tape_id::Integer,
    reuse_tape::Bool,
)
    if !(reuse_tape)
        create_tape(f, x, tape_id)
    end
    return TbadoubleModule.lagra_hess_vec(tape_id, m, n, x, dir, weights, res.data)
end

function vec_hess_vec!(
    res,
    f,
    m::Integer,
    n::Integer,
    x::Union{Cdouble,Vector{Cdouble}},
    dir::Vector{Cdouble},
    weights::CxxVector,
    tape_id::Integer,
    reuse_tape::Bool,
)
    if !(reuse_tape)
        create_tape(f, x, tape_id)
    end
    return TbadoubleModule.lagra_hess_vec(tape_id, m, n, x, dir, weights.data, res.data)
end

function vec_hess_vec!(
    res,
    f,
    m::Integer,
    n::Integer,
    x::Union{Cdouble,Vector{Cdouble}},
    dir::CxxVector,
    weights::CxxVector,
    tape_id::Integer,
    reuse_tape::Bool,
)
    if !(reuse_tape)
        create_tape(f, x, tape_id)
    end
    return TbadoubleModule.lagra_hess_vec(
        tape_id, m, n, x, dir.data, weights.data, res.data
    )
end

function vec_hess_mat!(
    res,
    f,
    m::Integer,
    n::Integer,
    x::Union{Cdouble,Vector{Cdouble}},
    dir::Matrix{Cdouble},
    weights::Vector{Cdouble},
    tape_id::Integer,
    reuse_tape::Bool,
)
    if !(reuse_tape)
        create_tape(f, x, tape_id)
    end
    res_tmp = CxxVector(n)
    return vec_hess_mat!(res, f, m, n, x, dir, weights, res_tmp, tape_id, reuse_tape)
end

function vec_hess_mat!(
    res,
    f,
    m::Integer,
    n::Integer,
    x::Union{Cdouble,Vector{Cdouble}},
    dir::CxxMatrix,
    weights::CxxVector,
    tape_id::Integer,
    reuse_tape::Bool,
)
    if !(reuse_tape)
        create_tape(f, x, tape_id)
    end
    res_tmp = CxxVector(n)
    return vec_hess_mat!(res, f, m, n, x, dir, weights, res_tmp, tape_id, reuse_tape)
end

function vec_hess_mat!(
    res,
    f,
    m::Integer,
    n::Integer,
    x::Union{Cdouble,Vector{Cdouble}},
    dir::Matrix{Cdouble},
    weights::Vector{Cdouble},
    res_tmp::CxxVector,
    tape_id::Integer,
    reuse_tape::Bool,
)
    if !(reuse_tape)
        create_tape(f, x, tape_id)
    end
    for i in axes(dir, 2)
        vec_hess_vec!(res_tmp, f, m, n, x, dir[:, i], weights, tape_id, true)
        for j in 1:n
            res[j, i] = res_tmp[j]
            res_tmp[j] = 0.0
        end
    end
end

function vec_hess_mat!(
    res,
    f,
    m::Integer,
    n::Integer,
    x::Union{Cdouble,Vector{Cdouble}},
    dir::CxxMatrix,
    weights::CxxVector,
    res_tmp::CxxVector,
    tape_id::Integer,
    reuse_tape::Bool,
)
    if !(reuse_tape)
        create_tape(f, x, tape_id)
    end
    for i in axes(dir, 2)
        vec_hess_vec!(res_tmp, f, m, n, x, dir[:, i], weights, tape_id, true)
        for j in 1:n
            res[j, i] = res_tmp[j]
            res_tmp[j] = 0.0
        end
    end
end

function vec_hess!(
    res,
    f,
    m::Integer,
    n::Integer,
    x::Union{Cdouble,Vector{Cdouble}},
    weights::Vector{Cdouble},
    tape_id::Integer,
    reuse_tape::Bool,
)
    dir = Matrix{Cdouble}(I, n, n)
    return vec_hess_mat!(res, f, m, n, x, dir, weights, tape_id, reuse_tape)
end

function vec_hess!(
    res,
    f,
    m::Integer,
    n::Integer,
    x::Union{Cdouble,Vector{Cdouble}},
    weights::CxxVector,
    tape_id::Integer,
    reuse_tape::Bool,
)
    dir = CxxMatrix(create_cxx_identity(n, n), n, n)
    return vec_hess_mat!(res, f, m, n, x, dir, weights, tape_id, reuse_tape)
end

function mat_hess_vec!(
    res,
    f,
    m::Integer,
    n::Integer,
    x::Union{Cdouble,Vector{Cdouble}},
    dir::Vector{Cdouble},
    weights::Matrix{Cdouble},
    tape_id::Integer,
    reuse_tape::Bool,
)
    cxx_weights = CxxMatrix(weights)
    num_weights = cxx_weights.dim1
    res_fos_tmp = CxxVector(m)
    nz_tmp = alloc_mat_short(num_weights, n)
    res_hov_tmp = CxxTensor(num_weights, n, 2)
    mat_hess_vec!(
        res,
        f,
        m,
        n,
        x,
        dir,
        cxx_weights,
        tape_id,
        reuse_tape,
        res_fos_tmp,
        nz_tmp,
        res_hov_tmp,
    )
    return free_mat_short(nz_tmp, num_weights)
end

function mat_hess_vec!(
    res,
    f,
    m::Integer,
    n::Integer,
    x::Union{Cdouble,Vector{Cdouble}},
    dir::Vector{Cdouble},
    weights::CxxMatrix,
    tape_id::Integer,
    reuse_tape::Bool,
)
    num_weights = weights.dim1
    res_fos_tmp = CxxVector(m)
    nz_tmp = alloc_mat_short(num_weights, n)
    res_hov_tmp = CxxTensor(num_weights, n, 2)
    mat_hess_vec!(
        res,
        f,
        m,
        n,
        x,
        dir,
        weights,
        tape_id,
        reuse_tape,
        res_fos_tmp,
        nz_tmp,
        res_hov_tmp,
    )
    return free_mat_short(nz_tmp, num_weights) 
end


function mat_hess_vec!(
    res,
    f,
    m::Integer,
    n::Integer,
    x::Union{Cdouble,Vector{Cdouble}},
    dir::Vector{Cdouble},
    weights::CxxMatrix,
    tape_id::Integer,
    reuse_tape::Bool,
    res_fos_tmp::CxxVector,
    nz_tmp::CxxPtr{CxxPtr{Cshort}},
    res_hov_tmp::CxxTensor,
)
    if !(reuse_tape)
        create_tape(f, x, tape_id)
    end
    num_weights = weights.dim1
    degree = 1
    keep = degree + 1
    TbadoubleModule.fos_forward(
        tape_id, m, n, keep, x, dir, Vector{Cdouble}(undef, m), res_fos_tmp.data
    )
    TbadoubleModule.hov_reverse(
        tape_id, m, n, degree, num_weights, weights.data, res_hov_tmp.data, nz_tmp
    )
    for i in 1:num_weights
        for j in 1:n
            res[i, j] = res_hov_tmp[i, j, degree + 1]
            res_hov_tmp[i, j, degree + 1] = 0.0
        end
    end
end

function mat_hess_vec!(
    res,
    f,
    m::Integer,
    n::Integer,
    x::Union{Cdouble,Vector{Cdouble}},
    dir::CxxVector,
    weights::CxxMatrix,
    tape_id::Integer,
    reuse_tape::Bool,
)
    num_weights = weights.dim1
    res_fos_tmp = CxxVector(m)
    nz_tmp = alloc_mat_short(num_weights, n)
    res_hov_tmp = CxxTensor(num_weights, n, 2)
    mat_hess_vec!(
        res, f, m, n, x, dir, weights, tape_id, reuse_tape, res_fos_tmp, nz_tmp, res_hov_tmp
    )
    return free_mat_short(nz_tmp, num_weights)
end


function mat_hess_vec!(
    res,
    f,
    m::Integer,
    n::Integer,
    x::Union{Cdouble,Vector{Cdouble}},
    dir::CxxVector,
    weights::CxxMatrix,
    tape_id::Integer,
    reuse_tape::Bool,
    res_fos_tmp::CxxVector,
    nz_tmp::CxxPtr{CxxPtr{Cshort}},
    res_hov_tmp::CxxTensor,
)
    if !(reuse_tape)
        create_tape(f, x, tape_id)
    end
    num_weights = weights.dim1
    degree = 1
    keep = degree + 1
    TbadoubleModule.fos_forward(
        tape_id, m, n, keep, x, dir.data, Vector{Cdouble}(undef, m), res_fos_tmp.data
    )
    TbadoubleModule.hov_reverse(
        tape_id, m, n, degree, num_weights, weights.data, res_hov_tmp.data, nz_tmp
    )
    for i in 1:num_weights
        for j in 1:n
            res[i, j] = res_hov_tmp[i, j, degree + 1]
            res_hov_tmp[i, j, degree + 1] = 0.0
        end
    end
end

function mat_hess_mat!(
    res,
    f,
    m::Integer,
    n::Integer,
    x::Union{Cdouble,Vector{Cdouble}},
    dir::Matrix{Cdouble},
    weights::Matrix{Cdouble},
    tape_id::Integer,
    reuse_tape::Bool;
    lower_triag::Bool=false,
)
    cxx_dir = CxxMatrix(dir)
    cxx_weights = CxxMatrix(weights)
    num_weights = cxx_weights.dim1
    res_tmp = CxxMatrix(num_weights, n)
    res_fos_tmp = CxxVector(m)
    nz_tmp = alloc_mat_short(num_weights, n)
    res_hov_tmp = CxxTensor(num_weights, n, 2)
    mat_hess_mat!(
        res,
        f,
        m,
        n,
        x,
        cxx_dir,
        cxx_weights,
        tape_id,
        reuse_tape,
        res_tmp,
        res_fos_tmp,
        nz_tmp,
        res_hov_tmp;
        lower_triag=lower_triag,
    )
    return free_mat_short(nz_tmp, num_weights)
end

function mat_hess_mat!(
    res,
    f,
    m::Integer,
    n::Integer,
    x::Union{Cdouble,Vector{Cdouble}},
    dir::CxxMatrix,
    weights::CxxMatrix,
    tape_id::Integer,
    reuse_tape::Bool;
    lower_triag::Bool=false,
)
    num_weights = weights.dim1
    res_tmp = CxxMatrix(num_weights, n)
    res_fos_tmp = CxxVector(m)
    nz_tmp = alloc_mat_short(num_weights, n)
    res_hov_tmp = CxxTensor(num_weights, n, 2)
    mat_hess_mat!(
        res,
        f,
        m,
        n,
        x,
        dir,
        weights,
        tape_id,
        reuse_tape,
        res_tmp,
        res_fos_tmp,
        nz_tmp,
        res_hov_tmp;
        lower_triag=lower_triag,
    )
    return free_mat_short(nz_tmp, num_weights)
end

function mat_hess_mat!(
    res,
    f,
    m::Integer,
    n::Integer,
    x::Union{Cdouble,Vector{Cdouble}},
    dir::CxxMatrix,
    weights::CxxMatrix,
    tape_id::Integer,
    reuse_tape::Bool,
    res_tmp::CxxMatrix,
    res_fos_tmp::CxxVector,
    nz_tmp::CxxPtr{CxxPtr{Cshort}},
    res_hov_tmp::CxxTensor;
    lower_triag::Bool=false,
)
    if !(reuse_tape)
        create_tape(f, x, tape_id)
    end
    for i in axes(dir, 2)
        mat_hess_vec!(
            res_tmp,
            f,
            m,
            n,
            x,
            dir[:, i],
            weights,
            tape_id,
            true,
            res_fos_tmp,
            nz_tmp,
            res_hov_tmp,
        )
        if lower_triag
            for j in 1:m
                for k in 1:n
                    if i <= k
                        res[j, k, i] = res_tmp[j, k]
                        res_tmp[j, k] = 0.0
                    end
                end
            end
        else
            for j in 1:weights.dim1
                for k in 1:n
                    res[j, k, i] = res_tmp[j, k]
                    res_tmp[j, k] = 0.0
                end
            end
        end
    end
end

function hessian!(
    res,
    f::Function,
    m::Integer,
    n::Integer,
    x::Union{Cdouble,Vector{Cdouble}},
    tape_id::Integer,
    reuse_tape::Bool,
)
    cxx_dir = CxxMatrix(create_cxx_identity(n, n), n, n)
    cxx_weights = CxxMatrix(create_cxx_identity(m, m), m, m)
    return mat_hess_mat!(
        res, f, m, n, x, cxx_dir, cxx_weights, tape_id, reuse_tape; lower_triag=true
    )
end

function hess_vec!(
    res,
    f,
    m::Integer,
    n::Integer,
    x::Union{Cdouble,Vector{Cdouble}},
    dir::Vector{Cdouble},
    tape_id::Integer,
    reuse_tape::Bool,
)
    weights = CxxMatrix(create_cxx_identity(m, m), m, m)
    return mat_hess_vec!(res, f, m, n, x, dir, weights, tape_id, reuse_tape)
end

function hess_vec!(
    res,
    f,
    m::Integer,
    n::Integer,
    x::Union{Cdouble,Vector{Cdouble}},
    dir::CxxVector,
    tape_id::Integer,
    reuse_tape::Bool,
)
    weights = CxxMatrix(create_cxx_identity(m, m), m, m)
    return mat_hess_vec!(res, f, m, n, x, dir, weights, tape_id, reuse_tape)
end

function mat_hess!(
    res,
    f,
    m::Integer,
    n::Integer,
    x::Union{Cdouble,Vector{Cdouble}},
    weights::Matrix{Cdouble},
    tape_id::Integer,
    reuse_tape::Bool,
)
    dir = Matrix{Cdouble}(I, n, n)
    return mat_hess_mat!(res, f, m, n, x, dir, weights, tape_id, reuse_tape)
end

function mat_hess!(
    res,
    f,
    m::Integer,
    n::Integer,
    x::Union{Cdouble,Vector{Cdouble}},
    weights::CxxMatrix,
    tape_id::Integer,
    reuse_tape::Bool,
)
    dir = CxxMatrix(create_cxx_identity(n, n), n, n)
    return mat_hess_mat!(res, f, m, n, x, dir, weights, tape_id, reuse_tape)
end

function hess_mat!(
    res,
    f,
    m::Integer,
    n::Integer,
    x::Union{Cdouble,Vector{Cdouble}},
    dir::Matrix{Cdouble},
    tape_id::Integer,
    reuse_tape::Bool,
)
    cxx_dir = CxxMatrix(dir)
    cxx_weights = CxxMatrix(create_cxx_identity(m, m), m, m)
    return mat_hess_mat!(res, f, m, n, x, cxx_dir, cxx_weights, tape_id, reuse_tape)
end

function hess_mat!(
    res,
    f,
    m::Integer,
    n::Integer,
    x::Union{Cdouble,Vector{Cdouble}},
    dir::CxxMatrix,
    tape_id::Integer,
    reuse_tape::Bool,
)
    cxx_weights = CxxMatrix(create_cxx_identity(m, m), m, m)
    return mat_hess_mat!(res, f, m, n, x, dir, cxx_weights, tape_id, reuse_tape)
end

function higher_order!(
    res,
    f,
    m::Integer,
    n::Integer,
    x::Vector{Cdouble},
    partials::Vector{Vector{Int64}},
    seed::CxxMatrix,
    num_seeds::Integer,
    tape_id::Integer,
    reuse_tape::Bool,
    adolc_format::Bool,
)
    if !reuse_tape
        create_tape(f, x, tape_id)
    end
    if adolc_format
        degree = length(partials[1])
    else
        degree = maximum(map(sum, partials))
    end
    res_tmp = CxxMatrix(m, binomial(num_seeds + degree, degree))
    tensor_eval(tape_id, m, n, degree, num_seeds, x, res_tmp.data, seed.data)
    if !adolc_format
        adolc_partial = zeros(Cint, degree)
    end
    for (i, partial) in enumerate(partials)
        if !adolc_format
            partial_to_adolc_format!(adolc_partial, partial, degree)
        end
        for j in 1:m
            if !adolc_format
                res[j, i] = res_tmp[j, tensor_address(degree, adolc_partial)]
            else
                res[j, i] = res_tmp[j, tensor_address(degree, partial)]
            end
        end
    end
end

function higher_order!(
    res::CxxMatrix,
    f,
    m::Integer,
    n::Integer,
    x::Vector{Cdouble},
    degree::Integer,
    seed::CxxMatrix,
    tape_id::Integer,
    reuse_tape::Bool,
)
    if !reuse_tape
        create_tape(f, x, tape_id)
    end
    num_seeds = seed.dim2
    return tensor_eval(tape_id, m, n, degree, num_seeds, x, res.data, seed.data)
end

function create_tape(
    f,
    x::Union{Cdouble,Vector{Cdouble}},
    tape_id::Integer;
    keep::Integer=0,
    enableMinMaxUsingAbs=false,
)
    if enableMinMaxUsingAbs
        TbadoubleModule.enableMinMaxUsingAbs()
    end

    trace_on(tape_id, keep)
    a = create_independent(x)
    b = f(a)
    dependent(b)
    trace_off()

    return length(b), length(x)
end

"""
    create_independent(x::Union{Cdouble, Vector{Cdouble}})

"""
function create_independent(x)
    n = length(x)
    a = n == 1 ? Adouble{TbAlloc}() : [Adouble{TbAlloc}() for _ in 1:n]
    a << x
    return a
end

function dependent(b)
    m = length(b)
    y = m == 1 ? Cdouble(0.0) : Vector{Cdouble}(undef, m) 
    return b >> y
end
