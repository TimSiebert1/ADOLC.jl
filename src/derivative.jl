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
        _, m, n = create_tape(f, x, tape_id; enableMinMaxUsingAbs=mode == :abs_normal)
    else
        m = ccall((:num_dependent, ADOLC_JLL_PATH), Cuint, (Cshort,), tape_id)
        n = ccall((:num_independent, ADOLC_JLL_PATH), Cuint, (Cshort,), tape_id)
    end
    res = allocator(tape_id, m, n, mode, size(dir, 2)[1], size(weights, 1)[1], x)
    if mode === :abs_normal
        derivative!(res)
    else
        derivative!(res, tape_id, m, n, x, mode; dir=dir, weights=weights)
    end
    return res
end

"""
    derivative(
        f::Function,
        x::Union{Cdouble,Vector{Cdouble}},
        param::Union{Cdouble,Vector{Cdouble}},
        mode::Symbol;
        dir=Vector{Cdouble}(),
        weights=Vector{Cdouble}(),
        tape_id::Integer=0,
        reuse_tape::Bool=false,
    )
This version of the [`derivative`](@ref) driver allows the definition of function parameters (`param`), which can be changed 
in subsequent calls without retaping. The given function `f` is expected to have the shape `f(x, param)`.

# Example:
```jldoctest
function f(x, param)
    x1 = x[1] * param[1]
    return [x1*x[2], x[2]] 
end
x = [-1.0, 1/2]
param = 3.0
dir = [2.0, -2.0]
res = derivative(f, x, param, :jac_vec, dir=dir, tape_id=1)

##res[1] == 9.0
##res[2] == -2.0

param = -3.0
x = [1.0, 1.0]
res = derivative(f, x, param, :jac_vec, dir=dir, tape_id=1, reuse_tape=true)
res 

# output

2-element CxxVector:
  0.0
 -2.0
```

"""
function derivative(
    f::Function,
    x::Union{Cdouble,Vector{Cdouble}},
    param::Union{Cdouble,Vector{Cdouble}},
    mode::Symbol;
    dir=Vector{Cdouble}(),
    weights=Vector{Cdouble}(),
    tape_id::Integer=0,
    reuse_tape::Bool=false,
)
    if !reuse_tape
        _, m, n = create_tape(
            f, x, param, tape_id; enableMinMaxUsingAbs=mode == :abs_normal
        )
    else
        set_param_vec(tape_id, param)
        m = ccall((:num_dependent, ADOLC_JLL_PATH), Cuint, (Cshort,), tape_id)
        n = ccall((:num_independent, ADOLC_JLL_PATH), Cuint, (Cshort,), tape_id)
    end
    res = allocator(tape_id, m, n, mode, size(dir, 2)[1], size(weights, 1)[1], x)
    if mode === :abs_normal
        derivative!(res)
    else
        derivative!(res, tape_id, m, n, x, mode; dir, weights)
    end
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
        _, m, n = create_tape(f, x, tape_id)
        reuse_tape = true
    else
        m = ccall((:num_dependent, ADOLC_JLL_PATH), Cuint, (Cshort,), tape_id)
        n = ccall((:num_independent, ADOLC_JLL_PATH), Cuint, (Cshort,), tape_id)
    end
    res = CxxMatrix(m, length(partials))
    derivative!(res, tape_id, m, n, x, partials; id_seed=id_seed, adolc_format=adolc_format)
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
        _, m, n = create_tape(f, x, tape_id)
        reuse_tape = true
    else
        m = ccall((:num_dependent, ADOLC_JLL_PATH), Cuint, (Cshort,), tape_id)
        n = ccall((:num_independent, ADOLC_JLL_PATH), Cuint, (Cshort,), tape_id)
    end

    res = CxxMatrix(m, length(partials))
    derivative!(res, tape_id, m, n, x, partials, seed; adolc_format=adolc_format)
    return res
end

"""
    derivative(
    f,
    x::Union{Cdouble, Vector{Cdouble}},
    degree::Integer;
    tape_id::Integer=0,
    reuse_tape::Bool=false
)
"""
function derivative(
    f,
    x::Union{Cdouble,Vector{Cdouble}},
    degree::Integer;
    tape_id::Integer=0,
    reuse_tape::Bool=false,
)
    if !reuse_tape
        _, m, n = create_tape(f, x, tape_id)
        reuse_tape = true
    else
        m = ccall((:num_dependent, ADOLC_JLL_PATH), Cuint, (Cshort,), tape_id)
        n = ccall((:num_independent, ADOLC_JLL_PATH), Cuint, (Cshort,), tape_id)
    end
    res = CxxMatrix(m, binomial(n + degree, degree))
    derivative!(res, tape_id, m, n, x, degree, CxxMatrix(create_cxx_identity(n, n), n, n))
    return res
end

"""
    derivative(
    f,
    x::Union{Cdouble, Vector{Cdouble}},
    degree::Integer,
    seed::CxxMatrix;
    tape_id::Integer=0,
    reuse_tape::Bool=false
)
"""
function derivative(
    f,
    x::Union{Cdouble,Vector{Cdouble}},
    degree::Integer,
    seed::CxxMatrix,
    tape_id::Integer=0,
    reuse_tape::Bool=false,
)
    if !reuse_tape
        _, m, n = create_tape(f, x, tape_id)
        reuse_tape = true
    else
        m = ccall((:num_dependent, ADOLC_JLL_PATH), Cuint, (Cshort,), tape_id)
        n = ccall((:num_independent, ADOLC_JLL_PATH), Cuint, (Cshort,), tape_id)
    end
    num_seeds = size(seed, 2)
    res = CxxMatrix(m, binomial(num_seeds + degree, degree))
    derivative!(res, tape_id, m, n, x, degree, seed)
    return res
end

"""
    function_and_derivative_value!(
        res::Vector,
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
Variant of the [derivative!](@ref) function, which stores the value of `f` at `x` in the first entry of `res`. The second entry 
stores the derivatives.

!!! note
    
    Currently, only first-order derivatives are supported!

# Example
```jldoctest

f(x) = sin(x)
jac_val = CxxVector([0.0])
func_val = [0.0]
res = [func_val, jac_val]
function_and_derivative_value!(res, f, 1, 1, float(π), :jac)
res 
# output

2-element Vector{AbstractVector{Float64}}:
 [1.2246467991473532e-16]
 [-1.0]
```
"""
function function_and_derivative_value!(
    res::Vector,
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
    return derivative!(
        res,
        f,
        m,
        n,
        x,
        mode;
        dir=dir,
        weights=weights,
        tape_id=tape_id,
        reuse_tape=reuse_tape,
    )
end

"""
    derivative!(
        res,
        tape_id::Integer
        m::Integer,
        n::Integer,
        x::Union{Cdouble,Vector{Cdouble}},
        mode::Symbol;
        dir::Union{Vector{Cdouble},Matrix{Cdouble}}=Vector{Cdouble}(),
        weights::Union{Vector{Cdouble},Matrix{Cdouble}}=Vector{Cdouble}(),
    )
A variant of the [`derivative`](@ref) driver for [first-](@ref "First-Order"),
[second-order](@ref "Second-Order") and [abs-normal-form](@ref "Abs-Normal-Form") 
computations that allows the user to provide a pre-allocated container for the result `res`.
The container can be allocated by leveraging the methods [`allocator`](@ref) or [`init_abs_normal_form`](@ref).
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
```jldoctest
f(x) = max(x[1]*x[2], x[1]^2)
x = [1.0, 1.0]
m = 1
n = 2
abs_normal_form = init_abs_normal_form(f, x)
derivative!(abs_normal_form, f, m, n, x, :abs_normal, tape_id=abs_normal_form.tape_id, reuse_tape=true)
abs_normal_form
# output

AbsNormalForm(0, 1, 2, 1, [1.0, 1.0], [1.0], [0.0], [0.0], [1.0], [1.5 0.5], [0.5;;], [1.0 -1.0], [0.0;;])
```
"""
function derivative!(
    res,
    tape_id::Integer,
    m::Integer,
    n::Integer,
    x::Union{Cdouble,Vector{Cdouble}},
    mode::Symbol;
    dir=Vector{Cdouble}(),
    weights=Vector{Cdouble}(),
)
    x = isa(x, Number) ? [x] : x
    if mode === :jac
        jac!(res, tape_id, m, n, x)
    elseif mode === :hess
        hessian!(res, tape_id, m, n, x)
    elseif mode === :jac_vec
        fos_forward!(res, tape_id, m, n, x, dir)
    elseif mode === :jac_mat
        fov_forward!(res, tape_id, m, n, x, dir)
    elseif mode === :hess_vec
        hess_vec!(res, tape_id, m, n, x, dir)
    elseif mode === :hess_mat
        hess_mat!(res, tape_id, m, n, x, dir)

    elseif mode === :vec_jac
        tmp_out = Vector{Cdouble}(undef, m)
        zos_forward!(tmp_out, tape_id, m, n, x)
        fos_reverse!(res, tape_id, m, n, weights)
    elseif mode === :mat_jac
        tmp_out = Vector{Cdouble}(undef, m)
        zos_forward!(tmp_out, tape_id, m, n, x)
        fov_reverse!(res, tape_id, m, n, weights)

    elseif mode === :vec_hess
        vec_hess!(res, tape_id, m, n, x, weights)
    elseif mode === :mat_hess
        mat_hess!(res, tape_id, m, n, x, weights)

    elseif mode === :vec_hess_vec
        vec_hess_vec!(res, tape_id, m, n, x, dir, weights)
    elseif mode === :mat_hess_vec
        mat_hess_vec!(res, tape_id, m, n, x, dir, weights)
    elseif mode === :vec_hess_mat
        vec_hess_mat!(res, tape_id, m, n, x, dir, weights)
    elseif mode === :mat_hess_mat
        mat_hess_mat!(res, tape_id, m, n, x, dir, weights)

    else
        throw(ArgumentError("mode $mode not implemented!"))
    end
end

function derivative!(abs_normal_form::AbsNormalForm)
    return abs_normal!(abs_normal_form)
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
    tape_id,
    m::Integer,
    n::Integer,
    x::Union{Cdouble,Vector{Cdouble}},
    partials::Vector{Vector{Int64}};
    id_seed::Bool=false,
    adolc_format::Bool=false,
)
    x = isa(x, Number) ? [x] : x
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
    return higher_order!(res, tape_id, m, n, x, partials, seed, n, adolc_format)
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
    tape_id,
    m::Integer,
    n::Integer,
    x::Vector{Cdouble},
    partials::Vector{Vector{Int64}},
    seed::CxxMatrix;
    adolc_format::Bool=false,
)
    return higher_order!(res, tape_id, m, n, x, partials, seed, size(seed, 2), adolc_format)
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
    tape_id,
    m::Integer,
    n::Integer,
    x::Vector{Cdouble},
    degree::Integer,
    seed::CxxMatrix,
)
    return higher_order!(res, tape_id, m, n, x, degree, seed)
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
    tape_id,
    m::Integer,
    n::Integer,
    x::Union{Cdouble,Vector{Cdouble}},
    degree::Integer,
)
    return higher_order!(
        res, tape_id, m, n, x, degree, CxxMatrix(create_cxx_identity(n, n), n, n)
    )
end

function jac!(
    res, tape_id::Integer, m::Integer, n::Integer, x::Union{Cdouble,Vector{Cdouble}}
)
    if m == 1
        tmp_out = Vector{Cdouble}(undef, m)
        zos_forward!(tmp_out, tape_id, m, n, x)
        fos_reverse!(res, tape_id, m, n, [1.0])
    else
        if n / 2 < m
            if n == 1
                fos_forward!(res, tape_id, m, n, x, [1.0])
            else
                dir = CxxMatrix(create_cxx_identity(n, n), n, n)
                fov_forward!(res, tape_id, m, n, x, dir)
            end
        else
            tmp_out = Vector{Cdouble}(undef, m)
            zos_forward!(tmp_out, tape_id, m, n, x)
            weights = CxxMatrix(create_cxx_identity(m, m), m, m)
            fov_reverse!(res, tape_id, m, n, weights)
        end
    end
end

function jac!(
    res::Vector, tape_id::Integer, m::Integer, n::Integer, x::Union{Cdouble,Vector{Cdouble}}
)
    if m == 1
        zos_forward!(res[1], tape_id, m, n, x)
        fos_reverse!(res[2], tape_id, m, n, [1.0])
    else
        if n / 2 < m
            if n == 1
                fos_forward!(res, tape_id, m, n, x, [1.0])
            else
                dir = CxxMatrix(create_cxx_identity(n, n), n, n)
                fov_forward!(res, tape_id, m, n, x, dir)
            end
        else
            zos_forward!(res[1], tape_id, m, n, x)
            weights = CxxMatrix(create_cxx_identity(m, m), m, m)
            fov_reverse!(res[2], tape_id, m, n, weights)
        end
    end
end

function zos_forward!(
    out::Vector{Cdouble}, tape_id::Integer, m::Integer, n::Integer, x::Vector{Cdouble}
)
    return ccall(
        (:zos_forward, ADOLC_JLL_PATH),
        Cint,
        (Cshort, Cint, Cint, Cint, Ptr{Cdouble}, Ptr{Cdouble}),
        tape_id,
        m,
        n,
        1,
        x,
        out,
    )
end

function fos_reverse!(res::CxxVector, tape_id::Integer, m::Integer, n::Integer, weights)
    return ccall(
        (:fos_reverse, ADOLC_JLL_PATH),
        Cint,
        (Cshort, Cint, Cint, Ptr{Cdouble}, Ptr{Cdouble}),
        tape_id,
        m,
        n,
        weights,
        res,
    )
end

function fov_reverse!(
    res, tape_id::Integer, m::Integer, n::Integer, weights::Matrix{Cdouble}
)
    cxx_weights = CxxMatrix(weights)
    return fov_reverse!(res, tape_id, m, n, cxx_weights)
end

function fov_reverse!(
    res::CxxMatrix, tape_id::Integer, m::Integer, n::Integer, weights::CxxMatrix
)
    num_weights = size(weights, 1)
    return ccall(
        (:fov_reverse, ADOLC_JLL_PATH),
        Cint,
        (Cshort, Cint, Cint, Cint, Ptr{Ptr{Cdouble}}, Ptr{Ptr{Cdouble}}),
        tape_id,
        m,
        n,
        num_weights,
        weights,
        res,
    )
end

function fos_forward!(
    res::Vector,
    tape_id::Integer,
    m::Integer,
    n::Integer,
    x::Vector{Cdouble},
    dir;
    keep::Integer=0,
)
    return ccall(
        (:fos_forward, ADOLC_JLL_PATH),
        Cint,
        (Cshort, Cint, Cint, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}),
        tape_id,
        m,
        n,
        keep,
        x,
        dir,
        res[1],
        res[2],
    )
end

function fos_forward!(
    res::CxxVector,
    tape_id::Integer,
    m::Integer,
    n::Integer,
    x::Vector{Cdouble},
    dir;
    keep::Integer=0,
)
    tmp_out = Vector{Cdouble}(undef, m)
    return fos_forward!([tmp_out, res], tape_id, m, n, x, dir; keep=keep)
end

function fov_forward!(
    res::Vector,
    tape_id::Integer,
    m::Integer,
    n::Integer,
    x::Vector{Cdouble},
    dir::CxxMatrix,
)
    num_dir = size(dir, 2)
    return ccall(
        (:fov_forward, ADOLC_JLL_PATH),
        Cint,
        (
            Cshort,
            Cint,
            Cint,
            Cint,
            Ptr{Cdouble},
            Ptr{Ptr{Cdouble}},
            Ptr{Cdouble},
            Ptr{Ptr{Cdouble}},
        ),
        tape_id,
        m,
        n,
        num_dir,
        x,
        dir,
        res[1],
        res[2],
    )
end
function fov_forward!(
    res::CxxMatrix,
    tape_id::Integer,
    m::Integer,
    n::Integer,
    x::Vector{Cdouble},
    dir::CxxMatrix,
)
    tmp_out = Vector{Cdouble}(undef, m)
    return fov_forward!([tmp_out, res], tape_id, m, n, x, dir)
end

function fov_forward!(
    res, tape_id, m::Integer, n::Integer, x::Vector{Cdouble}, dir::Matrix{Cdouble}
)
    cxx_dir = CxxMatrix(dir)
    return fov_forward!(res, tape_id, m, n, x, cxx_dir)
end

function vec_hess_vec!(
    res::CxxMatrix,
    res_grad,
    res_out,
    tape_id::Integer,
    m::Integer,
    n::Integer,
    x::Vector{Cdouble},
    dir,
    weights,
)
    rc = fos_forward!([res_out, res_grad], tape_id, m, n, x, dir; keep=2)
    if rc < 0
        throw("Not twice differentiable!")
    end
    return ccall(
        (:hos_reverse, ADOLC_JLL_PATH),
        Cint,
        (Cshort, Cint, Cint, Cint, Ptr{Cdouble}, Ptr{Ptr{Cdouble}}),
        tape_id,
        m,
        n,
        1,
        weights,
        res,
    )
end
function vec_hess_vec!(
    res::Vector, tape_id::Integer, m::Integer, n::Integer, x::Vector{Cdouble}, dir, weights
)
    res_tmp = CxxMatrix(n, 2)
    rc = vec_hess_vec!(res_tmp, res[2], res[1], tape_id, m, n, x, dir, weights)
    for i in 1:n
        res[3][i] = res_tmp[i, 2]
    end
    return rc
end

function vec_hess_vec!(
    res::CxxVector,
    tape_id::Integer,
    m::Integer,
    n::Integer,
    x::Vector{Cdouble},
    dir,
    weights,
)
    res_tangent = CxxVector(m)
    res_out = CxxVector(m)
    return vec_hess_vec!([res_out, res_tangent, res], tape_id, m, n, x, dir, weights)
end

function vec_hess_mat!(
    res_tmp::CxxMatrix,
    res::CxxMatrix,
    res_tangent::Vector{CxxVector},
    res_out,
    tape_id::Integer,
    m::Integer,
    n::Integer,
    x::Vector{Cdouble},
    dir,
    weights,
)
    rc = 0
    for (j, col) in enumerate(eachcol(dir))
        rc = vec_hess_vec!(res_tmp, res_tangent[j], res_out, tape_id, m, n, x, col, weights)
        for i in 1:n
            res[i, j] = res_tmp[i, 2]
        end
    end
    return rc
end

function vec_hess_mat!(
    res::Vector, tape_id::Integer, m::Integer, n::Integer, x::Vector{Cdouble}, dir, weights
)
    res_tmp = CxxMatrix(n, 2)
    return vec_hess_mat!(res_tmp, res[3], res[2], res[1], tape_id, m, n, x, dir, weights)
end

function vec_hess_mat!(
    res::CxxMatrix,
    tape_id::Integer,
    m::Integer,
    n::Integer,
    x::Vector{Cdouble},
    dir,
    weights,
)
    res_tmp = CxxMatrix(n, 2)
    res_tangent_tmp = CxxVector(n)
    res_out = CxxVector(m)
    rc = 0
    for j in 1:size(dir, 2)
        rc = vec_hess_vec!(
            res_tmp, res_tangent_tmp, res_out, tape_id, m, n, x, dir[:, j], weights
        )
        for i in 1:n
            res[i, j] = res_tmp[i, 2]
        end
    end
    return rc
end

function vec_hess!(res, tape_id, m::Integer, n::Integer, x::Vector{Cdouble}, weights)
    dir = Matrix{Cdouble}(I, n, n)
    return vec_hess_mat!(res, tape_id, m, n, x, dir, weights)
end

function vec_hess!(
    res::Vector, tape_id, m::Integer, n::Integer, x::Vector{Cdouble}, weights
)
    dir = Matrix{Cdouble}(I, n, n)
    res_tangent_tmp = [CxxVector(n) for _ in 1:n]
    rc = vec_hess_mat!([res[1], res_tangent_tmp, res[3]], tape_id, m, n, x, dir, weights)
    res[2] = res_tangent_tmp
    return rc
end

function mat_hess_vec!(
    res_tmp,
    res,
    res_grad,
    res_out,
    tape_id,
    m::Integer,
    n::Integer,
    x::Vector{Cdouble},
    dir,
    weights,
)
    num_weights = size(weights, 1)
    fos_forward!([res_out, res_grad], tape_id, m, n, x, dir; keep=2)
    rc = ccall(
        (:hov_reverse, ADOLC_JLL_PATH),
        Cint,
        (Cshort, Cint, Cint, Cint, Cint, Ptr{Ptr{Cdouble}}, Ptr{Ptr{Ptr{Cdouble}}}),
        tape_id,
        m,
        n,
        1,
        num_weights,
        isa(weights, Matrix) ? CxxMatrix(weights) : weights,
        res_tmp,
    )
    for i in 1:num_weights
        for j in 1:n
            res[i, j] = res_tmp[i, j, 2]
        end
    end
    return rc
end

function mat_hess_vec!(
    res_tmp,
    res,
    res_grad,
    res_out,
    tape_id,
    m::Integer,
    n::Integer,
    x::Vector{Cdouble},
    dir,
    weights,
    tensor_idx;
    lower_triag=false,
)
    num_weights = size(weights, 1)
    mat_short = ccall(
        (:alloc_short_mat, ADOLC_JLL_PATH), Ptr{Ptr{Cshort}}, (Cint, Cint), num_weights, n
    )
    cxx_weights = isa(weights, Matrix) ? CxxMatrix(weights) : weights
    fos_forward!([res_out, res_grad], tape_id, m, n, x, dir; keep=2)
    rc = ccall(
        (:hov_reverse, ADOLC_JLL_PATH),
        Cint,
        (
            Cshort,
            Cint,
            Cint,
            Cint,
            Cint,
            Ptr{Ptr{Cdouble}},
            Ptr{Ptr{Ptr{Cdouble}}},
            Ptr{Ptr{Cshort}},
        ),
        tape_id,
        m,
        n,
        1,
        num_weights,
        cxx_weights,
        res_tmp,
        mat_short,
    )
    ccall((:free_short_mat, ADOLC_JLL_PATH), Cvoid, (Ptr{Ptr{Cshort}},), mat_short)
    if lower_triag
        for i in 1:num_weights
            for j in 1:n
                if tensor_idx <= j
                    res[i, j, tensor_idx] = res_tmp[i, j, 2]
                end
            end
        end
    else
        for i in 1:num_weights
            for j in 1:n
                res[i, j, tensor_idx] = res_tmp[i, j, 2]
            end
        end
    end
    return rc
end

function mat_hess_vec!(
    res::Vector, tape_id, m::Integer, n::Integer, x::Vector{Cdouble}, dir, weights
)
    res_tmp = CxxTensor(size(weights, 1), n, 2)
    return mat_hess_vec!(res_tmp, res[3], res[2], res[1], tape_id, m, n, x, dir, weights)
end

function mat_hess_vec!(
    res, tape_id, m::Integer, n::Integer, x::Vector{Cdouble}, dir, weights
)
    res_out = CxxVector(m)
    res_grad = CxxVector(m)
    return mat_hess_vec!([res_out, res_grad, res], tape_id, m, n, x, dir, weights)
end

function _mat_hess_mat!(
    res_tmp,
    res,
    res_grad,
    res_out,
    tape_id,
    m::Integer,
    n::Integer,
    x::Vector{Cdouble},
    dir,
    weights,
    tensor_idx;
    lower_triag::Bool=false,
)
    return mat_hess_vec!(
        res_tmp,
        res,
        res_grad,
        res_out,
        tape_id,
        m,
        n,
        x,
        dir,
        weights,
        tensor_idx;
        lower_triag=lower_triag,
    )
end

function mat_hess_mat!(
    res_tmp,
    res,
    res_grad::Vector,
    res_out,
    tape_id,
    m::Integer,
    n::Integer,
    x::Vector{Cdouble},
    dir,
    weights;
    lower_triag::Bool=false,
)
    rc = 0
    for i in axes(dir, 2)
        rc = _mat_hess_mat!(
            res_tmp,
            res,
            res_grad[i],
            res_out,
            tape_id,
            m,
            n,
            x,
            dir[:, i],
            weights,
            i;
            lower_triag=lower_triag,
        )
    end
    return rc
end

function mat_hess_mat!(
    res_tmp,
    res,
    res_grad,
    res_out,
    tape_id,
    m::Integer,
    n::Integer,
    x::Vector{Cdouble},
    dir,
    weights;
    lower_triag::Bool=false,
)
    rc = 0
    for i in axes(dir, 2)
        rc = _mat_hess_mat!(
            res_tmp,
            res,
            res_grad,
            res_out,
            tape_id,
            m,
            n,
            x,
            dir[:, i],
            weights,
            i;
            lower_triag=lower_triag,
        )
    end
    return rc
end

function mat_hess_mat!(
    res::Vector,
    tape_id,
    m::Integer,
    n::Integer,
    x::Vector{Cdouble},
    dir,
    weights;
    lower_triag::Bool=false,
)
    num_weights = size(weights, 1)
    res_tmp = CxxTensor(num_weights, n, 2)
    return mat_hess_mat!(
        res_tmp,
        res[3],
        res[2],
        res[1],
        tape_id,
        m,
        n,
        x,
        dir,
        weights;
        lower_triag=lower_triag,
    )
end

function mat_hess_mat!(
    res,
    tape_id,
    m::Integer,
    n::Integer,
    x::Vector{Cdouble},
    dir,
    weights;
    lower_triag::Bool=false,
)
    res_out = CxxVector(m)
    res_grad = CxxVector(m)
    return mat_hess_mat!(
        [res_out, res_grad, res], tape_id, m, n, x, dir, weights; lower_triag=lower_triag
    )
end

function hessian!(res, tape_id, m::Integer, n::Integer, x::Vector{Cdouble})
    cxx_dir = CxxMatrix(create_cxx_identity(n, n), n, n)
    cxx_weights = CxxMatrix(create_cxx_identity(m, m), m, m)
    return mat_hess_mat!(res, tape_id, m, n, x, cxx_dir, cxx_weights; lower_triag=true)
end

function hess_vec!(
    res, tape_id, m::Integer, n::Integer, x::Union{Cdouble,Vector{Cdouble}}, dir
)
    weights = CxxMatrix(create_cxx_identity(m, m), m, m)
    return mat_hess_vec!(res, tape_id, m, n, x, dir, weights)
end

function mat_hess!(res, tape_id, m::Integer, n::Integer, x::Vector{Cdouble}, weights)
    dir = CxxMatrix(create_cxx_identity(n, n), n, n)
    return mat_hess_mat!(res, tape_id, m, n, x, dir, weights)
end

function hess_mat!(res, tape_id, m::Integer, n::Integer, x::Vector{Cdouble}, dir)
    cxx_weights = CxxMatrix(create_cxx_identity(m, m), m, m)
    return mat_hess_mat!(res, tape_id, m, n, x, dir, cxx_weights)
end

function higher_order!(
    res::CxxMatrix,
    tape_id,
    m::Integer,
    n::Integer,
    x::Vector{Cdouble},
    degree::Integer,
    seed::CxxMatrix,
)
    num_seeds = seed.dim2
    return ccall(
        (:tensor_eval, ADOLC_JLL_PATH),
        Cint,
        (
            Cshort,
            Cint,
            Cint,
            Cint,
            Cint,
            Ptr{Cdouble},
            Ptr{Ptr{Cdouble}},
            Ptr{Ptr{Cdouble}},
        ),
        tape_id,
        m,
        n,
        degree,
        num_seeds,
        x,
        res,
        seed,
    )
end

function higher_order!(
    res,
    tape_id,
    m::Integer,
    n::Integer,
    x::Vector{Cdouble},
    partials::Vector{Vector{Int64}},
    seed::CxxMatrix,
    num_seeds::Integer,
    adolc_format::Bool,
)
    if adolc_format
        degree = length(partials[1])
    else
        degree = maximum(map(sum, partials))
    end
    res_tmp = CxxMatrix(m, binomial(num_seeds + degree, degree))
    rc = ccall(
        (:tensor_eval, ADOLC_JLL_PATH),
        Cint,
        (
            Cshort,
            Cint,
            Cint,
            Cint,
            Cint,
            Ptr{Cdouble},
            Ptr{Ptr{Cdouble}},
            Ptr{Ptr{Cdouble}},
        ),
        tape_id,
        m,
        n,
        degree,
        num_seeds,
        x,
        res_tmp,
        seed,
    )
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
    return rc
end
