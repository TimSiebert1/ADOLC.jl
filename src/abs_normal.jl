using .array_types
function abs_normal_finalizer(problem)
    finalize(problem.x)
    finalize(problem.y)
    finalize(problem.z)

    finalize(problem.cz)
    finalize(problem.cy)

    finalize(problem.Y)
    finalize(problem.J)
    finalize(problem.Z)
    return finalize(problem.L)
end
mutable struct AbsNormalForm
    tape_id::Int64
    m::Int64
    n::Int64
    num_switches::Int32
    x::CxxVector
    y::CxxVector
    z::CxxVector
    cz::CxxVector
    cy::CxxVector
    Y::CxxMatrix
    J::CxxMatrix
    Z::CxxMatrix
    L::CxxMatrix

    function AbsNormalForm(tape_id::Int64, m, n, x::Vector{Float64}, y::Vector{Float64})
        num_switches = TbadoubleModule.get_num_switches(tape_id)
        z = CxxVector(num_switches)

        cz = CxxVector(num_switches)
        cy = CxxVector(length(y))

        Y = CxxMatrix(length(y), length(x))
        J = CxxMatrix(length(y), num_switches)
        Z = CxxMatrix(num_switches, length(x))
        L = CxxMatrix(num_switches, num_switches)

        problem = new(
            tape_id, m, n, num_switches, CxxVector(x), CxxVector(y), z, cz, cy, Y, J, Z, L
        )
        return finalizer(abs_normal_finalizer, problem)
    end
    AbsNormalForm() = new()
end

function abs_normal!(abs_normal_form::AbsNormalForm)
    return _abs_normal!(
        abs_normal_form.tape_id,
        abs_normal_form.z,
        abs_normal_form.cz,
        abs_normal_form.cy,
        abs_normal_form.Y,
        abs_normal_form.J,
        abs_normal_form.Z,
        abs_normal_form.L,
        abs_normal_form.m,
        abs_normal_form.n,
        abs_normal_form.num_switches,
        abs_normal_form.x,
        abs_normal_form.y,
    )
end

function _abs_normal!(
    tape_id::Int64,
    z_cxx::CxxVector,
    cz_cxx::CxxVector,
    cy_cxx::CxxVector,
    Y_cxx::CxxMatrix,
    J_cxx::CxxMatrix,
    Z_cxx::CxxMatrix,
    L_cxx::CxxMatrix,
    m::Integer,
    n::Integer,
    num_switches::Int32,
    x_cxx::CxxVector,
    y_cxx::CxxVector,
)
    cz = cz_cxx.data
    cy = cy_cxx.data
    x = x_cxx.data
    y = y_cxx.data
    z = z_cxx.data

    Y = Y_cxx.data
    J = J_cxx.data
    Z = Z_cxx.data
    L = L_cxx.data

    return TbadoubleModule.abs_normal(
        tape_id, m, n, num_switches, x, y, z, cz, cy, Y, J, Z, L
    )
end
"""
    init_abs_normal_form(
        f, x::Union{Cdouble,Vector{Cdouble}}; tape_id::Integer=0, reuse_tape=false
    )
"""
function init_abs_normal_form(
    f, x::Union{Cdouble,Vector{Cdouble}}; tape_id::Integer=0, reuse_tape=false
)
    if !reuse_tape
        m, n = create_tape(f, x, tape_id; enableMinMaxUsingAbs=true)
        reuse_tape = true
    else
        m = TbadoubleModule.num_dependents(tape_id)
        n = TbadoubleModule.num_independents(tape_id)
    end
    return AbsNormalForm(tape_id, m, n, x, Vector{Cdouble}(undef, m))
end
