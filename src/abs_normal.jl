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

    x::CxxVector{Float64}
    y::CxxVector{Float64}
    z::CxxVector{Float64}

    cz::CxxVector{Float64}
    cy::CxxVector{Float64}

    Y::CxxMatrix{Float64}
    J::CxxMatrix{Float64}
    Z::CxxMatrix{Float64}
    L::CxxMatrix{Float64}

    function AbsNormalForm(
        tape_id::Int64, m::Int64, n::Int64, x::Vector{Float64}, y::Vector{Float64}
    )
        num_switches = TbadoubleModule.get_num_switches(tape_id)
        z = CxxVector{Float64}(num_switches)

        cz = CxxVector{Float64}(num_switches)
        cy = CxxVector{Float64}(length(y))

        Y = CxxMatrix{Float64}(length(y), length(x))
        J = CxxMatrix{Float64}(length(y), num_switches)
        Z = CxxMatrix{Float64}(num_switches, length(x))
        L = CxxMatrix{Float64}(num_switches, num_switches)

        problem = new(
            tape_id,
            m,
            n,
            num_switches,
            CxxVector{Float64}(x),
            CxxVector{Float64}(y),
            z,
            cz,
            cy,
            Y,
            J,
            Z,
            L,
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
    z_cxx::CxxVector{Float64},
    cz_cxx::CxxVector{Float64},
    cy_cxx::CxxVector{Float64},
    Y_cxx::CxxMatrix{Float64},
    J_cxx::CxxMatrix{Float64},
    Z_cxx::CxxMatrix{Float64},
    L_cxx::CxxMatrix{Float64},
    m::Int64,
    n::Int64,
    num_switches::Int32,
    x_cxx::CxxVector{Float64},
    y_cxx::CxxVector{Float64},
)

    # use c++ double*
    cz = cz_cxx.data
    cy = cy_cxx.data
    x = x_cxx.data
    y = y_cxx.data
    z = z_cxx.data

    # use the c++ double**
    Y = Y_cxx.data
    J = J_cxx.data
    Z = Z_cxx.data
    L = L_cxx.data

    return TbadoubleModule.abs_normal(
        tape_id, m, n, num_switches, x, y, z, cz, cy, Y, J, Z, L
    )
end

function abs_normal!(
    tape_id::Int64,
    cz::Vector{Float64},
    cy::Vector{Float64},
    Y::Matrix{Float64},
    J::Matrix{Float64},
    Z::Matrix{Float64},
    L::Matrix{Float64},
    m::Int64,
    n::Int64,
    num_switches::Int64,
    x::Vector{Float64},
    y::Vector{Float64},
    z::Vector{Float64},
)

    # julia matrix to c++ matrix
    Y_cxx = CxxMatrix(Y)
    J_cxx = CxxMatrix(J)
    Z_cxx = CxxMatrix(Z)
    L_cxx = CxxMatrix(L)

    return abs_normal!(
        tape_id, cz, cy, Y_cxx, J_cxx, Z_cxx, L_cxx, m, n, num_switches, x, y, z
    )
end
