module ADOLC_wrap

include("array_types.jl")
include("AdoubleModule.jl")
include("TladoubleModule.jl")


using Main.ADOLC_wrap.array_types
using Main.ADOLC_wrap.AdoubleModule
using Main.ADOLC_wrap.TladoubleModule

struct AbsNormalProblem{T}
    m::Int64
    n::Int64
    num_switches::Int32

    x::CxxVector{T}
    y::CxxVector{T}
    z::CxxVector{T}

    cz::CxxVector{T}
    cy::CxxVector{T}

    Y::CxxMatrix{T} 
    J::CxxMatrix{T}
    Z::CxxMatrix{T} 
    L::CxxMatrix{T}


    function AbsNormalProblem{T}(tape_num::Int64, m::Int64, n::Int64, x::Vector{T}, y::Vector{T}) where T <: Real
        
        num_switches = AdoubleModule.get_num_switches(tape_num)
        z = CxxVector{Float64}(num_switches)

        cz = CxxVector{Float64}(num_switches)
        cy = CxxVector{Float64}(length(y))

        Y = CxxMatrix{Float64}(length(y), length(x))
        J = CxxMatrix{Float64}(length(y), num_switches)
        Z = CxxMatrix{Float64}(num_switches, length(x))
        L = CxxMatrix{Float64}(num_switches, num_switches)

        new{T}(m, n, num_switches, CxxVector{T}(x), CxxVector{T}(y), z, cz, cy, Y, J, Z, L)
    end
end
function abs_normal!(abs_normal_problem::AbsNormalProblem{T}, tape_num::Int64) where T <: Real
    _abs_normal!(
    abs_normal_problem.z,
    abs_normal_problem.cz,
    abs_normal_problem.cy,
    abs_normal_problem.Y, 
    abs_normal_problem.J,
    abs_normal_problem.Z, 
    abs_normal_problem.L,
    abs_normal_problem.m,
    abs_normal_problem.n,
    abs_normal_problem.num_switches,
    tape_num,
    abs_normal_problem.x,
    abs_normal_problem.y)
end

function _abs_normal!(
    z_cxx::CxxVector{T},
    cz_cxx::CxxVector{T},
    cy_cxx::CxxVector{T},
    Y_cxx::CxxMatrix{T}, 
    J_cxx::CxxMatrix{T},
    Z_cxx::CxxMatrix{T}, 
    L_cxx::CxxMatrix{T},
    m::Int64,
    n::Int64,
    num_switches::Int32,
    tape_num::Int64,
    x_cxx::CxxVector{T},
    y_cxx::CxxVector{T}
) where T <: Real

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

    AdoubleModule.abs_normal(tape_num, m, n, num_switches, x, y, z, cz, cy, Y, J, Z, L)
end


function abs_normal!(
    cz::Vector{Float64},
    cy::Vector{Float64},
    Y::Matrix{Float64}, 
    J::Matrix{Float64},
    Z::Matrix{Float64}, 
    L::Matrix{Float64},
    tape_num::Int64,
    m::Int64,
    n::Int64,
    num_switches::Int64,
    x::Vector{Float64},
    y::Vector{Float64},
    z::Vector{Float64}
)

    # julia matrix to c++ matrix
    Y_cxx = CxxMatrix(Y)
    J_cxx = CxxMatrix(J)
    Z_cxx = CxxMatrix(Z)
    L_cxx = CxxMatrix(L)

    abs_normal!(cz, cy, Y_cxx, J_cxx, Z_cxx, L_cxx, tape_num, m, n, num_switches, x, y, z)
end


function _gradient_tape_less(func, init_point::Vector{Float64})
    a = TladoubleModule.tladouble_vector_init(init_point)
    b = func(a)
    return TladoubleModule.get_gradient(b, length(init_point))
end

function _gradient_tape_based(func, init_point::Vector{Float64}, num_dependent::Int64, derivative_order::Int64)

    num_independent = length(init_point)
    y = Vector{Float64}(undef, num_dependent)
    a = [AdoubleModule.AdoubleCxx() for _ in 1:num_independent]
    tape_num = 1
    keep = 1
    trace_on(tape_num, keep)
    a << init_point
    b = func(a)
    b >> y
    trace_off(0)

    if derivative_order == 1
        
        U = myalloc2(num_dependent, num_dependent)
        for i in 1:num_dependent
            for j in 1:num_dependent
                U[i, j] = 0.0
                if i == j
                    U[i, i] = 1.0
                end
            end
        end

        Z = myalloc2(num_dependent, num_independent)
        fov_reverse(tape_num, num_dependent, num_independent, num_dependent, U, Z) 
        return Z

    else
        output = Vector{Any}(undef, num_independent)
        
        for direction in 1:num_independent
            # forward until last derivative_order
            X = myalloc2(num_independent, derivative_order)
            for i in 1:num_independent
                for j in 1:derivative_order
                    X[i, j] = 0.0
                end
            end
            X[direction, 1] = 1.0
            Y = myalloc2(num_dependent, derivative_order)
            hos_forward(tape_num, num_dependent, num_independent, 1, derivative_order+1, init_point, X, y, Y) 

            U = myalloc2(num_dependent, num_dependent)
            for i in 1:num_dependent
                for j in 1:num_dependent
                    U[i, j] = 0.0
                    if i == j 
                        U[i, i] = 1.0
                    end
                end
            end

            Z = myalloc3(num_dependent, num_independent, derivative_order+1)
            nz = alloc_mat_short(num_dependent, num_independent)
            hov_reverse(tape_num, num_dependent, num_independent, derivative_order, num_dependent, U, Z, nz) 
            output[direction] = Z
        end
        return output

    end
end


function gradient(func, init_point::Vector{Float64}, num_dependent::Int64; switch_point::Int64=100, mode=nothing, derivative_order::Int64=1)

    if mode === :tape_less
        return _gradient_tape_less(func, init_point)
    elseif mode === :tape_based
        return _gradient_tape_based(func, init_point, num_dependent, derivative_order)

    else 
        if mode === nothing
            mode = length(init_point) < switch_point ? :tape_less : :tape_based
            mode = derivative_order > 1 ? :tape_based : mode
            return gradient(func, init_point, num_dependent, switch_point=switch_point, mode=mode, derivative_order=derivative_order)
        else
            throw("Mode $(mode) is not implemented!")
        end
    end
end

export abs_normal!, AbsNormalProblem, gradient, _gradient_tape_based, _gradient_tape_less

end # module ADOLC_wrap
