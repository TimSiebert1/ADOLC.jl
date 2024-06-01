
function check_input_taylor_coeff(
    num_independents,
    derivative_order::Int64;
    num_directions = nothing,
    init_series = nothing,
)
    if num_directions == 1
        if derivative_order == 1
            @assert(
                init_series !== nothing,
                "For derivative_order=$(derivative_order) and 
                num_directions=$(num_directions) you have to provide an 
                init_series (as a vector) to initialize the taylor series 
                propagation!"
            )

            @assert(
                length(size(init_series)) == 1,
                "Please provide the init_series with length(size(init_series))=1
                and not $(length(size(init_series)))."
            )

            @assert(
                size(init_series)[1] == num_independents,
                "Please provide a init_series of length 
                $(num_independents) to initialize the taylor series 
                propagation. Each entry corresponds to one independant."
            )

        else
            if init_series !== nothing
                @assert(
                    length(size(init_series)) == 2,
                    "Please provide the init_series with length(size(init_series))==2
                    and not $(length(size(init_series)))."
                )
                @assert size(
                    init_series == (num_independents, derivative_order),
                    "The init_series has the wrong shape: $(size(init_series)) but must be
                    ($(num_independents), $(derivative_order)). Please provide the taylor 
                    coefficients of the init_series up to order derivative_order-1. 
                    In detail init_series must have the shape (num_independents, derivative_order)
                    and the i-th column corresponds to the i-1-th taylor coefficient 
                    of the init_series.",
                )
            end
        end
    end
    @assert(
        init_series !== nothing,
        "For derivative_order=$(derivative_order) you have to provide 
        an init_series (as a vector) to initialize the taylor series 
        propagation!"
    )
    @assert(
        length(size(init_series)) == derivative_order,
        "The input for init_series has the wrong shape! Please provide
        a vector of $(num_independents) to initialize the taylor
        series propagation. Each entry corresponds to one independant"
    )
end


function taylor_coeff(
    func,
    init_point,
    num_dependents,
    num_independents,
    derivative_order;
    num_directions = nothing,
    init_series = nothing,
)


    a = [Adouble{TbAlloc}() for _ in eachindex(init_point)]
    y0 = Vector{Float64}(undef, num_dependents)
    tape_num = 1
    keep = 0
    trace_on(tape_num, keep)
    a << init_point
    b = func(a)
    b >> y0
    trace_off(0)

    """
    check_input_taylor_coeff(num_independents, 
                            derivative_order,
                            num_directions=num_directions,
                            init_series=init_series)
    """

    if num_directions === nothing
        num_directions = num_independents

    elseif num_directions == 1
        if derivative_order == 1
            y1 = Vector{Float64}(undef, 2)
            fos_forward(
                tape_num,
                num_dependents,
                num_independents,
                keep,
                init_point,
                init_series,
                y0,
                y1,
            )
            return y0, y1
        else
            Y = myalloc2(num_dependents, derivative_order)
            hos_forward(
                tape_num,
                num_dependents,
                num_independents,
                derivative_order,
                0,
                init_point,
                init_series,
                y0,
                Y,
            )
            return y0, Y
        end
    else
        if derivative_order == 1
            if init_series === nothing
                init_series = myalloc2(num_independents, num_directions)
                for i = 1:num_independents
                    for j = 1:num_directions
                        init_series[i, j] = 0.0
                        if i == j
                            init_series[i, i] = 1.0
                        end
                    end
                end
            end
            Y = myalloc2(num_dependents, num_directions)
            fov_forward(
                tape_num,
                num_dependents,
                num_independents,
                num_directions,
                init_point,
                init_series,
                y0,
                Y,
            )
            return y0, Y
        else
            if init_series === nothing
                init_series = myalloc3(num_independents, num_directions, derivative_order)
                for i = 1:num_independents
                    for j = 1:derivative_order
                        for k = 1:num_directions
                            init_series[i, j, k] = 0.0
                        end
                    end
                end
                for k = 1:num_directions
                    init_series[k, k, 1] = 1.0
                end
            end
            Y = myalloc3(num_dependents, num_directions, derivative_order)
            hov_forward(
                tape_num,
                num_dependents,
                num_independents,
                derivative_order,
                num_independents,
                init_point,
                init_series,
                y0,
                Y,
            )
            return y0, Y
        end
    end
end
