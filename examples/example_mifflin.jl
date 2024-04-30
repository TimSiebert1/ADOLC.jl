using ADOLC

function demo_abs_normal()
    # Define the Mifflin 2 function
    function f(x)
        return -x[1] + 2 * (x[1]^2 + x[2]^2 - 1) + 1.75 * abs(x[1]^2 + x[2]^2 - 1)
    end

    # Define the derivative evaluation point x
    x = [-1.0, -1.0]

    # Initialize the AbsNormalForm object
    abs_normal_form = ADOLC.init_abs_normal_form(f, 1, 2, x, tape_id = 1)

    # Calculate the absolute normal form derivative
    derivative!(
        abs_normal_form,
        f,
        1,
        2,
        x,
        :abs_normal,
        tape_id = abs_normal_form.tape_id,
        reuse_tape = true,
    )

    println("AbsNormalForm at $x: ", abs_normal_form)

    # some computations
    # ....

    # new evaluation point
    x = [-0.5, 1.0]
    derivative!(
        abs_normal_form,
        f,
        1,
        2,
        x,
        :abs_normal,
        tape_id = abs_normal_form.tape_id,
        reuse_tape = true,
    )

    println("AbsNormalForm at $x: ", abs_normal_form)
end
demo_abs_normal()
