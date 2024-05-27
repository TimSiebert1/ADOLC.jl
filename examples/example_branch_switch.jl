using ADOLC

function demo_reuse_tape()
    function f(x)
        if x[1] == 1.0
            return x[1]^2 * x[2]
        else 
            return 1
        end
    end

    x0 = [1.0, 2.0]
    dir = [1.0, 2.0]

    res = [0.0]
    m = 1
    n = 2
    tape_id = 1

    derivative!(res, f, m, n, x0, :jac_vec, dir = dir, tape_id = tape_id)
    println(res[1])
    x0 = [-2.0, 4.0]
    derivative!(
                res,
                f,
                m,
                n,
                x0,
                :jac_vec,
                dir = dir,
                tape_id = tape_id,
                reuse_tape = true,
            )
    println(res[1])
end

demo_reuse_tape()
