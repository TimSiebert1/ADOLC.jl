using ADOLC

function demo_reuse_tape()
    function f(x)
        return x[1]^2 * x[2]
    end

    x0 = [1.0, 2.0]
    dir = [1.0, 2.0]

    res = [0.0]
    m = 1
    n = 2
    tape_id = 1
    max_iters = 100
    for i = 1:max_iters
        if i == 1
            derivative!(res, f, m, n, x0, :jac_vec, dir = dir, tape_id = tape_id)
        else
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
        end
        # do computations ....
        # update x0 ....
        # update dir ....
    end
end

demo_reuse_tape()
