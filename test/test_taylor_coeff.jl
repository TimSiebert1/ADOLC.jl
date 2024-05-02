
function func(x)
    return [x[1] * x[2]^3, x[1] + x[3] / x[2]]
end

x0 = [-1.0, 2.0, -1.0]






###### test 1 order scalar #######

x1 = [1.0, 0.0, 0.0]
x0 = [-1.0, 2.0, -1.0]
y0, y1 = taylor_coeff(func, x0, 2, 3, 1, init_series = x1, num_directions = 1)

@test y0[1] == -8.0
@test y0[2] == -1.5

@test y1[1] == 8.0
@test y1[2] == 1.0

x1 = [0.0, 1.0, 0.0]
y1 = Vector{Float64}(undef, 2)
y0, y1 = taylor_coeff(func, x0, 2, 3, 1, init_series = x1, num_directions = 1)

@test y0[1] == -8.0
@test y0[2] == -1.5

@test y1[1] == -12.0
@test y1[2] == 0.25


x1 = [0.0, 0.0, 1.0]
y1 = Vector{Float64}(undef, 2)
y0, y1 = taylor_coeff(func, x0, 2, 3, 1, init_series = x1, num_directions = 1)

@test y0[1] == -8.0
@test y0[2] == -1.5

@test y1[1] == 0.0
@test y1[2] == 0.5

####### test 1st order vector ########

m = 2
n = 3
p = 3
X = myalloc2(n, p)
for i = 1:n
    for j = 1:p
        X[i, j] = 0.0
        if i == j
            X[i, i] = 1.0
        end
    end
end

y0, Y = taylor_coeff(func, x0, m, n, 1, num_directions = p, init_series = X)


@test y0[1] == -8.0
@test y0[2] == -1.5


@test Y[1, 1] == 8.0
@test Y[2, 1] == 1.0
@test Y[1, 2] == -12.0
@test Y[2, 2] == 0.25
@test Y[1, 3] == 0.0
@test Y[2, 3] == 0.5



y0, Y = taylor_coeff(func, x0, m, n, 1, num_directions = p)
@test y0[1] == -8.0
@test y0[2] == -1.5


@test Y[1, 1] == 8.0
@test Y[2, 1] == 1.0
@test Y[1, 2] == -12.0
@test Y[2, 2] == 0.25
@test Y[1, 3] == 0.0
@test Y[2, 3] == 0.5


####### test higher order scalar #######



derivative_order = 3
X = myalloc2(n, derivative_order)
for i = 1:n
    for j = 1:derivative_order
        X[i, j] = 0.0
    end
end
X[2, 1] = 1.0
y0, Y = taylor_coeff(func, x0, m, n, derivative_order, num_directions = 1, init_series = X)



@test y0[1] == -8.0
@test y0[2] == -1.5

@test Y[1, 1] == -12.0
@test Y[1, 2] == -6.0
@test Y[1, 3] == -1.0
@test Y[2, 1] == 0.25
@test Y[2, 2] == -1 / 8
@test Y[2, 3] == 1 / 16



######### test higher order vector ########

derivative_order = 3
num_directions = 3
# first dim number of independants
# second dim number of directions
# third dim number of dervatives
X = myalloc3(n, num_directions, derivative_order)
for i = 1:n
    for j = 1:derivative_order
        for k = 1:num_directions
            X[i, j, k] = 0.0
        end
    end
end

# for the ith direction the ith component is set to 1
# this gives the partials w.r.t to ith component
for k = 1:num_directions
    X[k, k, 1] = 1.0
end


y0, Y = taylor_coeff(
    func,
    x0,
    m,
    n,
    derivative_order,
    num_directions = num_directions,
    init_series = X,
)


@test y0[1] == -8.0
@test y0[2] == -1.5



@test Y[1, 1, 1] == 8.0
@test Y[1, 1, 2] == 0.0
@test Y[1, 1, 3] == 0.0
@test Y[2, 1, 1] == 1.0
@test Y[2, 1, 2] == 0.0
@test Y[2, 1, 3] == 0.0


@test Y[1, 2, 1] == -12.0
@test Y[1, 2, 2] == -6.0
@test Y[1, 2, 3] == -1.0
@test Y[2, 2, 1] == 0.25
@test Y[2, 2, 2] == -1 / 8
@test Y[2, 2, 3] == 1 / 16


@test Y[1, 3, 1] == 0.0
@test Y[1, 3, 2] == 0.0
@test Y[1, 3, 3] == 0.0
@test Y[2, 3, 1] == 0.5
@test Y[2, 3, 2] == 0.0
@test Y[2, 3, 3] == 0.0

println("Done")
