function func(x)
    return [
        x[1] * x[2]^3,
        x[1] + x[3] / x[2]
    ]
end

x0 = [-1.0, 2.0, -1.0]
a = [Adouble{TbAlloc}() for _ in eachindex(x0)]
y0 = Vector{Float64}(undef, 2)
n = length(x0)
m = length(y0)
tape_num = 1
keep = 1
trace_on(tape_num, keep)
a << x0
b = func(a)
b >> y0
trace_off(0)

u = [1.0, 0.0]
z = Vector{Float64}(undef, 3)
fos_reverse(tape_num, m, n, u, z) 


@test z[1] == 8.0
@test z[2] == -12.0
@test z[3] == 0.0


u = [0.0, 1.0]
z = Vector{Float64}(undef, 3)
fos_reverse(tape_num, m, n, u, z)

@test z[1] == 1.0
@test z[2] == 0.25
@test z[3] == 0.5


q = 2
U = myalloc2(q, m)
for i in 1:q
    for j in 1:m
        U[i, j] = 0.0
        if i == j
            U[i, i] = 1.0
        end
    end
end

Z = myalloc2(q, n)

fov_reverse(tape_num, m, n, q, U, Z) 

@test Z[1, 1] == 8.0
@test Z[2, 1] == 1.0
@test Z[1, 2] == -12.0
@test Z[2, 2] == 0.25
@test Z[1, 3] == 0.0
@test Z[2, 3] == 0.5




d = 1
# preprun

X = myalloc2(n, d)
for i in 1:n
    for j in 1:d
        X[i, j] = 0.0
    end
end
X[1, 1] = 1.0
Y = myalloc2(m, d)
hos_forward(tape_num, m, n, 1, d+1, x0, X, y0, Y) 


u = [1.0, 0.0]
Z = myalloc2(n, d + 1)
hos_reverse(tape_num, m, n, d, u, Z) 

@test Z[1, 1] == 8.0
@test Z[2, 1] == -12.0
@test Z[3, 1] == 0.0
@test Z[1, 2] == 0.0
@test Z[2, 2] == 12.0
@test Z[3, 2] == 0.0


u = [0.0, 1.0]
Z = myalloc2(n, d + 1)
hos_reverse(tape_num, m, n, d, u, Z) 

@test Z[1, 1] == 1.0
@test Z[2, 1] == 0.25
@test Z[3, 1] == 0.5
@test Z[1, 2] == 0.0
@test Z[2, 2] == 0.0
@test Z[3, 2] == 0.0



d = 1
q = 2
U = myalloc2(q, m)
for i in 1:q
    for j in 1:m
        U[i, j] = 0.0
        if i == j 
            U[i, i] = 1.0
        end
    end
end

Z = myalloc3(q, n, d+1)
nz = alloc_mat_short(q, n)
hov_reverse(tape_num, m, n, d, q, U, Z, nz) 


@test Z[1, 1, 1] == 8.0
@test Z[1, 2, 1] == -12.0
@test Z[1, 3, 1] == 0.0
@test Z[1, 1, 2] == 0.0
@test Z[1, 2, 2] == 12.0
@test Z[1, 3, 2] == 0.0

@test Z[2, 1, 1] == 1.0
@test Z[2, 2, 1] == 0.25
@test Z[2, 3, 1] == 0.5
@test Z[2, 1, 2] == 0.0
@test Z[2, 2, 2] == 0.0
@test Z[2, 3, 2] == 0.0


X = myalloc2(n, d)
for i in 1:n
    for j in 1:d
        X[i, j] = 0.0
    end
end
X[2, 1] = 1.0
Y = myalloc2(m, d)
hos_forward(tape_num, m, n, d, d+1, x0, X, y0, Y) 


d = 1
q = 2
U = myalloc2(q, m)
for i in 1:q
    for j in 1:m
        U[i, j] = 0.0
        if i == j 
            U[i, i] = 1.0
        end
    end
end

Z = myalloc3(q, n, d+1)
nz = alloc_mat_short(q, n)
hov_reverse(tape_num, m, n, d, q, U, Z, nz) 


@test Z[1, 1, 1] == 8.0
@test Z[1, 2, 1] == -12.0
@test Z[1, 3, 1] == 0.0

@test Z[1, 1, 2] == 12.0
@test Z[1, 2, 2] == -12.0
@test Z[1, 3, 2] == 0.0

@test Z[2, 1, 1] == 1.0
@test Z[2, 2, 1] == 0.25
@test Z[2, 3, 1] == 0.5

@test Z[2, 1, 2] == 0.0
@test Z[2, 2, 2] == -0.25
@test Z[2, 3, 2] == -0.25

println("Done")