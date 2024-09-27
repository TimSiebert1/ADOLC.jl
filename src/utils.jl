"""
    tensor_address(degree::Integer, adolc_partial::Vector{Integer})
    tensor_address(degree::Integer, adolc_partial::Vector{Cint})

Generates the index (address) of the mixed-partial specified by `adolc_partial`
in an higher-order derivative tensor of derivative order `degree`.

!!! note 
    The partial has to be in [ADOLC-Format](@ref).
"""
function tensor_address(degree, adolc_partial)
    return tensor_address(degree, convert(Vector{Cint}, adolc_partial))
end

function tensor_address(degree::Integer, adolc_partial::Vector{Cint})
    # "+1" because c++ indexing is -1
    return ccall(
        (:tensor_address, ADOLC_JLL_PATH), Cint, (Cint, Ptr{Cint}), degree, adolc_partial
    ) + 1
end

"""
    partial_to_adolc_format(partial::Vector{I_1}, degree::I_2) where {I_1<:Integer, I_2<:Integer}

Transforms a given `partial` to the [ADOLC-Format](@ref). 

!!! note

    `partial` is required to be in the `Partial-format`

# Example:
```jldoctest

partial = [1, 0, 4]
degree = sum(partial)
partial_to_adolc_format(partial, degree)

# output

5-element Vector{Int32}:
 3
 3
 3
 3
 1
```
"""
function partial_to_adolc_format(
    partial::Vector{I_1}, degree::I_2
) where {I_1<:Integer,I_2<:Integer}
    res = Vector{Cint}(undef, degree)
    partial_to_adolc_format!(res, partial, degree)
    return res
end

"""
    partial_to_adolc_format!(res::Vector{Cint}, partial::Vector{I_1}, degree::I_2) where {I_1<:Integer, I_2<:Integer}
    partial_to_adolc_format!(res::Vector{Cint}, partial::Vector{Cint}, degree::I) where I <: Integer

    
Variant of [`partial_to_adolc_format`](@ref) that writes the result to `res`.
    

# Example:
```jldoctest
partial = [1, 3, 2, 0]
degree = sum(partial)
res = zeros(Int32, degree)
partial_to_adolc_format!(res, partial, degree)

# output

6-element Vector{Int32}:
 3
 3
 2
 2
 2
 1
```
"""
function partial_to_adolc_format!(
    res::Vector{Cint}, partial::Vector{I_1}, degree::I_2
) where {I_1<:Integer,I_2<:Integer}
    return partial_to_adolc_format!(res, convert(Vector{Cint}, partial), degree)
end

function partial_to_adolc_format!(
    res::Vector{Cint}, partial::Vector{Cint}, degree::I
) where {I<:Integer}
    idx = 1
    for i in eachindex(partial)
        for _ in 1:partial[i]
            res[idx] = i
            idx += 1
        end
    end
    for i in idx:degree
        res[i] = 0
    end
    return sort!(res; rev=true)
end

"""
    seed_idxs_partial_format(partials::Vector{Vector{I}}) where I <: Integer

Extracts the actually required derivative directions of `partials` and returns them 
ascendet sorted. 

!!! note
    `partials` has to be in [Partial-Format](@ref).

# Example
```jldoctest

partials = [[1, 0, 0, 0, 3], [1, 0, 1, 0, 0], [0, 0, 3, 0, 0]]
seed_idxs_partial_format(partials)

# output

3-element Vector{Int64}:
 1
 3
 5
```
"""
function seed_idxs_partial_format(partials::Vector{Vector{I}}) where {I<:Integer}
    seed_idxs = Vector{I}()
    for partial in partials
        for i in eachindex(partial)
            if partial[i] != 0
                if !(i in seed_idxs)
                    push!(seed_idxs, i)
                end
            end
        end
    end
    sort!(seed_idxs)
    return seed_idxs
end

"""
    seed_idxs_adolc_format(partials::Vector{Vector{I}}) where I <: Integer


Extracts the actually required derivative directions of `partials` and returns them 
ascendet sorted. 

!!! note

    `partials` has to be in [ADOLC-Format](@ref).

# Example
```jldoctest

partials = [[5, 5, 5, 1], [3, 1, 0, 0], [3, 3, 3, 0]]
seed_idxs_adolc_format(partials)

# output

3-element Vector{Int64}:
 1
 3
 5
```
"""
function seed_idxs_adolc_format(partials::Vector{Vector{I}}) where {I<:Integer}
    seed_idxs = Vector{I}()
    for partial in partials
        for i in partial
            if i != 0
                if !(i in seed_idxs)
                    push!(seed_idxs, i)
                end
            end
        end
    end
    sort!(seed_idxs)
    return seed_idxs
end
"""
    partial_format_to_seed_space(partials::Vector{Vector{I_1}}, seed_idxs::Vector{I_2}) where {I_1 <: Integer, I_2 <: Integer}
    partial_format_to_seed_space(partials::Vector{Vector{I}}) where I <: Integer

Converts `partials` in [Partial-Format](@ref) to `partials` of the same format but with (possible) reduced number 
of derivatives directions. The `seed_idxs` is expected to store the result of [`seed_idxs_partial_format(seed_idxs)`](@ref).
Without `seed_idxs` the function first calls [`seed_idxs_partial_format(seed_idxs)`](@ref) to get the indices.

# Examples
```jldoctest

partials = [[0, 1, 1], [0, 2, 0]]
seed_idxs = seed_idxs_partial_format(partials)
partial_format_to_seed_space(partials, seed_idxs)

# output

2-element Vector{Vector{Int64}}:
 [1, 1]
 [2, 0]
```
Without `seed_idxs`
```jldoctest

partials = [[0, 1, 1], [0, 2, 0]]
partial_format_to_seed_space(partials)

# output

2-element Vector{Vector{Int64}}:
 [1, 1]
 [2, 0]
```
"""
function partial_format_to_seed_space(
    partials::Vector{Vector{I_1}}, seed_idxs::Vector{I_2}
) where {I_1<:Integer,I_2<:Integer}
    seed_space_partials = Vector{Vector{Int64}}(undef, length(partials))
    for (i, partial) in enumerate(partials)
        seed_space_partials[i] = zeros(length(seed_idxs))
        for j in eachindex(partial)
            if partial[j] != 0
                seed_space_partials[i][indexin(j, seed_idxs)[1]] = partial[j]
            end
        end
    end
    return seed_space_partials
end

function partial_format_to_seed_space(partials::Vector{Vector{I}}) where {I<:Integer}
    seed_idxs = seed_idxs_partial_format(partials)
    return partial_format_to_seed_space(partials, seed_idxs)
end

"""
    adolc_format_to_seed_space(partials::Vector{Vector{I_1}}, seed_idxs::Vector{I_2}) where {I_1 <: Integer, I_2 <: Integer}
    adolc_format_to_seed_space(partials::Vector{Vector{I}}) where I <: Integer

Same as [`partial_format_to_seed_space`](@ref) but with [ADOLC-Format](@ref).

# Examples
```jldoctest

partials = [[3, 2], [2, 2]]
seed_idxs = seed_idxs_adolc_format(partials)
adolc_format_to_seed_space(partials, seed_idxs)

# output

2-element Vector{Vector{Int64}}:
 [2, 1]
 [1, 1]
```
Without `seed_idxs`
```jldoctest

partials = [[3, 2], [2, 2]]
seed_idxs = seed_idxs_adolc_format(partials)
adolc_format_to_seed_space(partials, seed_idxs)

# output

2-element Vector{Vector{Int64}}:
 [2, 1]
 [1, 1]
```
"""
function adolc_format_to_seed_space(
    partials::Vector{Vector{I_1}}, seed_idxs::Vector{I_2}
) where {I_1<:Integer,I_2<:Integer}
    new_partials = Vector{Vector{Int64}}(undef, length(partials))
    for (i, partial) in enumerate(partials)
        new_partials[i] = zeros(length(partial))
        for j in eachindex(partial)
            if partial[j] != 0
                new_partials[i][j] = indexin(partial[j], seed_idxs)[1]
            else # since adolc_format is sorted, first zero means everything afterwards is zero
                break
            end
        end
    end
    return new_partials
end

function adolc_format_to_seed_space(partials::Vector{Vector{I}}) where {I<:Integer}
    seed_idxs = seed_idxs_adolc_format(partials)
    return adolc_format_to_seed_space(partials, seed_idxs)
end

"""
    allocator(m::Integer, n::Integer, mode::Symbol, num_dir::Integer, num_weights::Integer)

"""
function allocator(tape_id, m, n, mode, num_dir, num_weights, x)
    if mode === :jac
        if m > 1
            return CxxMatrix(m, n)
        else
            return CxxVector(n)
        end
    elseif mode === :hess
        return CxxTensor(m, n, n)
    elseif mode === :jac_vec
        return CxxVector(m)
    elseif mode === :jac_mat
        return CxxMatrix(m, num_dir)
    elseif mode === :vec_jac
        return CxxVector(n)
    elseif mode === :mat_jac
        return CxxMatrix(num_weights, n)

    elseif mode === :hess_vec
        return CxxMatrix(m, n)
    elseif mode === :hess_mat
        return CxxTensor(m, n, num_dir)
    elseif mode === :vec_hess
        return CxxMatrix(n, n)
    elseif mode === :mat_hess
        return CxxTensor(num_weights, n, n)
    elseif mode === :vec_hess_vec
        return CxxVector(n)
    elseif mode === :mat_hess_vec
        return CxxMatrix(num_weights, n)
    elseif mode === :vec_hess_mat
        return CxxMatrix(n, num_dir)
    elseif mode === :mat_hess_mat
        return CxxTensor(num_weights, n, num_dir)
    elseif mode === :abs_normal
        return init_abs_normal_form(tape_id, x)
    end
end

"""
    jl_allocator(m::Int64, n::Int64, mode::Symbol, num_dir::Int64, num_weights::Int64)

"""
function jl_allocator(m, n, mode, num_dir, num_weights)
    if mode === :jac
        if m > 1
            return Matrix{Float64}(undef, m, n)
        else
            return Vector{Float64}(undef, n)
        end
    elseif mode === :hess
        return Array{Float64}(undef, m, n, n)
    elseif mode === :jac_vec
        return Vector{Float64}(undef, m)
    elseif mode === :jac_mat
        return Matrix{Float64}(undef, m, num_dir)
    elseif mode === :vec_jac
        return Vector{Float64}(undef, n)
    elseif mode === :mat_jac
        return Matrix{Float64}(undef, num_weights, n)

    elseif mode === :hess_vec
        return Matrix{Float64}(undef, m, n)
    elseif mode === :hess_mat
        return Array{Float64}(undef, m, n, num_dir)
    elseif mode === :vec_hess
        return Matrix{Float64}(undef, n, n)
    elseif mode === :mat_hess
        return Array{Float64}(undef, num_weights, n, n)
    elseif mode === :vec_hess_vec
        return Vector{Float64}(undef, n)
    elseif mode === :mat_hess_vec
        return Matrix{Float64}(undef, num_weights, n)
    elseif mode === :vec_hess_mat
        return Matrix{Float64}(undef, n, num_dir)
    elseif mode === :mat_hess_mat
        return Array{Float64}(undef, num_weights, n, num_dir)
    end
end
