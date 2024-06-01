
function tensor_address(degree::Int64, adolc_partial::Vector{Int32})
    # "+1" because c++ indexing is -1
    return Int64(TbadoubleModule.tensor_address(degree, adolc_partial)) + 1
end

function partial_to_adolc_scheme!(res::Vector{Int32}, partial::Vector{Int64}, degree::Int64)
    idx = 1
    for i in eachindex(partial)
        for _ = 1:partial[i]
            res[idx] = i
            idx += 1
        end
    end
    for i in idx:degree
        res[i] = 0
    end
    sort!(res, rev = true)
end

function create_cxx_identity(n::Int64, m::Int64)
    I = myalloc2(n, m)
    for i = 1:n
        for j = 1:m
            I[i, j] = 0.0
            if i == j
                I[i, i] = 1.0
            end
        end
    end
    return I
end

function create_partial_cxx_identity(n::Int64, idxs::Vector{Int64})
    m = length(idxs)
    I = myalloc2(n, m)
    for j = 1:m
        for i = 1:n
            I[i, j] = 0.0
        end
        I[idxs[j], j] = 1.0
    end
    return I
end

function partials_to_seed_space(partials::Vector{Vector{Int64}}, seed_idxs::Vector{Int64})
    new_partials = Vector{Vector{Int64}}(undef, length(partials))
    for (i, partial) in enumerate(partials)
        new_partials[i] = zeros(length(seed_idxs))
        for j in eachindex(partial)
            if partial[j] != 0
                new_partials[i][indexin(j, seed_idxs)[1]] = partial[j]
            end
        end
    end
    return new_partials
end

function adolc_scheme_to_seed_space(partials::Vector{Vector{Int64}}, seed_idxs::Vector{Int64})
    new_partials = Vector{Vector{Int64}}(undef, length(partials))
    for (i, partial) in enumerate(partials)
        new_partials[i] = zeros(length(partial))
        for j in eachindex(partial)
            if partial[j] != 0
                new_partials[i][j] = indexin(partial[j], seed_idxs)[1]
            else # since adolc_scheme is sorted, first zero means everything afterward is zero
                break
            end
        end
    end
    return new_partials
end

function get_seed_idxs(partials::Vector{Vector{Int64}})
    seed_idxs = Vector{Int64}()
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

function get_seed_idxs_adolc_scheme(partials::Vector{Vector{Int64}})
    seed_idxs = Vector{Int64}()
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

function build_tensor(
    derivative_order::Int64,
    num_dependents::Int64,
    num_independents::Int64,
    CxxTensor,
)

    # allocate the output (julia) tensor 
    tensor = Array{Float64}(
        undef,
        [num_independents for _ = 1:derivative_order]...,
        num_dependents,
    )


    # creates all index-pairs; the i-th entry specifies the i-th directional derivative w.r.t x_i
    # e.g. (1, 1, 3, 4) gives the derivative w.r.t x_1, x_1, x_3, x_4
    # this is used as index for the tensor and to get the address from the compressed vector
    idxs = vec(
        collect(
            Iterators.product(Iterators.repeated(1:num_independents, derivative_order)...),
        ),
    )

    # build the tensor
    for idx in idxs
        for component = 1:num_dependents
            tensor[idx..., component] =
                CxxTensor[component, tensor_address2(derivative_order, idx)]
        end
    end
    return tensor
end


