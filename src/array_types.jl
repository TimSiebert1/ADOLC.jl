module array_types
using ADOLC_jll
using CxxWrap

@wrapmodule(() -> libadolc_wrap, :julia_module_array_types)

function __init__()
    @initcxx
end

function Base.getindex(cxx_ptr_ptr_ptr::CxxPtr{CxxPtr{CxxPtr{Cdouble}}}, dim::Int64)
    return getindex_tens(cxx_ptr_ptr_ptr, dim)
end
function Base.getindex(
    cxx_ptr_ptr_ptr::CxxPtr{CxxPtr{CxxPtr{Cdouble}}}, dim::Int64, row::Int64, col::Int64
)
    return getindex_tens(cxx_ptr_ptr_ptr, dim, row, col)
end
function Base.setindex!(
    cxx_ptr_ptr_ptr::CxxPtr{CxxPtr{CxxPtr{Cdouble}}},
    val::Cdouble,
    dim::Int64,
    row::Int64,
    col::Int64,
)
    return setindex_tens(cxx_ptr_ptr_ptr, val, dim, row, col)
end

function Base.getindex(cxx_ptr_ptr::CxxPtr{CxxPtr{Cdouble}}, row::Int64, col::Int64)
    return getindex_mat(cxx_ptr_ptr, row, col)
end
function Base.setindex!(
    cxx_ptr_ptr::CxxPtr{CxxPtr{Cdouble}}, val::Cdouble, row::Int64, col::Int64
)
    return setindex_mat(cxx_ptr_ptr, val, row, col)
end

Base.getindex(cxx_ptr::CxxPtr{Cdouble}, row::Int64) = getindex_vec(cxx_ptr, row)
function Base.setindex!(cxx_ptr::CxxPtr{Cdouble}, val::Cdouble, row::Int64)
    return setindex_vec(cxx_ptr, val, row)
end

Base.getindex(cxx_ptr::CxxPtr{Cshort}, row::Int64) = getindex_vec(cxx_ptr, row)
function Base.setindex!(cxx_ptr::CxxPtr{Cshort}, val::Cshort, row::Int64)
    return setindex_vec(cxx_ptr, val, row)
end

function cxx_tensor_finalizer(cxx_tensor)
    return myfree3(cxx_tensor.data)
end

"""
    mutable struct CxxTensor <: AbstractArray{Cdouble, 3}

Wrapper for `CxxPtr{CxxPtr{CxxPtr{Cdouble}}}`, which is used as `double***` in C++.
"""
mutable struct CxxTensor <: AbstractArray{Cdouble,3}
    data::CxxPtr{CxxPtr{CxxPtr{Cdouble}}}
    dim1::Integer
    dim2::Integer
    dim3::Integer

    function CxxTensor(dim1::Integer, dim2::Integer, dim3::Integer)
        cxx_tensor = new(myalloc3(dim1, dim2, dim3), dim1, dim2, dim3)
        return finalizer(cxx_tensor_finalizer, cxx_tensor)
    end
    function CxxTensor(
        x::CxxPtr{CxxPtr{CxxPtr{Cdouble}}}, dim1::Integer, dim2::Integer, dim3::Integer
    )
        cxx_tensor = new(x, dim1, dim2, dim3)
        return finalizer(cxx_tensor_finalizer, cxx_tensor)
    end
    function CxxTensor(jl_tensor::AbstractArray{Cdouble,3})
        dim1, dim2, dim3 = size(jl_tensor)
        cxx_tensor = new(myalloc3(dim1, dim2, dim3), dim1, dim2, dim3)
        for k in 1:dim3
            for j in 1:dim2
                for i in 1:dim1
                    cxx_tensor[i, j, k] = jl_mat[i, j, k]
                end
            end
        end
        return finalizer(cxx_tensor_finalizer, cxx_tensor)
    end
end

Base.size(cxx_tensor::CxxTensor) = (cxx_tensor.dim1, cxx_tensor.dim2, cxx_tensor.dim3)
function Base.axes(cxx_tensor::CxxTensor)
    return (
        Base.OneTo(cxx_tensor.dim1),
        Base.OneTo(cxx_tensor.dim2),
        Base.OneTo(cxx_tensor.dim3),
    )
end
function Base.axes(cxx_tensor::CxxTensor, dim::Integer)
    if dim == 1
        return Base.OneTo(cxx_tensor.dim1)
    elseif dim == 2
        return Base.OneTo(cxx_tensor.dim2)
    elseif dim == 3
        return Base.OneTo(cxx_tensor.dim3)
    else
        return Base.OneTo(1)
    end
end
function Base.setindex!(
    cxx_tensor::CxxTensor, val::Number, dim1::Integer, dim2::Integer, dim3::Integer
)
    return setindex!(cxx_tensor.data, Cdouble(val), dim1, dim2, dim3)
end
function Base.getindex(cxx_tensor::CxxTensor, dim1::Integer, dim2::Integer, dim3::Integer)
    return getindex(cxx_tensor.data, dim1, dim2, dim3)
end

function cxx_mat_finalizer(cxx_mat)
    return myfree2(cxx_mat.data)
end

"""
    mutable struct CxxMatrix <: AbstractMatrix{Cdouble}

Wrapper for `CxxPtr{CxxPtr{Cdouble}}`, which is used as `double**` in C++.
"""
mutable struct CxxMatrix <: AbstractMatrix{Cdouble}
    data::CxxPtr{CxxPtr{Cdouble}}
    dim1::Integer
    dim2::Integer
    function CxxMatrix(dim1::Integer, dim2::Integer)
        cxx_mat = new(myalloc2(dim1, dim2), dim1, dim2)
        return finalizer(cxx_mat_finalizer, cxx_mat)
    end

    function CxxMatrix(x::CxxPtr{CxxPtr{Cdouble}}, dim1::Integer, dim2::Integer)
        cxx_mat = new(x, dim1, dim2)
        return finalizer(cxx_mat_finalizer, cxx_mat)
    end

    function CxxMatrix(jl_mat::AbstractMatrix{Cdouble})
        dim1, dim2 = size(jl_mat)
        cxx_mat = new(myalloc2(dim1, dim2), dim1, dim2)
        for j in 1:dim2
            for i in 1:dim1
                cxx_mat[i, j] = jl_mat[i, j]
            end
        end
        return finalizer(cxx_mat_finalizer, cxx_mat)
    end
end

Base.size(cxx_mat::CxxMatrix) = (cxx_mat.dim1, cxx_mat.dim2)
Base.axes(cxx_mat::CxxMatrix) = (Base.OneTo(cxx_mat.dim1), Base.OneTo(cxx_mat.dim2))
function Base.axes(cxx_mat::CxxMatrix, dim::Integer)
    if dim == 1
        return Base.OneTo(cxx_mat.dim1)
    elseif dim == 2
        return Base.OneTo(cxx_mat.dim2)
    else
        return Base.OneTo(1)
    end
end
function Base.setindex!(cxx_mat::CxxMatrix, val::Number, dim1::Integer, dim2::Integer)
    return setindex!(cxx_mat.data, Cdouble(val), dim1, dim2)
end
function Base.getindex(cxx_mat::CxxMatrix, dim1::Integer, dim2::Integer)
    return getindex(cxx_mat.data, dim1, dim2)
end

#######################################

function cxx_vec_finalizer(cxx_vec)
    return free_vec_double(cxx_vec.data)
end

"""
    mutable struct CxxVector <: AbstractVector{Cdouble}
Wrapper of a `double*` (`CxxPtr{Cdouble}`).
"""
mutable struct CxxVector <: AbstractVector{Cdouble}
    data::CxxPtr{Cdouble}
    dim::Integer
    function CxxVector(dim::Integer)
        cxx_vec = new(alloc_vec_double(dim), dim)
        return finalizer(cxx_vec_finalizer, cxx_vec)
    end
    function CxxVector(x::CxxPtr{Cdouble}, dim::Integer)
        cxx_vec = new(x, dim)
        return finalizer(cxx_vec_finalizer, cxx_vec)
    end

    function CxxVector(jl_vec::Vector{Cdouble})
        dim = size(jl_vec)[1]
        cxx_vec = new(alloc_vec_double(dim), dim)
        for i in 1:dim
            cxx_vec[i] = jl_vec[i]
        end
        return finalizer(cxx_vec_finalizer, cxx_vec)
    end
end

function Base.axes(cxx_vec::CxxVector, dim::Integer)
    return dim == 1 ? Base.OneTo(cxx_vec.dim) : Base.OneTo(1)
end
Base.axes(cxx_vec::CxxVector) = (Base.OneTo(cxx_vec.n),)
Base.size(cxx_vec::CxxVector) = (cxx_vec.dim,)
Base.getindex(cxx_vec::CxxVector, dim::Integer) = getindex(cxx_vec.data, dim)
function Base.setindex!(cxx_vec::CxxVector, val::Number, dim::Integer)
    return setindex!(cxx_vec.data, Cdouble(val), dim)
end

##### until here

function mat_to_cxx(cxx_mat::CxxMatrix, mat::Matrix{Float64})
    if cxx_mat.dim1 != size(mat, 1) || cxx_mat.dim2 != size(mat, 2)
        throw("dimension mistmatch!")
    end
    for j in 1:size(mat, 2)
        for i in 1:size(mat, 1)
            cxx_mat[i, j] = mat[i, j]
        end
    end
end

function vec_to_cxx(cxx_vec::CxxVector, vec::Vector{Float64})
    if cxx_vec.dim != size(vec, 1)
        throw("dimension mistmatch!")
    end
    for i in 1:size(vec, 1)
        cxx_vec[i] = vec[i]
    end
end

function jl_mat_to_cxx_mat(mat::Matrix{Float64})
    mat_cxx = myalloc2(size(mat)...)
    for i in 1:size(mat, 1)
        for j in 1:size(mat, 2)
            mat_cxx[i, j] = mat[i, j]
        end
    end
    return mat_cxx
end

function cxx_mat_to_jl_mat(mat_cxx::CxxPtr{CxxPtr{Float64}}, dim1::Integer, dim2::Integer)
    jl_mat = Matrix{Float64}(undef, dim1, dim2)
    for i in 1:dim2
        for j in 1:dim1
            jl_mat[j, i] = mat_cxx[j, i]
        end
    end
    return jl_mat
end

"""
    cxx_mat_to_jl_mat!(
        jl_mat::Matrix{Float64}, mat_cxx::CxxPtr{CxxPtr{Float64}}, dim1::Integer, dim2::Integer
    )
"""
function cxx_mat_to_jl_mat!(
    jl_mat::Matrix{Float64}, mat_cxx::CxxPtr{CxxPtr{Float64}}, dim1::Integer, dim2::Integer
)
    for i in 1:dim2
        for j in 1:dim1
            jl_mat[j, i] = mat_cxx[j, i]
        end
    end
    return jl_mat
end

function cxx_vec_to_jl_vec(cxx_vec::CxxPtr{Float64}, dim::Integer)
    jl_vec = Vector{Float64}(undef, dim)
    for i in 1:dim
        jl_vec[i] = cxx_vec[i]
    end
    return jl_vec
end

"""
    cxx_vec_to_jl_vec!(jl_vec::Vector{Float64}, cxx_vec::CxxPtr{Float64}, dim::Integer)

"""
function cxx_vec_to_jl_vec!(jl_vec::Vector{Float64}, cxx_vec::CxxPtr{Float64}, dim::Integer)
    for i in 1:dim
        jl_vec[i] = cxx_vec[i]
    end
    return jl_vec
end

function cxx_tensor_to_jl_tensor(
    cxx_tensor::CxxPtr{CxxPtr{CxxPtr{Float64}}}, dim1::Integer, dim2::Integer, dim3::Integer
)
    jl_tensor = Array{Float64}(undef, dim1, dim2, dim3)
    for i in 1:dim3
        for j in 1:dim2
            for k in 1:dim1
                jl_tensor[k, j, i] = cxx_tensor[k, j, i]
            end
        end
    end
    return jl_tensor
end

"""
    cxx_tensor_to_jl_tensor!(
        jl_tensor::Array{Float64,3},
        cxx_tensor::CxxPtr{CxxPtr{CxxPtr{Float64}}},
        dim1::Integer,
        dim2::Integer,
        dim3::Integer
    )

"""
function cxx_tensor_to_jl_tensor!(
    jl_tensor::Array{Float64,3},
    cxx_tensor::CxxPtr{CxxPtr{CxxPtr{Float64}}},
    dim1::Integer,
    dim2::Integer,
    dim3::Integer,
)
    for i in 1:dim3
        for j in 1:dim2
            for k in 1:dim1
                jl_tensor[k, j, i] = cxx_tensor[k, j, i]
            end
        end
    end
    return jl_tensor
end

function cxx_res_to_jl_res(
    cxx_res, m::Integer, n::Integer, mode::Symbol, num_dir::Integer, num_weights::Integer
)
    if mode === :jac
        if m > 1
            return cxx_mat_to_jl_mat(cxx_res, m, n)
        else
            return cxx_vec_to_jl_vec(cxx_res, n)
        end
    elseif mode === :hess
        return cxx_tensor_to_jl_tensor(cxx_res, m, n, n)
    elseif mode === :jac_vec
        return cxx_vec_to_jl_vec(cxx_res, m)
    elseif mode === :jac_mat
        return cxx_mat_to_jl_mat(cxx_res, m, num_dir)
    elseif mode === :vec_jac
        return cxx_vec_to_jl_vec(cxx_res, n)
    elseif mode === :mat_jac
        return cxx_mat_to_jl_mat(cxx_res, num_weights, n)

    elseif mode === :hess_vec
        return cxx_mat_to_jl_mat(cxx_res, m, n)
    elseif mode === :hess_mat
        return cxx_tensor_to_jl_tensor(cxx_res, m, n, num_dir)
    elseif mode === :vec_hess
        return cxx_mat_to_jl_mat(cxx_res, n, n)
    elseif mode === :mat_hess
        return cxx_tensor_to_jl_tensor(cxx_res, num_weights, n, n)

    elseif mode === :vec_hess_vec
        return cxx_vec_to_jl_vec(cxx_res, n)
    elseif mode === :mat_hess_vec
        return cxx_mat_to_jl_mat(cxx_res, num_weights, n)
    elseif mode === :vec_hess_mat
        return cxx_mat_to_jl_mat(cxx_res, n, num_dir)
    elseif mode === :mat_hess_mat
        return cxx_tensor_to_jl_tensor(cxx_res, num_weights, n, num_dir)
    elseif mode === :abs_normal
        return nothing
    end
end

"""
    cxx_res_to_jl_res!(
        jl_res, cxx_res, m::Integer, n::Integer, mode::Symbol, num_dir::Integer, num_weights::Integer
    )

"""
function cxx_res_to_jl_res!(
    jl_res,
    cxx_res,
    m::Integer,
    n::Integer,
    mode::Symbol,
    num_dir::Integer,
    num_weights::Integer,
)
    if mode === :jac
        if m > 1
            return cxx_mat_to_jl_mat!(jl_res, cxx_res, m, n)
        else
            return cxx_vec_to_jl_vec!(jl_res, cxx_res, n)
        end
    elseif mode === :hess
        return cxx_tensor_to_jl_tensor!(jl_res, cxx_res, m, n, n)
    elseif mode === :jac_vec
        return cxx_vec_to_jl_vec!(jl_res, cxx_res, m)
    elseif mode === :jac_mat
        return cxx_mat_to_jl_mat!(jl_res, cxx_res, m, num_dir)
    elseif mode === :vec_jac
        return cxx_vec_to_jl_vec!(jl_res, cxx_res, n)
    elseif mode === :mat_jac
        return cxx_mat_to_jl_mat!(jl_res, cxx_res, num_weights, n)

    elseif mode === :hess_vec
        return cxx_mat_to_jl_mat!(jl_res, cxx_res, m, n)
    elseif mode === :hess_mat
        return cxx_tensor_to_jl_tensor!(jl_res, cxx_res, m, n, num_dir)
    elseif mode === :vec_hess
        return cxx_mat_to_jl_mat!(jl_res, cxx_res, n, n)
    elseif mode === :mat_hess
        return cxx_tensor_to_jl_tensor!(jl_res, cxx_res, num_weights, n, n)

    elseif mode === :vec_hess_vec
        return cxx_vec_to_jl_vec!(jl_res, cxx_res, n)
    elseif mode === :mat_hess_vec
        return cxx_mat_to_jl_mat!(jl_res, cxx_res, num_weights, n)
    elseif mode === :vec_hess_mat
        return cxx_mat_to_jl_mat!(jl_res, cxx_res, n, num_dir)
    elseif mode === :mat_hess_mat
        return cxx_tensor_to_jl_tensor!(jl_res, cxx_res, num_weights, n, num_dir)
    elseif mode === :abs_normal
        return nothing
    end
end

export CxxMatrix,
    CxxVector,
    CxxTensor,
    myalloc3,
    myalloc2,
    alloc_vec_double,
    alloc_vec_short,
    alloc_vec,
    alloc_mat_short,
    jl_mat_to_cxx_mat,
    cxx_mat_to_jl_mat,
    cxx_vec_to_jl_vec,
    cxx_tensor_to_jl_tensor,
    cxx_mat_to_jl_mat!,
    cxx_vec_to_jl_vec!,
    cxx_tensor_to_jl_tensor!,
    cxx_res_to_jl_res,
    cxx_res_to_jl_res!,
    vec_to_cxx

export myfree3, myfree2, free_vec_double, free_vec_short, free_mat_short
end # module arry_types
