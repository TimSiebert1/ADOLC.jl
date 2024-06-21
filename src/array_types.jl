module array_types
using ADOLC_jll
using CxxWrap

@wrapmodule(() -> libadolc_wrap, :julia_module_array_types)

function __init__()
    @initcxx
end

#CxxMatrix{T} = CxxPtr{CxxPtr{T}} where T <: Real
#CxxTensor{T} = CxxPtr{CxxPtr{CxxPtr{T}}} where T <: Real

###### Array getter and setter ###### 
Base.getindex(X::CxxPtr{CxxPtr{CxxPtr{Cdouble}}}, dim::Int64) = getindex_tens(X, dim)
function Base.getindex(
    X::CxxPtr{CxxPtr{CxxPtr{Cdouble}}}, dim::Int64, row::Int64, col::Int64
)
    return getindex_tens(X, dim, row, col)
end
function Base.setindex!(
    X::CxxPtr{CxxPtr{CxxPtr{Cdouble}}}, val::Cdouble, dim::Int64, row::Int64, col::Int64
)
    return setindex_tens(X, val, dim, row, col)
end

function Base.getindex(X::CxxPtr{CxxPtr{Cdouble}}, row::Int64, col::Int64)
    return getindex_mat(X, row, col)
end
function Base.setindex!(X::CxxPtr{CxxPtr{Cdouble}}, val::Cdouble, row::Int64, col::Int64)
    return setindex_mat(X, val, row, col)
end

Base.getindex(X::CxxPtr{Cdouble}, row::Int64) = getindex_vec(X, row)
Base.setindex!(X::CxxPtr{Cdouble}, val::Cdouble, row::Int64) = setindex_vec(X, val, row)

Base.getindex(X::CxxPtr{Cshort}, row::Int64) = getindex_vec(X, row)
Base.setindex!(X::CxxPtr{Cshort}, val::Cshort, row::Int64) = setindex_vec(X, val, row)

################################################

function cxx_mat_finalizer(t)
    return myfree2(t.data)
end

###### ptr-ptr wrapper #########################

mutable struct CxxMatrix{T} <: AbstractMatrix{T}
    """
    Wrapper for c++ double** or short** data
    """
    data::CxxPtr{CxxPtr{T}}
    n::Int64
    m::Int64
    function CxxMatrix{T}(n::Integer, m::Integer) where {T<:Real}
        check_type_mat(T)
        x = new{T}(alloc_mat(T, n, m), n, m)
        return finalizer(cxx_mat_finalizer, x)
    end
    function CxxMatrix{T}(Y::AbstractMatrix{T}) where {T<:Real}
        check_type_mat(T)
        n, m = size(Y)
        X = new{T}(alloc_mat(T, n, m), n, m)
        for i in 1:n
            for j in 1:m
                X[i, j] = Y[i, j]
            end
        end
        return finalizer(cxx_mat_finalizer, X)
    end
end

Base.size(A::CxxMatrix) = (A.n, A.m)
Base.setindex!(A::CxxMatrix{T}, v, i, j) where {T} = setindex!(A.data, T(v), i, j)
Base.getindex(A::CxxMatrix, i, j) = getindex(A.data, i, j)

function check_type_mat(::Type{T}) where {T<:Real}
    if !(T <: Union{Cdouble})
        throw("Not implemented for type $T")
    end
end

alloc_mat(::Type{Float64}, n::Integer, m::Integer) = myalloc2(n, m)

function Base.axes(X::CxxMatrix{T}, n::Int64) where {T<:Real}
    if n == 1
        return Base.OneTo(X.n)
    elseif n == 2
        return Base.OneTo(X.m)
    else
        return Base.OneTo(1)
    end
end

Base.axes(X::CxxMatrix{T}) where {T<:Real} = (Base.OneTo(X.n), Base.OneTo(X.m))

Base.size(X::CxxMatrix{T}) where {T<:Real} = (X.n, X.m)
function Base.getindex(X::CxxMatrix{T}, row::Int64, col::Int64) where {T<:Real}
    return getindex(X.data, row, col)
end
function Base.setindex!(X::CxxMatrix{T}, val::T, row::Int64, col::Int64) where {T<:Real}
    return setindex!(X.data, val, row, col)
end

#######################################

alloc_vec(::Type{Cdouble}, n::Integer) = alloc_vec_double(n)
alloc_vec(::Type{Cshort}, n::Integer) = alloc_vec_short(n)

function cxx_vec_finalizer(x)
    return free_vec_double(x.data)
end

mutable struct CxxVector{T} <: AbstractVector{T}
    data::CxxPtr{T}
    n::Int64
    function CxxVector{T}(n::Integer) where {T<:Real}
        check_type_vec(T)
        x = new{T}(alloc_vec(T, n), n)
        return finalizer(cxx_vec_finalizer, x)
    end
    function CxxVector{T}(y::Vector{T}) where {T<:Real}
        check_type_vec(T)
        n = length(y)
        x = new{T}(alloc_vec(T, n), n)
        for i in 1:n
            x[i] = y[i]
        end
        return finalizer(cxx_vec_finalizer, x)
    end
end

function check_type_vec(::Type{T}) where {T<:Real}
    if !(T <: Union{Cdouble,Cshort})
        throw("Not implemented for type $T")
    end
end

function mat_to_cxx(cxx_mat::CxxMatrix{Float64}, mat::Matrix{Float64})
    if cxx_mat.n != size(mat, 1) || cxx_mat.m != size(mat, 2)
        throw("dimension mistmatch!")
    end
    for i in 1:size(mat, 1)
        for j in 1:size(mat, 2)
            cxx_mat[i, j] = mat[i, j]
        end
    end
end

function vec_to_cxx(cxx_vec::CxxVector{Float64}, vec::Vector{Float64})
    if cxx_vec.n != size(vec, 1)
        throw("dimension mistmatch!")
    end
    for i in 1:size(vec, 1)
        cxx_vec[i] = vec[i]
    end
end

function Base.axes(X::CxxVector{T}, n::Int64) where {T<:Real}
    return n == 1 ? Base.OneTo(X.n) : Base.OneTo(1)
end
Base.axes(X::CxxVector{T}) where {T<:Real} = (Base.OneTo(X.n),)

Base.size(X::CxxVector{T}) where {T<:Real} = X.n
Base.getindex(X::CxxVector{T}, row::Int64) where {T<:Real} = getindex(X.data, row)
function Base.setindex!(X::CxxVector{T}, val::T, row::Int64) where {T<:Real}
    return setindex!(X.data, val, row)
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

function cxx_mat_to_jl_mat(mat_cxx::CxxPtr{CxxPtr{Float64}}, dim1, dim2)
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
        jl_mat::Matrix{Float64}, mat_cxx::CxxPtr{CxxPtr{Float64}}, dim1, dim2
    )
"""
function cxx_mat_to_jl_mat!(
    jl_mat::Matrix{Float64}, mat_cxx::CxxPtr{CxxPtr{Float64}}, dim1, dim2
)
    for i in 1:dim2
        for j in 1:dim1
            jl_mat[j, i] = mat_cxx[j, i]
        end
    end
    return jl_mat
end

function cxx_vec_to_jl_vec(cxx_vec::CxxPtr{Float64}, dim)
    jl_vec = Vector{Float64}(undef, dim)
    for i in 1:dim
        jl_vec[i] = cxx_vec[i]
    end
    return jl_vec
end

"""
    cxx_vec_to_jl_vec!(jl_vec::Vector{Float64}, cxx_vec::CxxPtr{Float64}, dim)

"""
function cxx_vec_to_jl_vec!(jl_vec::Vector{Float64}, cxx_vec::CxxPtr{Float64}, dim)
    for i in 1:dim
        jl_vec[i] = cxx_vec[i]
    end
    return jl_vec
end

function cxx_tensor_to_jl_tensor(
    cxx_tensor::CxxPtr{CxxPtr{CxxPtr{Float64}}}, dim1, dim2, dim3
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
        dim1,
        dim2,
        dim3,
    )

"""
function cxx_tensor_to_jl_tensor!(
    jl_tensor::Array{Float64,3},
    cxx_tensor::CxxPtr{CxxPtr{CxxPtr{Float64}}},
    dim1,
    dim2,
    dim3,
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
    cxx_res, m::Int64, n::Int64, mode::Symbol, num_dir::Int64, num_weights::Int64
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
        jl_res, cxx_res, m::Int64, n::Int64, mode::Symbol, num_dir::Int64, num_weights::Int64
    )

"""
function cxx_res_to_jl_res!(
    jl_res, cxx_res, m::Int64, n::Int64, mode::Symbol, num_dir::Int64, num_weights::Int64
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
