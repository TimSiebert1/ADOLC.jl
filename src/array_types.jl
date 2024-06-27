module array_types
using ADOLC_jll
using CxxWrap

@wrapmodule(() -> libadolc_wrap, :julia_module_array_types)

function __init__()
    @initcxx
end

function Base.getindex(cxx_ptr_ptr_ptr::CxxPtr{CxxPtr{CxxPtr{Cdouble}}}, dim)
    return getindex_tens(cxx_ptr_ptr_ptr, dim)
end
function Base.getindex(cxx_ptr_ptr_ptr::CxxPtr{CxxPtr{CxxPtr{Cdouble}}}, dim::Int64)
    return getindex_tens(cxx_ptr_ptr_ptr, dim)
end
function Base.getindex(cxx_ptr_ptr_ptr::CxxPtr{CxxPtr{CxxPtr{Cdouble}}}, dim, row, col)
    return getindex_tens(cxx_ptr_ptr_ptr, dim, row, col)
end
function Base.getindex(
    cxx_ptr_ptr_ptr::CxxPtr{CxxPtr{CxxPtr{Cdouble}}}, dim::Int64, row::Int64, col::Int64
)
    return getindex_tens(cxx_ptr_ptr_ptr, dim, row, col)
end
function Base.setindex!(
    cxx_ptr_ptr_ptr::CxxPtr{CxxPtr{CxxPtr{Cdouble}}}, val, dim, row, col
)
    return setindex_tens(cxx_ptr_ptr_ptr, Cdouble(val), dim, row, col)
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
function Base.getindex(cxx_ptr_ptr::CxxPtr{CxxPtr{Cdouble}}, row, col)
    return getindex_mat(cxx_ptr_ptr, row, col)
end
function Base.getindex(cxx_ptr_ptr::CxxPtr{CxxPtr{Cdouble}}, row::Int64, col::Int64)
    return getindex_mat(cxx_ptr_ptr, row, col)
end
function Base.setindex!(cxx_ptr_ptr::CxxPtr{CxxPtr{Cdouble}}, val, row, col)
    return setindex_mat(cxx_ptr_ptr, Cdouble(val), row, col)
end
function Base.setindex!(
    cxx_ptr_ptr::CxxPtr{CxxPtr{Cdouble}}, val::Cdouble, row::Int64, col::Int64
)
    return setindex_mat(cxx_ptr_ptr, val, row, col)
end
Base.getindex(cxx_ptr::CxxPtr{Cdouble}, row::Int64) = getindex_vec(cxx_ptr, row)
Base.getindex(cxx_ptr::CxxPtr{Cdouble}, row) = getindex_vec(cxx_ptr, row)
function Base.setindex!(cxx_ptr::CxxPtr{Cdouble}, val::Cdouble, row::Int64)
    return setindex_vec(cxx_ptr, val, row)
end
function Base.setindex!(cxx_ptr::CxxPtr{Cdouble}, val, row)
    return setindex_vec(cxx_ptr, Cdouble(val), row)
end
Base.getindex(cxx_ptr::CxxPtr{Cshort}, row::Int64) = getindex_vec(cxx_ptr, row)
Base.getindex(cxx_ptr::CxxPtr{Cshort}, row) = getindex_vec(cxx_ptr, row)
function Base.setindex!(cxx_ptr::CxxPtr{Cshort}, val::Cshort, row::Int64)
    return setindex_vec(cxx_ptr, val, row)
end
function Base.setindex!(cxx_ptr::CxxPtr{Cshort}, val, row)
    return setindex_vec(cxx_ptr, Cshort(val), row)
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
                    cxx_tensor[i, j, k] = jl_tensor[i, j, k]
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

    function CxxVector(jl_vec::AbstractVector{Cdouble})
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
Base.axes(cxx_vec::CxxVector) = (Base.OneTo(cxx_vec.dim),)
Base.size(cxx_vec::CxxVector) = (cxx_vec.dim,)
Base.getindex(cxx_vec::CxxVector, dim::Integer) = getindex(cxx_vec.data, dim)
function Base.setindex!(cxx_vec::CxxVector, val::Number, dim::Integer)
    return setindex!(cxx_vec.data, Cdouble(val), dim)
end

function jl_vec_to_cxx_vec!(cxx_vec::CxxVector, jl_vec::AbstractVector{Cdouble})
    if cxx_vec.dim != size(jl_vec, 1)
        throw(DimensionMismatch("Size of cxx_vec not equal to size of jl_vec"))
    end
    for i in 1:(cxx_vec.dim)
        cxx_vec[i] = jl_vec[i]
    end
end
function jl_mat_to_cxx_mat!(cxx_mat::CxxMatrix, jl_mat::AbstractMatrix{Cdouble})
    if cxx_mat.dim1 != size(jl_mat, 1) || cxx_mat.dim2 != size(jl_mat, 2)
        throw(DimensionMismatch("Size of cxx_mat not equal to size of jl_mat"))
    end
    for j in 1:(cxx_mat.dim2)
        for i in 1:(cxx_mat.dim1)
            cxx_mat[i, j] = jl_mat[i, j]
        end
    end
end
function jl_tensor_to_cxx_tensor!(
    cxx_tensor::CxxTensor, jl_tensor::AbstractArray{Cdouble,3}
)
    if cxx_tensor.dim1 != size(jl_tensor, 1) ||
        cxx_tensor.dim2 != size(jl_tensor, 2) ||
        cxx_tensor.dim3 != size(jl_tensor, 3)
        throw(DimensionMismatch("Size of cxx_tensor not equal to size of jl_tensor"))
    end
    for k in 1:(cxx_tensor.dim3)
        for j in 1:(cxx_tensor.dim2)
            for i in 1:(cxx_tensor.dim1)
                cxx_tensor[i, j, k] = jl_tensor[i, j, k]
            end
        end
    end
end

"""
    jl_res_to_cxx_res!(cxx_res::CxxVector, jl_res::AbstractVector{Cdouble}) 
    jl_res_to_cxx_res!(cxx_res::CxxMatrix, jl_res::AbstractMatrix{Cdouble})
    jl_res_to_cxx_res!(cxx_res::CxxTensor, jl_res::AbstractArray{Cdouble, 3})

Copies the values of a `AbstractVector{Cdouble}`, `AbstractMatrix{Cdouble}` or `AbstractArray{Cdouble, 3}` to
a corresponding `CxxVector`, `CxxMatrix` or `CxxTensor`.
"""
function jl_res_to_cxx_res!(cxx_res::CxxVector, jl_res::AbstractVector{Cdouble})
    return jl_vec_to_cxx_vec!(cxx_res, jl_res)
end
function jl_res_to_cxx_res!(cxx_res::CxxMatrix, jl_res::AbstractMatrix{Cdouble})
    return jl_mat_to_cxx_mat!(cxx_res, jl_res)
end
function jl_res_to_cxx_res!(cxx_res::CxxTensor, jl_res::AbstractArray{Cdouble,3})
    return jl_tensor_to_cxx_tensor!(cxx_res, jl_res)
end

"""
    jl_res_to_cxx_res(jl_res::AbstractVector{Cdouble}) 
    jl_res_to_cxx_res(jl_res::AbstractMatrix{Cdouble})
    jl_res_to_cxx_res(jl_res::AbstractArray{Cdouble, 3})

Creates a `CxxVector`, `CxxMatrix` or `CxxTensor` and copies the values from the corresponding input
`AbstractVector{Cdouble}`, `AbstractMatrix{Cdouble}` or `AbstractArray{Cdouble, 3}` to it.
"""
function jl_res_to_cxx_res(jl_res::AbstractVector{Cdouble})
    cxx_res = CxxVector(size(jl_res)...)
    jl_vec_to_cxx_vec!(cxx_res, jl_res)
    return cxx_res
end
function jl_res_to_cxx_res(jl_res::AbstractMatrix{Cdouble})
    cxx_res = CxxMatrix(size(jl_res)...)
    jl_mat_to_cxx_mat!(cxx_res, jl_res)
    return cxx_res
end
function jl_res_to_cxx_res(jl_res::AbstractArray{Cdouble,3})
    cxx_res = CxxTensor(size(jl_res)...)
    jl_tensor_to_cxx_tensor!(cxx_res, jl_res)
    return cxx_res
end

function cxx_vec_to_jl_vec!(jl_vec::AbstractVector{Cdouble}, cxx_vec::CxxVector)
    if cxx_vec.dim != size(jl_vec, 1)
        throw(DimensionMismatch("Size of cxx_vec not equal to size of jl_vec"))
    end
    for i in 1:(cxx_vec.dim)
        jl_vec[i] = cxx_vec[i]
    end
end
function cxx_mat_to_jl_mat!(jl_mat::AbstractMatrix{Cdouble}, cxx_mat::CxxMatrix)
    if cxx_mat.dim1 != size(jl_mat, 1) || cxx_mat.dim2 != size(jl_mat, 2)
        throw(DimensionMismatch("Size of cxx_mat not equal to size of jl_mat"))
    end
    for i in 1:(cxx_mat.dim2)
        for j in 1:(cxx_mat.dim1)
            jl_mat[j, i] = cxx_mat[j, i]
        end
    end
end
function cxx_tensor_to_jl_tensor!(
    jl_tensor::AbstractArray{Cdouble,3}, cxx_tensor::CxxTensor
)
    if cxx_tensor.dim1 != size(jl_tensor, 1) ||
        cxx_tensor.dim2 != size(jl_tensor, 2) ||
        cxx_tensor.dim3 != size(jl_tensor, 3)
        throw(DimensionMismatch("Size of cxx_tensor not equal to size of jl_tensor"))
    end
    for k in 1:(cxx_tensor.dim3)
        for j in 1:(cxx_tensor.dim2)
            for i in 1:(cxx_tensor.dim1)
                jl_tensor[i, j, k] = cxx_tensor[i, j, k]
            end
        end
    end
end
"""
    cxx_res_to_jl_res!(jl_res::AbstractVector{Cdouble}, cxx_res::CxxVector)
    cxx_res_to_jl_res!(jl_res::AbstractMatrix{Float64}, cxx_res::CxxMatrix)
    cxx_res_to_jl_res!(jl_res::AbstractArray{Float64, 3}, cxx_res::CxxTensor)

Copies the entries of a `CxxVector`, `CxxMatrix` or `CxxTensor` 
to a `AbstractVector{Cdouble}`, `AbstractMatrix{Cdouble}` or `AbstractArray{Cdouble, 3}`.
"""
function cxx_res_to_jl_res!(jl_res::AbstractVector{Cdouble}, cxx_res::CxxVector)
    return cxx_vec_to_jl_vec!(jl_res, cxx_res)
end
function cxx_res_to_jl_res!(jl_res::AbstractMatrix{Float64}, cxx_res::CxxMatrix)
    return cxx_mat_to_jl_mat!(jl_res, cxx_res)
end
function cxx_res_to_jl_res!(jl_res::AbstractArray{Float64,3}, cxx_res::CxxTensor)
    return cxx_tensor_to_jl_tensor!(jl_res, cxx_res)
end

"""
    jl_res_to_cxx_res(cxx_res::CxxVector) 
    jl_res_to_cxx_res(cxx_res::CxxMatrix)
    jl_res_to_cxx_res(cxx_res::CxxTensor)

Creates a `Vector{Cdouble}`, `Matrix{Cdouble}` or `Array{Cdouble, 3}`
and copies the values from the corresponding input `CxxVector`, `CxxMatrix` or `CxxTensor` to it.
"""
function cxx_res_to_jl_res(cxx_res::CxxVector)
    jl_res = Vector{Cdouble}(undef, size(cxx_res)...)
    cxx_vec_to_jl_vec!(jl_res, cxx_res)
    return jl_res
end

function cxx_res_to_jl_res(cxx_res::CxxMatrix)
    jl_res = Matrix{Cdouble}(undef, size(cxx_res)...)
    cxx_mat_to_jl_mat!(jl_res, cxx_res)
    return jl_res
end

function cxx_res_to_jl_res(cxx_res::CxxTensor)
    jl_res = Array{Cdouble,3}(undef, size(cxx_res)...)
    cxx_tensor_to_jl_tensor!(jl_res, cxx_res)
    return jl_res
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
    cxx_res_to_jl_res,
    cxx_res_to_jl_res!,
    jl_res_to_cxx_res!,
    jl_res_to_cxx_res

export myfree3, myfree2, free_vec_double, free_vec_short, free_mat_short
end # module arry_types
