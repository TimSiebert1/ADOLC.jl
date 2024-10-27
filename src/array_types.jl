module array_types
using ADOLC_jll
function Base.getindex(cxx_ptr_ptr_ptr::Ptr{Ptr{Ptr{Cdouble}}}, dim)
    return ccall(
        (:getindex_tens, adolc_interface_lib),
        Ptr{Ptr{Cdouble}},
        (Ptr{Ptr{Ptr{Cdouble}}}, Cint),
        cxx_ptr_ptr_ptr,
        dim - 1,
    )
end
function Base.getindex(cxx_ptr_ptr_ptr::Ptr{Ptr{Ptr{Cdouble}}}, dim::Int64)
    return ccall(
        (:getindex_tens, adolc_interface_lib),
        Ptr{Ptr{Cdouble}},
        (Ptr{Ptr{Ptr{Cdouble}}}, Cint),
        cxx_ptr_ptr_ptr,
        dim - 1,
    )
end
function Base.getindex(cxx_ptr_ptr_ptr::Ptr{Ptr{Ptr{Cdouble}}}, dim, row, col)
    return ccall(
        (:getindex_tens, adolc_interface_lib),
        Cdouble,
        (Ptr{Ptr{Ptr{Cdouble}}}, Cint, Cint, Cint),
        cxx_ptr_ptr_ptr,
        dim - 1,
        row - 1,
        col - 1,
    )
end
function Base.getindex(
    cxx_ptr_ptr_ptr::Ptr{Ptr{Ptr{Cdouble}}}, dim::Int64, row::Int64, col::Int64
)
    return ccall(
        (:getindex_tens, adolc_interface_lib),
        Cdouble,
        (Ptr{Ptr{Ptr{Cdouble}}}, Cint, Cint, Cint),
        cxx_ptr_ptr_ptr,
        dim - 1,
        row - 1,
        col - 1,
    )
end
function Base.setindex!(cxx_ptr_ptr_ptr::Ptr{Ptr{Ptr{Cdouble}}}, val, dim, row, col)
    return ccall(
        (:setindex_tens, adolc_interface_lib),
        Cvoid,
        (Ptr{Ptr{Ptr{Cdouble}}}, Cdouble, Cint, Cint, Cint),
        cxx_ptr_ptr_ptr,
        Cdouble(val),
        dim - 1,
        row - 1,
        col - 1,
    )
end
function Base.setindex!(
    cxx_ptr_ptr_ptr::Ptr{Ptr{Ptr{Cdouble}}},
    val::Cdouble,
    dim::Int64,
    row::Int64,
    col::Int64,
)
    return ccall(
        (:setindex_tens, adolc_interface_lib),
        Cvoid,
        (Ptr{Ptr{Ptr{Cdouble}}}, Cdouble, Cint, Cint, Cint),
        cxx_ptr_ptr_ptr,
        Cdouble(val),
        dim - 1,
        row - 1,
        col - 1,
    )
end
function Base.getindex(cxx_ptr_ptr::Ptr{Ptr{Cdouble}}, row, col)
    return ccall(
        (:getindex_mat, adolc_interface_lib),
        Cdouble,
        (Ptr{Ptr{Cdouble}}, Cint, Cint),
        cxx_ptr_ptr,
        row - 1,
        col - 1,
    )
end
function Base.getindex(cxx_ptr_ptr::Ptr{Ptr{Cdouble}}, row::Int64, col::Int64)
    return ccall(
        (:getindex_mat, adolc_interface_lib),
        Cdouble,
        (Ptr{Ptr{Cdouble}}, Cint, Cint),
        cxx_ptr_ptr,
        row - 1,
        col - 1,
    )
end
function Base.setindex!(cxx_ptr_ptr::Ptr{Ptr{Cdouble}}, val, row, col)
    return ccall(
        (:setindex_mat, adolc_interface_lib),
        Cvoid,
        (Ptr{Ptr{Cdouble}}, Cdouble, Cint, Cint),
        cxx_ptr_ptr,
        Cdouble(val),
        row - 1,
        col - 1,
    )
end
function Base.setindex!(
    cxx_ptr_ptr::Ptr{Ptr{Cdouble}}, val::Cdouble, row::Int64, col::Int64
)
    return ccall(
        (:setindex_mat, adolc_interface_lib),
        Cvoid,
        (Ptr{Ptr{Cdouble}}, Cdouble, Cint, Cint),
        cxx_ptr_ptr,
        Cdouble(val),
        row - 1,
        col - 1,
    )
end
function Base.getindex(cxx_ptr::Ptr{Cdouble}, row::Int64)
    return ccall(
        (:getindex_vec, adolc_interface_lib), Cdouble, (Ptr{Cdouble}, Cint), cxx_ptr, row - 1
    )
end
function Base.getindex(cxx_ptr::Ptr{Cdouble}, row)
    return ccall(
        (:getindex_vec, adolc_interface_lib), Cdouble, (Ptr{Cdouble}, Cint), cxx_ptr, row - 1
    )
end
function Base.setindex!(cxx_ptr::Ptr{Cdouble}, val::Cdouble, row::Int64)
    return ccall(
        (:setindex_vec, adolc_interface_lib),
        Cvoid,
        (Ptr{Cdouble}, Cdouble, Cint),
        cxx_ptr,
        val,
        row - 1,
    )
end
function Base.setindex!(cxx_ptr::Ptr{Cdouble}, val, row)
    return ccall(
        (:setindex_vec, adolc_interface_lib),
        Cvoid,
        (Ptr{Cdouble}, Cdouble, Cint),
        cxx_ptr,
        val,
        row - 1,
    )
end

function cxx_tensor_finalizer(x)
    return ccall((:myfree3, adolc_interface_lib), Cvoid, (Ptr{Ptr{Ptr{Cvoid}}},), x.data)
end
function alloc_tensor(dim1, dim2, dim3)
    return ccall(
        (:myalloc3, adolc_interface_lib),
        Ptr{Ptr{Ptr{Cdouble}}},
        (Cint, Cint, Cint),
        dim1,
        dim2,
        dim3,
    )
end
"""
    mutable struct CxxTensor <: AbstractArray{Cdouble, 3}

Wrapper for `Ptr{Ptr{Ptr{Cdouble}}}`, which is used as `double***` in C++.
"""
mutable struct CxxTensor <: AbstractArray{Cdouble,3}
    data::Ptr{Ptr{Ptr{Cdouble}}}
    dim1::Integer
    dim2::Integer
    dim3::Integer

    function CxxTensor(dim1::Integer, dim2::Integer, dim3::Integer)
        cxx_tensor = new(alloc_tensor(dim1, dim2, dim3), dim1, dim2, dim3)
        return finalizer(cxx_tensor_finalizer, cxx_tensor)
    end
    function CxxTensor(
        x::Ptr{Ptr{Ptr{Cdouble}}}, dim1::Integer, dim2::Integer, dim3::Integer
    )
        cxx_tensor = new(x, dim1, dim2, dim3)
        return finalizer(cxx_tensor_finalizer, cxx_tensor)
    end
    function CxxTensor(jl_tensor::AbstractArray{Cdouble,3})
        dim1, dim2, dim3 = size(jl_tensor)
        cxx_tensor = new(alloc_tensor(dim1, dim2, dim3), dim1, dim2, dim3)
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
    return ccall((:myfree2, adolc_interface_lib), Cvoid, (Ptr{Ptr{Cdouble}},), cxx_mat.data)
end
function alloc_matrix(dim1, dim2)
    return ccall((:myalloc2, adolc_interface_lib), Ptr{Ptr{Cdouble}}, (Cint, Cint), dim1, dim2)
end

"""
    mutable struct CxxMatrix <: AbstractMatrix{Cdouble}

Wrapper for `Ptr{Ptr{Cdouble}}`, which is used as `double**` in C++.
"""
mutable struct CxxMatrix <: AbstractMatrix{Cdouble}
    data::Ptr{Ptr{Cdouble}}
    dim1::Integer
    dim2::Integer
    function CxxMatrix(dim1::Integer, dim2::Integer)
        cxx_mat = new(alloc_matrix(dim1, dim2), dim1, dim2)
        return finalizer(cxx_mat_finalizer, cxx_mat)
    end

    function CxxMatrix(x::Ptr{Ptr{Cdouble}}, dim1::Integer, dim2::Integer)
        cxx_mat = new(x, dim1, dim2)
        return finalizer(cxx_mat_finalizer, cxx_mat)
    end

    function CxxMatrix(jl_mat::AbstractMatrix{Cdouble})
        dim1, dim2 = size(jl_mat)
        cxx_mat = new(alloc_matrix(dim1, dim2), dim1, dim2)
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
    return ccall((:myfree1, adolc_interface_lib), Cvoid, (Ptr{Cdouble},), cxx_vec.data)
end
alloc_vector(dim) = ccall((:myalloc1, adolc_interface_lib), Ptr{Cdouble}, (Cint,), dim)
"""
    mutable struct CxxVector <: AbstractVector{Cdouble}
Wrapper of a `double*` (`Ptr{Cdouble}`).
"""
mutable struct CxxVector <: AbstractVector{Cdouble}
    data::Ptr{Cdouble}
    dim::Integer
    function CxxVector(dim::Integer)
        cxx_vec = new(alloc_vector(dim), dim)
        return finalizer(cxx_vec_finalizer, cxx_vec)
    end
    function CxxVector(x::Ptr{Cdouble}, dim::Integer)
        cxx_vec = new(x, dim)
        return finalizer(cxx_vec_finalizer, cxx_vec)
    end

    function CxxVector(jl_vec::AbstractVector{Cdouble})
        dim = size(jl_vec)[1]
        cxx_vec = new(alloc_vector(dim), dim)
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

Base.unsafe_convert(::Type{Ptr{Ptr{Ptr{Cdouble}}}}, x::CxxTensor) = x.data
Base.unsafe_convert(::Type{Ptr{Ptr{Cdouble}}}, x::CxxMatrix) = x.data
Base.unsafe_convert(::Type{Ptr{Cdouble}}, x::CxxVector) = x.data

function Array{Cdouble,3}(x::CxxTensor)
    return unsafe_wrap(
        Array{Cdouble,3}, Ptr{Cdouble}(x.data), (x.dim1, x.dim2, x.dim3); own=true
    )
end
function Matrix{Cdouble}(x::CxxMatrix)
    return unsafe_wrap(Matrix{Cdouble}, Ptr{Cdouble}(x.data), (x.dim1, x.dim2); own=true)
end
Vector{Cdouble}(x::CxxVector) = unsafe_wrap(Vector{Cdouble}, x.data, x.dim; own=true)

"""
    create_cxx_identity(n::I_1, m::I_2) where {I_1 <: Integer, I_2 <: Integer}

Creates a identity matrix of shape (`n`, `m`) of type CxxPtr{CxxPtr{Float64}} (wrapper of C++'s double**).


# Example
```jldoctest
id = CxxMatrix(create_cxx_identity(2, 4), 2, 4)


# output

2×4 CxxMatrix:
 1.0  0.0  0.0  0.0
 0.0  1.0  0.0  0.0
```
"""
function create_cxx_identity(n::I_1, m::I_2) where {I_1<:Integer,I_2<:Integer}
    I = alloc_matrix(n, m)
    for i in 1:n
        for j in 1:m
            I[i, j] = 0.0
            if i == j
                I[i, i] = 1.0
            end
        end
    end
    return I
end

"""
    create_partial_cxx_identity(n::I_1, idxs::Vector{I_2}) where {I_1 <: Integer, I_2 <: Integer}

Creates a matrix of shape (`n`, `length(idxs)`) of type CxxPtr{CxxPtr{Float64}} (wrapper of C++'s double**).
The columns are canonical basis vectors corresponding to the entries of `idxs`. The order of the basis vectors
is defined by the order of the indices in `idxs`. Details about the application can be found in this [guide](@ref "Seed-Matrix").

!!! warning
    The number of rows `n` must be smaller than the maximal index of `idxs`!

!!! warning
    The values of `idxs` must be non-negative!

# Examples
```jldoctest
n = 4
idxs = [1, 3]
id = CxxMatrix(create_partial_cxx_identity(n, idxs), n, length(idxs))
# output

4×2 CxxMatrix:
 1.0  0.0
 0.0  0.0
 0.0  1.0
 0.0  0.0
```
The order in `idxs` defines the order of the basis vectors.
```jldoctest
n = 3
idxs = [3, 0, 1]
id = CxxMatrix(create_partial_cxx_identity(n, idxs), n, length(idxs))


# output

3×3 CxxMatrix:
 0.0  0.0  1.0
 0.0  0.0  0.0
 1.0  0.0  0.0
```
"""
function create_partial_cxx_identity(
    n::I_1, idxs::Vector{I_2}
) where {I_1<:Integer,I_2<:Integer}
    if n < maximum(idxs)
        throw(
            "ArgumentError: The number of rows must be greater than the largest index: $n < $(maximum(idxs)).",
        )
    end
    m = length(idxs)
    I = alloc_matrix(n, m)
    for j in 1:m
        for i in 1:n
            I[i, j] = 0.0
        end
        if idxs[j] > 0
            I[idxs[j], j] = 1.0
        end
    end
    return I
end

export CxxMatrix, CxxVector, CxxTensor, create_cxx_identity, create_partial_cxx_identity
export alloc_tensor, alloc_matrix, alloc_vector
end # module arry_types
