module array_types
    using ADOLC_jll
    using CxxWrap


    @wrapmodule(()->libadolc_wrap, :julia_module_array_types)

    function __init__()
    @initcxx
    end


##### raw CxxPtr utilities ###################

Base.getindex(X::CxxPtr{CxxPtr{CxxPtr{Cdouble}}}, dim::Int64, row::Int64, col::Int64) = getindex_tens(X, dim, row, col)
Base.setindex!(X::CxxPtr{CxxPtr{CxxPtr{Cdouble}}}, val::Cdouble, dim::Int64, row::Int64, col::Int64) = setindex_tens(X, val, dim, row, col)


Base.getindex(X::CxxPtr{CxxPtr{Cdouble}}, row::Int64, col::Int64) = getindex_mat(X, row, col)
Base.setindex!(X::CxxPtr{CxxPtr{Cdouble}}, val::Cdouble, row::Int64, col::Int64) = setindex_mat(X, val, row, col)


Base.getindex(X::CxxPtr{Cdouble}, row::Int64) = getindex_vec(X, row)
Base.setindex!(X::CxxPtr{Cdouble}, val::Cdouble, row::Int64) = setindex_vec(X, val, row)

Base.getindex(X::CxxPtr{Cshort}, row::Int64) = getindex_vec(X, row)
Base.setindex!(X::CxxPtr{Cshort}, val::Cshort, row::Int64) = setindex_vec(X, val, row)

################################################




###### double** wrappe #########################

struct CxxMatrix{T} <: AbstractMatrix{T} 
    """
    Wrapper for c++ double** data
    """
    data::CxxPtr{CxxPtr{T}}
    n::Int64 
    m::Int64
    function CxxMatrix{T}(n::Integer, m::Integer) where T <: Real
        check_type_mat(T)
        new{T}(alloc_mat(T, n, m), n, m)
    end
    function CxxMatrix{T}(Y::Matrix{T}) where T <: Real
        check_type_mat(T)
        n, m = size(Y)
        X = new{T}(alloc_mat(T, n, m), n, m)
        for i in 1:n
            for j in 1:m
                X[i, j] = Y[i, j]
            end
        end
        return X
    end
end

function check_type_mat(::Type{T}) where T <: Real
    if !(T <: Union{Cdouble})
       throw("Not implemented for type $T")
    end
end

alloc_mat(::Type{Float64}, n::Integer, m::Integer) = myalloc2(n, m)

function Base.axes(X::CxxMatrix{T}, n::Int64) where T <: Real
    if n == 1 
        return Base.OneTo(X.n)
    elseif n==2
        return Base.OneTo(X.m)
    else 
        return Base.OneTo(1)
    end
end

Base.axes(X::CxxMatrix{T}) where T <: Real = (Base.OneTo(X.n), Base.OneTo(X.m))

Base.size(X::CxxMatrix{T}) where T <: Real = (X.n, X.m)
Base.getindex(X::CxxMatrix{T}, row::Int64, col::Int64) where T <: Real = getindex(X.data, row, col)
Base.setindex!(X::CxxMatrix{T}, val::T, row::Int64, col::Int64) where T <: Real = setindex!(X.data, val, row, col)


#######################################

alloc_vec(::Type{Cdouble}, n::Integer) = alloc_vec_double(n)
alloc_vec(::Type{Cshort}, n::Integer) = alloc_vec_short(n)

struct CxxVector{T} <: AbstractVector{T}
    data::CxxPtr{T}
    n::Int64 
    function CxxVector{T}(n::Integer) where T <: Real
        check_type_vec(T)
        new{T}(alloc_vec(T, n), n)
    end
    function CxxVector{T}(y::Vector{T}) where T <: Real
        check_type_vec(T)
        n = length(y)
        x = new{T}(alloc_vec(T, n), n)
        for i in 1:n
            x[i] = y[i]
        end   
        return x
    end
    
end

function check_type_vec(::Type{T}) where T <: Real
    if !(T <: Union{Cdouble, Cshort})
       throw("Not implemented for type $T")
    end
end



Base.axes(X::CxxVector{T}, n::Int64) where T <: Real = n == 1 ? Base.OneTo(X.n) : Base.OneTo(1)
Base.axes(X::CxxVector{T}) where T <: Real = (Base.OneTo(X.n),)

Base.size(X::CxxVector{T}) where T <: Real = X.n
Base.getindex(X::CxxVector{T}, row::Int64) where T <: Real = getindex(X.data, row)
Base.setindex!(X::CxxVector{T}, val::T, row::Int64) where T <: Real = setindex!(X.data, val, row)

export CxxMatrix, CxxVector, myalloc3, myalloc2, myalloc1, alloc_vec_double, alloc_vec_short, alloc_vec, alloc_mat_short

end # module arry_types
