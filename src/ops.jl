Base.getindex(X::CxxPtr{CxxPtr{Float64}}, row::Int64, col::Int64) = getindex_mat(X, row, col)
Base.setindex!(X::CxxPtr{CxxPtr{Float64}}, val::Float64, row::Int64, col::Int64) = setindex_mat(X, val, row, col)
Base.getindex(X::CxxPtr{Float64}, row::Int64) = getindex_vec(X, row)
Base.setindex!(X::CxxPtr{Float64}, val::Float64, row::Int64) = setindex_vec(X, val, row)
  


# convient inits for independant and dependants
function init_independent_vec(a::Vector{Main.ADOLC_wrap.adoubleAllocated}, x::Vector{Float64})
    for i in 1:length(x)
        a[i] << x[i]
    end
end
Base.:<<(a::Vector{Main.ADOLC_wrap.adoubleAllocated}, x::Vector{Float64}) = init_independent_vec(a, x)

function init_dependent_vec(a::Vector{Main.ADOLC_wrap.adoubleAllocated}, x::Vector{Float64})
    for i in 1:length(x)
        a[i] >> x[i]
    end
end
Base.:>>(a::Vector{Main.ADOLC_wrap.adoubleAllocated}, x::Vector{Float64}) = init_dependent_vec(a, x)



