Base.getindex(X::CxxPtr{CxxPtr{Float64}}, row::Int64, col::Int64) = getindex_mat(X, row, col)
Base.setindex!(X::CxxPtr{CxxPtr{Float64}}, val::Float64, row::Int64, col::Int64) = setindex_mat(X, val, row, col)
Base.getindex(X::CxxPtr{Float64}, row::Int64) = getindex_vec(X, row)
Base.setindex!(X::CxxPtr{Float64}, val::Float64, row::Int64) = setindex_vec(X, val, row)
  