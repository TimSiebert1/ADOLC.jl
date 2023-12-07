#include <jlcxx/jlcxx.hpp>
#include <adolc/adolc.h>

JLCXX_MODULE julia_module_array_types(jlcxx::Module &types)
{

    types.method("alloc_vec_double", myalloc1);
    types.method("alloc_vec_short", [](int i)
                 { return new short[i]; });
    types.method("myalloc2", myalloc2);

    // utils for accessing matrices or vectors
    types.method("getindex_mat", [](double **A, const int &row, const int &col)
                 { return A[row - 1][col - 1]; });
    types.method("setindex_mat", [](double **A, const double val, const int &row, const int &col)
                 { A[row - 1][col - 1] = val; });
    types.method("getindex_vec", [](const double *A, const int &row)
                 { return A[row - 1]; });
    types.method("setindex_vec", [](double *A, const double val, const int &row)
                 { A[row - 1] = val; });

    types.method("setindex_vec", [](short *A, const short &val, const int &row)
                 { A[row - 1] = val; });
    types.method("getindex_vec", [](const short *A, const int &row)
                 { return A[row - 1]; });
}
