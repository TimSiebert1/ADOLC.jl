#include <jlcxx/jlcxx.hpp>
#include <adolc/adolc.h>
#include <iostream>

adouble add(adouble const &a, adouble const &b)
{
  return a + b;
}

adouble add_right(adouble const &a, double const v)
{
  return a + v;
}

adouble add_left(double const v, adouble const &a)
{
  return v + a;
}

adouble diff(adouble const &a, adouble const &b)
{
  return a - b;
}

adouble diff_right(adouble const &a, double const v)
{
  return a - v;
}

adouble diff_left(double const v, adouble const &a)
{
  return v - a;
}
adouble diff_unary(adouble const &a)
{
  return (-1) * a;
}

adouble mult(adouble const &a, adouble const &b)
{
  return a * b;
}

adouble mult_right(adouble const &a, double const v)
{
  return a * v;
}

adouble mult_left(double const v, adouble const &a)
{
  return v * a;
}

adouble div2(adouble const &a, adouble const &b)
{
  return a / b;
}

adouble div_right(adouble const &a, double const v)
{
  return a / v;
}

adouble div_left(double const v, adouble const &a)
{
  return v / a;
}

adouble fabs2(adouble const &a)
{
  return fabs(a);
}

void assign(adouble &x, double val)
{
  x <<= val;
}

double dassign(adouble &x, double &val)
{
  x >>= val;
  return val;
}
adouble power(adouble x, int n)
{
  adouble z = 1;

  if (n > 0) /* Recursion and branches */
  {
    int nh = n / 2;   /* that do not depend on  */
    z = power(x, nh); /* adoubles are fine !!!! */
    z *= z;
    if (2 * nh != n)
      z *= x;
    return z;
  } /* end if */
  else
  {
    if (n == 0) /* The local adouble z dies */
      return z; /* as it goes out of scope. */
    else
      return 1 / power(x, -n);
  } /* end else */
}

double *alloc_vec(int size)
{
  double *A = new double[size];
  return A;
}

void write_out(double *A)
{
  for (auto i = 0; i < 2; i++)
  {
    std::cout << A[i] << std::endl;
  }
}
adouble sqrt2(adouble const &a)
{
  return sqrt(a);
}
adouble exp2(adouble const &a)
{
  return exp(a);
}
adouble fmax2(adouble const &a, adouble const &b)
{
  return fmax(a, b);
}

JLCXX_MODULE define_julia_module(jlcxx::Module &types)
{
  types.add_type<adouble>("adouble")
      .constructor<double>();

  types.method("getValue", [](adouble &a)
               { return a.getValue(); });
  types.method("gradient", [](int tag, int n, double *x, double *g)
               { return gradient(tag, n, x, g); });
  types.method("alloc_vec", alloc_vec);
  types.method("myalloc2", myalloc2);
  types.method("trace_on", [](int tag)
               { return trace_on(tag); });
  types.method("trace_off", trace_off);
  types.method("forward", [](short tag, int m, int n,
                             int d,
                             int keep,
                             double **X,
                             double **Y)
               { forward(tag,
                         m,
                         n,
                         d,
                         keep,
                         X,
                         Y); });
  types.method("reverse2", [](short tag,
                              int m,
                              int n,
                              int d,
                              double *u,
                              double **Z)
               { reverse(tag, m, n, d, u, Z); });

  // basic arithmetic operations
  types.set_override_module(jl_base_module);

  types.method("+", add);
  types.method("+", add_right);
  types.method("+", add_left);

  types.method("-", diff);
  types.method("-", diff_right);
  types.method("-", diff_left);
  types.method("-", diff_unary);

  types.method("*", mult);
  types.method("*", mult_left);
  types.method("*", mult_right);

  types.method("/", div2);
  types.method("/", div_left);
  types.method("/", div_right);

  types.method("<<", assign);
  types.method(">>", dassign);

  types.method("^", [](adouble x, int n)
               { return power(x, n); });

  types.method("max", [](adouble const &a, adouble const &b)
               { return fmax2(a, b); });
  types.method("abs", [](adouble const &a)
               { return fabs2(a); });
  types.method("sqrt", [](adouble const &a)
               { return sqrt2(a); });
  types.method("exp", [](adouble const &a)
               { return exp2(a); });
  types.unset_override_module();

  // utils for accessing matrices
  types.method("getindex_mat", [](double **A, const int &row, const int &col)
               { return A[row - 1][col - 1]; });
  types.method("setindex_mat", [](double **A, const double val, const int &row, const int &col)
               { A[row - 1][col - 1] = val; });
  types.method("getindex_vec", [](double *A, const int &row)
               { return A[row - 1]; });
  types.method("setindex_vec", [](double *A, const double val, const int &row)
               { A[row - 1] = val; });
}

int main()
{
  adouble c = 5.0;
  adouble d = 4.0;
  double **X = myalloc2(1, 3);
  double *A;
  A = alloc_vec(2);
  write_out(A);
  condassign(d, c - d, c, d);
  std::cout << d.getValue() << std::endl;
  std::cout << A[1] << std::endl;
}