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
adouble fmin2(adouble const &a, adouble const &b)
{
  return fmin(a, b);
}

JLCXX_MODULE Adouble_module(jlcxx::Module &types)
{
  types.add_type<adouble>("AdoubleCxx", jlcxx::julia_type("AbstractFloat", "Base"))
      .constructor<double>();

  types.method("getValue", [](adouble &a)
               { return a.getValue(); });
  types.method("gradient", [](int tag, int n, double *x, double *g)
               { return gradient(tag, n, x, g); });
  types.method("trace_on", [](int tag)
               { return trace_on(tag); });
  types.method("trace_on", [](int tag, int keep)
               { return trace_on(tag, keep); });
  types.method("trace_off", trace_off);

  // easy to use drivers

  types.method("jacobian", jacobian);
  types.method("hessian", hessian);
  types.method("vec_jac", vec_jac);
  types.method("jac_vec", jac_vec);
  types.method("hess_vec", hess_vec);
  types.method("hess_mat", hess_mat);
  types.method("lagra_hess_vec", lagra_hess_vec);
  types.method("jac_solv", jac_solv);

  types.method("ad_forward", [](short tag, int m, int n,
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
  types.method("ad_reverse", [](short tag,
                                int m,
                                int n,
                                int d,
                                double *u,
                                double **Z)
               { reverse(tag, m, n, d, u, Z); });

  types.method("zos_forward", zos_forward);
  types.method("fos_forward", fos_forward);
  types.method("hos_forward", hos_forward);

  types.method("fov_forward", fov_forward);
  types.method("hov_forward", hov_forward);

  types.method("fos_reverse", fos_reverse);
  types.method("hos_reverse", hos_reverse);

  types.method("fov_reverse", fov_reverse);
  types.method("hov_reverse", hov_reverse);

  // easy to use higher order drivers
  types.method("tensor_address", tensor_address);
  types.method("tensor_eval", tensor_eval);

  // pointwise-smooth functions
  types.method("enableMinMaxUsingAbs", enableMinMaxUsingAbs);
  types.method("get_num_switches", get_num_switches);
  types.method("zos_pl_forward", zos_pl_forward);
  types.method("fos_pl_forward", fos_pl_forward);
  types.method("fov_pl_forward", fov_pl_forward);
  types.method("abs_normal", abs_normal);

  //--------------------

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
  types.method("min", [](adouble const &a, adouble const &b)
               { return fmin2(a, b); });
  types.method("abs", [](adouble const &a)
               { return fabs2(a); });
  types.method("sqrt", [](adouble const &a)
               { return sqrt2(a); });
  types.method("exp", [](adouble const &a)
               { return exp2(a); });
  types.unset_override_module();
}
