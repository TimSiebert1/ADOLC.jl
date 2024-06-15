# ADOLC.jl

*A Julia wrapper of the algorithmic differentiation package ADOL-C*

## Derivative Modes

# First-Order
| Mode          | Formula                  |
|:---------------:|:--------------------------:|
| jac         | $$Df(x)$$  |
| jac_vec    |  $$Df(x)\dot{v}$$            |
| jac_mat    |  $$Df(x)\dot{V}$$            |
| vec_jac    | $$\bar{z}^T Df(x)$$         |
| mat_jac    | $$\bar{Z}^T Df(x)$$       |


# Second-Order
| Mode             | Formula                       |
|:------------------:|:-------------------------------:|
| hess           | $$D^2f(x)$$               |
| hess_vec      | $$D^2f(x) \dot{v}$$          |
| hess_mat      | $$D^2f(x)  \dot{V}$$         |
| vec_hess      | $$\bar{z}^T D^2f(x)$$       |
| mat_hess      | $$\bar{Z}^T D^2f(x)$$       |
| vec_hess_vec | $$\bar{z}^T D^2f(x)  \dot{v}$$  |
| vec_hess_mat | $$\bar{z}^T D^2f(x)  \dot{V}$$ |
| mat_hess_mat | $$\bar{Z}^T D^2f(x)  \dot{V}$$  |
| mat_hess_vec | $$\bar{Z}^T D^2f(x)  \dot{v}$$  |


# Abs-Normal-Form
| Mode             | Formula                       |
|:------------------:|:-------------------------------:|
| abs_norm           | $$\Delta f(x)$$               |

## Higher-Order 

# ADOLC-Format
```math
\frac{\partial f(x)}{\partial x}
```

# Partial-Format
```math
\frac{\partial f(x)}{\partial x}
```

# Seed-Space

## Memory handling

## API Reference
```@contents
Pages = ["lib/reference.md"]
Depth = 1
```


## Index
```@index
Pages = ["reference]
```