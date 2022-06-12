# Optimization algorithms for the minimum covariate imbalance problem

In this repository we implemented and experimented with several optimization
algorithms for the minimum covariate imbalance problem. The problem is proven
to be NP-hard when $P$ (the number of covariates) is strictly greater than 2.

We focused on some fast methods for the case $P = 2$, though some of the
functions can be used also in a more general case.

## Tools

We employed several different tools in order to evaluate the best one for the
problem. All the code is written in Python, with the support of some NumPy
functions here and there. We used the following optimization engines:
- Gurobi
- Google OR-Tools
- NetworkX (minimum network flow solver)

## Overview

We consider a so-called *treatment sample* of size $n$ represented by
$\ell_{p,i}$ such that $p \in \{1, \dots, P\}$ is the index of a covariate and
$i \in k_p$ is the level among the allowed levels for the $p$-th covariate.
The objective of the problem is the identification, among another set called
*control sample*, of a subset $S$ of size $n$ such that the following holds:

$$S = \min_{T: |T| = n} \sum_{p=1}^P \sum_{i=1}^{k_p} ||T \cap L_{p,i}'| - \ell_{p,i}|$$

where $T \cap L'_{p,i}$ is the subset of $T$ such that all the elements are in
the $i$-th level with respect to the $p$-th covariate.

## Models implemented

- MIP model ([1], Section 2)
  - Gurobi
- Alternative MIP model ([1], Section 3)
  - Gurobi
- MCNF model ([1], Section 4)
  - Gurobi
  - NetworkX
  - Google OR-Tools
- General (`q != n`) MCNF model ([1], Section 6)
  - NetworkX
  - Gurobi

## Running the code

First of all we generate randomly a problem using the utility functions inside
the module `utils`:

```python
from utils import generate_problems

n = 5
n_prime = 15
k0 = k1 = 5

l, L_prime = generate_problems(n, n_prime, k0, k1)
```

You can now use any method from the modules in the repository in order to solve
the problem:

```python
# brute force solver
from brute_force import brute_force
brute_force(l, L_prime)

# MIP model
from mip_formulation import min_imbalance_solver, min_imbalance_solver_alt, min_imbalance_solver_mcnf
min_imbalance_solver(l, L_prime)
min_imbalance_solver_alt(l, L_prime)
min_imbalance_solver_mcnf(l, L_prime)

# MCNF model
from minimum_network_flow import min_imbalance_solver_networkx, min_imbalance_solver_google
min_imbalance_solver_networkx(l, L_prime)
min_imbalance_solver_google(l, L_prime)
```

## Benchmarks

### Increasing `n` and `k1, k2` (`n' = 500`, `k1 = k2 = n/2`)
![1](https://user-images.githubusercontent.com/8464342/173231385-72e6c808-6050-4203-a330-dd35437c62c0.png)

### Increasing `n'` (`n = 50`, `k1, k2 = 50`)
![2](https://user-images.githubusercontent.com/8464342/173231392-fdc6dbe3-4568-4cc2-b0d3-2f6bbe2631fd.png)

### Increasing `k1, k2` (`n = 100`, `n' = 1.000.000`)
![3](https://user-images.githubusercontent.com/8464342/173231394-1d44401c-6b3d-47e0-9f50-996de39331ba.png)

### Legend

- Gurobi:
  - `Integer` : MIP formulation in [1]
  - `Integer` : Alternative MIP formulation in [1]
  - `Integer MCNF` : MCNF formulation implemented like a MIP
- Google OR-Tools:
  - `MCNF OR` : MCFN formulation
- NetworkX:
  - `MCNF NX` : MCFN formulation

## Reference

[1] Network flow methods for the minimum covariate imbalance problem

Dorit S. Hochbaum, Xu Rao, Jason Sauppe

https://arxiv.org/pdf/2007.06828.pdf
