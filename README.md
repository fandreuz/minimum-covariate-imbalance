# Optimization algorithms for the minimum covariate imbalance problem

In this repository we implemented and experimented with several optimization
algorithms for the minimum covariate imbalance problem. The problem is proven
to be NP-hard when $P$ (the number of covariates) is strictly greater than 2.

We focused on some fast methods for the case $P = 2$, though some of the
functions can be used also in a more general case.

## Technlogies involved

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

$$ S = \min_{T: |T| = n} \sum_{p=1}^P \sum_{i=1}^{k_p} ||T \cap L'_{p,i}| - \ell_{p,i}| $$

where $T \cap L'_{p,i}$ is the subset of $T$ such that all the elements are in
the $i$-th level with respect to the $p$-th covariate.

## Algorithms

TODO

## Benchmarks

TODO

## Reference

Network flow methods for the minimum covariate imbalance problem

Dorit S. Hochbaum, Xu Rao, Jason Sauppe

https://arxiv.org/pdf/2007.06828.pdf
