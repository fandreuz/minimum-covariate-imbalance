from brute_force import extract_k, extract_n, extract_n_prime
from utils import print_time

import numpy as np
import gurobipy as gb

@print_time
def solve(problem):
    problem.optimize()

# each k[i] consecutive rows of A contain 1 in the j-th
# column if for the (i+1)-th covariate we have that
# z[j] belongs to the k-th bucket of the covariate i,
# where k is the offset from k[i]+sum(k[:i])
def compute_A(L_prime, k, n_prime):
    A = np.zeros((sum(k), n_prime), dtype=int)
    current_row = 0
    for p in range(len(L_prime)):
        Lp_prime = L_prime[p]
        for i in range(k[p]):
            A[current_row, Lp_prime[i]] = 1
            current_row += 1
    return A


def min_imbalance_solver(l, L_prime, verbose=False):
    min_imbalance = gb.Model()
    min_imbalance.modelSense = gb.GRB.MINIMIZE
    min_imbalance.setParam("outputFlag", 0)

    n = extract_n(l)
    k = extract_k(l)
    P = len(k)
    l = np.array(np.concatenate(l))
    n_prime = extract_n_prime(L_prime, k)

    # 1e
    z = min_imbalance.addMVar(n_prime, vtype=gb.GRB.BINARY)
    y = min_imbalance.addMVar(sum(k))

    A = compute_A(L_prime, k, n_prime)

    # 1b
    min_imbalance.addConstr(A @ z - l <= y)
    # 1c
    min_imbalance.addConstr(l - A @ z <= y)
    # 1d
    min_imbalance.addConstr(gb.quicksum(z) == n)

    # 1a
    min_imbalance.setObjective(gb.quicksum(y))

    solve(min_imbalance)

    if verbose:
        if min_imbalance.status == 2:
            print("OK")
        else:
            print("Bad things happened")
    return z.x, sum(y.x)


def min_imbalance_solver_alt(l, L_prime, verbose=False):
    min_imbalance = gb.Model()
    min_imbalance.modelSense = gb.GRB.MINIMIZE
    min_imbalance.setParam("outputFlag", 0)

    n = extract_n(l)
    k = extract_k(l)
    P = len(k)
    l = np.array(np.concatenate(l))
    n_prime = extract_n_prime(L_prime, k)

    assert P == 2

    # 2f
    z = min_imbalance.addMVar(n_prime, vtype=gb.GRB.BINARY)
    # 2e
    e = min_imbalance.addMVar(sum(k), lb=0.0)
    d = min_imbalance.addMVar(sum(k), lb=0.0)

    A = compute_A(L_prime, k, n_prime)

    for p in range(2):
        # smallest index for the covariate p
        bottom_index = sum(k[:p])
        # biggest index for the covariate p
        top_index = k[p] + bottom_index

        sl = slice(bottom_index, top_index)
        # 2b/2c
        min_imbalance.addConstr(A[sl] @ z + d[sl] - e[sl] == l[sl])

    # 2d
    min_imbalance.addConstr(gb.quicksum(e[: k[0]]) - gb.quicksum(d[: k[0]]) == 0)

    # 2a
    min_imbalance.setObjective(gb.quicksum(e) + gb.quicksum(d))

    solve(min_imbalance)

    if verbose:
        if min_imbalance.status == 2:
            print("OK")
        else:
            print("Bad things happened")
    return z.x, sum(e.x) + sum(d.x)


def compute_U(A, k):
    U = np.empty((k[0], k[1]), dtype=int)

    A1 = A[: k[0]] > 0
    A2 = A[k[0] :] > 0

    return np.count_nonzero(np.logical_and(A1[:, None], A2[None]), axis=-1)


def X_to_Z(A, k, X):
    U = np.empty((k[0], k[1]), dtype=int)

    A1 = A[: k[0]] > 0
    A2 = A[k[0] :] > 0

    B = np.logical_and(A1[:, None], A2[None])

    z = np.zeros(B.shape[2])
    for i1 in range(B.shape[0]):
        for i2 in range(B.shape[1]):
            # i1 \cap i2
            intersection = int(X[i1 * k[1] + i2])
            # we take the first intersection values in the intersection
            take = np.where(B[i1, i2])[0][:intersection]
            z[take] = 1
    return z


def min_imbalance_solver_mcnf(l, L_prime, verbose=False):
    min_imbalance = gb.Model()
    min_imbalance.modelSense = gb.GRB.MINIMIZE
    min_imbalance.setParam("outputFlag", 0)

    n = extract_n(l)
    k = extract_k(l)
    P = len(k)
    l = np.array(np.concatenate(l))
    n_prime = extract_n_prime(L_prime, k)

    assert P == 2

    A = compute_A(L_prime, k, n_prime)
    U = compute_U(A, k)

    # 3g (i2 changes faster than i1)
    x = min_imbalance.addMVar(U.shape[0] * U.shape[1], lb=0.0, ub=U.flatten(order="C"))
    # 3f
    e = min_imbalance.addMVar(sum(k), lb=0.0)
    d = min_imbalance.addMVar(sum(k), lb=0.0)

    for p in range(2):
        # 3b
        for i1 in range(k[0]):
            x_i1 = gb.quicksum(x[k[1] * i1 : k[1] * (i1 + 1)])
            min_imbalance.addConstr(x_i1 + d[i1] - e[i1] == l[i1])

        # 3c
        for i2 in range(k[1]):
            x_i2 = gb.quicksum(x[i2 :: k[1]])
            min_imbalance.addConstr(
                -x_i2 - d[k[0] + i2] + e[k[0] + i2] == -l[k[0] + i2]
            )

        # 3d
        min_imbalance.addConstr(gb.quicksum(e[: k[0]]) - gb.quicksum(d[: k[0]]) == 0)

        # 3e
        min_imbalance.addConstr(gb.quicksum(e[k[0] :]) - gb.quicksum(d[k[0] :]) == 0)

    # 3a
    min_imbalance.setObjective(gb.quicksum(e) + gb.quicksum(d))

    solve(min_imbalance)

    if verbose:
        if min_imbalance.status == 2:
            print("OK")
        else:
            print("Bad things happened")
    return X_to_Z(A, k, x.x), sum(e.x) + sum(d.x)
