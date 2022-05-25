from itertools import combinations, product
import numpy as np

# samples from the control samples are indexed from 0 to n_prime
def extract_n_prime(L_prime, k):
    M = 0
    for i in range(k[0]):
        if len(L_prime[0][i]) > 0:
            M = max(M, np.max(L_prime[0][i]))
    return int(M) + 1


# number of treatment samples (and also cardinality of S)
def extract_n(l):
    return int(sum(l[0]))


def extract_k(l):
    return list(map(len, l))


def compute_imbalance(map_L_prime_to_value, S, l):
    s = 0
    for p in range(P):
        S = list(S)
        values, counts = np.unique(map_L_prime_to_value[p][S], return_counts=True)
        # we might be missing some values in values
        full_counts = np.zeros(len(l[p]), dtype=int)
        full_counts[values] = counts
        s += np.sum(np.abs(full_counts - l[p]))
    return s


def brute_force(l, L_prime):
    k = extract_k(l)
    n_prime = extract_n_prime(L_prime, k)
    n = extract_n(l)

    z = np.arange(n_prime)

    P = len(k)

    map_L_prime_to_value = []
    for p in range(P):
        ls = np.empty(n_prime, dtype=int)
        for i in range(k[p]):
            ls[L_prime[p][i]] = i
        map_L_prime_to_value.append(ls)

    m = 10000000
    S_m = None
    for S in combinations(z, n):
        mi = compute_imbalance(map_L_prime_to_value, S, l)
        if mi < m:
            m = mi
            S_m = [S]
        elif mi == m:
            S_m.append(S)
    return S_m, m
