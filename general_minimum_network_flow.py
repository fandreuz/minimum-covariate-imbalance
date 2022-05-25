from linear_formulation import compute_A, compute_U
from brute_force import extract_k, extract_n, extract_n_prime
from minimum_network_flow import infinity, convert_networkx_to_ortools

from ortools.graph import pywrapgraph
import networkx as nx
import numpy as np
from functools import partial
from math import floor


def compute_zero_cost_arc_capacity(li, q, n):
    return floor(q / n * li)


def compute_one_capacity_arc_weight(li, q, n):
    t = q / n * li
    return 1 - (t - floor(t))


def binary_search(ls, left, right, condition):
    if left == right:
        return left

    m = floor((left + right) / 2)
    v = condition(ls[m])
    if v == 0:
        return m
    if v < 0:
        return binary_search(ls, m + 1, right, condition)
    return binary_search(ls, left, m, condition)


def condition(power, t):
    k = power * t
    if k / 10 == int(k / 10):
        return 1
    if k == int(k):
        return 0
    return -1


# find the smallest power of 10 which should be used to multiply in order to obtain
# integer weights for all the arcs
def minimum_scaling(ls, max_power=10):
    powers = np.power(10, np.arange(max_power + 1))

    top = 0
    for t in ls:
        if t == int(t):
            continue
        cnd = partial(condition, t=t)
        tp = binary_search(powers, 0, len(powers), cnd)
        if tp >= max_power:
            print("Ceiling reached with {}".format(t))
            tp = max_power - 1
        top = max(top, tp)
    return top


# we set all data explicitly because this will be used by another solver.
# max_power is the maximum exponent such that 10^max_power is an allowed multiplier to
# scale weights in the graph in order to have integer weights.
# this function returns the graph, and the multiplier used for the weights
def general_min_imbalance_network(l, L_prime, q, max_power):
    G = nx.DiGraph()

    G.add_node(0, demand=-q)
    G.add_node(1, demand=q)

    n = extract_n(l)
    k = extract_k(l)
    P = len(k)
    assert P == 2

    all_weights = [q / n] + [
        compute_one_capacity_arc_weight(l[p][i], q, n) * 2 / q
        for p in range(P)
        for i in range(k[p])
    ]
    powers = np.power(10, np.arange(max_power + 1))
    top = minimum_scaling(all_weights, max_power)

    for i in range(k[0]):
        G.add_node((0, i), demand=0)
    for i in range(k[1]):
        G.add_node((1, i), demand=0)

    # cost zero
    for i in range(k[0]):
        G.add_edge(
            0,
            (0, i),
            weight=1,
            capacity=compute_zero_cost_arc_capacity(l[0][i], q, n) * powers[top],
        )
    for i in range(k[1]):
        G.add_edge(
            (1, i),
            1,
            weight=1,
            capacity=compute_zero_cost_arc_capacity(l[1][i], q, n) * powers[top],
        )

    # capacity one
    for i in range(k[0]):
        G.add_edge(
            0,
            (0, i),
            weight=compute_one_capacity_arc_weight(l[0][i], q, n) * 2 / q * powers[top],
            capacity=1,
        )
    for i in range(k[1]):
        G.add_edge(
            (1, i),
            1,
            weight=compute_one_capacity_arc_weight(l[1][i], q, n) * 2 / q * powers[top],
            capacity=1,
        )

    # capacity infinity
    for i in range(k[0]):
        G.add_edge(0, (0, i), weight=2 / q * powers[top], capacity=infinity)
    for i in range(k[1]):
        G.add_edge((1, i), 1, weight=2 / q * powers[top], capacity=infinity)

    # x_{i1,i2}
    l = np.array(np.concatenate(l))
    n_prime = extract_n_prime(L_prime, k)
    A = compute_A(L_prime, k, n_prime)
    U = compute_U(A, k)

    for i1 in range(k[0]):
        for i2 in range(k[1]):
            if U[i1, i2] > 0:
                G.add_edge((0, i1), (1, i2), capacity=U[i1, i2], weight=0)

    return G, powers[top]


def general_min_imbalance_networkx_extract_result(G, dc):
    s = 0
    for source, dests in dc.items():
        # we are interested in counting only d and e
        for dest, value in dests.items():
            if not isinstance(dest, tuple) or not isinstance(source, tuple):
                s += value * G.get_edge_data(source, dest)["weight"]
    return s


def general_min_imbalance_solver_networkx(l, L_prime, q, verbose=False, max_power=10):
    net, scale = general_min_imbalance_network(l, L_prime, q, max_power=max_power)

    if verbose:
        for node in net.nodes(data=True):
            print("- o {}".format(node))
        for edg in net.edges(data=True):
            print("- > {}".format(edg))

    dc = nx.min_cost_flow(net)
    return general_min_imbalance_networkx_extract_result(net, dc) / scale


def general_min_imbalance_solver_google(l, L_prime, q, max_power=10):
    G, scale = general_min_imbalance_network(l, L_prime, q, max_power=max_power)

    min_cost_flow = pywrapgraph.SimpleMinCostFlow()
    convert_networkx_to_ortools(G, min_cost_flow)

    status = min_cost_flow.Solve()
    if status == min_cost_flow.INFEASIBLE:
        print("Infeasible")
    elif status == min_cost_flow.UNBALANCED:
        print("Unbalanced")
    else:
        return min_cost_flow.OptimalCost() / (scale)
