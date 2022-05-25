from linear_formulation import compute_A, compute_U
from brute_force import extract_k, extract_n, extract_n_prime

from ortools.graph import pywrapgraph
import networkx as nx
import numpy as np

infinity = 100000000

# we set all data explicitly because this will be used by another solver
def min_imbalance_network(l, L_prime):
    G = nx.DiGraph()

    G.add_node(0, demand=0)
    G.add_node(1, demand=0)

    n = extract_n(l)
    k = extract_k(l)
    P = len(k)
    assert P == 2

    for i in range(k[0]):
        G.add_node((0, i), demand=-l[0][i])
    for i in range(k[1]):
        G.add_node((1, i), demand=l[1][i])

    # excess
    for i in range(k[0]):
        G.add_edge(0, (0, i), weight=1, capacity=infinity)
    for i in range(k[1]):
        G.add_edge((1, i), 1, weight=1, capacity=infinity)

    # deficit
    for i in range(k[0]):
        G.add_edge((0, i), 0, weight=1, capacity=infinity)
    for i in range(k[1]):
        G.add_edge(1, (1, i), weight=1, capacity=infinity)

    # x_{i1,i2}
    l = np.array(np.concatenate(l))
    n_prime = extract_n_prime(L_prime, k)
    A = compute_A(L_prime, k, n_prime)
    U = compute_U(A, k)

    for i1 in range(k[0]):
        for i2 in range(k[1]):
            if U[i1, i2] > 0:
                G.add_edge((0, i1), (1, i2), capacity=U[i1, i2], weight=0)

    return G


def min_imbalance_networkx_extract_result(dc):
    s = 0
    for source, dests in dc.items():
        # we are interested in counting only d and e
        for dest, value in dests.items():
            if not isinstance(dest, tuple) or not isinstance(source, tuple):
                s += value
    return s


def min_imbalance_solver_networkx(l, L_prime, verbose=False):
    net = min_imbalance_network(l, L_prime)

    if verbose:
        for node in net.nodes(data=True):
            print("- o {}".format(node))
        for edg in net.edges(data=True):
            print("- > {}".format(edg))

    dc = nx.min_cost_flow(net)
    return min_imbalance_networkx_extract_result(dc)


def convert_networkx_to_ortools(net, ortools_flow_obj):
    mapping = {}
    i = 0

    # weight and capacity
    for source, dest, data in net.edges(data=True):
        if source not in mapping:
            mapping[source] = i
            i += 1
        if dest not in mapping:
            mapping[dest] = i
            i += 1

        assert int(data["capacity"]) == data["capacity"]
        assert int(data["weight"]) == data["weight"]

        arc = ortools_flow_obj.AddArcWithCapacityAndUnitCost(
            mapping[source],
            mapping[dest],
            int(data["capacity"]),
            int(data["weight"]),
        )

    for node, data in net.nodes(data=True):
        assert int(data["demand"]) == data["demand"]
        ortools_flow_obj.SetNodeSupply(mapping[node], -int(data["demand"]))


def min_imbalance_solver_google(l, L_prime):
    G = min_imbalance_network(l, L_prime)

    min_cost_flow = pywrapgraph.SimpleMinCostFlow()
    convert_networkx_to_ortools(G, min_cost_flow)

    status = min_cost_flow.Solve()
    if status == min_cost_flow.INFEASIBLE:
        print("Infeasible")
    elif status == min_cost_flow.UNBALANCED:
        print("Unbalanced")
    else:
        return min_cost_flow.OptimalCost()
