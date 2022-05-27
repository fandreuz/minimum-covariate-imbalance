from minimum_network_flow import (
    min_imbalance_solver_networkx,
    min_imbalance_solver_google,
)
from linear_formulation import (
    min_imbalance_solver_mcnf,
    min_imbalance_solver_alt,
    min_imbalance_solver,
)

from utils import generate_problems
from rich.progress import track
from itertools import product

data = list(product(range(1000, 10001, 1000), range(10000, 1000001, 10000)))
for n, nprime in track(data, description="Computing all cases..."):
    if nprime < n + 1000:
        continue

    for k1, k2 in [(3, 3), (50, 50), (100, 100)]:
        while True:
            print("n {}, n_prime {}, k1 {}, k2 {}".format(n, nprime, k1, k2))
            l, L_prime = generate_problems(n, nprime, k1, k2)

            if min_imbalance_solver(l, L_prime) is None:
                print("Timeout!")
                continue

            min_imbalance_solver_alt(l, L_prime)
            min_imbalance_solver_mcnf(l, L_prime)
            min_imbalance_solver_networkx(l, L_prime)
            min_imbalance_solver_google(l, L_prime)
            print("-----")
            break
