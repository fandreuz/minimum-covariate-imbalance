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

for n in range(5, 101, 5):
    for nprime in range(n+5, 101, 5):
        for k1, k2 in [(3,3), (4,4), (5,5), (3,10)]:
            l, L_prime = generate_problems(n, nprime, k1, k2)

            min_imbalance_solver(l, L_prime)
            min_imbalance_solver_alt(l, L_prime)
            min_imbalance_solver_mcnf(l, L_prime)
            min_imbalance_solver_networkx(l, L_prime)
            min_imbalance_solver_google(l, L_prime)
            print('-----')
