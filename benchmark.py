from minimum_network_flow import (
    min_imbalance_solver_networkx,
    min_imbalance_solver_google,
)
from linear_formulation import (
    min_imbalance_solver_mcnf,
    min_imbalance_solver_alt,
    min_imbalance_solver,
    compute_A,
    compute_U,
)

from utils import generate_problems
from time import time

time_file = open("benchmark.txt", "w")

for n in range(500, 5001, 500):
    for nprime in [10000, 25000, 50000, 75000, 100000]:
        for func in [
            lambda x: int(x / 3),
            lambda x: int(x / 2),
            lambda x: x,
            lambda x: x * 2,
            lambda x: x * 3,
        ]:
            k1 = func(n)
            k2 = func(n)
            k = (k1, k2)

            while True:
                print(
                    "n {}, n_prime {}, k1 {}, k2 {}".format(n, nprime, k1, k2)
                )
                l, L_prime = generate_problems(n, nprime, k1, k2)

                start = time()
                try:
                    A = compute_A(L_prime, k, nprime)
                except:
                    print(
                        "An error occurred while computing A. L_prime: {}, k: {}".format(
                            L_prime, j
                        )
                    )
                    continue
                print("A computed in {}s".format(time() - start))

                start = time()
                U = compute_U(A, k1)
                print("U computed in {}s".format(time() - start))

                if (
                    min_imbalance_solver(l, L_prime, A=A, time_file=time_file)
                    is None
                ):
                    print("Timeout!")
                    continue

                min_imbalance_solver_alt(l, L_prime, A=A, time_file=time_file)
                min_imbalance_solver_mcnf(l, L_prime, A=A, time_file=time_file)
                min_imbalance_solver_networkx(
                    l, L_prime, A=A, U=U, time_file=time_file
                )
                min_imbalance_solver_google(
                    l, L_prime, A=A, U=U, time_file=time_file
                )
                print("-----")
                break
