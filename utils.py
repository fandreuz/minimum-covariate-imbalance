from time import time
import numpy as np
from functools import wraps


def print_time(time_file):
    def decorator_print_time(func):
        @wraps(func)
        def wrapper_print_time(*args, **kwargs):
            start = time()
            value = func(*args, **kwargs)
            end = time() - start

            s = "# {}".format(end)
            if time_file:
                time_file.write(s + "\n")
            else:
                print(s)
            return value

        return wrapper_print_time

    return decorator_print_time


def group_by(a):
    a = a[a[:, 0].argsort()]
    return np.split(a[:, 1], np.unique(a[:, 0], return_index=True)[1][1:])


# P is the number of covariates
# k is the number of levels per covariate
# n is the dimension of the sample
def random_sample(P, n, k, probability_generator=None):
    if probability_generator is None:

        def probability_generator(arr):
            return np.ones_like(arr) / len(arr)

    ls = [None for p in range(P)]
    for p in range(P):
        arr = np.arange(k[p])
        probabilities = probability_generator(arr)

        # randomly select a level for each control value
        choice = np.random.choice(arr, size=n, p=probabilities)
        indexed_choices = np.hstack([choice[:, None], np.arange(n)[:, None]])
        gb = group_by(indexed_choices)

        un = np.unique(choice)
        # add missing levels
        i = 0
        for u in un:
            for j in range(i, u):
                gb.insert(i, [])
            i = u + 1
        for j in range(i, k[p]):
            gb.append([])

        ls[p] = gb
    return ls


def random_Lprime(P, n_prime, k, probability_generator=None):
    return random_sample(
        P, n_prime, k, probability_generator=probability_generator
    )


def print_Lprime(L_prime):
    for p in range(len(L_prime)):
        print(p, L_prime[p])


def random_l(P, n, k, probability_generator=None):
    ls = random_sample(P, n, k, probability_generator=probability_generator)
    return list(map(lambda l: list(map(len, l)), ls))


# with P=2
def generate_problems(n, n_prime, k1, k2, probability_generator=None):
    l = random_l(2, n, (k1, k2), probability_generator=probability_generator)
    L_prime = random_Lprime(
        2, n_prime, (k1, k2), probability_generator=probability_generator
    )
    return l, L_prime
