from time import time

def print_time(func):
    def wrapper(*args, **kwargs):
        start = time()
        value = func(*args, **kwargs)
        print(time() - start)
        return value
    return wrapper
