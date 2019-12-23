import time


def map_maybe(f, lst):
    return [f(x) if x is not None else None for x in lst]


def measure(f):
    t0 = time.time()
    result = f()
    duration = time.time() - t0
    return duration, result


def one_based_range(n):
    return range(1, 1 + n)
