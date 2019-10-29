def map_maybe(f, lst):
    return [f(x) if x is not None else None for x in lst]
