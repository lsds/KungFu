class _SynchronousSGDOptimizer(object):
    def __init__(self, optimizer, named_parameters, op):
        pass

    def zero_grad(self):
        print('TODO: zero_grad')
        pass

    def step(self):
        print('TODO: step')
        pass


def SynchronousSGDOptimizer(optimizer, named_parameters, op=None):
    if op is None:
        op = 'sum'
    return _SynchronousSGDOptimizer(optimizer, named_parameters, op)
