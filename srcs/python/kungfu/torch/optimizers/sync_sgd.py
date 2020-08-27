import torch
from kungfu.torch.ops import (inplace_all_reduce_async_op,
                              inplace_all_reduce_op, wait_all_handles)


class _SynchronousSGDOptimizer(torch.optim.Optimizer):
    def __init__(self, param_groups, named_parameters, op):
        super(self.__class__, self).__init__(param_groups)
        self._named_parameters = named_parameters
        self._op = op

    def sync_gradients(self):
        # FIXME: make sure order is consistent across all workers
        handles = []
        for name, p in self._named_parameters:
            if p.requires_grad:
                if '.cuda.' in p.grad.type():
                    h = inplace_all_reduce_async_op(p.grad, name, self._op)
                    handles.append(h)
                else:
                    inplace_all_reduce_op(p.grad, self._op)
        wait_all_handles(handles)

    def step(self, closure=None):
        self.sync_gradients()
        return super(self.__class__, self).step(closure)


def SynchronousSGDOptimizer(optimizer, named_parameters, op=None):
    clazz = type(optimizer.__class__.__name__, (optimizer.__class__, ),
                 dict(_SynchronousSGDOptimizer.__dict__))
    return clazz(optimizer.param_groups, list(named_parameters), op)
