import torch

from .clib import ops


class Allreduce(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, name):
        print('Allreduce::forward')
        print('ctx: %s' % (ctx))
        print('x: %s' % (x))
        # ctx.average = average
        # ctx.op = op
        # handle =
        output = ops.all_reduce(x)
        print('output: %s' % (output))
        y, = output
        print('y: %s' % (y))
        return y
        # return synchronize(handle)

    # @staticmethod
    # def backward(ctx, grad_output):
    #     print('Allreduce::backward')
    #     print(ctx)
    #     return ops.all_reduce(grad_output, average=ctx.average,
    #                           op=ctx.op), None, None, None


def all_reduce(x, name=None):
    print('calling python all_reduce')
    return Allreduce.apply(x, name)
