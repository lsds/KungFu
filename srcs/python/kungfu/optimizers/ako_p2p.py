import tensorflow as tf

from kungfu.ops import send_to, _tensor_size, _bin_pack
from .core import KungFuOptimizer

def _send_with_return(rank, t):
    with tf.control_dependencies([send_to(rank, t)]):
        return tf.identity(t)

def _get_self_rank():
    import os
    return int(os.getenv('KUNGFU_TEST_SELF_RANK'))


def _get_num_peers():
    import json, os
    cluster_spec = json.loads(os.getenv('KUNGFU_CLUSTER_SPEC'))
    return len(cluster_spec['Peers'])

class AkoP2P(KungFuOptimizer):
    """An optimizer that negotiates using the AllReduce operator."""

    def __init__(self,
                 optimizer,
                 name=None,
                 use_locking=False,
                 device_dense='',
                 device_sparse='',
                 fraction=0.1):
        super(AkoP2P, self).__init__(optimizer, name, use_locking,
                                                device_dense, device_sparse)
        self.fraction = fraction

    def _negotiate_grads_by_strategy(self, grads_and_vars_to_negotiate):
        """Send grads to peers according to Ako algorithm"""
        gradients, variables = list(zip(*grads_and_vars_to_negotiate))

        total_size = sum([_tensor_size(t) for t in gradients])
        budget = int(self.fraction * total_size)
        indexes, num_partitions = _bin_pack(
            dict((t.name, _tensor_size(t)) for t in gradients), budget)
        groups = [[] for _ in range(num_partitions)]
        for t in gradients:
            groups[indexes[t.name]].append(t)

        send_ops = []
        for dest_rank in range(_get_num_peers()):
            k = dest_rank % num_partitions
            group_k = groups[k]
            for g in group_k:
                print("Gradient name: " + str(g.name))
                send_op = send_to(dest_rank, g)
                send_ops.append(send_op)
        with tf.control_dependencies(send_ops):
            id_grads = [tf.identity(g) for g in gradients]
            return list(zip(id_grads, variables))
