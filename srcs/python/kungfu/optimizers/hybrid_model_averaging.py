import tensorflow as tf

from kungfu.ops import broadcast, save_model, request_model, model_averaging_with_schedule
from .core import KungFuOptimizer


def _get_self_rank():
    import os
    return int(os.getenv('KUNGFU_TEST_SELF_RANK'))


def _get_num_peers():
    import json, os
    cluster_spec = json.loads(os.getenv('KUNGFU_CLUSTER_SPEC'))
    return len(cluster_spec['Peers'])

def _try_as_int(s):
    try:
        return int(s)
    except ValueError:
        return float(s)

def _parse_schedule(schedule, num_train, batch_size):
    # schedule is of the form
    # x;e1;y;e2;z..., where x, y, z in {kf, p2p}
    tokens = schedule.split(",")
    print("Num train: " + str(num_train))
    print("Batch size: " + str(batch_size))
    to_gs = lambda epoch: int(epoch * num_train / (batch_size * _get_num_peers()))
    pairs = [(to_gs(_try_as_int(t.split(":")[0])), t.split(":")[1])
             for t in tokens]
    steps, strategies = zip(*pairs)
    steps, strategies = list(steps), list(strategies)

    print("Steps: " + str(steps))
    print("Strategies: " + str(strategies))
    return steps, strategies



class HybridPeerModelAveraging(KungFuOptimizer):
    """An optimizer that negotiates using the AllReduce operator."""

    def __init__(self,
                 optimizer,
                 schedule,
                 num_train,
                 batch_size,
                 all_reduce_interval=10,
                 model_averaging_device="cpu",
                 request_mode="sync",
                 peer_selection_strategy="random",
                 name=None,
                 use_locking=False,
                 device_dense='',
                 device_sparse=''):
        super(HybridPeerModelAveraging, self).__init__(optimizer, name, use_locking,
                                                 device_dense, device_sparse)
        print("Constructing HybridPeerModelAveragingOptimizer with schedule " + schedule + ".")
        print("Constructing HybridPeerModelAveragingOptimizer with num_train = " + \
              str(num_train) + ", batch_size= " + str(batch_size))

        self.request_mode = request_mode
        self.model_averaging_device = model_averaging_device
        self.peer_selection_strategy = peer_selection_strategy
        self.all_reduce_interval = all_reduce_interval
        self.steps, self.strategies = _parse_schedule(schedule, num_train, batch_size)

    @staticmethod
    def get_initializer():
        g = tf.get_default_graph()
        ops = []
        # TODO: auto inject tf.global_variables_initializer
        # with tf.control_dependencies([tf.global_variables_initializer()]):
        variables = g.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        for v in variables:
            ops.append(tf.assign(v, broadcast(v)))
        with tf.control_dependencies(ops):
            return save_model(tf.trainable_variables())

    def apply_gradients(self, grads_and_vars, **kwargs):
        """Calls this same method on the underlying optimizer."""

        grads, variables = zip(*grads_and_vars)

        if self.model_averaging_device == 'cpu':
            apply_avg_model = model_averaging_with_schedule(
                [i for i in range(_get_num_peers())], variables,
                self.request_mode, self.peer_selection_strategy,
                self.steps, self.strategies,
                self.all_reduce_interval)

            apply_op = self._optimizer.apply_gradients(grads_and_vars,
                                                       **kwargs)
            save_model_op = save_model(variables)

            with tf.control_dependencies([apply_avg_model]):
                with tf.control_dependencies([apply_op]):
                    with tf.control_dependencies([save_model_op]):
                        return tf.group(apply_op)
        elif self.model_averaging_device == 'gpu':
            raise Exception(
                "PeerModelAveraging optimizer does not support GPU request model type."
            )
        else:
            raise Exception(
                "PeerModelAveraging optimizer does not support provided request model type."
            )

    def _negotiate_grads_by_strategy(self, grads_and_vars_to_negotiate):
        return grads_and_vars_to_negotiate
