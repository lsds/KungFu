from kungfu.internal import _get_num_peers, _get_other_ranks, _get_self_rank

from .adapt import (get_init_version, get_start_step, propose_update,
                    start_step, update_cluster)
from .collective import (all_reduce, all_reduce_gpu, barrier, broadcast,
                         cpu_group_all_reduce, global_variance,
                         gpu_group_all_reduce, group_all_reduce)
from .loader import _has_gpu, _init_lib, _op_lib
from .local import save_variable, save_variables
from .p2p import request_variable, request_variable_with_template


def _tensor_size(t):
    return t.shape.num_elements() * t.dtype.size


# metadata APIs
def current_rank():
    return _init_lib.kungfu_rank()


def current_cluster_size():
    return _init_lib.kungfu_cluster_size()


def peer_info(version):
    """
    Inputs:
        version : A scalar tensor of int32,
        will use current version if version < 0.
    Returns:
        a pair of scalar tensors of int32: (rank, cluster_size).
    """
    return _op_lib.kungfu_get_peer_info(version)


# TODO: group ops by category


def get_peer_latencies(local_step=None):
    """Returns the vector V of round-trip time from this peer to all other peers.

    For the peer of rank i, V[j] is the RTT from i to j (j != i), V[i] = 0.
    """
    # FIXME: don't require input
    if local_step is None:
        import tensorflow as tf
        local_step = tf.Variable(tf.zeros([], tf.int64), trainable=False)
    return _op_lib.kungfu_get_peer_latencies(local_step,
                                             cluster_size=_get_num_peers())


def global_minimum_spanning_tree(self_weights):
    """Compute the minimum spanning tree.

    self_weights: a vector of length n,
        where n is the number of peers in the cluster.
        All self_weights vectors from n peers are gathered to a matrix W of
        n x n. The MST is then computed based on (W + W^T)/2.
    returns:
        a matrix m of (n - 1) x 2,
        where (m[i][0], m[i][1]) is the i-th edge of the tree.
    """
    return _op_lib.kungfu_minimum_spanning_tree(self_weights)


def get_neighbour_mask(edges):
    """Compute a bool vector of neighbours for the current peer.

    For the peer of rank i, v[j] = true if (i, j) is an edge of the MST,
    otherwise v[j] = false.
    """
    return _op_lib.kungfu_get_neighbour_mask(edges,
                                             self_rank=_get_self_rank(),
                                             cluster_size=_get_num_peers())


def round_robin(mask):
    return _op_lib.kungfu_round_robin(mask)


def model_averaging(peer_ranks, variables, mode, peer_selection_strategy):
    import tensorflow as tf
    var_sizes = [var.shape.num_elements() for var in variables]

    # Remove self rank from the list
    peer_ranks.remove(_get_self_rank())

    if mode == 'async':
        print(
            "Applying model averaging with a model requested asynchronously.")
        model_averaging = _op_lib.async_model_averaging(
            variables,
            self_rank=_get_self_rank(),
            ranks=peer_ranks,
            var_type_size=variables[0].dtype.size,
            var_sizes=var_sizes,
            peer_selection_strategy=peer_selection_strategy)
    elif mode == 'sync':
        print("Applying model averaging with a model requested synchronously.")
        model_averaging = _op_lib.model_averaging(
            variables,
            self_rank=_get_self_rank(),
            ranks=peer_ranks,
            var_type_size=variables[0].dtype.size,
            var_sizes=var_sizes,
            peer_selection_strategy=peer_selection_strategy)
    else:
        raise Exception("Invalid type of model request mode.")

    return model_averaging


def request_model(peer_ranks, variables, mode, peer_selection_strategy):
    import tensorflow as tf
    var_shapes = [var.shape for var in variables]

    var_sizes = [var.shape.num_elements() for var in variables]

    # Remove self rank from the list
    peer_ranks.remove(_get_self_rank())

    if mode == 'async':
        print("Request a model asynchronously.")
        request_model = _op_lib.async_request_model(
            variables,
            self_rank=_get_self_rank(),
            ranks=peer_ranks,
            var_type_size=variables[0].dtype.size,
            var_sizes=var_sizes,
            shapes=var_shapes,
            peer_selection_strategy=peer_selection_strategy)
    elif mode == 'sync':
        print("Request a model synchronously.")
        request_model = _op_lib.request_model(
            variables,
            self_rank=_get_self_rank(),
            ranks=peer_ranks,
            var_type_size=variables[0].dtype.size,
            var_sizes=var_sizes,
            shapes=var_shapes,
            peer_selection_strategy=peer_selection_strategy)
    else:
        raise Exception("Invalid type of model request mode")

    return request_model


def adaptive_request_variables(variables, window_size):
    ranks = _get_other_ranks()
    if len(ranks) == 0:
        return variables
    return _op_lib.adaptive_request_variables(
        variables,
        dtype=variables[0].dtype,
        shapes=[v.shape for v in variables],
        names=[v.name for v in variables],
        ranks=ranks,
        window_size=window_size)


def _parse_schedule(schedule, batch_size, num_train):
    # schedule is of the form
    # f1;e1;f2;e2;f3;e3
    tokens = schedule.split(",")
    print("Num train: " + str(num_train))
    print("Batch size: " + str(batch_size))
    to_gs = lambda epoch: int(epoch * num_train /
                              (batch_size * _get_num_peers()))
    pairs = [(to_gs(int(t.split(":")[0])), float(t.split(":")[1]))
             for t in tokens]
    steps, fractions = zip(*pairs)

    print("Steps: " + str(steps))
    print("Fractions: " + str(fractions))
    return steps, fractions


def compute_partitions(fraction, ts, total_size, tensor_partition_idx_vars,
                       num_partitions_var):
    import math
    import tensorflow as tf
    budget = int(math.floor(fraction * total_size))
    indexes, new_num_partitions = _bin_pack(
        dict((t.name, _tensor_size(t)) for t in ts), budget)
    print("Fraction: " + str(fraction))
    print("Size indices: " + str(len(indexes.values())))

    assign_partitions = tf.assign(num_partitions_var, new_num_partitions)

    assign_idx_vars = []
    for k in indexes.keys():
        # k is tensor name
        assign_idx_var = tf.assign(tensor_partition_idx_vars[k], indexes[k])
        assign_idx_vars.append(assign_idx_var)
    with tf.control_dependencies(assign_idx_vars + [assign_partitions]):
        return tf.constant(True, dtype=tf.bool)


def adaptive_partial_exchange_with_cpu_allreduce(ts,
                                                 batch_size,
                                                 num_train,
                                                 schedule,
                                                 accumulate=False,
                                                 average="none"):
    import tensorflow as tf
    print("Using piecewise partitioning schedule: " + schedule)
    steps, fractions = _parse_schedule(schedule, int(batch_size),
                                       int(num_train))

    total_size = sum([_tensor_size(t) for t in ts])
    global_step = tf.Variable(
        tf.zeros([], dtype=tf.int64)
    )  # tf.train.get_or_create_global_step(graph=tf.get_default_graph())
    increment_global_step_op = tf.assign(global_step, global_step + 1)

    # Dynamic partition info
    tensor_partition_idx_vars = dict(
        (t.name, tf.Variable(tf.ones([], dtype=tf.int64))) for t in ts)
    num_partitions_var = tf.Variable(tf.ones([], dtype=tf.int64))

    # Reverse both
    steps = steps[::-1]
    fractions = fractions[::-1]

    cases = []
    for i in range(len(steps)):
        cases.append((tf.greater_equal(global_step - 1, steps[i]),
                      lambda frac=fractions[i]: compute_partitions(
                          frac, ts, total_size, tensor_partition_idx_vars,
                          num_partitions_var)))

    bin_pack_case = tf.case(cases,
                            exclusive=False,
                            default=lambda: tf.constant(True, dtype=tf.bool))

    with tf.control_dependencies([bin_pack_case]):
        partial_negotiated_ts = []
        for tensor in ts:
            partition_idx_var = tensor_partition_idx_vars[tensor.name]
            mod_op = tf.mod(global_step - 1, num_partitions_var)
            equal_op = tf.equal(mod_op, partition_idx_var)

            negotiated_grad = tf.cond(
                equal_op,
                lambda tensor=tensor, partition_idx_var=partition_idx_var,
                num_partitions_var=num_partitions_var: all_reduce(tensor),
                lambda tensor=tensor: tf.identity(tensor))
            partial_negotiated_ts.append(negotiated_grad)

        with tf.control_dependencies([increment_global_step_op]):
            return [tf.identity(pnt) for pnt in partial_negotiated_ts]


def _bin_pack(sizes, budget, adjust_budget=False):
    lst = list(reversed(sorted([(size, name)
                                for name, size in sizes.items()])))
    if adjust_budget:
        budget = max(budget, lst[0][0])
    else:
        if lst[0][0] > budget:
            print("Suggested budget fraction is %f" %
                  (lst[0][0] / sum([s[1] for s in sizes.items()])))
            raise RuntimeError("Budget is too small %f. Largest tensor is %f" %
                               (budget, lst[0][0]))

    budgets = []
    indexes = dict()
    for size, name in lst:
        ok = False
        for i, b in enumerate(budgets):
            if b >= size:
                budgets[i] -= size
                indexes[name] = i
                ok = True
                break
        if not ok:
            budgets.append(budget - size)
            indexes[name] = len(budgets) - 1
    return indexes, len(budgets)


def partial_exchange_with_gpu_allreduce(ts,
                                        fraction=0.3,
                                        accumulate=False,
                                        average="none"):
    import math
    import tensorflow as tf
    total_size = sum([_tensor_size(t) for t in ts])
    print("Total Size of All Gradients: " + str(total_size))
    print("The fraction is: " + str(fraction))
    budget = int(math.floor(fraction * total_size))
    indexes, num_partitions = _bin_pack(
        dict((t.name, _tensor_size(t)) for t in ts), budget)
    print("The bucket budget is: " + str(budget))

    gs = tf.Variable(tf.zeros([], dtype=tf.int64))
    advance_gs = tf.assign(gs, gs + 1)

    name_order = dict((t.name, i) for i, t in enumerate(ts))

    # Construct groups
    groups = [[] for _ in range(num_partitions)]
    for t in ts:
        groups[indexes[t.name]].append(t)

    # Start all groups
    reordered_cond_ops = [None] * len(ts)
    for i, partition in enumerate(groups):
        negotiated_partition = tf.cond(
            tf.equal(tf.mod(gs - 1, num_partitions), i),
            lambda partition=partition: gpu_group_all_reduce(partition),
            lambda partition=partition: partition)
        if len(partition) == 1:
            negotiated_partition = [negotiated_partition]
        for i in range(len(partition)):
            grad = partition[i]
            negotiated_grad = negotiated_partition[i]
            reordered_cond_ops[name_order[grad.name]] = negotiated_grad

    with tf.control_dependencies([advance_gs]):
        return reordered_cond_ops


def _concat(ts):
    import tensorflow as tf
    return tf.concat([tf.reshape(t, [-1]) for t in ts], -1)


def cpu_group_all_reduce_variance_monitor(grads, batch_small):
    import tensorflow as tf
    negotiated_grads = [all_reduce(t) for t in grads]
    noise_op = get_global_gradient_noise_operator(batch_small, _concat(grads),
                                                  _concat(negotiated_grads))
    with tf.control_dependencies([noise_op]):
        return [tf.identity(g) for g in negotiated_grads]


# deprecated
def get_global_gradient_noise_operator(batch_small,
                                       concat_grad,
                                       concat_negotiated_grad,
                                       alpha=0.6):
    import tensorflow as tf
    import json, os
    cluster_spec = json.loads(os.getenv('KUNGFU_CLUSTER_SPEC'))
    num_workers = len(cluster_spec['Peers'])
    if num_workers == 0:
        raise "Cluster spec KUNGFU_CLUSTER_SPEC is invalid"
    batch_big = batch_small * num_workers

    # Take average over workers
    G_big = tf.div(concat_negotiated_grad, num_workers)
    G_small = concat_grad

    G_sq_small = tf.square(tf.norm(G_small))
    G_sq_big = tf.square(tf.norm(G_big))

    G_biased = 1 / (batch_big - batch_small) * (batch_big * G_sq_big -
                                                batch_small * G_sq_small)
    S_biased = 1 / (1 / batch_small - 1 / batch_big) * (G_sq_small - G_sq_big)

    global_noise_op = _op_lib.gradient_noise(G_biased, S_biased, alpha=alpha)

    return global_noise_op


def global_gradient_noise_scale(batch_small,
                                concat_grad,
                                concat_negotiated_grad,
                                alpha=0.6):
    import tensorflow as tf
    _, np = peer_info(tf.constant(-1, dtype=tf.int32))
    cluster_size = tf.cast(np, dtype=tf.float32)
    batch_small = tf.cast(batch_small, dtype=tf.float32)

    batch_big = batch_small * cluster_size
    # Take average over workers
    G_big = tf.div(concat_negotiated_grad, cluster_size)
    G_small = concat_grad

    G_sq_small = tf.square(tf.norm(G_small))
    G_sq_big = tf.square(tf.norm(G_big))

    G_biased = 1 / (batch_big - batch_small) * (batch_big * G_sq_big -
                                                batch_small * G_sq_small)
    S_biased = 1 / (1 / batch_small - 1 / batch_big) * (G_sq_small - G_sq_big)

    return _op_lib.gradient_noise(G_biased, S_biased, alpha=alpha)
