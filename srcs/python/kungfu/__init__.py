from kungfu.optimizers.core import lazy_load_op_lib


def distributed_variables_initializer():
    import tensorflow as tf
    _op_lib = lazy_load_op_lib()
    g = tf.get_default_graph()
    ops = []
    # TODO: auto inject tf.global_variables_initializer
    # with tf.control_dependencies([tf.global_variables_initializer()]):
    for v in g.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
        ops.append(tf.assign(v, _op_lib.broadcast(v)))
    return tf.group(ops)
