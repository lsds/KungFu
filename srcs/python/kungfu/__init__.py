from .ops import broadcast


def distributed_variables_initializer():
    import tensorflow as tf
    g = tf.get_default_graph()
    ops = []
    # TODO: auto inject tf.global_variables_initializer
    # with tf.control_dependencies([tf.global_variables_initializer()]):
    for v in g.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
        ops.append(tf.assign(v, broadcast(v)))
    return tf.group(ops)
