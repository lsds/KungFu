import tensorflow as tf
from tensorflow.python.eager import context
from tensorflow.python.platform import tf_logging as logging


def create_placeholder_for(t):
    return tf.placeholder(dtype=t.dtype, shape=t.shape)


def create_assign_op_for(t):
    p = create_placeholder_for(t)
    op = t.assign(p)
    return op, p


def get_global_variable(name, graph=None):
    graph = graph or tf.get_default_graph()
    global_variable_tensor = None
    global_variable_tensors = graph.get_collection(name)
    if len(global_variable_tensors) == 1:
        global_variable_tensor = global_variable_tensors[0]
    elif not global_variable_tensors:
        try:
            global_variable_tensor = graph.get_tensor_by_name(name + ':0')
        except KeyError:
            return None
    else:
        logging.error('Multiple tensors in %s collection.' % (name))
        return None

    return global_variable_tensor


def create_global_variable(name, shape, dtype, graph=None, init=None):
    graph = graph or tf.get_default_graph()
    if get_global_variable(name, graph) is not None:
        raise ValueError('"%s" already exists.' % (name))
    if context.executing_eagerly():
        raise ValueError('TODO: support eager mode.')

    if init is None:
        init = tf.zeros_initializer()
    else:
        shape = None  # If initializer is a constant, do not specify shape.

    with graph.as_default() as g, g.name_scope(None):
        return tf.get_variable(
            name,
            shape=shape,
            dtype=dtype,
            initializer=init,
            trainable=False,
            # aggregation=variables.VariableAggregation.ONLY_FIRST_REPLICA,
            collections=[
                tf.GraphKeys.GLOBAL_VARIABLES,
                # tf.GraphKeys.LOCAL_VARIABLES,  # FIXME: use KUNGFU_VARIABLES
                name,
            ],
        )


def get_or_create_global_variable(name, shape, dtype, graph=None, init=None):
    graph = graph or tf.get_default_graph()
    global_variable_tensor = get_global_variable(name, graph)
    if global_variable_tensor is None:
        global_variable_tensor = create_global_variable(name,
                                                        shape=shape,
                                                        dtype=dtype,
                                                        graph=graph,
                                                        init=init)
    else:
        pass
        # FIXME: check type <dtype: 'int32'> != <dtype: 'int32_ref'>
    return global_variable_tensor


def eval_global_variable(name, sess=None, graph=None):
    global_variable_tensor = get_global_variable(name, graph)
    if global_variable_tensor is None:
        raise RuntimeError('"%s" not exist' % (name))
    return sess.run(global_variable_tensor)


class GraphKeys(object):
    BATCH_SIZE = "kungfu_batch_size"
    GRADIENT_NOISE_SCALE = "kungfu_gradient_noise_scale"

    TOTAL_SAMPLES = "kungfu_total_samples"

    # TRAINED_STEPS = "kungfu_trained_steps"
    TRAINED_SAMPLES = "kungfu_trained_samples"


def get_or_create_batch_size(init=None):
    return get_or_create_global_variable(GraphKeys.BATCH_SIZE,
                                         shape=[],
                                         dtype=tf.int32,
                                         init=init)


def get_batch_size():
    """Returns the batch size tensor."""
    return get_global_variable(GraphKeys.BATCH_SIZE)


def eval_batch_size(sess=None):
    """Returns the value of batch size in given session."""
    return eval_global_variable(GraphKeys.BATCH_SIZE, sess)


def eval_gradient_noise_scale(sess=None):
    """Returns the value of gradient noise scale in given session."""
    return eval_global_variable(GraphKeys.GRADIENT_NOISE_SCALE, sess)


def create_setter(v):
    op, place = create_assign_op_for(v)

    def set_value(sess, value):
        sess.run(op, feed_dict={place: value})

    return set_value
