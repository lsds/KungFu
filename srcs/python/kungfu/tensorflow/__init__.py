from kungfu.tensorflow.optimizers.grad_noise_scale import gns

import tensorflow as tf

__all__ = [
    # 'batch_size',
    'gns',
]

# TODO: put it in a dict
_batch_size_tensor = None
_batch_size_place = None
_set_batch_size_op = None


def get_or_create_batch_size_tensor(init_batch_size=None):
    global _batch_size_tensor
    global _batch_size_place
    global _set_batch_size_op
    if _batch_size_tensor is None:
        if init_batch_size is None:
            init_batch_size = 32
        _batch_size_tensor = tf.Variable(init_batch_size,
                                         dtype=tf.int64,
                                         name='kungfu_local_batch_size',
                                         trainable=False)
        _batch_size_place = tf.placeholder(dtype=tf.int64)
        _set_batch_size_op = tf.assign(_batch_size_tensor, _batch_size_place)
    else:
        if init_batch_size is not None:
            print(
                'Warning: get_or_create_batch_size_tensor called with init value after batch_size_tensor is created'
            )
    return _batch_size_tensor


def get_batch_size_tensor():
    global _batch_size_tensor
    assert (_batch_size_tensor is not None)
    return _batch_size_tensor


def batch_size(sess):
    return sess.run(get_batch_size_tensor())


def set_batch_size(sess, new_batch_size):
    global _batch_size_place
    global _set_batch_size_op
    sess.run(_set_batch_size_op, feed_dict={_batch_size_place: new_batch_size})
