import tensorflow as tf

_tf_major_version = int(tf.__version__.split('.')[0])

if _tf_major_version == 1:
    # TensorFlow 1.x
    _tf_optimizer = tf.train.Optimizer
    _tf_assign = tf.assign
    _tf_mod = tf.mod
    _tf_hook = tf.train.SessionRunHook
elif _tf_major_version == 2:
    # TensorFlow 2.x
    _tf_optimizer = tf.compat.v1.train.Optimizer
    _tf_assign = tf.compat.v1.assign
    _tf_mod = tf.math.floormod
    _tf_hook = tf.estimator.SessionRunHook
else:
    raise RuntimeError('Not sure what TensorFlow version to use.')
