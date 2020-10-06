import tensorflow as tf
from kungfu.tensorflow.ops.testing import fake_error


def test_error(has_error):
    x = tf.Variable(has_error)
    y = fake_error(x)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        try:
            v = sess.run(y)
            print(v)
        except Exception as e:
            print(e)
            pass


test_error(False)
test_error(True)
