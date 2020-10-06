import tensorflow as tf
from kungfu.tensorflow.ops.testing import fake_error
from tensorflow.python.util import deprecation

deprecation._PRINT_DEPRECATION_WARNINGS = False


def try_catch_error(error_code):
    has_error = error_code != 0
    x = tf.Variable(error_code, dtype=tf.int32)
    y = fake_error(x)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        catched = False
        try:
            v = sess.run(y)
        except tf.errors.OpError:
            catched = True
        # except tf.errors.OutOfRangeError:
        #     catched = True
        # except Exception as e:
        #     print(e)
        #     print(e.__class__)
        #     catched = True

        assert (has_error) == catched
    print('OK, can handle error code %d' % (error_code))


def test_errors():
    errors = {
        'OK': 0,
        'CANCELLED': 1,
        'UNKNOWN': 2,
        'INVALID_ARGUMENT': 3,
        'DEADLINE_EXCEEDED': 4,
        'NOT_FOUND': 5,
        'ALREADY_EXISTS': 6,
        'PERMISSION_DENIED': 7,
        'UNAUTHENTICATED': 16,
        'RESOURCE_EXHAUSTED': 8,
        'FAILED_PRECONDITION': 9,
        'ABORTED': 10,
        'OUT_OF_RANGE': 11,
        'UNIMPLEMENTED': 12,
        'INTERNAL': 13,
        'UNAVAILABLE': 14,
        'DATA_LOSS': 15,
    }
    for k, v in errors.items():
        try_catch_error(v)
    # try_catch_error(10000) will fail


# test_errors()
