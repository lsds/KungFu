import tensorflow as tf


def show_tensorflow_info():
    print('tensorflow info:')
    print('Version: %s' % tf.__version__)
    print('Lib: %s' % tf.sysconfig.get_lib())
    print('Include: %s' % tf.sysconfig.get_include())
    print('Compile Flags: %s' % tf.sysconfig.get_compile_flags())
    print('Link Flags: %s' % tf.sysconfig.get_link_flags())


def test_cublas():
    x = tf.Variable(tf.ones(shape=[100, 100]))
    w = tf.Variable(tf.ones(shape=[100, 100]))
    y = tf.matmul(x, w)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        v = sess.run(y)
        # print(v.sum())
    print('cublas OK')


def test_cudnn():
    x = tf.Variable(tf.ones(shape=[2, 3, 3, 3]))
    w = tf.Variable(tf.ones(shape=[3, 3, 3, 3]))
    y = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        v = sess.run(y)
        # print(v.sum())
    print('cudnn OK')


show_tensorflow_info()
print('testing tensorflow features')
test_cublas()
test_cudnn()
print('test tensorflow features done')
