"""A collection of common used layers."""

import tensorflow as tf

param_size = 0


def show_size(s):
    Ki = 1024
    Mi = Ki * Ki
    Gi = Mi * Ki
    if s > Gi:
        return '%.2fGi' % (float(s) / Gi)
    elif s > Mi:
        return '%.2fMi' % (float(s) / Mi)
    elif s > Ki:
        return '%.2fKi' % (float(s) / Ki)
    else:
        return '%d' % s


def _get_size(v):
    n = 1
    for d in v.shape:
        n *= int(d)
    if str(v.dtype) == "<dtype: 'float32_ref'>":
        n *= 4
    else:
        print('unknown dtype: %s' % str(v.dtype))
    return n


def _new_variable(initial):
    v = tf.Variable(initial)
    global param_size
    param_size += _get_size(v)
    return v


def new_conv_kernel(shape):
    print('new %s :: %s' % ('conv kernel', shape))
    initial = tf.truncated_normal(shape, stddev=0.1)
    return _new_variable(initial)


def new_bias(shape):
    print('new %s :: %s' % ('biss', shape))
    initial = tf.constant(0.1, shape=shape)
    return _new_variable(initial)


def new_dense_weight(shape):
    print('new %s :: %s' % ('dense weight', shape))
    initial = tf.truncated_normal(shape, stddev=0.1)
    return _new_variable(initial)


def conv2d(x, W, strides=None):
    if not strides:
        strides = [1, 1]
    r, s = strides
    return tf.nn.conv2d(x, W, strides=[1, r, s, 1], padding='SAME')


def max_pool(x, ksize):
    r, s = ksize
    return tf.nn.max_pool(
        x, ksize=[1, r, s, 1], strides=[1, r, s, 1], padding='SAME')


class Layer(object):
    pass


def seq_apply(layers, x):
    y = x
    for l in layers:
        y = l(y)
        # print('new %s :: %s' % ('layer', y.shape))
    return y


class Pool(Layer):
    def __init__(self, ksize=None):
        if ksize is None:
            ksize = (2, 2)
        self._ksize = ksize

    def __call__(self, x):
        y = max_pool(x, self._ksize)
        return y


class Dense(Layer):
    def __init__(self, logits, bias=True, act=None):
        self._logits = logits
        self._bias = bias
        self._act = act

    def __call__(self, x):
        if len(x.shape) == 4:
            _n, h, w, c = x.shape
            dims = [int(d) for d in [h, w, c]]
            input_size = dims[0] * dims[1] * dims[2]
            print('flatten :: %s -> %d' % (dims, input_size))
            x_flat = tf.reshape(x, [-1, input_size])
        elif len(x.shape) == 2:
            x_flat = x
            _n, m = x.shape
            input_size = int(m)
        else:
            raise RuntimeError('invalid input size')

        w = new_dense_weight((input_size, self._logits))
        y = tf.matmul(x_flat, w)
        if self._bias:
            b = new_bias((self._logits, ))
            y = tf.add(y, b)
        if self._act:
            y = self._act(y)
        return y


class Conv(Layer):
    def __init__(self, ksize, n_filters, strides=None):
        if strides is None:
            strides = [1, 1]
        self._ksize = ksize
        self._n_filters = n_filters
        self._strides = strides

    def _get_channel_size(self, shape):
        _n, _h, _w, c = shape
        return int(c)

    def __call__(self, x):
        r, s = self._ksize
        c, d = self._get_channel_size(x.shape), self._n_filters
        w = new_conv_kernel((r, s, c, d))
        y = conv2d(x, w, self._strides)
        return y
