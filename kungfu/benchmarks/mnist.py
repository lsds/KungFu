import tensorflow as tf

from .layers import Dense, seq_apply


def slp(input_size, class_number):
    x = tf.placeholder(tf.float32, [None, input_size])
    y = Dense(class_number, act=tf.nn.softmax)(x)
    return x, y


def mlp(input_size, class_number, hidden_layers=None):
    if hidden_layers is None:
        hidden_layers = [800, 600]

    all_layers = [Dense(n, act=tf.nn.relu) for n in hidden_layers] + [
        Dense(class_number, act=tf.nn.softmax),
    ]
    x = tf.placeholder(tf.float32, [None, input_size])
    y = seq_apply(all_layers, x)
    return x, y
