from .loader import _op_lib


def global_noise_scale(batch_small, batch_big, tensor, avg_tensor, alpha=0.6):
    import tensorflow as tf
    G_big = avg_tensor
    G_small = tensor

    G_sq_small = tf.square(tf.norm(G_small))
    G_sq_big = tf.square(tf.norm(G_big))

    G_biased = 1 / (batch_big - batch_small) * (batch_big * G_sq_big -
                                                batch_small * G_sq_small)
    S_biased = 1 / (1 / batch_small - 1 / batch_big) * (G_sq_small - G_sq_big)

    return _op_lib.noise_scale(G_biased, S_biased, alpha=alpha)
