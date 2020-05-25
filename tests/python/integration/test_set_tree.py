import numpy as np
import tensorflow as tf


def gen_binary_tree(n, permu):
    father = [i for i in range(n)]
    father[permu[0]] = permu[0]
    for i in range(n):
        j, k = i * 2 + 1, i * 2 + 2
        if j < n:
            father[permu[j]] = permu[i]
        if k < n:
            father[permu[k]] = permu[i]
    return father


def gen_tree(n):
    permu = np.random.permutation(n)
    tree = gen_binary_tree(n, permu)
    return tree


def main():
    from kungfu import current_cluster_size
    from kungfu.tensorflow.ops import all_reduce, broadcast
    from kungfu.tensorflow.ops.adapt import set_tree

    np = current_cluster_size()

    tree_place = tf.placeholder(dtype=tf.int32, shape=(np, ))
    set_tree_op = set_tree(broadcast(tree_place))

    magic = 32
    x = tf.Variable(list(range(magic)), dtype=tf.int32)
    y = all_reduce(x)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        for _ in range(10):
            v = sess.run(y)
            assert (v.sum() == np * magic * (magic - 1) / 2)
            # print(v)

            tree = gen_tree(np)
            sess.run(set_tree_op, feed_dict={
                tree_place: tree,
            })
    print('test set_tree OK')


main()
