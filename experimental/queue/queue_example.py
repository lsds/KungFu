from kungfu.tensorflow.ops.queue import new_queue
from kungfu.python import current_rank
import tensorflow as tf


def main():
    rank = current_rank()
    src = 1
    dst = 2

    if rank in [src, dst]:
        q = new_queue(src, dst)
        print(q)

        x = tf.Variable([1, 2, 3], dtype=tf.int32)

        if rank == src:
            print(x)
            q.put(x)
        elif rank == dst:
            y = q.get(x.dtype, x.shape)
            print(y)

        q2 = new_queue(dst, src)
        if rank == dst:
            print(x)
            q2.put(x)
        elif rank == src:
            y = q2.get(x.dtype, x.shape)
            print(y)


main()
