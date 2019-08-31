import tensorflow as tf

from kungfu.ops import peer_info


def get_rank():
    rank, _ = peer_info(tf.constant(-1, dtype=tf.int32))
    with tf.Session() as sess:
        return int(sess.run(rank))


class Reporter(object):
    def __init__(self, fmt, hdrs=None):
        self._records = []
        self._format = fmt
        self._headers = hdrs

    def __call__(self, row):
        print(self._format % tuple(row))
        self._records.append(row)

    def save(self, filename):
        with open(filename, 'w') as f:
            if self._headers:
                f.write(','.join(self._headers) + '\n')
            for row in self._records:
                f.write((self._format % tuple(row)) + '\n')
