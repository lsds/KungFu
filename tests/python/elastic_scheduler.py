from kungfu.ops import current_cluster_size, get_init_checkpoint, peer_info, resize_cluster
import tensorflow as tf


def _parse_schedule(config):
    schedule = []
    t = 0
    for p in config.split(','):
        kv = p.split(':')
        n, val = (int(kv[0]), int(kv[1]))
        schedule.append((t, t + n, val))
        t += n
    return schedule, t


def _get_cluster_size(i, sch):
    for s, e, n in sch:
        if s <= i and i < e:
            return n
    raise RuntimeError('no schedule defined for %d' % (i))


def _get_schedule(stage, schedule):
    return tf.case(
        [
            (
                tf.logical_and(tf.less_equal(a, stage), tf.less(stage, b)),  #
                lambda c=c: tf.constant(c),
            ) for a, b, c in schedule
        ],
        default=lambda: tf.constant(0),
        strict=True)


class ElasticScheduler(object):
    def __init__(self, config):

        self.schedule, self.max_stage = _parse_schedule(config)
        print(self.schedule)

        self.init_stage = self.restore(get_init_checkpoint())
        print('init stage: %d' % (self.init_stage))

        self.current_np = current_cluster_size()
        init_np = _get_cluster_size(self.init_stage, self.schedule)
        if self.current_np != init_np:
            print(
                '[W] init cluster size (np=%d) is not consistent with schedule (np=%d)'
                % (self.current_np, init_np))

        #
        self.ckpt = tf.placeholder(tf.string)
        self.new_size = tf.placeholder(tf.int32)
        self.resize_op = resize_cluster(self.ckpt, self.new_size)

    def restore(self, checkpoint):
        return int(checkpoint)

    def create_op(self):
        _, np = peer_info()
        stage = tf.Variable(self.init_stage, dtype=tf.int32, trainable=False)
        next_stage = stage + 1
        new_np = _get_schedule(next_stage, self.schedule)
        ckpt_tensor = tf.as_string(next_stage)
        resize_op = resize_cluster(ckpt_tensor, new_np)

        adapt_op = tf.cond(
            tf.less(next_stage, self.max_stage),  #
            lambda: resize_op,
            # lambda: tf.cond(
            #     tf.equal(new_np, np),  #
            #     lambda: True,
            #     lambda: resize_op),
            lambda: True)

        with tf.control_dependencies([adapt_op]):
            with tf.control_dependencies([tf.assign_add(stage, 1)]):
                return tf.identity(adapt_op)

    def run(self, sess, stage):
        next_stage = stage + 1
        if next_stage < self.max_stage:
            new_np = _get_cluster_size(next_stage, self.schedule)
            if new_np != self.current_np:
                keep = sess.run(self.resize_op,
                                feed_dict={
                                    self.ckpt: str(next_stage),
                                    self.new_size: new_np,
                                })
                self.current_np = new_np
                if not keep:
                    return False
        return True
