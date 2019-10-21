from kungfu.ops import current_cluster_size, get_init_checkpoint, resize_cluster
import tensorflow as tf


def parse_schedule(config):
    schedule = []
    t = 0
    for p in config.split(','):
        kv = p.split(':')
        n, val = (int(kv[0]), int(kv[1]))
        schedule.append((t, t + n, val))
        t += n
    return schedule, t


def get_cluster_size(i, sch):
    for s, e, n in sch:
        if s <= i and i < e:
            return n
    raise RuntimeError('no schedule defined for %d' % (i))


class ElasticScheduler(object):
    def __init__(self, config):

        self.schedule, self.max_step = parse_schedule(config)
        print(self.schedule)

        self.ckpt = tf.placeholder(tf.string)
        self.new_size = tf.placeholder(tf.int32)
        self.resize_op = resize_cluster(self.ckpt, self.new_size)

        self.init_step = self.restore(get_init_checkpoint())
        self.current_np = current_cluster_size()
        init_np = get_cluster_size(self.init_step, self.schedule)
        if self.current_np != init_np:
            print(
                '[W] init cluster size (np=%d) is not consistent with schedule (np=%d)'
                % (self.current_np, init_np))

    def restore(self, checkpoint):
        return int(checkpoint)

    def run(self, sess, step):
        next_step = step + 1
        if next_step < self.max_step:
            new_np = get_cluster_size(next_step, self.schedule)
            if new_np != self.current_np:
                keep = sess.run(self.resize_op,
                                feed_dict={
                                    self.ckpt: str(next_step),
                                    self.new_size: new_np,
                                })
                self.current_np = new_np
                if not keep:
                    return False
        return True
