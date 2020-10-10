import kungfu.tensorflow as kf
import tensorflow as tf
from kungfu.tensorflow.initializer import BroadcastGlobalVariablesOp
from kungfu.tensorflow.ops import resize
from kungfu.tensorflow.policy import BasePolicy
from kungfu.tensorflow.variables import GraphKeys


class ScheduledElasticPolicy(BasePolicy):
    def __init__(self, _step_schedule):
        self._step_schedule = _step_schedule
        self._need_sync = True

    def before_train(self):
        self._size_place = tf.placeholder(dtype=tf.uint32, shape=[])
        self._resize_op = resize(self._size_place)
        self._sync_op = BroadcastGlobalVariablesOp()

    def before_epoch(self, sess):
        pass

    def before_step(self, sess):
        if self._need_sync:
            print('running sync')
            sess.run(self._sync_op)
            print('finish sync')
            self._need_sync = False

    def after_step(self, sess):
        step = sess.run(tf.train.get_global_step())
        if step in self._step_schedule:
            new_cluster_size = self._step_schedule[step]
            print('resize to %s' % (new_cluster_size))
            self._need_sync = self._resize(sess, new_cluster_size)
            print('after resize to %s, need sync: %s' %
                  (new_cluster_size, self._need_sync))

    def _resize(self, sess, new_size):
        return sess.run(self._resize_op, feed_dict={
            self._size_place: new_size,
        })

    def after_epoch(self, sess):
        pass
        # self._trained_epochs += 1
