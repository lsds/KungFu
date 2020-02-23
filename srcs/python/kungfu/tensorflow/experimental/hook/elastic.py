import tensorflow as tf
from kungfu.tensorflow.initializer import BroadcastGlobalVariablesOp
from kungfu.tensorflow.ops import all_reduce, resize_cluster_from_url


class ElasticHook(tf.train.SessionRunHook):
    def __init__(self, max_step):
        self._max_step = max_step
        self._need_sync = True

    def begin(self):
        self._step = 0
        self._step_place = tf.placeholder(dtype=tf.int32, shape=())
        self._sync_step_op = all_reduce(self._step_place, op='max')

        self._sync_op = BroadcastGlobalVariablesOp()
        self._resize_op = resize_cluster_from_url()

    def after_create_session(self, sess, coord):
        pass

    def before_run(self, run_context):
        if self._step >= self._max_step:  # shouldn't happen
            print('request_stop before kungfu_step: %d' % (self._step))
            # run_context.request_stop()
            # FIXME: force quit

        if self._need_sync:
            self._step = run_context.session.run(
                self._sync_step_op, feed_dict={self._step_place: self._step})
            run_context.session.run(self._sync_op)
            self._need_sync = False

    def after_run(self, run_context, run_values):
        changed, keep = run_context.session.run(
            self._resize_op, feed_dict={self._step_place: self._step})
        if not keep:
            run_context.request_stop()
            self._exit_reasion = 'resize'
            return
        if changed:
            print('changed on step %d' % (self._step))
            self._need_sync = True
        self._step += 1
        if self._step >= self._max_step:
            self._exit_reasion = 'finished'
            print('request_stop on kungfu_step: %d' % (self._step))
            run_context.request_stop()

    def end(self, sess):
        print('stopped at step: %d due to %s' %
              (self._step, self._exit_reasion))
