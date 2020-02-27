import time

import tensorflow as tf
from kungfu._utils import show_duration
from kungfu.tensorflow.initializer import BroadcastGlobalVariablesOp
from kungfu.tensorflow.ops import all_reduce, resize_cluster_from_url


class ElasticHook(tf.train.SessionRunHook):
    def __init__(self, max_step, debug=True):
        self._max_step = max_step
        self._need_sync = True
        self._debug = debug
        self._exit_reason = None

    def begin(self):
        self._step = 0
        self._step_place = tf.placeholder(dtype=tf.int32, shape=())
        self._sync_step_op = all_reduce(self._step_place, op='max')

        self._sync_state_op = BroadcastGlobalVariablesOp()
        self._resize_op = resize_cluster_from_url()

        if self._debug:
            self._t0 = time.time()
            self._t_last = self._t0

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
            run_context.session.run(self._sync_state_op)
            self._need_sync = False

    def after_run(self, run_context, run_values):
        self._step += 1
        changed, keep = run_context.session.run(self._resize_op)
        if not keep:
            run_context.request_stop()
            self._exit_reason = 'resize'
            return
        if changed:
            self._need_sync = True
        if self._debug:
            self._log_rate()
        if self._step >= self._max_step:
            self._exit_reason = 'finished'
            print('request_stop on kungfu_step: %d' % (self._step))
            run_context.request_stop()

    def _log_rate(self):
        log_period = 100
        if (self._step % log_period == 0):
            now = time.time()
            duration = now - self._t_last
            self._t_last = now
            rate = duration / log_period
            remain = (self._max_step - self._step) * rate
            print(
                'current step: %d, max step: %d, %.3fs / step, %fs since last, %s since begin, about %s remain'
                % (self._step, self._max_step, rate, duration,
                   show_duration(now - self._t0), show_duration(remain)))

    def end(self, sess):
        if self._exit_reason is None:
            # raise RuntimeError('unknown exit reason') # FIXME: doesn't work
            print('unknown exit reason!')
            exit(1)  # cause all workers to stop
        print('stopped at step %d due to %s' % (self._step, self._exit_reason))
