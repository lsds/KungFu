import tensorflow as tf
from kungfu._utils import show_duration
from kungfu.tensorflow.initializer import BroadcastGlobalVariablesOp
from kungfu.tensorflow.ops import (all_reduce, current_cluster_size,
                                   resize_cluster_from_url)


class ElasticHook(tf.train.SessionRunHook):
    def __init__(self, local_batch_size, epochs, epoch_size):
        self._local_batch_size = local_batch_size
        self._total_samples = epoch_size * epochs
        self._need_sync = True
        self._exit_reason = None

    def begin(self):
        self._step = 0
        self._trained_samples = 0
        self._trained_samples_place = tf.placeholder(dtype=tf.int32, shape=())
        self._sync_offset_op = all_reduce(self._trained_samples_place,
                                          op='max')

        self._sync_state_op = BroadcastGlobalVariablesOp()
        self._resize_op = resize_cluster_from_url()

    def _do_sync_offset(self, sess):
        new_offset = sess.run(
            self._sync_offset_op,
            feed_dict={self._trained_samples_place: self._trained_samples})
        print('sync offset %d -> %d on step %d' %
              (self._trained_samples, new_offset, self._step))
        self._trained_samples = new_offset

    def before_run(self, run_context):
        if self._need_sync:
            self._do_sync_offset(run_context.session)
            run_context.session.run(self._sync_state_op)
            self._need_sync = False

    def after_run(self, run_context, run_values):
        self._step += 1
        np = current_cluster_size()
        self._trained_samples += self._local_batch_size * np
        changed, keep = run_context.session.run(self._resize_op)
        if not keep:
            run_context.request_stop()
            self._exit_reason = 'change cluster'
            return
        if changed:
            self._need_sync = True
        if self._trained_samples >= self._total_samples:
            self._exit_reason = 'finished'
            run_context.request_stop()

    def end(self, sess):
        if self._exit_reason is None:
            # raise RuntimeError('unknown exit reason') # FIXME: doesn't work
            print('unknown exit reason!')
            exit(1)  # cause all workers to stop
        print('stopped after trained %d samples in %d steps due to %s' %
              (self._trained_samples, self._step, self._exit_reason))
