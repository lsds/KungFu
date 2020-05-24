import time

from kungfu._utils import show_duration
from kungfu.tensorflow.initializer import BroadcastGlobalVariablesOp
from kungfu.tensorflow.ops import (all_reduce, current_cluster_size,
                                   resize_cluster_from_url)

import tensorflow as tf


class ResizeProfiler():
    def __init__(self):
        self._begin = None
        self._new = True
        self._records = []

    def begin(self):
        assert (self._begin is None)
        self._new = False
        self._begin = time.time()
        self._resizing = True

    def end(self):
        if self._new:
            return
        assert (self._begin is not None)
        dur = time.time() - self._begin

        print('resize took %s', show_duration(dur))
        self._records.append((dur, 'OK'))
        self._begin = None

    def cancel(self):
        assert (self._begin is not None)
        self._begin = None

    def report(self):
        for idx, (dur, msg) in enumerate(self._records):
            print('resize #%d took %s' % (idx, show_duration(dur)))

    def __del__(self):
        assert (self._begin is None)


class ElasticHook(tf.train.SessionRunHook):
    def __init__(self, local_batch_size, epochs, epoch_size):
        self._local_batch_size = local_batch_size
        self._total_samples = epoch_size * epochs
        self._need_sync = True
        self._exit_reason = None
        self._profiler = ResizeProfiler()

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
            self._profiler.end()

    def after_run(self, run_context, run_values):
        self._step += 1
        np = current_cluster_size()
        self._trained_samples += self._local_batch_size * np

        self._profiler.begin()
        changed, keep = run_context.session.run(self._resize_op)
        if not keep:
            run_context.request_stop()
            self._exit_reason = 'change cluster'
            self._profiler.end()
            return
        if changed:
            self._need_sync = True
        else:
            self._profiler.cancel()

        if self._trained_samples >= self._total_samples:
            self._exit_reason = 'finished'
            run_context.request_stop()

    def end(self, sess):
        self._profiler.report()
        if self._exit_reason is None:
            # raise RuntimeError('unknown exit reason') # FIXME: doesn't work
            print('unknown exit reason!')
            exit(1)  # cause all workers to stop
        print('stopped after trained %d samples in %d steps due to %s' %
              (self._trained_samples, self._step, self._exit_reason))
