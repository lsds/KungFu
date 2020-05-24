import os
import time

import numpy as np
import tensorflow as tf


def _cluster_size():
    if os.getenv('KUNGFU_SELF_SPEC'):
        from kungfu import current_cluster_size
        return current_cluster_size()
    else:
        try:
            import horovod.tensorflow as hvd
            return hvd.size()
        except:
            return 1


class LogPerfHook(tf.train.SessionRunHook):
    def __init__(self, batch_size, warmup_steps=5, drop_last=1):
        self._batch_size = batch_size
        self._warmup_steps = warmup_steps
        self._drop_last = drop_last
        self._step = 0
        self._durations = []
        self._t_last = time.time()

    def before_run(self, run_context):
        if self._step == 0:
            from kungfu._utils import _since_job_start, show_duration
            print('_since_job_start: %s' % (show_duration(_since_job_start())))

    def after_run(self, run_context, run_values):
        self._step += 1
        t1 = time.time()
        dur = t1 - self._t_last
        step_per_sec = 1 / dur
        sample_per_sec = step_per_sec * self._batch_size
        print('local step %d, took %.3fs, %.2f samples / sec' %
              (self._step, dur, sample_per_sec))
        if self._step > self._warmup_steps:
            self._durations.append(dur)
        self._t_last = t1

    def end(self, run_context):
        durations = list(self._durations)
        durations = durations[:len(durations) - self._drop_last]
        ds = np.array(durations)
        mean_duration = ds.mean()
        # print('durations: %s' % (durations))
        print('mean_duration: %.3fs' % (mean_duration))
        step_per_sec = 1 / mean_duration
        sample_per_sec = step_per_sec * self._batch_size
        print('RESULT: %.2f samples / sec, batch size: %d, cluster size %d' %
              (sample_per_sec, self._batch_size, _cluster_size()))


class LogStepHook(tf.train.SessionRunHook):
    def __init__(self):
        self._step = 0

    def before_run(self, run_context):
        print('%s::%s %d steps' % ('LogStepHook', 'before_run', self._step))

    def after_run(self, run_context, run_values):
        self._step += 1
        print('%s::%s after %d steps' %
              ('LogStepHook', 'after_run', self._step))

    def end(self, run_context):
        print('%s::%s after %d steps' % ('LogStepHook', 'end', self._step))


class ProfileResizeHook(tf.train.SessionRunHook):
    def __init__(self, schedule):
        from kungfu import current_rank
        self._rank = current_rank()
        self._step = 0

        self._schedule = schedule

    def before_run(self, run_context):
        pass

    def after_run(self, run_context, run_values):
        self._step += 1
        if self._rank != 0:
            return

        if self._step in self._schedule:
            new_size = self._schedule[self._step]
            print('ProfileResizeHook step %d, new_size: %d' %
                  (self._step, new_size))
            from kungfu.ext import propose_new_size
            propose_new_size(new_size)

    def end(self, run_context):
        pass
