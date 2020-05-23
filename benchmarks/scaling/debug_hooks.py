import time

import numpy as np
import tensorflow as tf


class LogPerfHook(tf.train.SessionRunHook):
    def __init__(self, batch_size, warmup_steps=10, drop_last=1):
        self._batch_size = batch_size
        self._warmup_steps = warmup_steps
        self._drop_last = drop_last
        self._step = 0
        self._durations = []
        self._t_last = time.time()

    def before_run(self, run_context):
        pass

    def after_run(self, run_context, run_values):
        self._step += 1
        t1 = time.time()
        dur = t1 - self._t_last
        print('step %d, took %.3fs' % (self._step, dur))
        if self._step > self._warmup_steps:
            self._durations.append(dur)
        self._t_last = t1

    def end(self, run_context):
        durations = list(self._durations)
        durations = durations[:len(durations) - self._drop_last]
        ds = np.array(durations)
        mean_duration = ds.mean()
        step_per_sec = 1 / mean_duration
        sample_per_sec = step_per_sec * self._batch_size
        print('%.2f samples / sec, batch size: %d' %
              (sample_per_sec, self._batch_size))


class LogStepHook(tf.train.SessionRunHook):
    def __init__(self):
        self._step = 0

    def before_run(self, run_context):
        self._step += 1
        print('%s::%s before_run %d steps' %
              ('LogStepHook', 'after_run', self._step))

    def after_run(self, run_context, run_values):
        print('%s::%s after %d steps' %
              ('LogStepHook', 'after_run', self._step))

    def end(self, run_context):
        print('%s::%s after %d steps' % ('LogStepHook', 'end', self._step))
