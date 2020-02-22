import os

import numpy as np
import tensorflow as tf
from kungfu.tensorflow.initializer import BroadcastGlobalVariablesOp
from kungfu.tensorflow.ops import (all_reduce, consensus, counter,
                                   resize_cluster, step_based_schedule)


class KungFuElasticTrainHook(tf.train.SessionRunHook):
    def __init__(self, schedule, max_step, model_dir, save_final_model=False):
        self._schedule = schedule
        self._max_step = max_step
        self._model_dir = model_dir
        self._save_final_model = save_final_model
        self._need_sync = True

    def _build_resize_op(self, config, step):
        new_size = step_based_schedule(config, step)
        resize_op = resize_cluster(new_size)
        return resize_op

    def begin(self):
        self._kungfu_step = tf.Variable(0, trainable=False, dtype=tf.int64)
        self._advance = tf.assign_add(self._kungfu_step, 1)
        self._sync_op = BroadcastGlobalVariablesOp()

        self._step = 0
        self._step_place = tf.placeholder(dtype=tf.int32, shape=())
        self._sync_step_op = all_reduce(self._step_place, op='max')
        self._resize_op = self._build_resize_op(self._schedule,
                                                self._step_place)

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
            return
        if changed:
            print('changed on %d' % (self._step))
            self._need_sync = True
        self._step += 1
        if self._step >= self._max_step:
            print('request_stop on kungfu_step: %d' % (self._step))
            run_context.request_stop()

    def end(self, sess):
        print('stopped at step: %d' % (self._step))
        if self._save_final_model:
            self.save(sess, 'final')

    def save(self, sess, idx):
        vs = tf.global_variables()
        d = dict()
        for t in vs:
            v = sess.run(t)
            d[t.name] = v
        np.savez(os.path.join(self._model_dir, 'variables-%s.npz' % (idx)),
                 **d)
