import time
import tensorflow as tf
from kungfu.ops import (all_reduce, broadcast, get_init_version,
                        propose_update, save_variable, start_step,
                        update_cluster, request_variable)


class InitHook(tf.train.SessionRunHook):
    def __init__(self, version, global_step):
        def _request(target, version, v):
            return tf.assign(
                v, request_variable(0, version, v.name, v.shape, v.dtype))

        bcast_ops = []
        save_ops = []
        request_ops = []
        for v in tf.trainable_variables():
            bcast_ops.append(tf.assign(v, broadcast(v)))
            save_ops.append(save_variable(global_step, v))
            request_ops.append(_request(0, global_step - 1, v))
        self._bcast_ops = bcast_ops
        self._save_ops = save_ops
        self._request_ops = request_ops
        self._tf_init = tf.global_variables_initializer()
        self._global_step = global_step

    def after_create_session(self, sess, coord):
        sess.run(self._tf_init)
        if sess.run(self._global_step) == 0:
            sess.run(self._bcast_ops)
            sess.run(self._save_ops)
        else:
            sess.run(self._request_ops)


class AdaptHook(tf.train.SessionRunHook):
    def __init__(self, version, global_step, max_step, mon_op, new_size):
        self._version = version
        self._global_step = global_step

        self._update_step = tf.Variable(tf.constant(-1, dtype=tf.int64))
        self._set_update_step = tf.assign(self._update_step,
                                          self._global_step + 1)

        self._should_update = tf.equal(self._global_step, self._update_step)
        self._update_cluster_op = update_cluster(self._version)
        self._inc_version = tf.assign_add(self._version, 1)

        isnt_last_step = tf.less(global_step + 1, max_step)
        self._should_propose = tf.logical_and(mon_op, isnt_last_step)
        self._propose_update_op = propose_update(self._update_step,
                                                 self._version, new_size)

        save_ops = []
        for v in tf.trainable_variables():
            save_ops.append(save_variable(global_step, v))
        self._save_ops = save_ops

    def before_run(self, run_context):
        should_update = run_context.session.run(self._should_update)
        if should_update:
            exist = run_context.session.run(self._update_cluster_op)
            if not exist:
                print('peer not in cluster any more')
                run_context.request_stop()

    def after_run(self, run_context, run_values):
        should_propose = run_context.session.run(self._should_propose)
        if should_propose:
            run_context.session.run(self._inc_version)
            run_context.session.run(self._set_update_step)
            accepted, keep = run_context.session.run(self._propose_update_op)
            print('propose result: (%s, %s)' % (accepted, keep))
            if accepted:
                run_context.session.run(self._save_ops)
            else:
                print('propose update if rejected')
            if not keep:
                print('peer not in new cluster')
                run_context.request_stop()


class GlobalStepHook(tf.train.SessionRunHook):
    def __init__(self, global_step, limit):
        self._global_step = global_step
        self._limit = limit
        self._inc_global_step = tf.assign_add(global_step, 1)

    def after_run(self, run_context, run_values):
        global_step = run_context.session.run(self._inc_global_step)
        if global_step >= self._limit:
            run_context.request_stop()


def _create_version_tensor():
    init_version = get_init_version()
    version_tensor = tf.Variable(tf.constant(init_version, tf.int32),
                                 trainable=False)
    return version_tensor


def mon_op(global_step):
    # 0 1 2 3 4 5 6 7 8 9 10 11
    #     x     x     x      x
    return tf.equal(tf.mod(global_step, 3), 2)


def compute_new_size(global_step):
    # 0 1 2 3 4 5 6 7 8 9 10 11 12
    # 1 2 3 4 5 6 1 2 3 4 5  6
    #     x     x     x      x
    # 2 2 2 3 3 3 6 6 6 3 3  3  6
    return tf.cast(tf.mod(global_step, 6) + 1, dtype=tf.int32)


class StopWatch():
    def __init__(self):
        self._last = time.time()

    def __call__(self):
        t = time.time()
        d = t - self._last
        self._last = t
        return d


def show_duration(duration):
    if duration < 1e-3:
        return '%.2fms' % (duration * 1e3)
    return '%.2fs' % duration


def log_duration(duration, name):
    print('%s took %s' % (name, show_duration(duration)))


class Trainer(object):
    def __init__(self, adapt=False):
        self._adapt = adapt

    def train(self, max_step, train_op, handle=None):
        version = _create_version_tensor()
        global_step = tf.Variable(start_step(version), trainable=False)

        hooks = []
        if self._adapt:
            hooks.append(
                AdaptHook(version, global_step, max_step, mon_op(global_step),
                          compute_new_size(global_step)))
        hooks.extend([
            GlobalStepHook(global_step, max_step),
            InitHook(version, global_step),
        ])

        watch = StopWatch()
        with tf.train.MonitoredSession(hooks=hooks) as sess:
            log_duration(watch(), 'create session')
            n = 0
            while not sess.should_stop():
                print('step %d ----------------------------------' % (n))
                result = sess.run(train_op)
                if handle:
                    handle(result)
                log_duration(watch(), 'train_step %d' % n)
                n += 1
        print('done')
