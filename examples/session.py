import tensorflow as tf
from kungfu.ops import broadcast, get_init_version, save_variable, propose_update, start_step, update_cluster, all_reduce


class InitHook(tf.train.SessionRunHook):
    def __init__(self, version, global_step):
        bcast_ops = []
        save_ops = []
        for v in tf.trainable_variables():
            bcast_ops.append(tf.assign(v, broadcast(v)))
            save_ops.append(save_variable(global_step, v))
        self._bcast_ops = bcast_ops
        self._save_ops = save_ops

        self._reset_global_step = tf.assign(global_step, start_step(version))
        self._tf_init = tf.global_variables_initializer()

        self._global_step = global_step

    def after_create_session(self, sess, coord):
        print('InitHook::after_create_session')

        print('running tf init')
        sess.run(self._tf_init)

        print('running reset global step')
        sess.run(self._reset_global_step)

        global_step = sess.run(self._global_step)
        print('init global_step = %d' % (global_step))

        # print('running broadcast ops')
        # sess.run(self._bcast_ops)

        print('running save ops')
        sess.run(self._save_ops)


class AdaptHook(tf.train.SessionRunHook):
    def __init__(self, version, global_step, mon_op, new_size):
        self._version = version
        self._global_step = global_step

        self._update_step = tf.Variable(tf.constant(-1, dtype=tf.int64))
        self._set_update_step = tf.assign(self._update_step,
                                          self._global_step + 1)

        self._should_update = tf.equal(self._global_step, self._update_step)
        self._update_cluster_op = update_cluster(self._version)
        self._inc_version = tf.assign_add(self._version, 1)

        self._should_propose = mon_op
        self._propose_update_op = propose_update(self._update_step,
                                                 self._version, new_size)

    def before_run(self, run_context):
        print('AdaptHook::before_run')
        should_update = run_context.session.run(self._should_update)
        if should_update:
            version = run_context.session.run(self._version)
            print('update to version %d' % (version))
            exist = run_context.session.run(self._update_cluster_op)
            if not exist:
                print('peer not in cluster any more')
                run_context.request_stop()
                print('peer not in cluster any more, request_stop called!')

    def after_run(self, run_context, run_values):
        print('AdaptHook::after_run')
        should_propose = run_context.session.run(self._should_propose)
        if should_propose:
            run_context.session.run(self._inc_version)
            run_context.session.run(self._set_update_step)
            accepted, keep = run_context.session.run(self._propose_update_op)
            print('propose result: (%s, %s)' % (accepted, keep))
            if not accepted:
                print('propose update if rejected')
            if not keep:
                print('peer not in new cluster')
                run_context.request_stop()


class GlobalStepHook(tf.train.SessionRunHook):
    def __init__(self, global_step, limit):
        self._global_step = global_step
        self._limit = limit
        self._inc_global_step = tf.assign_add(global_step, 1)

    def before_run(self, run_context):
        print('GlobalStepHook::before_run')

    def after_run(self, run_context, run_values):
        print('GlobalStepHook::after_run')
        run_context.session.run(self._inc_global_step)
        global_step = run_context.session.run(self._global_step)
        if global_step >= self._limit:
            run_context.request_stop()


class DebugHook(tf.train.SessionRunHook):
    def __init__(self, global_step):
        self._global_step = global_step
        one = tf.Variable(tf.ones([], dtype=tf.int32))
        self._np = all_reduce(one)

    def before_run(self, run_context):
        print('DebugHook::before_run')

    def after_run(self, run_context, run_values):
        print('DebugHook::after_run')
        if run_context.stop_requested:
            print('stop_requested!')
            return

        print('DebugHook::after_run global step : %d' %
              tf.train.global_step(run_context.session, self._global_step))
        v = run_context.session.run(self._np)
        print('np=%d' % (v))


def _create_version_tensor():
    init_version = get_init_version()
    version_tensor = tf.Variable(tf.constant(init_version, tf.int32))
    return version_tensor


def mon_op(global_step):
    # 0 1 2 3 4 5 6 7 8 9 10 11
    #     x     x     x      x
    return tf.equal(tf.mod(global_step, 3), 2)


def compute_new_size(global_step):
    # 0 1 2 3 4 5 6 7 8 9 10 11
    # 1 2 3 4 5 6 1 2 3 4 5  6
    #     x     x     x      x
    # 2 2 2 3 3 3 6 6 6 3 3  3
    return tf.cast(tf.mod(global_step, 6) + 1, dtype=tf.int32)


# public
def kungfu_train(max_step, train_step):
    global_step = tf.Variable(tf.zeros([], tf.int64), trainable=False)
    version = _create_version_tensor()

    hooks = [
        AdaptHook(version, global_step, mon_op(global_step),
                  compute_new_size(global_step)),  # optional
        # DebugHook(global_step),
        GlobalStepHook(global_step, max_step),
        InitHook(version, global_step),
    ]

    with tf.train.MonitoredSession(hooks=hooks) as sess:
        n = 0
        while not sess.should_stop():
            print('step : %d ------------' % (n))
            train_step(sess)
            n += 1

    print('done')
