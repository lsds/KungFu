import kungfu.tensorflow as kf
import tensorflow as tf
from kungfu.python import current_cluster_size, detached
from kungfu.tensorflow.variables import GraphKeys, create_global_variable


# PolicyHook is a SessionRunHook class to run kungfu policies in tensorflow estimator
class PolicyHook(tf.estimator.SessionRunHook):
    def __init__(self, policies, epoch_size, epoch_num, init_batch_size=None):
        self._policies = policies
        self._epoch_size = epoch_size
        self._epoch_num = epoch_num
        self._total_samples = int(epoch_size * epoch_num)
        self._init_batch_size = init_batch_size

        self._trained_epochs = 0
        self._last_trained_epochs = -1

    @property
    def policies(self):
        return self._policies

    def begin(self):
        self._trained_samples = create_global_variable(
            GraphKeys.TRAINED_SAMPLES, shape=[], dtype=tf.int64)
        self._set_trained_samples = kf.create_setter(self._trained_samples)

        kf.get_or_create_batch_size(self._init_batch_size)
        total_samples = create_global_variable(GraphKeys.TOTAL_SAMPLES,
                                               shape=[],
                                               dtype=tf.int64,
                                               init=tf.constant(
                                                   self._total_samples,
                                                   dtype=tf.int64))
        for policy in self._policies:
            policy.before_train()

    def after_create_session(self, session, coord):
        pass

    def before_run(self, run_context):
        if self._trained_epochs > self._last_trained_epochs:
            for policy in self._policies:
                policy.before_epoch(run_context.session)
            self._last_trained_epochs = self._trained_epochs

        for policy in self._policies:
            policy.before_step(run_context.session)

    def after_run(self, run_context, run_values):
        sess = run_context.session
        bs = self.get_batch_size(sess)
        trained_samples = sess.run(self._trained_samples)
        trained_samples += bs * current_cluster_size()
        self._set_trained_samples(sess, trained_samples)
        self._trained_epochs = int(trained_samples / self._epoch_size)

        for policy in reversed(self._policies):
            policy.after_step(sess)

        if self._trained_epochs > self._last_trained_epochs:
            for policy in reversed(self._policies):
                policy.after_epoch(sess)

        if trained_samples >= self._total_samples:
            # print('%s' % 'request_stop ...')
            run_context.request_stop()

        if detached():
            run_context.request_stop()

    def end(self, session):
        for policy in reversed(self._policies):
            policy.after_train(session)

    def get_batch_size(self, sess):
        return kf.eval_batch_size(sess)
