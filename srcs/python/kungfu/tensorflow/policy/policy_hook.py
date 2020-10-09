import kungfu.tensorflow as kf
import tensorflow as tf
from kungfu.python import current_cluster_size
from kungfu.tensorflow.variables import GraphKeys, create_global_variable


# PolicyHook is a SessionRunHook class to run kungfu policies in tensorflow estimator
class PolicyHook(tf.estimator.SessionRunHook):
    def __init__(self, policies, epoch_size, epoch_num, init_batch_size=None):
        self._policies = policies
        self._epoch_size = epoch_size
        self._epoch_num = epoch_num
        self._total_samples = int(epoch_size * epoch_num)
        self._init_batch_size = init_batch_size

        self._trained_samples = 0
        self._trained_steps = 0
        self._trained_epochs = 0
        self._last_trained_epochs = -1

    @property
    def policies(self):
        return self._policies

    def begin(self):
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

    def after_run(self, run_context, run_values):
        bs = self.get_batch_size(run_context.session)
        self._trained_steps += 1
        self._trained_samples += bs * current_cluster_size()
        self._trained_epochs = int(self._trained_samples / self._epoch_size)

        for policy in self._policies:
            policy.after_step(run_context.session)

        if self._trained_epochs > self._last_trained_epochs:
            for policy in self._policies:
                policy.after_epoch(run_context.session)

        if self._trained_samples >= self._total_samples:
            # print('%s' % 'request_stop ...')
            run_context.request_stop()

        # print('%s' % 'after_run')

    def end(self, session):
        # print('%s' % 'end')
        pass

    def get_batch_size(self, sess):
        return kf.eval_batch_size(sess)
