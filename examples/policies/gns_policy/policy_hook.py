import kungfu.tensorflow as kf
import tensorflow as tf
from kungfu.python import current_cluster_size
from kungfu.tensorflow.v1.helpers.utils import must_get_tensor_by_name


# PolicyHook is a adaptor class to run kungfu policies in tensorflow estimator
class PolicyHook(tf.estimator.SessionRunHook):
    def __init__(self, policy):
        self._policy = policy
        self._epoch_size = policy.epoch_size
        self._epoch_num = policy.epoch_num
        self._total_samples = policy.total_samples
        self._batch_size = policy.init_batch_size

        self._trained_samples = 0
        self._cluster_size = current_cluster_size()
        self._trained_steps = 0
        self._trained_epochs = 0
        self._last_trained_epochs = -1

        print('%s' % 'init')

    def begin(self):
        print('%s' % 'begin')

        variables = []  # TODO
        params = []  # TODO
        self._policy.before_train(variables, params)

    def after_create_session(self, session, coord):
        print('%s' % 'after_create_session')

    def before_run(self, run_context):
        # print('%s' % 'before_run')

        variables = []  # TODO
        params = []  # TODO
        if self._trained_epochs > self._last_trained_epochs:
            self._policy.before_epoch(variables, params)
            self._last_trained_epochs = self._trained_epochs

    def after_run(self, run_context, run_values):
        variables = []  # TODO
        params = []  # TODO
        grads = []  # TODO

        self._policy.after_step(run_context.session, variables, params, grads)
        self._trained_steps += 1
        self._trained_samples += self._batch_size * self._cluster_size
        self._trained_epochs = int(self._trained_samples / self._epoch_size)

        if self._trained_epochs > self._last_trained_epochs:
            self._policy.after_epoch(run_context.session, variables, params)

        if self._trained_samples >= self._total_samples:
            print('%s' % 'request_stop ...')
            run_context.request_stop()

        # print('%s' % 'after_run')

    def end(self, session):
        print('%s' % 'end')
