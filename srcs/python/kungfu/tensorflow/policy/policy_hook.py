import tensorflow as tf
import kungfu as kf

class PolicyHook(tf.estimator.SessionRunHook):
    def __init__(self, policy: kf.policy.Policy, batch_size, num_training_samples):
        self._policy = policy
        self._batch_size = batch_size
        self._num_training_samples = num_training_samples

    def after_create_session(self, session, coord):
        self._policy.before_train(vars=None, params=None) # here or in begin() ?

    def after_run(self, run_context, run_values):
        global_step = global_step = run_context.session.run(self._global_step)
        self._policy.after_step(vars=None, params=None)
        if self.completed_epoch(global_step):
            self._policy.after_epoch(vars=None, params=None)

    def before_run(self, run_context):
        global_step = global_step = run_context.session.run(self._global_step)
        if self.completed_epoch(global_step) or global_step == 0:
            self._policy.before_epoch(vars=None, params=None)
        self._policy.before_step(vars=None, params=None)

    def begin(self):
        self._global_step = tf.train.get_or_create_global_step()

    def end(self, session):
        self._policy.after_train(vars=None, params=None)

    def completed_epoch(self, global_step):
        num_processed_samples = global_step * self._batch_size
        num_processed_samples = num_processed_samples % self._num_training_samples
        if (global_step != 0 and
            num_processed_samples >= self._num_training_samples and
            num_processed_samples < self._num_training_samples + self._batch_size):
            return True
        return False
