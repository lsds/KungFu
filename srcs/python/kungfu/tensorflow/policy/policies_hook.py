import tensorflow as tf
from kungfu.policy import Policy

class PoliciesHook(tf.estimator.SessionRunHook):
    def __init__(self, policies: [Policy], batch_size, num_training_samples):
        self._policies = policies
        self._batch_size = batch_size
        self._num_training_samples = num_training_samples

    def after_create_session(self, session, coord):
        [policy.before_train() for policy in self._policies]

    def after_run(self, run_context, run_values):
        global_step = run_context.session.run(self._global_step)
        [policy.after_step(global_step) for policy in self._policies]
        if self.completed_epoch(global_step):
            [policy.after_epoch() for policy in self._policies]

    def before_run(self, run_context):
        global_step = global_step = run_context.session.run(self._global_step)
        if self.completed_epoch(global_step) or global_step == 0:
            [policy.before_epoch() for policy in self._policies]
        [policy.before_step() for policy in self._policies]

    def begin(self):
        self._global_step = tf.train.get_or_create_global_step()

    def end(self, session):
        [policy.after_train() for policy in self._policies]

    def completed_epoch(self, global_step):
        num_processed_samples = global_step * self._batch_size
        num_processed_samples = num_processed_samples % self._num_training_samples
        if (global_step != 0 and
            num_processed_samples >= self._num_training_samples and
            num_processed_samples < self._num_training_samples + self._batch_size):
            return True
        return False
