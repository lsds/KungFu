import kungfu.tensorflow as kf
from kungfu.tensorflow.policy import BasePolicy
from kungfu.tensorflow.variables import GraphKeys


class ScheduledBatchSizePolicyExample(BasePolicy):
    def __init__(self, schedule):
        self._schedule = schedule

    def before_train(self):
        self._trained_steps = 0
        self._trained_epochs = 0

        self._set_batch_size = kf.create_setter(kf.get_or_create_batch_size())

    def before_epoch(self, sess):
        pass

    def before_step(self, sess):
        pass

    def after_step(self, sess):
        self._trained_steps += 1

        if self._trained_steps in self._schedule:
            new_batch_size = self._schedule[self._trained_steps]
            self._set_batch_size(sess, new_batch_size)

    def after_epoch(self, sess):
        self._trained_epochs += 1
