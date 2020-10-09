import kungfu.tensorflow as kf
from kungfu.tensorflow.variables import GraphKeys

from .base_policy import BasePolicy


class ScheduledBatchSizePolicyExample(BasePolicy):
    def __init__(self, schedule):
        self._schedule = schedule

    def before_train(self):
        self._trained_steps = 0
        self._trained_epochs = 0
        self._set_batch_size, self._batch_size_place = kf.create_assign_op_for(
            kf.get_or_create_batch_size())

    def before_epoch(self):
        pass

    def after_step(self, sess):
        self._trained_steps += 1

        if self._trained_steps in self._schedule:
            new_batch_size = self._schedule[self._trained_steps]
            sess.run(self._set_batch_size,
                     feed_dict={self._batch_size_place: new_batch_size})

    def after_epoch(self, sess):
        self._trained_epochs += 1
        # pass
