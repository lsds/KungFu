from policy_hook import PolicyHook


class KungFuPolicy(object):
    def __init__(self, epoch_size, epoch_num, init_batch_size):
        self._epoch_size = epoch_size
        self._epoch_num = epoch_num
        self._total_samples = int(epoch_size * epoch_num)
        self._init_batch_size = init_batch_size

    @property
    def epoch_size(self):
        return self._epoch_size

    @property
    def epoch_num(self):
        return self._epoch_num

    @property
    def total_samples(self):
        return self._total_samples

    @property
    def init_batch_size(self):
        return self._init_batch_size

    def get_tensorflow_hook(self):
        return PolicyHook(self)

    def before_train(self, vars, params):
        print('%s' % ('before_train'))

    def before_epoch(self, vars, params):
        print('%s' % ('before_epoch'))

    def after_step(self, sess, vars, params, grads):
        print('%s' % ('after_step'))

    def after_epoch(self, sess, vars, params):
        print('%s' % ('after_epoch'))
