import tensorflow as tf
from kungfu.python import current_cluster_size
from kungfu.tensorflow.optimizers.grad_noise_scale import get_gns_tensor
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
        batch_size_tensor = must_get_tensor_by_name('kungfu_device_batch_size')
        batch_size_place = tf.placeholder(dtype=tf.int64)
        set_batch_size_op = tf.assign(batch_size_tensor, batch_size_place)
        print('%s' % (batch_size_tensor))

        variables = []  # TODO
        params = []  # TODO
        self._policy.before_train(variables, params)

    def after_create_session(self, session, coord):
        print('%s' % 'after_create_session')

    def before_run(self, run_context):
        print('%s' % 'before_run')

        variables = []  # TODO
        params = []  # TODO
        if self._trained_epochs > self._last_trained_epochs:
            self._policy.before_epoch(variables, params)
            self._last_trained_epochs = self._trained_epochs

    def after_run(self, run_context, run_values):
        variables = []  # TODO
        params = []  # TODO
        grads = []  # TODO

        self._policy._sess = run_context.session
        self._policy.after_step(variables, params, grads)

        self._trained_steps += 1
        self._trained_samples += self._batch_size * self._cluster_size
        self._trained_epochs = int(self._trained_samples / self._epoch_size)

        if self._trained_epochs > self._last_trained_epochs:
            self._policy.after_epoch(variables, params)

        if self._trained_samples >= self._total_samples:
            print('%s' % 'request_stop ...')
            run_context.request_stop()

        print('%s' % 'after_run')

    def end(self, session):
        print('%s' % 'end')


class KungFuPolicy(object):
    def get_tensorflow_hook(self):
        return PolicyHook(self)


class GNSPolicy(KungFuPolicy):
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

    def before_train(self, vars, params):
        print('%s' % ('before_train'))

        self._gns = get_gns_tensor()
        print(self._gns)

        # self.prev = None
        # self.ema = kf.ema(0.01) # Local GNS EMA
        # self.sync = True # Worker state synchronisation flag

    def before_epoch(self, vars, params):
        print('%s' % ('before_epoch'))
        # if self.sync:
        #     for v in vars:
        #     v = kf.broadcast(v, 0) # Communication 15 self.sync = False

    def after_step(self, vars, params, grads):
        print('%s' % ('after_step'))
        # sess = tf.get_default_session()
        gns = self._sess.run(self._gns)
        print('gns = %s' % (gns))

        # gns = kf.gns(grads, avg_grads) # Monitoring
        # self.ema.update(gns)

    def after_epoch(self, vars, params):
        print('%s' % ('after_epoch'))

        # gns = self.ema.value()
        # avg = kf.allreduce(gns) / kf.size() # Communication
        # if self.prev is None:
        #     self.prev = avg
        # elif avg > self.prev:
        #     new_size = int(kf.size() * avg / self.prev)
        # if new_size != kf.size():
        #     kf.resize(new_size) # Adaptation
        #     self.sync = True
        #     self.prev = avg
