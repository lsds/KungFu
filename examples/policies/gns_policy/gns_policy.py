import tensorflow as tf
from kungfu.python import current_cluster_size
from kungfu.tensorflow.optimizers.grad_noise_scale import get_gns_tensor
from kungfu.tensorflow.v1.helpers.utils import must_get_tensor_by_name

from policy_hook import PolicyHook


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
