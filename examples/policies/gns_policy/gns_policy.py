import tensorflow as tf
from kungfu.python import current_cluster_size
from kungfu.tensorflow.optimizers.grad_noise_scale import get_gns_tensor
from kungfu.tensorflow.v1.helpers.utils import must_get_tensor_by_name

from base_policy import KungFuPolicy
from ema import EMA


def get_batch_size_tensor():
    return must_get_tensor_by_name('kungfu_device_batch_size')


class GNSPolicy(KungFuPolicy):
    def __init__(self, *args, **kwargs):
        super(GNSPolicy, self).__init__(*args, **kwargs)

        self._ratio_ema = EMA(0.9, 2)
        self._prev_ave = None
        self._prev = None

    def before_train(self, vars, params):
        print('%s' % ('before_train'))

        self._last_gns = get_gns_tensor()
        self._device_batch_size = get_batch_size_tensor()

        self._need_sync = True  # Worker state synchronisation flag

        # self.prev = None

    def _run_sync_op(self):
        pass

    def before_epoch(self, vars, params):
        print('%s' % ('before_epoch'))
        if self._need_sync:
            self._run_sync_op()
            self._need_sync = False

    def after_step(self, vars, params, grads):
        print('%s' % ('after_step'))

        bs = self._sess.run(self._device_batch_size)
        gns = self._sess.run(self._last_gns)
        gns_abs = abs(gns)
        ratio = gns_abs / bs
        self._ratio_ema.update(ratio)

        print('gns = %s' % (gns))

        # gns = kf.gns(grads, avg_grads) # Monitoring
        # self.ema.update(gns)

    def after_epoch(self, vars, params):
        print('%s' % ('after_epoch'))
        ave = self._ratio_ema.get()
        if self._prev_ave is not None:
            pass
        self._prev_ave = ave

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
