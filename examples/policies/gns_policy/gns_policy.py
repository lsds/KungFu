import kungfu.tensorflow as kf
import tensorflow as tf
from kungfu.python import current_cluster_size
from kungfu.tensorflow.ops import all_reduce

from base_policy import KungFuPolicy
from ema import EMA


class GNSPolicy(KungFuPolicy):
    def __init__(self, *args, **kwargs):
        super(GNSPolicy, self).__init__(*args, **kwargs)

        self._ratio_ema = EMA(0.9, 2)
        self._prev_ave = None
        self._prev = None

    def before_train(self, vars, params):
        print('%s' % ('before_train'))

        self._gns_place = tf.placeholder(dtype=tf.float32)
        self._gns_ave = all_reduce(self._gns_place)

        self._need_sync = True  # Worker state synchronisation flag

    def _run_sync_op(self):
        pass

    def before_epoch(self, vars, params):
        # print('%s' % ('before_epoch'))
        if self._need_sync:
            self._run_sync_op()
            self._need_sync = False

    def after_step(self, sess, vars, params, grads):
        # print('%s' % ('after_step'))

        # bs = sess.run(self._device_batch_size)
        bs = kf.batch_size(sess)
        gns = kf.gradient_noise_scale(sess)  #  get monitored valued
        gns_abs = abs(gns)
        ratio = gns_abs / bs
        self._ratio_ema.update(ratio)

        print('gns = %s' % (gns))

    def after_epoch(self, sess, vars, params):
        # print('%s' % ('after_epoch'))
        gns_ema = self._ratio_ema.get()

        # TODO: run all_reduce without operator
        ave = sess.run(self._gns_ave, feed_dict={self._gns_place: gns_ema})
        print('after epoch: ave: %s' % (ave))
        if self._prev_ave is not None:
            bs = kf.batch_size(sess)
            if ave > self._prev_ave:
                new_bs = min(bs * 2, 1024)
                print('change batch size %d -> %d' % (bs, new_bs))
                kf.set_batch_size(sess, new_bs)
        self._prev_ave = ave
