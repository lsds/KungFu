import os
import time

import numpy as np
import tensorflow as tf
from kungfu._utils import map_maybe
from kungfu.tensorflow.compat import _tf_assign, _tf_hook
from kungfu.tensorflow.initializer import BroadcastGlobalVariablesOp
from kungfu.tensorflow.ops import (all_reduce, counter, current_cluster_size,
                                   group_all_reduce)
from kungfu.tensorflow.optimizers.core import _KungFuAlgorithm


class NetMonHook(tf.estimator.SessionRunHook):

    interference_threshold = 0.25

    def __init__(self):
        self._cur_step = 0
        self._avg_step_dur = 0


    def begin(self):
        self.__setup_summary_writer()
        self._avg_step_dur_tensor = tf.Variable(0.,trainable=False)
        self._net_cong_mon_tensor = tf.Variable(np.uint8(0), trainable=False)
        tf.summary.scalar(name='CMA', tensor=self._avg_step_dur_tensor)
        tf.summary.scalar(name='congestion', tensor=self._net_cong_mon_tensor)
        self._net_cong_mon_place = tf.placeholder(tf.uint8)
        self._cma_place = tf.placeholder(tf.float32)
        self._net_cong_mon_assign_op = tf.assign(self._net_cong_mon_tensor, self._net_cong_mon_place)
        self._cma_assign_op = tf.assign(self._avg_step_dur_tensor, self._cma_place)

        #get Ada optimizer cond variable handle 
        self._cond_var_Ada = tf.get_default_graph().get_tensor_by_name('cond_var_Ada:0')
        self._cond_var_Ada_setTrue = tf.assign(self._cond_var_Ada, True)
        self._cond_var_Ada_setFalse = tf.assign(self._cond_var_Ada, False)

        #create AllReduce tensor
        self._global_avg_step_dur_tensor = tf.Variable(0.,trainable=False)
        self._global_avg_step_dur_tensor_place = tf.placeholder(tf.float32)
        self._global_avg_step_dur_tensor_place_assign_op = tf.assign(self._global_avg_step_dur_tensor, self._global_avg_step_dur_tensor_place)

        #create AllReduce operator
        self._global_avg_step_dur_allreduce_op = all_reduce(self._global_avg_step_dur_tensor)

    def after_create_session(self, sess, coord):
        pass

    def before_run(self, run_context):
        self._cur_step +=1
        self._step_start_time = time.time()

    def after_run(self, run_context, run_values):
        step_dur = time.time() - self._step_start_time
        
        if self._cur_step == 1:
            self._avg_step_dur = step_dur

            # self._avg_step_dur_tensor.assign(step_dur)
            run_context.session.run(self._cma_assign_op, feed_dict={
                self._cma_place: step_dur,
            })
            return

        #update global avg step dur tensor
        run_context.session.run(self._global_avg_step_dur_tensor_place_assign_op, feed_dict={
            self._global_avg_step_dur_tensor_place: step_dur,
        })

        #perform allreduce to communicate cma between peers
        global_aggr_avg = self.__cma_allreduce(run_context)

        #check for network interference: 
        #If Global CMA average is deviating more that a predefined value from the last step CMA
        #trigger a network interference flag action
        if global_aggr_avg - self._avg_step_dur > self.interference_threshold*self._avg_step_dur:
            print("WARNINIG: Network congestion detected !")
            
            #update network congestion monitor
            run_context.session.run(self._net_cong_mon_assign_op, feed_dict={
                self._net_cong_mon_place: 1,
            })

            #TODO: change for more intricate triggering algorithm
            run_context.session.run(self._cond_var_Ada_setFalse)

        else:
            #update network congestion monitor
            run_context.session.run(self._net_cong_mon_assign_op, feed_dict={
                self._net_cong_mon_place: 0,
            })

            #TODO: change for more intricate triggering algorithm
            run_context.session.run(self._cond_var_Ada_setTrue)

        #Calculation of Cumulative Moving Average (CMA)
        self._avg_step_dur = ((self._avg_step_dur * (self._cur_step-1)) + global_aggr_avg) / self._cur_step

        run_context.session.run(self._cma_assign_op, feed_dict={
            self._cma_place: self._avg_step_dur,
        })
    
    def end(self, sess):
        pass

    def __setup_summary_writer(self):
        cma_log_dir = 'mnist/model'
        self._cma_summary_writer = tf.summary.FileWriter(cma_log_dir)

    def __cma_allreduce(self, run_context):
        cluster_size = current_cluster_size()

        #perform CMA AllReduce
        return (run_context.session.run(self._global_avg_step_dur_allreduce_op)) / cluster_size


class CustomAdaptiveSGD(_KungFuAlgorithm):
    def __init__(self, change_step, alpha):
        self._num_workers = current_cluster_size()
        self._alpha = alpha
        self._change_step = change_step
        self._global_step = tf.train.get_or_create_global_step()
        self._cond_var_Ada_var = tf.Variable(False, trainable=False, name='cond_var_Ada')

    def _ssgd(self, apply_grads_func, gradients, variables, **kwargs):
        sum_grads = group_all_reduce(gradients)
        avg_grads = map_maybe(lambda g: g / self._num_workers, sum_grads)

        # We need to re-zip gradients and variables as grads_and_vars can be only unzipped once.
        grads_and_vars = zip(avg_grads, variables)

        return apply_grads_func(grads_and_vars, **kwargs)

    def _sma(self, apply_grads_func, gradients, variables, **kwargs):
        # It is important to apply model averaging every iteration [2]
        sum_vars = group_all_reduce(variables)
        avg_vars = [v / self._num_workers for v in sum_vars]

        # TODO: Apply momentum to the averaged model [2]
        assign_ops = [
            _tf_assign(v, (1 - self._alpha) * v + self._alpha * avg_v)
            for v, avg_v in zip(variables, avg_vars)
        ]

        # We need to re-zip gradients and variables as grads_and_vars can be only unzipped once.
        grads_and_vars = zip(gradients, variables)

        # We can overlap model averaging and local SGD [2].
        with tf.control_dependencies(assign_ops):
            return apply_grads_func(grads_and_vars, **kwargs)

    def apply_gradients(self, apply_grads_func, grads_and_vars, **kwargs):
        g, v = list(zip(*grads_and_vars))

        return tf.cond(self._cond_var_Ada_var,
                       lambda: self._sma(apply_grads_func, g, v, **kwargs),
                       lambda: self._ssgd(apply_grads_func, g, v, **kwargs))
