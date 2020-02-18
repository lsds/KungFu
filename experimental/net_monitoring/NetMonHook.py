import os

import time
import numpy as np
import tensorflow as tf
from kungfu.tensorflow.initializer import BroadcastGlobalVariablesOp
from kungfu.tensorflow.ops import (current_cluster_size, all_reduce)


class NetMonHook(tf.estimator.SessionRunHook):
    def __init__(self):
        self._cur_step = 0
        self._avg_step_dur = 0


    def begin(self):
        self.__setup_summary_writer()
        self._avg_step_dur_tensor = tf.Variable(0.,trainable=False)
        tf.summary.scalar(name='CMA', tensor=self._avg_step_dur_tensor)
        self._cma_place = tf.placeholder(tf.float32)
        self._cma_assign_op = tf.assign(self._avg_step_dur_tensor, self._cma_place)

        #create AllReduce tensor
        self._global_avg_step_dur_tensor = tf.Variable(0.,trainable=False)
        self._allreduce_place = tf.placeholder(tf.float32)
        self._allreduce_op = tf.assign(self._global_avg_step_dur_tensor, self._allreduce_place)

        #create AllReduce operator
        self._global_avg_step_dur_op = all_reduce(self._global_avg_step_dur_tensor)

    def after_create_session(self, sess, coord):
        pass

    def before_run(self, run_context):
        self._cur_step +=1
        self._step_start_time = time.time()

    def after_run(self, run_context, run_values):
        step_dur = time.time() - self._step_start_time
        #print('Step Time: ', step_dur,' sec')
        if self._cur_step == 1:
            self._avg_step_dur = step_dur

            # self._avg_step_dur_tensor.assign(step_dur)
            run_context.session.run(self._cma_assign_op, feed_dict={
                self._cma_place: step_dur,
            })
            return

        #update CMA tensor
        run_context.session.run(self._allreduce_op, feed_dict={
            self._allreduce_place: step_dur,
        })

        #perform allreduce to communicate cma between peers
        global_aggr_avg = self.__cma_allreduce(run_context)

        #check for network interference: 
        #If Global CMA average is deviating more that a predefined value from the last step CMA
        #trigger a network interference flag action
        if global_aggr_avg - self._avg_step_dur > 0.1*self._avg_step_dur:
            print("Network congestion detected")

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
        return (run_context.session.run(self._global_avg_step_dur_op)) / cluster_size