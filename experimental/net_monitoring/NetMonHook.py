import os

import time
import numpy as np
import tensorflow as tf
from kungfu.tensorflow.initializer import BroadcastGlobalVariablesOp


class NetMonHook(tf.estimator.SessionRunHook):
    def __init__(self):
        self._cur_step = 0
        self._avg_step_dur = 0


    def begin(self):
        self.__setup_summary_writer()
        self._avg_step_dur_tensor = tf.Variable(0.)
        tf.summary.scalar(name='CMA', tensor=self._avg_step_dur_tensor)
        self._place = tf.placeholder(tf.float32)
        self._assign_op = tf.assign(self._avg_step_dur_tensor, self._place)

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
            run_context.session.run(self._assign_op, feed_dict={
                self._place: 1,
            })
        else:
            #Calculation of Cumulative Moving Average (CMA)
            self._avg_step_dur = ((self._avg_step_dur * (self._cur_step-1)) + step_dur) / self._cur_step

            run_context.session.run(self._assign_op, feed_dict={
                self._place: self._avg_step_dur,
            })
        #print('cur cma = ', self._avg_step_dur)
        #print('cur cma tensor = ', run_context.session.run(self._avg_step_dur_tensor))
    
    def end(self, sess):
        pass

    def __setup_summary_writer(self):
        cma_log_dir = 'mnist/model'
        self._cma_summary_writer = tf.summary.FileWriter(cma_log_dir)