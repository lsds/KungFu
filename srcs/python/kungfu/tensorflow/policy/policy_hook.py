import tensorflow as tf

class PolicyHook(tf.estimator.SessionRunHook):
    def after_create_session(self, session, coord):
        pass

    def after_run(self, run_context, run_values):
        pass

    def before_run(self, run_context):
        pass

    def begin(self):
        pass

    def end(self, session):
        pass
