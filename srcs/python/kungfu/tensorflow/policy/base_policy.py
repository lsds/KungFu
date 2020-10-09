class BasePolicy(object):
    def __init__(self, *args, **kwargs):
        pass

    def before_train(self, vars, params):
        pass

    def before_epoch(self, vars, params):
        pass

    def after_step(self, sess, vars, params, grads):
        pass

    def after_epoch(self, sess, vars, params):
        pass
