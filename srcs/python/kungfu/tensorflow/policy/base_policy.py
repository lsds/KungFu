import abc
from abc import abstractmethod


class BasePolicy(abc.ABC):
    def __init__(self, *args, **kwargs):
        pass

    # @abstractmethod
    def before_train(self):
        pass

    # @abstractmethod
    def before_epoch(self, sess):
        pass

    # @abstractmethod
    def before_step(self, sess):
        pass

    # @abstractmethod
    def after_step(self, sess):
        pass

    # @abstractmethod
    def after_epoch(self, sess):
        pass

    # @abstractmethod
    def after_train(self, sess):
        pass
