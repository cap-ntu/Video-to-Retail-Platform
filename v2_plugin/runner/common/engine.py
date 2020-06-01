import abc


class BaseEngine(abc.ABC):

    def __init__(self, config):
        self.config = config

    @abc.abstractmethod
    def single_predict(self):
        raise NotImplementedError('Method `single_predict` not implemented')

    @abc.abstractmethod
    def batch_predict(self):
        raise NotImplementedError('Method `batch_predict` not implemented')

    # noinspection PyMethodMayBeStatic
    def pre_process(self, x, *args, **kwargs):
        return x

    # noinspection PyMethodMayBeStatic
    def post_process(self, x, *args, **kwargs):
        return x
