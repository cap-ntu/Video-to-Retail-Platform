import abc


class BaseEngine(abc.ABC):

    @abc.abstractmethod
    def batch_predict(self):
        raise NotImplementedError()

    def pre_process(self, x, *args, **kwargs):
        return x

    def post_process(self, x, *args, **kwargs):
        return x
