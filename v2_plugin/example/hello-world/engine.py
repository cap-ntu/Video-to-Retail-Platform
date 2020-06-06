from common.engine import BaseEngine


class Engine(BaseEngine):
    @classmethod
    def single_predict(cls, *args, **kwargs):
        return 'hello'

    @classmethod
    def batch_predict(cls, *args, **kwargs):
        return ['hello'] * 8
