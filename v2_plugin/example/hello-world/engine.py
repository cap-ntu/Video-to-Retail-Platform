from common.engine import BaseEngine


class Engine(BaseEngine):
    def single_predict(self, *args, **kwargs):
        return 'hello'

    def batch_predict(self, *args, **kwargs):
        return ['hello'] * 8
