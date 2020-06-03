from common.engine import BaseEngine


class Engine(BaseEngine):
    def single_predict(self, **kwargs):
        print('Hello world from single_predict.')

    def batch_predict(self, **kwargs):
        print('Hello world from batch predict.')
