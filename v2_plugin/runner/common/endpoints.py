import abc

from fastapi import APIRouter

from common.engine import BaseEngine


class BaseEndPoints(abc.ABC):

    router = APIRouter()

    def __init__(self, engine: BaseEngine):
        self.engine = engine
        self.predict = self.router.post('/predict')(self.predict)

    @abc.abstractmethod
    async def predict(self):
        raise NotImplementedError('Endpoint method `predict` does not implemented.')
