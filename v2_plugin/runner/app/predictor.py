from abc import ABC

from common.endpoints import BaseEndPoints
from common.servicer import BaseServicer


class PredictorServicer(BaseServicer, ABC):
    pass


class PredictorEndPoints(BaseEndPoints, ABC):
    pass
