import json
from typing import List, Iterable, Tuple

import numpy as np
from common.endpoints import BaseEndPoints
from common.servicer import BaseServicer
from pydantic import BaseModel
from utils import type_deserializer


class PredictorServicer(BaseServicer):

    predict_func = 'single_predict'

    def grpc_decode(self, buffer: Iterable, meta) -> Tuple[np.ndarray, dict]:
        meta: dict = json.loads(meta)
        shape = meta['shape']
        dtype = type_deserializer(meta['dtype'])

        buffer = list(buffer)

        assert len(buffer) == 1, 'Currently only support batch = 1.'

        buffer = np.fromstring(buffer[0], dtype=dtype).reshape(shape)

        return buffer, meta


class Input(BaseModel):
    raw_input: List[List[List[int]]]
    dtype: str


class PredictorEndPoints(BaseEndPoints):
    async def predict(self, inputs: Input):
        np_array = np.array(inputs.raw_input, dtype=type_deserializer(inputs.dtype))
        result = self.engine.single_predict(np_array)

        return {'result': result}
