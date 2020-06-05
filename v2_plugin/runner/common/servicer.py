import abc
import json
from functools import partial
from typing import Iterable, Tuple, Union, Callable, Any

import numpy as np
from toolz import compose

from common.engine import BaseEngine
from protos import infer_pb2_grpc, infer_pb2
from utils import type_deserializer

try:
    import torch
except ImportError:
    torch = None


class BaseServicer(infer_pb2_grpc.InferProtoServicer, abc.ABC):
    torch_flag: bool = False
    predict_func = 'batch_predict'

    def __init__(self, engine: BaseEngine):

        self.engine = engine

        # type checking
        assert hasattr(self.engine, self.predict_func), \
            f'Wrong value for `predict_func`, engine does not have attribute {self.predict_func}'
        assert isinstance(getattr(self.engine, self.predict_func), Callable), \
            f'`{self.predict_func}` is not callable attribute of engine {self.engine}'

    def grpc_decode(self, buffer: Iterable, meta) -> Tuple[Union[np.ndarray, Any], dict]:
        meta: dict = json.loads(meta)
        shape = meta['shape']
        dtype = type_deserializer(meta['dtype'])

        decode_pipeline = compose(
            partial(np.reshape, newshape=shape),
            partial(np.fromstring, dtype=dtype),
        )

        buffer = list(map(decode_pipeline, buffer))

        buffer = np.stack(buffer)

        if self.torch_flag:
            buffer = torch.from_numpy(buffer)

        return buffer, meta

    def Infer(self, request, context):
        raw_input = request.raw_input
        meta = request.meta
        inputs, _ = self.grpc_decode(raw_input, meta=meta)

        result = getattr(self.engine, self.predict_func)(inputs)

        result = self.engine.post_process(result)

        return infer_pb2.InferResponse(json=json.dumps(result))

    def StreamInfer(self, request_iterator, context):
        for request in request_iterator:
            response = self.Infer(request, context)
            yield response

    def Stop(self, request, context):
        del self.engine

        return infer_pb2.StopResponse(status=True)
