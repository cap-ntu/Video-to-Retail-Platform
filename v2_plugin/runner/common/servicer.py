import abc
import json
import os
from functools import partial
from typing import Iterable, Tuple, Union, Callable, Any

import numpy as np
import torch
from toolz import compose

from hysia.utils.misc import obtain_device, str_to_type
from protos import api2msl_pb2_grpc, api2msl_pb2


class BaseServicer(api2msl_pb2_grpc.Api2MslServicer, abc.ABC):
    torch_flag: bool = False
    predict_func = 'batch_predict'

    def __init__(self, config, suppress=False):

        cuda, device_num = obtain_device(config.device)

        if cuda:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(device_num)
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = ''

        self.logger.info(f'Using {"CUDA:" if cuda else "CPU"}{os.environ["CUDA_VISIBLE_DEVICES"]}')

        self.device = (cuda, device_num)
        self.name = config.name

        if not suppress:  # do not run `load_engine`
            self.engine = self.load_engine(config)

            # type checking
            assert hasattr(self.engine, self.predict_func), \
                f'Wrong value for `predict_func`, engine does not have attribute {self.predict_func}'
            assert isinstance(getattr(self.engine, self.predict_func), Callable), \
                f'`{self.predict_func}` is not callable attribute of engine {self.engine}'

    @abc.abstractmethod
    def load_engine(self, config) -> Any:
        raise NotImplementedError('Method `load_engine` is not implemented.')

    def grpc_decode(self, buffer: Iterable, meta) -> Tuple[Union[torch.Tensor, np.ndarray, Any], dict]:
        meta: dict = json.loads(meta)
        shape = meta['shape']
        dtype = str_to_type(meta['dtype'])

        decode_pipeline = compose(
            partial(np.reshape, newshape=shape),
            partial(np.fromstring, dtype=dtype),
        )

        buffer = list(map(decode_pipeline, buffer))

        buffer = np.stack(buffer)

        if self.torch_flag:
            buffer = torch.from_numpy(buffer)

        return buffer, meta

    def post_processing(self, x):
        return x

    def Infer(self, request, context):
        raw_input = request.raw_input
        meta = request.meta
        inputs, _ = self.grpc_decode(raw_input, meta=meta)

        result = getattr(self.engine, self.predict_func)(inputs)

        result = self.post_processing(result)

        return api2msl_pb2.InferResponse(json=json.dumps(result))

    def StreamInfer(self, request_iterator, context):
        for request in request_iterator:
            response = self.Infer(request, context)
            yield response

    def Stop(self, request, context):
        del self.engine

        return api2msl_pb2.StopResponse(status=True)
