import json

import numpy as np
import torch
from toolz import compose

from v2_plugin.protos.infer_pb2 import InferRequest
from v2_plugin.runner.utils import type_serializer


def make_request(model_name, inputs, meta=None):
    inputs = list(inputs)
    example = inputs[0]
    if meta is None:
        meta = dict()

    if isinstance(example, np.ndarray):
        to_byte = bytes
    elif isinstance(example, torch.Tensor):
        to_byte = compose(bytes, torch.Tensor.numpy)
    else:
        raise ValueError(
            'Argument `image` is expected to be an iterative numpy array, or an iterative torch Tensor')

    raw_input = list(map(to_byte, inputs))
    shape = example.shape
    dtype = type_serializer(example.dtype)
    meta.update({'shape': shape, 'dtype': dtype})

    return InferRequest(model_name=model_name, raw_input=raw_input, meta=json.dumps(meta))
