import json
import logging
import re
from collections import defaultdict
from typing import Tuple, Optional

import yaml


class _Struct(object):
    def __init__(self, d):
        for a, b in d.items():
            if isinstance(b, (list, tuple)):
                setattr(self, a, [_Struct(x) if isinstance(x, dict) else x for x in b])
            else:
                setattr(self, a, _Struct(b) if isinstance(b, dict) else b)


class ObjectEncoder(json.JSONEncoder):
    def default(self, o: object):
        if isinstance(o, object):
            return o.__dict__
        else:
            return json.JSONEncoder.default(self, o)


def dict_to_object(dictionary: dict):
    return _Struct(dictionary)


def object_to_dict(struct: object):
    return json.loads(json.dumps(struct, cls=ObjectEncoder))


def load_config():
    with open('config.yml', 'r') as stream:
        try:
            config = yaml.safe_load(stream)
            config = dict_to_object(config)
        except yaml.YAMLError as e:
            print(e)
            exit(1)

    return config


def obtain_device(device: str) -> Tuple[bool, Optional[int]]:
    """Obtain device (CUDA device or CPU) and device number from device string.

    Args:
        device (str): Device string. For example, 'cpu', 'cuda', 'cuda:1'.

    Returns:
        Tuple[bool, Optional[int]]: Tuple of CUDA flag and cuda device number (`None` if CUDA flag is `False`).
    """

    device = device.lower()

    device_num = None
    if device == 'cpu':
        cuda = False
    else:
        # match something like cuda, cuda:0, cuda:1
        matched = re.match(r'^cuda(?::([0-9]+))?$', device)
        if matched is None:  # load with CPU
            logging.warning('Wrong device specification, using `cpu`.')
            cuda = False
        else:  # load with CUDA
            cuda = True
            device_num = matched.groups()[0]
            if device_num is None:
                device_num = 0

    return cuda, device_num


def type_serializer(dtype):
    """Serialize data type to string."""
    import torch
    import numpy as np

    mapper = defaultdict(
        lambda x: 'INVALID',
        {
            np.dtype(np.uint8): 'UINT8',
            np.dtype(np.float32): 'FP32',
            torch.float32: 'FP32',
            torch.int64: 'INT64',
        }
    )

    return mapper[dtype]


def type_deserializer(dtype_string: str):
    """Deserialize data type from string."""
    import numpy as np

    mapper = {
        'UINT8': np.uint8,
        'FP32': np.float32,
        'INT64': np.int64,
    }

    return mapper[dtype_string]
