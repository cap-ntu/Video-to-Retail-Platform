import logging
import re
from pathlib import Path
from typing import Union, Tuple, Optional

import tensorflow as tf

from utils.perf import StreamSuppressor


def load_tf_graph(graph_pb_path: Union[Path, str]) -> tf.Graph:

    graph_pb_path = str(graph_pb_path)

    with StreamSuppressor():
        tf_graph = tf.Graph()
        with tf_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(graph_pb_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
    return tf_graph


def obtain_device(device: str) -> Tuple[bool, Optional[int]]:

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
