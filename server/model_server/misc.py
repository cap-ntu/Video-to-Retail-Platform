from pathlib import Path
from typing import Union

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
