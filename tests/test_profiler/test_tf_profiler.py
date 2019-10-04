import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
this_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(this_dir, ".."))
import tensorflow as tf

import _init_paths
from slim.nets import resnet_v2
from slim.nets import alexnet
from slim.nets import inception

from core.profiler.tf_profiler import profile


class ResNet:
    def __init__(self):
        batch_size = 2
        height, width = 299, 299
        num_classes = 1000

        eval_inputs = tf.random_uniform((batch_size, height, width, 3))
        logits, _ = resnet_v2.resnet_v2_101(
            eval_inputs,
            num_classes,
            is_training=False
        )
        self.predictions = tf.argmax(logits, 1)

    @profile
    def predict(self):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            run_meta = tf.RunMetadata()

            predictions = sess.run(
                self.predictions,
                options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
                run_metadata=run_meta,
            )
            return predictions, run_meta


class AlexNet:
    def __init__(self):
        batch_size = 1
        height, width = 224, 224

        inputs = tf.random_uniform((batch_size, height, width, 3))
        self.logits, _ = alexnet.alexnet_v2(inputs)

    @profile
    def predict(self):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            run_meta = tf.RunMetadata()

            predictions = sess.run(
                self.logits,
                options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
                run_metadata=run_meta,
            )
            return predictions, run_meta


class Inception:
    def __init__(self):
        batch_size = 2
        height, width = 299, 299
        num_classes = 1000

        eval_inputs = tf.random_uniform((batch_size, height, width, 3))
        logits, _ = inception.inception_v3(
            eval_inputs,
            num_classes,
            is_training=False
        )
        self.predictions = tf.argmax(logits, 1)

    @profile
    def predict(self):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            run_meta = tf.RunMetadata()

            predictions = sess.run(
                self.predictions,
                options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
                run_metadata=run_meta,
            )
            return predictions, run_meta


if __name__ == "__main__":
    print("Profiling ResNet...")
    res_net = ResNet()
    res_net.predict()
    tf.reset_default_graph()

    print("Profiling AlexNet..")
    alex_net = AlexNet()
    alex_net.predict()
    tf.reset_default_graph()

    print("Profiling Inception..")
    inception_net = Inception()
    inception_net.predict()
    tf.reset_default_graph()
