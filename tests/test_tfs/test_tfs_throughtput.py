import sys
import time
import threading

import grpc
import numpy
import soundfile as sf
import tensorflow as tf

import _init_paths
import audioset.vggish_input as vggish_input

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc


tf.app.flags.DEFINE_integer('concurrency', 1, 'concurrent inference requests limit')
tf.app.flags.DEFINE_integer('num_tests', 100, 'Number of test sample')
tf.app.flags.DEFINE_string('server', '0.0.0.0:8500', 'PredictionService host:port')
tf.app.flags.DEFINE_string('work_dir', '/tmp', 'Working directory')
FLAGS = tf.app.flags.FLAGS


class _ResultCounter(object):

    def __init__(self, num_tests, concurrency):
        self._num_tests = num_tests
        self._concurrency = concurrency
        self._error = 0
        self._done = 0
        self._active = 0
        self._condition = threading.Condition()
        self._start_time = -1
        self._end_time = 0

    def inc_done(self):
        with self._condition:
            self._done += 1
            if self._done == self._num_tests:
                self.set_end_time(time.time())
            self._condition.notify()

    def dec_active(self):
        with self._condition:
            self._active -= 1
            self._condition.notify()

    def throttle(self):
        with self._condition:
            if self._start_time == -1:
                self._start_time = time.time()
            while self._active == self._concurrency:
                self._condition.wait()
            self._active += 1
    
    def set_start_time(self, start_time):
        self._start_time = start_time

    def set_end_time(self, end_time):
        self._end_time = end_time

    def get_throughput(self):
        if self._end_time == 0:
            self.set_end_time(time.time())
        print(self._end_time - self._start_time)
        return self._num_tests / (self._end_time - self._start_time)


def time_to_sample(t, sr, factor):
    return round(sr * t / factor)


def _create_rpc_callback(label, result_counter):
    
    def _callback(result_future):

        exception = result_future.exception()
        if exception:
            # result_counter.inc_error()
            print(exception)
        else:
            print('normal')
            sys.stdout.write('.')
            sys.stdout.flush()
            response = numpy.array(result_future.result().outputs['output'].float_val)
        result_counter.inc_done()
        result_counter.dec_active()
    return _callback


def inference(hostport, work_dir, concurrency, num_tests):
    
    audio_path = 'test_DB/test_airport.wav'
    num_secs = 1
    sc_start = 0
    sc_end = 2000
    
    wav_data, sr = sf.read(audio_path, dtype='int16')
    assert wav_data.dtype == numpy.int16, 'Bad sample type: %r' % wav_data.dtype
    samples = wav_data / 32768.0  # Convert to [-1.0, +1.0]

    sc_center = time_to_sample((sc_start + sc_end) / 2, sr, 1000.0)
    # print('Center is {} when sample_rate is {}'.format(sc_center, sr))
    data_length = len(samples)
    data_width = time_to_sample(num_secs, sr, 1.0)
    half_input_width = int(data_width / 2)
    if sc_center < half_input_width:
        pad_width = half_input_width - sc_center
        samples = numpy.pad(samples, [(pad_width, 0), (0, 0)], mode='constant', constant_values=0)
        sc_center += pad_width
    elif sc_center + half_input_width > data_length:
        pad_width = sc_center + half_input_width - data_length
        samples = numpy.pad(samples, [(0, pad_width), (0, 0)], mode='constant', constant_values=0)
    samples = samples[sc_center - half_input_width: sc_center + half_input_width]
    audio_input = vggish_input.waveform_to_examples(samples, sr)
    print(audio_input.dtype)
    audio_input = audio_input.astype(numpy.float32)
    channel = grpc.insecure_channel(hostport)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    result_counter = _ResultCounter(num_tests, concurrency)
    for _ in range(num_tests):
        request = predict_pb2.PredictRequest()
        request.model_spec.name = 'vgg'
        request.model_spec.signature_name = 'prediction'
        print(audio_input.shape)
        request.inputs['input'].CopyFrom(tf.contrib.util.make_tensor_proto(audio_input, shape=audio_input.shape))
        result_counter.throttle()
        result_future = stub.Predict.future(request, 5.0)
        result_future.add_done_callback(_create_rpc_callback(None, result_counter))
    return result_counter.get_throughput()


def main(_):
    if FLAGS.num_tests > 10000:
        print('num_tests should not be greater than 10k')
        return

    if not FLAGS.server:
        print('please specify server host:port')
        return

    tfs_throughput = inference(FLAGS.server, FLAGS.work_dir, FLAGS.concurrency, FLAGS.num_tests)

    print('\n TFS Thoughput: %s requests/sec' % (tfs_throughput))


if __name__ == '__main__':
    tf.app.run()
