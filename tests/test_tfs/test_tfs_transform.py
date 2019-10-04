import _init_paths
import tensorflow as tf
import soundfile as sf
import numpy as np
from baselines.TFS.utils.transform_pb_to_server_model import *
import audioset.vggish_params as vggish_params
import audioset.vggish_input as vggish_input

input_tensor_name = vggish_params.INPUT_TENSOR_NAME
output_tensor_name = vggish_params.OUTPUT_TENSOR_NAME

PATH_TO_TEST_AUDIO = 'test_DB/audios/BBT0624.wav'
PATH_TO_VGG_GRAPH = '../weights/audioset/vggish_fr.pb'


def time_to_sample(self, t, sr, factor):
    return round(sr * t / factor)

transform(PATH_TO_VGG_GRAPH, input_tensor_name, output_tensor_name)

# Scene start&end timestamp in milliseconds
# sc_start = 0
# sc_end = 2000
# num_secs = 1
# vgg_graph = tf.Graph()
# with vgg_graph.as_default():
#     od_graph_def = tf.GraphDef()
#     with tf.gfile.GFile(PATH_TO_VGG_GRAPH, 'rb') as fid:
#         serialized_graph = fid.read()
#         od_graph_def.ParseFromString(serialized_graph)
#         tf.import_graph_def(od_graph_def, name='')
#
# wav_data, sr = sf.read(PATH_TO_TEST_AUDIO, dtyppwe='int16')
# assert wav_data.dtype == np.int16, 'Bad sample type: %r' % wav_data.dtype
# samples = wav_data / 32768.0  # Convert to [-1.0, +1.0]
#
# sc_center = time_to_sample((sc_start + sc_end) / 2, sr, 1000.0)
# # print('Center is {} when sample_rate is {}'.format(sc_center, sr))
# data_length = len(samples)
# data_width = time_to_sample(num_secs, sr, 1.0)
# half_input_width = int(data_width / 2)
# if sc_center < half_input_width:
#     pad_width = half_input_width - sc_center
#     samples = np.pad(samples, [(pad_width, 0), (0, 0)], mode='constant', constant_values=0)
#     sc_center += pad_width
# elif sc_center + half_input_width > data_length:
#     pad_width = sc_center + half_input_width - data_length
#     samples = np.pad(samples, [(0, pad_width), (0, 0)], mode='constant', constant_values=0)
# samples = samples[sc_center - half_input_width: sc_center + half_input_width]
# input_batch = vggish_input.waveform_to_examples(samples, sr)
