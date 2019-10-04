import math

import numpy as np
import tensorflow as tf

from .audio.audio_util import load_audio, preprocess
from .audio.dataset_util import get_dataconfig


class SoundNetClassifier(object):
    INPUT_TENSOR_NAME = 'Placeholder:0'
    OUTPUT_TENSOR_NAME = 'SoundNet/retrain4/dense/BiasAdd:0'

    def __init__(self, graph, database='dcase2018'):
        with graph.as_default():
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
            self.database = database
            self.num_secs = 5
            self.sr = 22050
            self.config = {
                'load_size': 5 * self.sr,
                'batch_size': 64,
                'sample_size': 5 * self.sr,
                'phase': 'test'
            }
            self.features_tensor = graph.get_tensor_by_name(
                self.INPUT_TENSOR_NAME)
            self.classify_tensor = graph.get_tensor_by_name(
                self.OUTPUT_TENSOR_NAME)

    def time_to_sample(self, t, sr, factor):
        return round(sr * t / factor)

    def classify_frame(self, audio_path, fr):
        """
        Predict scene labels of input audio stream
        If you wang to use this function, don't pass 'param_G' parameter during initialization
        :parameter:
            audio_path: path to audio stream from uploaded video
            fr: frame rate of the video

        :return:
            {
                labels: ['bus', 'train_station', ...] # list of labels linked with number
                confidences:[
                                [0.1, 0.1, 0.1, ... , 0.1] # Confidence of of a snippet consists of several frames
                                ...
                                [0.1, 0.1, 0.1, ... , 0.1]
                            ]
            }
        """
        input_sample, _ = load_audio(audio_path, self.sr, mono=True)
        return self.classify_frame_with_data(input_sample, fr)

    def classify_frame_with_data(self, input_sample, fr):
        sound_sample = preprocess(input_sample, self.config, is_train=False)
        hop = int(math.floor(self.sr / fr))
        data_len = sound_sample.shape[1]
        sample_size = int(self.config['sample_size'])
        frame_count = int(math.ceil((data_len - sample_size) / hop))
        input = np.empty([frame_count, sample_size, 1, 1])
        count = 0
        values = np.empty([frame_count, 5])
        indices = np.empty([frame_count, 5], dtype=np.int32)
        batch_size = self.config['batch_size']

        for j in range(0, data_len - sample_size, hop):
            input[count] = sound_sample[:, j:j + sample_size, :, :]
            count += 1

        output_tensor = tf.nn.top_k(tf.nn.softmax(self.classify_tensor),
                                    k=5, sorted=True)
        for j in range(0, frame_count, batch_size):
            # print("Running:{}".format(j))
            batch_input = input[j: (j + batch_size)]
            values[j:j + batch_size], indices[j:j + batch_size] \
                = self.sess.run(output_tensor, feed_dict={self.features_tensor: batch_input})

        labels = np.array(get_dataconfig(self.database)['labels'])
        values = values.tolist()
        labels = labels[indices].tolist()

        output = []
        for i in range(len(values)):
            item = {}
            item['labels'] = labels[i]
            item['scores'] = values[i]
            output.append(item)
        return output

    def classify_scene(self, audio_path, sc_start, sc_end):
        """

        """
        wav_data, sr = load_audio(audio_path, sr=self.sr, mono=True)
        return self.classify_scene_with_data(wav_data, sr, sc_start, sc_end)

    def classify_scene_with_data(self, wav_data, sr, sc_start, sc_end):
        samples = preprocess(wav_data, config=self.config, is_train=False)
        sc_center = self.time_to_sample((sc_start + sc_end) / 2, sr, 1000.0)
        # print('Center is {} when sample_rate is {}'.format(sc_center, sr))
        data_length = len(samples)
        data_width = self.time_to_sample(self.num_secs, sr, 1.0)
        half_input_width = int(data_width / 2)
        if sc_center < half_input_width:
            pad_width = int(half_input_width - sc_center)
            samples = np.pad(samples, [(0, 0), (pad_width, 0), (0, 0), (0, 0)], mode='constant', constant_values=0)
            sc_center += pad_width
        elif sc_center + half_input_width > data_length:
            pad_width = int(sc_center + half_input_width - data_length)
            samples = np.pad(samples, [(0, 0), (0, pad_width), (0, 0), (0, 0)], mode='constant', constant_values=0)
        samples = samples[:, int(sc_center - half_input_width): int(sc_center + half_input_width)]
        return self.classify(samples)

    def classify(self, audio_data):
        return self.sess.run(tf.argmax(self.classify_tensor, axis=1), feed_dict={self.features_tensor: audio_data})
