import numpy as np
import tensorflow as tf
import soundfile as sf
import audioset.vggish_params as vggish_params
import audioset.vggish_input as vggish_input
import audioset.vggish_postprocess as vggish_postprocess


class AudiosetFeatureExtractor:

    def __init__(self, graph, pca_path, num_secs=1):
        with graph.as_default():
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
            self.pca_path = pca_path
            self.num_secs = num_secs
            self.features_tensor = graph.get_tensor_by_name(
                vggish_params.INPUT_TENSOR_NAME)
            self.embedding_tensor = graph.get_tensor_by_name(
                vggish_params.OUTPUT_TENSOR_NAME)

    def time_to_sample(self, t, sr, factor):
        return round(sr * t / factor)

    def extract(self, audio_path, sc_start, sc_end):

        wav_data, sr = sf.read(audio_path, dtype='int16')
        assert wav_data.dtype == np.int16, 'Bad sample type: %r' % wav_data.dtype
        samples = wav_data / 32768.0  # Convert to [-1.0, +1.0]

        sc_center = self.time_to_sample((sc_start + sc_end) / 2, sr, 1000.0)
        # print('Center is {} when sample_rate is {}'.format(sc_center, sr))
        data_length = len(samples)
        data_width = self.time_to_sample(self.num_secs, sr, 1.0)
        half_input_width = int(data_width / 2)
        if sc_center < half_input_width:
            pad_width = half_input_width - sc_center
            samples = np.pad(samples, [(pad_width, 0), (0, 0)], mode='constant', constant_values=0)
            sc_center += pad_width
        elif sc_center + half_input_width > data_length:
            pad_width = sc_center + half_input_width - data_length
            samples = np.pad(samples, [(0, pad_width), (0, 0)], mode='constant', constant_values=0)
        samples = samples[sc_center - half_input_width: sc_center + half_input_width]
        input_batch = vggish_input.waveform_to_examples(samples, sr)
        [embedding_batch] = self.sess.run([self.embedding_tensor], feed_dict={self.features_tensor: input_batch})

        pproc = vggish_postprocess.Postprocessor(self.pca_path)
        postprocessed_batch = pproc.postprocess(embedding_batch)
        return postprocessed_batch
