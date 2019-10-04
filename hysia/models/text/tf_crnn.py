
import numpy as np
import tensorflow as tf
import os.path as ops
import cv2
from crnn.global_configuration.config import cfg
from crnn.utils import data_utils

class TF_CRNN:
    '''
    This is for detecting text in the image.
    '''
    def __init__(self, graph):
        '''

        :param graph:
        '''
        with graph.as_default():
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
            self.input_data = self.sess.graph.get_tensor_by_name('Placeholder:0')
            # self.output_ctc = self.sess.graph.get_tensor_by_name('CTCBeamSearchDecoder:0')
            self.output_rnn = self.sess.graph.get_tensor_by_name('shadow/LSTMLayers/transpose_time_major:0')
            print('===============Finish loading CRNN tensorflow version================')

    def resize(self, image):
        w = image.shape[1]
        h = image.shape[0]
        scale = h / 32
        w = int(w / scale)
        # if w / 100 > h / 32:
        #     total = int(w * 32 / 100) - h
        #     image = np.pad(image, [(pad_h, total - pad_h), (0, 0), (0, 0)], mode='constant', constant_values=0)
        # elif w / 100 < h / 32:
        #     total = int(h * 100 / 32) - w
        #     pad_w = int(total / 2)
        #     image = np.pad(image, [(0, 0), (pad_w, total - pad_w), (0, 0)], mode='constant', constant_values=0)
        image = cv2.resize(image, (w, 32))
        image = np.expand_dims(image, axis=0).astype(np.float32)
        return image, w

    def crop_image(self, image, bboxes):
        input_shape = list(cfg.TEST.DATA_SHAPE)
        input_shape.insert(0, len(bboxes))
        input_img = np.empty(input_shape, dtype=np.float32)
        w = image.shape[1]
        h = image.shape[0]

        for i, bbox in enumerate(bboxes):
            [min_y, min_x, max_y, max_x] = bbox
            min_x = int(min_x * w)
            max_x = int(max_x * w)
            min_y = int(min_y * h)
            max_y = int(max_y * h)
            temp = self.resize(image[min_y:max_y, min_x:max_x])
            input_img[i] = np.asarray(temp, dtype=np.float32)
        return input_img, len(bboxes)

    def detect(self, img, bboxes):
        '''

        :param img:
        :return: final result:
        '''
        # input_image, bbox_count = self.crop_image(img, bboxes)

        w = img.shape[1]
        h = img.shape[0]

        result = []
        cropped = []
        for i, bbox in enumerate(bboxes):
            [min_y, min_x, max_y, max_x] = bbox
            min_x = int(min_x * w)
            max_x = int(max_x * w)
            min_y = int(min_y * h)
            max_y = int(max_y * h)

            temp, new_w = self.resize(img[min_y:max_y, min_x:max_x])

            codec = data_utils.TextFeatureIO(char_dict_path=ops.join(cfg.PATH.CHAR_DICT_DIR, 'char_dict.json'),
                                             ord_map_dict_path=ops.join(cfg.PATH.CHAR_DICT_DIR, 'ord_map.json'))
            decodes, _ = tf.nn.ctc_beam_search_decoder(inputs=self.output_rnn, sequence_length= (new_w / 4) * np.ones(1),
                                                   merge_repeated=False)
            preds = self.sess.run(decodes, feed_dict={self.input_data: temp})

            preds = codec.writer.sparse_tensor_to_str(preds[0])
            result.append(preds[0])
            cropped.append(temp[0])
        return result, cropped