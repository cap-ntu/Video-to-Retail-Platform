# @Time    : 24/10/18 4:46 PM
# @Author  : Huaizheng Zhang
# @Site    : zhanghuaizheng.info
# @File    : tf_detector.py
from collections import defaultdict
from itertools import islice
from operator import itemgetter

import cv2
import numpy as np
import tensorflow as tf

from third.object_detection.utils import label_map_util
from third.object_detection.utils import ops as utils_ops


class TF_SSD(object):
    """
    This is a Dectector comes from tensorflow object detector module.

    https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb
    """

    def __init__(self, graph, label, num_class):
        """
        :param graph: a tensorflow graph:
        :param label: the label for this dataset:
        :param num_class: how many class in this dataset:
        """

        with graph.as_default():
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
            # Predefine image size as required by SSD
            self.image_shape = [365, 640, 3]
            # Predefine confidence threshold
            self.thresh = 0.3
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                        tensor_name)
            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes, self.image_shape[0], self.image_shape[1])
                detection_masks_reframed = tf.cast(
                    tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(
                    detection_masks_reframed, 0)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
            self.image_tensor = image_tensor
            self.tensor_dict = tensor_dict

            label_map = label_map_util.load_labelmap(label)
            categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=num_class,
                                                                        use_display_name=True)
            self.category_index = label_map_util.create_category_index(categories)

            print('===============Finish loading SSD detector================')

    def detect(self, img):
        '''
        :param img: a image tensor:

        :return: a dict containts class name, boxes and confidences:
        '''
        if img.shape != self.image_shape:
            img = cv2.resize(img, (self.image_shape[0], self.image_shape[1]))
        # Run inference
        output_dict = self.sess.run(self.tensor_dict, feed_dict={self.image_tensor: np.expand_dims(img, 0)})
        # All outputs are float32 numpy arrays, so convert types as appropriate
        # Apply threshold on detections
        keep = np.where(output_dict['detection_scores'][0] >= self.thresh)
        output_dict['num_detections'] = keep[0].size
        output_dict['detection_classes'] = output_dict[
            'detection_classes'][0][keep].astype(np.uint8).tolist()
        output_dict['detection_classes_names'] = [self.category_index[cls_id]['name'] for cls_id in
                                                  output_dict['detection_classes']]
        output_dict['detection_boxes'] = (output_dict['detection_boxes'][0][keep]).tolist()
        output_dict['detection_scores'] = (output_dict['detection_scores'][0][keep]).tolist()
        if 'detection_masks' in output_dict:
            output_dict['detection_masks'] = (output_dict['detection_masks'][0]).tolist()
        return output_dict

    def batch_predict(self, tensor):
        """
        :param tensor: a image tensor:

        :return: a dict containts class name, boxes and confidences:
        """
        # Run inference
        output_dict = self.sess.run(self.tensor_dict, feed_dict={self.image_tensor: tensor})
        # All outputs are float32 numpy arrays, so convert types as appropriate
        # Apply threshold on detections
        result = defaultdict(list)

        indices_bool = output_dict['detection_scores'] >= self.thresh
        row_indices, col_indices = np.where(indices_bool)

        result['num_detections'] = np.sum(indices_bool, axis=1).tolist()
        detection_classes = output_dict['detection_classes'][row_indices, col_indices].astype(np.uint8).tolist()
        detection_boxes = output_dict['detection_boxes'][row_indices, col_indices, :].tolist()
        detection_scores = output_dict['detection_scores'][row_indices, col_indices].tolist()
        detection_classes_names = list(itemgetter(*detection_classes)(self.category_index))

        if 'detection_masks' in output_dict:
            result['detection_masks'] = output_dict['detection_masks'][row_indices, col_indices, :].tolist()

        # obtain all result in a sequence
        for length in result['num_detections']:
            result['detection_classes'].append(list(islice(detection_classes, length)))
            result['detection_boxes'].append(list(islice(detection_boxes, length)))
            result['detection_scores'].append(list(islice(detection_scores, length)))
            result['detection_classes_names'].append(list(islice(detection_classes_names, length)))

        return dict(result)
