# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import cv2
import numpy as np


class COCODemoHelper(object):
    """Draw bbox from predictions. Adapted from
    https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/demo/predictor.py.
    Under MIT licence.
    """

    palette = np.array([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])

    @classmethod
    def compute_colors_for_labels(cls, labels):
        """
        Simple function that adds fixed colors depending on the class
        """
        colors = labels[:, None] * cls.palette
        colors = (colors % 255).astype("uint8")
        return colors

    @classmethod
    def overlay_boxes(cls, image, predictions):
        """
        Adds the predicted boxes on top of the image

        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (dict): the result of the computation by the model.
                It should contain the field `labels`.
        """
        labels = np.array(predictions['label_ids'])
        boxes = predictions['boxes']

        colors = cls.compute_colors_for_labels(labels).tolist()

        for box, color in zip(boxes, colors):
            top_left, bottom_right = box[:2], box[2:]
            image = cv2.rectangle(
                image, tuple(top_left), tuple(bottom_right), tuple(color), 1
            )

        return image

    @classmethod
    def overlay_class_names(cls, image, predictions):
        """
        Adds detected class names and scores in the positions defined by the
        top-left corner of the predicted bounding box
        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `scores` and `labels`.
        """
        scores = predictions['scores']
        labels = predictions['labels']
        boxes = predictions['boxes']

        template = '{}: {:.2f}'
        for box, score, label in zip(boxes, scores, labels):
            x, y = box[:2]
            s = template.format(label, score)
            cv2.putText(
                image, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1
            )

        return image

    @classmethod
    def overlay_mask(cls, image, predictions):
        """
        Adds the instances contours for each predicted object.
        Each label has a different color.

        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (dict): the result of the computation by the model.
                It should contain the field `mask` and `labels`.
        """
        masks = np.array(predictions['mask'], dtype=np.bool)
        labels = np.array(predictions['label_ids'])

        colors = cls.compute_colors_for_labels(labels).tolist()

        for mask, color in zip(masks, colors):
            thresh = mask[0, :, :, None].astype(np.uint8)
            contours, hierarchy = find_contours(
                thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
            image = cv2.drawContours(image, contours, -1, color, 3)

        composite = image

        return composite


def find_contours(*args, **kwargs):
    """Copy from https://github.com/facebookresearch/maskrcnn-benchmark/maskrcnn-benchmark/utils/cv2_util.py.
    Under MIT License.

    Wraps cv2.findContours to maintain compatiblity between versions
    3 and 4

    Returns:
        contours, hierarchy
    """
    if cv2.__version__.startswith('4'):
        contours, hierarchy = cv2.findContours(*args, **kwargs)
    elif cv2.__version__.startswith('3'):
        _, contours, hierarchy = cv2.findContours(*args, **kwargs)
    else:
        raise AssertionError(
            'cv2 must be either version 3 or 4 to call this method')

    return contours, hierarchy
