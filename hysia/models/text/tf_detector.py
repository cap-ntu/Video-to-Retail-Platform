# @Time    : 24/10/18 6:03 PM
# @Author  : Huaizheng Zhang
# @Site    : zhanghuaizheng.info
# @File    : tf_detector.py

import numpy as np
import tensorflow as tf
import cv2
from ctpn.lib.fast_rcnn.config import cfg, cfg_from_file
from ctpn.lib.fast_rcnn.test import _get_blobs
from ctpn.lib.text_connector.detectors import TextDetector
from ctpn.lib.text_connector.text_connect_cfg import Config as TextLineCfg
from ctpn.lib.rpn_msr.proposal_layer_tf import proposal_layer


class TF_CTPN:
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
            self.input_img = self.sess.graph.get_tensor_by_name('Placeholder:0')
            self.output_cls_prob = self.sess.graph.get_tensor_by_name('Reshape_2:0')
            self.output_box_pred = self.sess.graph.get_tensor_by_name('rpn_bbox_pred/Reshape_1:0')
            print('===============Finish loading CTPN tensorflow version================')

    def resize_im(self, im, scale, max_scale=None):
        '''

        :param im:
        :param scale:
        :param max_scale:
        :return: resize image:
        '''
        f = float(scale) / min(im.shape[0], im.shape[1])
        if max_scale != None and f * max(im.shape[0], im.shape[1]) > max_scale:
            f = float(max_scale) / max(im.shape[0], im.shape[1])
        return cv2.resize(im, None, None, fx=f, fy=f, interpolation=cv2.INTER_LINEAR), f

    def draw_boxes(self, img, boxes, scale):
        '''

        :param img:
        :param boxes:
        :param scale:
        :return: dict results:
        '''
        detections = {"text_bboxes": []}
        for box in boxes:
            if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
                continue
            if box[8] >= 0.9:
                color = (0, 255, 0)
            elif box[8] >= 0.8:
                color = (255, 0, 0)
            cv2.line(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
            cv2.line(img, (int(box[0]), int(box[1])), (int(box[4]), int(box[5])), color, 2)
            cv2.line(img, (int(box[6]), int(box[7])), (int(box[2]), int(box[3])), color, 2)
            cv2.line(img, (int(box[4]), int(box[5])), (int(box[6]), int(box[7])), color, 2)

            min_x = min(int(box[0] / scale), int(box[2] / scale), int(box[4] / scale), int(box[6] / scale)) / (
                    img.shape[1] / scale)
            min_y = min(int(box[1] / scale), int(box[3] / scale), int(box[5] / scale), int(box[7] / scale)) / (
                    img.shape[0] / scale)
            max_x = max(int(box[0] / scale), int(box[2] / scale), int(box[4] / scale), int(box[6] / scale)) / (
                    img.shape[1] / scale)
            max_y = max(int(box[1] / scale), int(box[3] / scale), int(box[5] / scale), int(box[7] / scale)) / (
                    img.shape[0] / scale)
            detections["text_bboxes"].append([min_y, min_x, max_y, max_x])
        return detections

    def detect(self, img):
        '''

        :param img:
        :return: final result:
        '''
        img, scale = self.resize_im(img, scale=TextLineCfg.SCALE, max_scale=TextLineCfg.MAX_SCALE)
        blobs, im_scales = _get_blobs(img, None)
        if cfg.TEST.HAS_RPN:
            im_blob = blobs['data']
            blobs['im_info'] = np.array(
                [[im_blob.shape[1], im_blob.shape[2], im_scales[0]]],
                dtype=np.float32)
        cls_prob, box_pred = self.sess.run([self.output_cls_prob, self.output_box_pred],
                                           feed_dict={self.input_img: blobs['data']})
        rois, _ = proposal_layer(cls_prob, box_pred, blobs['im_info'], 'TEST', anchor_scales=cfg.ANCHOR_SCALES)

        scores = rois[:, 0]
        boxes = rois[:, 1:5] / im_scales[0]
        textdetector = TextDetector()
        boxes = textdetector.detect(boxes, scores[:, np.newaxis], img.shape[:2])
        return self.draw_boxes(img, boxes, scale)