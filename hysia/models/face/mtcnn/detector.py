# Author: Wang Yongjie
# Email : yongjie.wang@ntu.edu.sg


import sys
import argparse
import time

import tensorflow as tf
import cv2
import numpy as np

from mtcnn.tools import detect_face
#import mtcnn.tools.detect_face

class mtcnn_detector():

    def __init__(self, m_path, threshold, minisize, factor):
        """
        mtcnn_detector.__init__(m_path, threshold, minisize, factor):
        create an mtcnn detector

        Parameters:
            m_path: mtcnn model path
            threshold: threshold for pnet, rnet, onet, list
            minisize: the minimal size of face
            factor: shrink scales of next pyramid image

        Returns:
            An mtcnn detector satisfying the specified requirements

        """
        self.m_path = m_path
        self.threshold = threshold
        self.minisize = minisize
        self.factor = factor
        self.load_param()

    def load_param(self):
        """
        mtcnn.load_param()
        load network graph and feed pretrained weight into networks

        Parameters:
            No parameter

        Returns:
            pnet_func: return the function that outputs the last layer of pnet
            rnet_func: return the function that outputs the last layer of onet
            onet_func: return the function that outputs the last layer of rnet

        """
        with tf.gfile.GFile(self.m_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name='')

        config = tf.ConfigProto()
        config.allow_soft_placement = True
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config, graph=graph)
        placeholder = graph.get_tensor_by_name("Placeholder:0")
        softmax = graph.get_tensor_by_name('softmax/Reshape_1:0')
        pnet_conv4_2 = graph.get_tensor_by_name('pnet/conv4-2/BiasAdd:0')
        placeholder_1 = graph.get_tensor_by_name('Placeholder_1:0')
        softmax_1 = graph.get_tensor_by_name('softmax_1/softmax:0')
        rnet_conv5_2 = graph.get_tensor_by_name('rnet/conv5-2/rnet/conv5-2:0')
        softmax_2 = graph.get_tensor_by_name('softmax_2/softmax:0')
        onet_conv6_2 = graph.get_tensor_by_name('onet/conv6-2/onet/conv6-2:0')
        onet_conv6_3 = graph.get_tensor_by_name('onet/conv6-3/onet/conv6-3:0')
        placeholder_2 = graph.get_tensor_by_name('Placeholder_2:0')

        def pnet_fun(img): return self.sess.run(
            (softmax, pnet_conv4_2),
            feed_dict={
                placeholder: img})

        def rnet_fun(img): return self.sess.run(
            (softmax_1, rnet_conv5_2),
            feed_dict={
                placeholder_1: img})

        def onet_fun(img): return self.sess.run(
            (softmax_2, onet_conv6_2, onet_conv6_3),
            feed_dict={placeholder_2: img})

        self.pnet_fun, self.rnet_fun, self.onet_fun = pnet_fun, rnet_fun, onet_fun

    def detect(self, img):
        """
        mtcnn.detect(img):
        detect the bounding boxes of face and features point(nose, eyes, mouth) on face


        Parameters:
            img: input image loaded by opencv python interface

        Returns:
            boundingboxes: bounding boxes of face location
            points:        key points of eyes, nose and mouth

        """
        start_time = time.time()
        boundingboxes, points = detect_face(img, self.minisize, self.pnet_fun, self.rnet_fun, self.onet_fun,
                                            self.threshold, self.factor)
        duration = time.time() - start_time
        return boundingboxes, points, duration
