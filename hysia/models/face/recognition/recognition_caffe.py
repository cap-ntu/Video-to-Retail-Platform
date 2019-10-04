# Author: Wang Yongjie
# Email:  yongjie.wang@ntu.edu.sg

import os
os.environ['GLOG_minloglevel'] = '2'
import caffe
import cv2
import copy
import numpy as np
from models.face.mtcnn.detector import mtcnn_detector
from models.face.alignment.alignment import transform
import pickle
import scipy.spatial.distance

class recog(object):
    def __init__(self, prototxt, model, saved_feature):
        """
        recog.__init__(self, prototxt, model, saved_feature):
        initialize face recognition class:

        Parameters:
            prototxt: caffe prototxt file
            model   : caffe pretrained model
            saved_feature : the feature datasets to be recongized

        Returns:
            A face recognition class

        """
        self.prototxt = prototxt
        self.model = model
        self.saved_feature = saved_feature


    def init_caffe_env(self):
        """
        initialize  caffe environment for face recognition
        """
        self.Net = caffe.Net(self.prototxt, self.model, caffe.TEST)
        caffe.set_mode_gpu()

    def init_mtcnn_detector(self, models, threshold, minisize, factor):
        '''
        Parameters:
            m_path: mtcnn model path
            threshold: threshold for pnet, rnet, onet, list 
            minisize: the minimal size of face
            factor: shrink scales of next pyramid image

        Returns:
            An mtcnn detector satisfying the specified requirements
        '''
        self.detector = mtcnn_detector(models, threshold, minisize, factor)

    def load_feature(self):
        """ load_feature(self, ):
        load pre-extracted face feature
        """
        f = open(self.saved_feature, 'rb')
        self.save_features = pickle.load(f)
        self.single_feature = []
        for i in range(len(self.save_features)):
            self.single_feature.append(self.save_features[i]['FEATURE'])


    def get_indentity(self, image, role = False):
        """
        recog.get_indentity(img):
        obtain the face locations and indentities in input image

        Parameters:
            img: input image loaded by OpenCV python interface

        Returns:
            boundingboxes: face loacations
            name_lists   : identities 
        """

        if isinstance(image, type(None)):
            print("image is empty")
            return 
        rectangles, points, _ = self.detector.detect(image)
        print(rectangles)
        print(points)

        name_lists = []

        if len(rectangles) == 0:
            print("no face detected")
            return rectangles, name_lists

        for i in range(len(rectangles)):
            point = points[:, i]
            wrapped = transform(image, point, 96, 112)
            if isinstance(wrapped, type(None)):
                print("subzone is empty")
                continue
            wrapped = cv2.cvtColor(wrapped, cv2.COLOR_BGR2RGB)
            wrapped = (wrapped - 127.5) / 128
            wrapped = np.transpose(wrapped, (2, 0, 1))
            img = np.expand_dims(wrapped, axis = 0)
            self.Net.blobs['data'].data[...] = img
            self.Net.forward()
            feature = copy.copy(self.Net.blobs['fc5'].data[0])
            feature = feature.reshape(1, -1)
            distances = scipy.spatial.distance.cdist(self.single_feature, feature, 'cosine').reshape(-1)
            topK = np.argsort(distances)
            max_distance = distances[topK[0]]

            # TODO calculate the cosin similarity.
            if max_distance < 0.58 and self.save_features[topK[0]]['ROLE'] == self.save_features[topK[0]]['ROLE']:
                if role:
                    name_lists.append(self.save_features[topK[0]]['ROLE'])
                else:
                    name_lists.append(self.save_features[topK[0]]['CELEBRITY'])
            else:
                name_lists.append("unknown")


        return rectangles, name_lists

