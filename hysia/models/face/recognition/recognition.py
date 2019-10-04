# Author: Wang Yongjie
# Email:  yongjie.wang@ntu.edu.sg

import tensorflow as tf
import cv2
import numpy as np
from models.face.mtcnn.detector import mtcnn_detector
from models.face.alignment.alignment import transform
import pickle
import scipy.spatial.distance


class recog(object):
    def __init__(self, pb_file, saved_feature):
        """
        """
        self.pb_file = pb_file
        self.saved_feature = saved_feature

    def init_tf_env(self):
        config = tf.ConfigProto()
        config.allow_soft_placement = True
        config.gpu_options.allow_growth = True
        with tf.gfile.GFile(self.pb_file, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name='')

        self.sess = tf.Session(config=config, graph=graph)
        self.input_image = graph.get_tensor_by_name("img_inputs:0")
        self.phase_train_placeholder = graph.get_tensor_by_name("dropout_rate:0")
        self.embeddings = graph.get_tensor_by_name("resnet_v1_50/E_BN2/Identity:0")

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

    def get_indentity(self, image, role=False, feat=False):
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

        name_lists = []
        features = []

        if len(rectangles) == 0:
            # print("no face detected")
            return rectangles, name_lists

        for i in range(len(rectangles)):
            point = points[:, i]
            wrapped = transform(image, point, 112, 112)
            if isinstance(wrapped, type(None)):
                print("subzone is empty")
                continue
            wrapped = (wrapped - 127.5) / 128
            img = np.expand_dims(wrapped, axis=0)
            feature = self.sess.run(self.embeddings, feed_dict={self.input_image: img, self.phase_train_placeholder: 1})
            features.append(feature)
            # feature = feature[0].reshape(1, -1)
            distances = scipy.spatial.distance.cdist(self.single_feature, feature, 'cosine').reshape(-1)
            topK = np.argsort(distances)
            max_distance = distances[topK[0]]

            # TODO calculate the cosin similarity. 
            if max_distance < 0.6869 and self.save_features[topK[0]]['ROLE'] == self.save_features[topK[1]]['ROLE']:
                if role:
                    name_lists.append(self.save_features[topK[0]]['ROLE'])
                else:
                    name_lists.append(self.save_features[topK[0]]['CELEBRITY'])
            else:
                name_lists.append("unknown")

        if feat is True:
            return rectangles, name_lists, features

        else:

            return rectangles, name_lists
