# Author: Wang Yongjie
# Email:  yongjie.wang@ntu.edu.sg

import torch
import cv2
import numpy as np
from models.face_pytorch.mtcnn.detector import mtcnn_detector
from models.face_pytorch.alignment.alignment import transform
from models.face_pytorch.recognition.model_irse import *
import pickle
import scipy.spatial.distance


class recog(object):
    def __init__(self, pth_file, saved_feature):
        """
        """
        self.pth_file = pth_file
        self.saved_feature = saved_feature

    def init_pytorch_env(self):
        # load backbone from a checkpoint
        self.backbone = IR_50([112, 112])
        self.backbone.load_state_dict(torch.load(self.pth_file))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.backbone.to(self.device)
        # extract features
        self.backbone.eval() # set to evaluation mode

    def init_mtcnn_detector(self, pnet, rnet, onet, minisize):
        '''
        Parameters:
            pnet:   mtcnn pnet model
            rnet:   mtcnn rnet model
            onet:   mtcnn onet model
            minisize: the minimal size of face

        Returns:
            An mtcnn detector satisfying the specified requirements
        '''
        self.detector = mtcnn_detector(pnet, rnet, onet, minisize)

    def load_feature(self):
        """ load_feature(self, ):
        load pre-extracted face feature
        """
        f = open(self.saved_feature, 'rb')
        self.save_features = pickle.load(f)
        self.single_feature = []
        for i in range(len(self.save_features)):
            self.single_feature.append(self.save_features[i]['FEATURE'])

    def get_indentity(self, image, role=False):
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
        rectangles, points = self.detector.detect(image)

        name_lists = []
        features = []

        if len(rectangles) == 0:
            # print("no face detected")
            return rectangles, name_lists

        for i in range(len(rectangles)):
            point = points[i, :]
            wrapped = transform(image, point, 112, 112)
            if isinstance(wrapped, type(None)):
                print("subzone is empty")
                continue
            img = wrapped[...,::-1] # BGR to RGB
            img = img.swapaxes(1, 2).swapaxes(0, 1)
            img = np.reshape(img, [1, 3, 112, 112])
            img = np.array(img, dtype = np.float32)
            img = (img - 127.5) / 128
            img = torch.from_numpy(img)

            feature = self.backbone(img.to(self.device))
            feature = feature.cpu().detach().numpy()
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

        return rectangles, name_lists, features
