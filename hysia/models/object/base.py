#!/usr/bin/env python
# @Time    : 6/10/18 1:41 PM
# @Author  : Huaizheng Zhang
# @Site    : zhanghuaizheng.info
# @File    : base.py


class BaseDetector(object):
    '''
    This class is a base class of object detection
    '''
    def __init__(self):
        '''
        Init object detector

        :arg

        :return
        '''
        pass

    def load_paprameters(self):
        '''
        To load pre-trained parameters
        :return:
        '''
        pass

    def output_features(self):
        '''
        Recive a layer name, then return the features from this layer
        :return:
        '''
        pass

    def detect(self):
        '''
        output class id; class; confidence bounding box;
        :return:
        '''
        pass