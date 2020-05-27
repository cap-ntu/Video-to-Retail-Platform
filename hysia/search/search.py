# @Time    : 15/11/18 8:26 PM
# @Author  : Huaizheng Zhang
# @Site    : zhanghuaizheng.info
# @File    : search.py


import os
import time

import cv2
import numpy as np
from PIL import Image

from models.nlp.sentence import TF_Sentence
from models.scene.detector import scene_visual
from search.index import BasicIndex, SubtitleIndex

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


class BasicSearch(object):
    '''
    This is the product-content match module.
    '''

    def __init__(self, database):
        '''

        :param database: the directory path
        '''
        try:
            self.database = database
            self.image_machine = BasicIndex.init_size(self.database)
            self.sentence_machine = SubtitleIndex(512, self.database)

            self.image_machine.index()
            print('Finish loading image GPU indexer')

            self.sentence_machine.index()
            print('Finish loading sentence GPU indexer')

        except ValueError:
            print("Please specify the pkl directory!")

        # TODO load scene/imagenet/faster-rcnn model to extract query image feature.
        self.scene_model = scene_visual('resnet50',
                                        os.path.join(THIS_DIR, '../../weights/places365/{}.pth'),
                                        os.path.join(THIS_DIR, '../../weights/places365/categories.txt'),
                                        'cuda:0')

        self.sentence_model = TF_Sentence(
            os.path.join(THIS_DIR, '../../weights/sentence'))

    def search(self, image_query, subtitle_query=None, audio_query=None, face_query=None, topK=30):
        '''

        :param image_query:
        :param subtitle_query:
        :param audio_query:
        :param face_query:
        :return:
        '''

        # search subtitle first.
        if (image_query is not None) and (subtitle_query is not None):
            product_description_vector = self.sentence_model.encode(subtitle_query)
            tic = time.time()
            _, idx, _ = self.sentence_machine.search(np.array(product_description_vector).astype(np.float32), topK)
            toc = time.time()
            print("Searching subtitle takes " + str((toc - tic)) + " ms")

            temp = cv2.imread(image_query)
            q_tensor = Image.fromarray(cv2.cvtColor(temp, cv2.COLOR_BGR2RGB))
            q_vec = self.scene_model.extract_vec(q_tensor, True)
            print(q_vec.shape)

            tic = time.time()
            self.image_machine.re_index(idx)
            results, _, _ = self.image_machine.search(q_vec, topK, 2)
            toc = time.time()
            print("Searching image takes " + str((toc - tic)) + " ms")

        elif subtitle_query is not None:
            product_description_vector = self.sentence_model.encode(subtitle_query)
            tic = time.time()
            results, _, _ = self.sentence_machine.search(np.array(product_description_vector).astype(np.float32), topK)
            toc = time.time()
            print("Searching subtitle takes " + str((toc - tic)) + " ms")

        elif image_query is not None:
            temp = cv2.imread(image_query)
            q_tensor = Image.fromarray(cv2.cvtColor(temp, cv2.COLOR_BGR2RGB))
            q_vec = self.scene_model.extract_vec(q_tensor, True)
            print(q_vec.shape)

            tic = time.time()
            results, _, _ = self.image_machine.search(q_vec, topK)
            toc = time.time()
            print("Searching image takes " + str((toc - tic)) + " ms")
        else:
            results = []

        return results


class DatabasePklSearch(BasicSearch):
    '''
    This is for pkl search and then connect to database to do more operations.
    '''

    def __init__(self, database):
        """
        :param database: the path contains many .pkl file
        """
        try:
            self.database = database

            self.image_machine = BasicIndex(2048, self.database)
            print('Finish loading scene-image GPU indexer')

            self.sentence_machine = BasicIndex(512, self.database)
            print('Finish loading scene-sentence GPU indexer')

        except ValueError:
            print("Please specify the pkl directory!")

        # TODO load scene/imagenet/faster-rcnn model to extract query image feature.
        self.scene_model = scene_visual('resnet50',
                                        os.path.join(THIS_DIR, '../../weights/places365/{}.pth'),
                                        os.path.join(THIS_DIR, '../../weights/places365/categories.txt'),
                                        'cuda:0')

        self.sentence_model = TF_Sentence(os.path.join(THIS_DIR, '../../weights/sentence'))

    def search(self, image_query, subtitle_query=None, audio_query=None, face_query=None, topK=30, tv_name=None):
        '''

        :param image_query:
        :param subtitle_query:
        :param audio_query:
        :param face_query:
        :return:
        '''

        if tv_name is not None:
            self.image_machine.index(tv_name=tv_name)
            self.sentence_machine.index(feature='SUBTITLE_FEATURE', tv_name=tv_name)
        else:
            self.image_machine.index()
            self.sentence_machine.index(feature='SUBTITLE_FEATURE')

        # search subtitle first.
        if (image_query is not None) and (subtitle_query is not None):
            product_description_vector = self.sentence_model.encode(subtitle_query)
            tic = time.time()
            _, idx, _ = self.sentence_machine.search(np.array(product_description_vector).astype(np.float32), topK)
            toc = time.time()
            print("Searching subtitle takes " + str((toc - tic)) + " ms")

            temp = cv2.imread(image_query)
            q_tensor = Image.fromarray(cv2.cvtColor(temp, cv2.COLOR_BGR2RGB))
            q_vec = self.scene_model.extract_vec(q_tensor, True)
            # print(q_vec.shape)

            tic = time.time()
            self.image_machine.re_index(idx)
            results, _, _ = self.image_machine.search(q_vec, topK, 2)
            toc = time.time()
            print("Searching image takes " + str((toc - tic)) + " ms")

        elif subtitle_query is not None:
            product_description_vector = self.sentence_model.encode(subtitle_query)
            tic = time.time()
            results, _, _ = self.sentence_machine.search(np.array(product_description_vector).astype(np.float32), topK)
            toc = time.time()
            print("Searching subtitle takes " + str((toc - tic)) + " ms")

        elif image_query is not None:
            temp = cv2.imread(image_query)
            q_tensor = Image.fromarray(cv2.cvtColor(temp, cv2.COLOR_BGR2RGB))
            q_vec = self.scene_model.extract_vec(q_tensor, True)
            # print(q_vec.shape)

            tic = time.time()
            results, _, _ = self.image_machine.search(q_vec, topK)
            toc = time.time()
            print("Searching image takes " + str((toc - tic)) + " ms")
        else:
            results = []

        # TODO better reset design.
        self.sentence_machine.reset_GPU()
        print("Reset sentence GPU indexer")
        self.image_machine.reset_GPU()
        print("Reset product image GPU indexer")

        return results

    def add_pkl(self):
        '''
        Every time we finish processing one video, we should add a pkl file
        :return:
        '''
        pass
