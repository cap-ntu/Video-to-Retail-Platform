# @Time    : 22/11/18 5:33 PM
# @Author  : Huaizheng Zhang
# @Site    : zhanghuaizheng.info
# @File    : product_search.py

import os
import time

import cv2
from PIL import Image

from models.scene.detector import scene_visual
from .index import BasicIndex
from .search import BasicSearch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


class ProductSearch(BasicSearch):
    '''
    For product image retrieval
    '''
    def __init__(self, database="/data/disk2/hysia_data/Stanford_Online_Products/"):

        self.database = database
        self.scene_model = scene_visual('resnet50',
                                        os.path.join(THIS_DIR, '../../weights/places365/{}.pth'),
                                        os.path.join(THIS_DIR, '../../weights/places365/categories.txt'),
                                        'cuda:0')
        self.image_machine = BasicIndex.init_size(self.database)
        self.image_machine.index()

        self.scene_query = None
        self.object_querys = None
        self.audio_querys = None

    def search(self, timestamp, video_path, k=5):
        '''

        :param timestamp: ms
        :param video_path: video absolute path
        :param k: default is 5
        :return: top 5 results
        '''
        tic = time.time()
        self.__get_query_image(timestamp, video_path)
        q_vec = self.scene_model.extract_vec(self.scene_query, True)
        print(q_vec.shape)

        results, _, _ = self.image_machine.search(q_vec, k)
        toc = time.time()

        print("Searching image takes " + str((toc - tic)) + " ms")
        return results



    # TODO combine to videos which has been processed
    # This is just for test
    def __get_query_image(self, timestamp, video_path, save_image=False):

        vidcap = cv2.VideoCapture(video_path)
        vidcap.set(cv2.CAP_PROP_POS_MSEC, timestamp)
        success, image = vidcap.read()
        if not save_image:
            if success:
                # TODO: the format from array
                self.scene_query = Image.fromarray(image)
        else:
            video_name = os.path.splitext(video_path)[0]
            self.scene_query = "{}_{}.jpg".format(video_name, timestamp)
            if success:
                cv2.imwrite(self.scene_query, image)

            temp = cv2.imread(self.scene_query)
            # temp = cv2.imread('test_beach.jpg')
            self.scene_query = Image.fromarray(cv2.cvtColor(temp, cv2.COLOR_BGR2RGB))
