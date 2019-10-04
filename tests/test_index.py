# @Time    : 14/11/18 3:03 PM
# @Author  : Huaizheng Zhang
# @Site    : zhanghuaizheng.info
# @File    : text_index.py

import _init_paths
from dataset.TVQA.index import TVQA_indexer
from search.index import BasicIndex, SubtitleIndex

from models.scene.detector import scene_visual
from models.nlp.sentence import TF_Sentence
import cv2
from PIL import Image
import numpy as np
import time


if __name__ == '__main__':
    '''
    Index a dir first
    Actually, the videos processed by our system should be saved in a directory (feature, scene, object, face ....)
    Then we can index them and then we can do search
    However, we do not have so many videos to process, 
    so we use TVQA dataset to demonstrate our product-content match module.
    '''
    VIDEO_DATA_PATH = "/data/disk2/hysia_data/UNC_TVQA_DATASET"
    #
    # a = TVQA_indexer(VIDEO_DATA_PATH)
    # a.index()


    # TODO load scene/imagenet/faster-rcnn model to extract query image feature.
    scene_model = scene_visual('resnet50', '../weights/places365/{}.pth', '../weights/places365/categories.txt',
                               'cuda:0')

    temp = cv2.imread('test_sofa2.jpg')
    q_tensor = Image.fromarray(cv2.cvtColor(temp, cv2.COLOR_BGR2RGB))
    q_vec = scene_model.extract_vec(q_tensor, True)
    print(q_vec.shape)

    gpu_machine = BasicIndex.init_size(VIDEO_DATA_PATH)
    gpu_machine.index()
    tic = time.time()
    results = gpu_machine.search(q_vec, 10)
    toc = time.time()
    print("It takes " + str((toc - tic)) + " ms")

    sentence_model = TF_Sentence('../weights/sentence/96e8f1d3d4d90ce86b2db128249eb8143a91db73')
    product_description_vector = sentence_model.encode('Nike sports shirts is very good.')

    sentence_machine = SubtitleIndex(512, VIDEO_DATA_PATH)
    sentence_machine.index()
    tic = time.time()
    results = sentence_machine.search(np.array(product_description_vector).astype(np.float32), 30)
    toc = time.time()
    print("It takes " + str((toc - tic)) + " ms")


    print(results)
    # show results
    for i in results:

        frame = cv2.imread(i['IMAGE'])
        cv2.imshow('show_img', frame)
        cv2.waitKey(0)


