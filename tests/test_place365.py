#!/usr/bin/env python
# @Time    : 19/10/18 4:22 PM
# @Author  : Huaizheng Zhang
# @Site    : zhanghuaizheng.info
# @File    : detector.py


import _init_paths
from models.scene.detector import scene_visual
import cv2
from PIL import Image
import numpy as np


if __name__ == '__main__':
    scene_model = scene_visual('resnet50', '../weights/places365/{}.pth', '../weights/places365/categories.txt', 'cuda:0')
    for i in ['test1.jpg', 'test2.jpg']:
        temp = scene_model.detect(i)
        print(temp)

    with open('test1.jpg', 'rb') as f:
        img = Image.open(f)
        img.convert('RGB')
    temp = scene_model.detect(img, True)
    print(temp)

    # Test vector extraction and cosine similarity
    with open('test_sofa1.jpg', 'rb') as f:
        img = Image.open(f)
        img.convert('RGB')
    q_vec = scene_model.extract_vec(img, True)
    print(type(q_vec))


    # temp = cv2.imread('test_beach.jpg')
    # search_tensor = Image.fromarray(cv2.cvtColor(temp, cv2.COLOR_BGR2RGB))
    with open('test1.jpg', 'rb') as f:
        img = Image.open(f)
        img.convert('RGB')
    search_vec = scene_model.extract_vec(img, True)

    scores = np.dot(search_vec.T, q_vec)
    print(scores)

