# @Time    : 22/11/18 3:28 PM
# @Author  : Huaizheng Zhang
# @Site    : zhanghuaizheng.info
# @File    : test_imagenet.py

import _init_paths
from models.object.pretrained_imagenet import Img2Vec
import cv2
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


if __name__ == '__main__':
    img = Image.open('test_sofa1.jpg')
    model = Img2Vec(cuda=True)
    q_vec = model.extract_vec(img, True)


    temp = Image.open('test_beach.jpg')
    search_vec = model.extract_vec(temp, True)

    scores = np.dot(search_vec.T, q_vec)
    print(scores)

    cos = cosine_similarity(np.reshape(q_vec, (1,-1)), np.reshape(search_vec,(1, -1)))
    print(cos)