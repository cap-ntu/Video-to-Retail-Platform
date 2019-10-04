#Author  : wang yongjie
#Email   : yongjie.wang@ntu.edu.sg

import _init_paths
import cv2
import numpy as np
from skimage import transform as trans

from models.face.mtcnn.detector import mtcnn_detector
from models.face.alignment.alignment import transform

if __name__ == "__main__":

    model = '../weights/mtcnn/mtcnn.pb'

    threshold = [0.6, 0.7, 0.9]
    factor = 0.7
    minisize = 20
    test = mtcnn_detector(model, threshold, minisize, factor)
    img = cv2.imread("./test3.jpg")
    cols, rows, channel = img.shape
    rectangles, points, duration = test.detect(img)
    wrapped = transform(img, points, 96, 112)
    cv2.imwrite("wrapped.jpg", wrapped)

