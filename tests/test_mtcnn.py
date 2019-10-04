#!/usr/bin/env python
# @Author  : wang yongjie
# @Site    : yongjie.wang@ntu.edu.sg
# @File    : test_mtcnn.py


import _init_paths
import sys
import cv2

from models.face.mtcnn.detector import mtcnn_detector

if __name__ == "__main__":

    print(sys.path)
    model = '../weights/mtcnn/mtcnn.pb'

    threshold = [0.6, 0.7, 0.9]
    factor = 0.7
    minisize = 20
    test = mtcnn_detector(model, threshold, minisize, factor)
    img = cv2.imread("./test1.jpg")
    rectangles, points, duration = test.detect(img)
    for rec in rectangles:
        cv2.rectangle(img, (int(rec[0]), int(rec[1])), (int(rec[2]), int(rec[3])), (0, 0, 255), 1)
    cv2.imwrite('result.jpg', img)
