# Author: wang yongjie
# Email:  yongjie.wang@ntu.edu.sg

import tensorflow as tf
import os
import cv2
import numpy as np
from models.face.mtcnn.detector import mtcnn_detector
from models.face.alignment.alignment import transform

def crop_align(input_dir, output_dir, model, threshold, minisize, factor):
    '''
    crop_align(datasets, model, threshold, minisize, factor):
    parameters:
        input_dir:  input directory of raw images
        output_dir: output directory of cropped images
        model:      mtcnn model file
        threshold:  threshold for pnet, onet, rnet
        minisize:   the minisize of face to detect
        factor: 

    returns:
        return the datasets after crop and align

    '''

    detector = mtcnn_detector(model, threshold, minisize, factor)

    for item in os.listdir(input_dir):
        subdir = os.path.join(input_dir, item)
        out_sub_dir = os.path.join(output_dir, item)
        cnt = 0
        if not os.path.exists(out_sub_dir):
            os.mkdir(out_sub_dir)

        for subitem in os.listdir(subdir):
            image = os.path.join(subdir, subitem)
            tmp = '%06d.jpg'%cnt
            cnt = cnt + 1
            save_image_name = os.path.join(out_sub_dir, tmp)
            print("processing %s  saved name %s"%(image, save_image_name))

            img = cv2.imread(image)
            if isinstance(img, type(None)):
                print(image + "not exist")
                continue

            rectangles, points, _ = detector.detect(img)
            if len(rectangles) == 0:
                continue
            scores = rectangles[:, 4]
            s_sort = np.argsort(scores)
            location = s_sort[-1]
            point = points[:, location] # choose the highest score face
            wrapped = transform(img, point, 112, 112)
            cv2.imwrite(save_image_name, wrapped)
