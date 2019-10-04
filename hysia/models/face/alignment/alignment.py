#Author: wang yongjie
#Email : yongjie.wang@ntu.edu.sg

import cv2
import numpy as np
from skimage import transform as trans

#The referenced key points are taken from known points in sphereface(https://github.com/wy1iu/sphereface/blob/master/preprocess/code/face_align_demo.m#L22)

#https://zhuanlan.zhihu.com/p/29515986

def transform(img, landmarks, width = 96, height = 112):
    '''
    transform(img):
        apply the affine transformation to align the facial key points

    Parameters: 
        img: the face area after detection, array_like
        landmarks:  the landmarks(eyes, nose, mouth), aray_like

    Returns:
        return the image after alignment
    '''

    dst = np.array([
        [30.2946, 51.6963],
        [65.5318, 51.5014],
        [48.0252, 71.7366],
        [33.5493, 92.3655],
        [62.7299, 92.2041]], dtype = np.float32)

    tform = trans.SimilarityTransform()
    tform.estimate(np.array(landmarks).reshape(5,2), dst)
    M = tform.params[0:2, :]
    wrapped = cv2.warpAffine(img, M, (width, height), borderValue = 0.0)

    return wrapped

