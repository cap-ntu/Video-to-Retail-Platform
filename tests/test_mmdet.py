#!/usr/bin/env python
# @Time    : 16/10/18 11:21 AM
# @Author  : Huaizheng Zhang
# @Site    : zhanghuaizheng.info
# @File    : test_images.py

import _init_paths
import cv2
from models.object.pt_detector import PT_MMdetector
from mmdet.apis import show_result

cfg_path = '../third/mmdet/configs/faster_rcnn_features.py'
model_path = '../weights/mmdetection/faster_rcnn_r50_fpn_1x_20181010-3d1b3351.pth'


model = PT_MMdetector(model_path, 'cuda:0', cfg_path)

img = cv2.imread('test1.jpg')
result = model.detect(img)


show_result(img, result, score_thr=0.8, out_file='mmdetect_result.jpg')

# test a list of images
# imgs = ['test1.jpg', 'test2.jpg']
# for i, result in enumerate(inference_detector(model, imgs, cfg, device='cuda:0')):
#     print(i, imgs[i])
#     show_result(imgs[i], result)