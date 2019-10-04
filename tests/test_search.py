# @Time    : 15/11/18 9:00 PM
# @Author  : Huaizheng Zhang
# @Site    : zhanghuaizheng.info
# @File    : test_search.py


import _init_paths
from search.search import BasicSearch
import cv2
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

VIDEO_DATA_PATH = "/data/disk2/hysia_data/UNC_TVQA_DATASET"

search_machine = BasicSearch(VIDEO_DATA_PATH)

results = search_machine.search(image_query=None, subtitle_query='The sofa is so comfortable.', face_query=None)

print(results)

results = search_machine.search(image_query='test_sofa2.jpg', subtitle_query=None,
                                face_query=None)
print(results)

results = search_machine.search(image_query='test_sofa2.jpg', subtitle_query='The sofa is so comfortable.',
                                face_query=None)

# for i in results:
#     frame = cv2.imread(i['IMAGE'])
#     cv2.imshow('show_img', frame)
#     cv2.waitKey(0)

print(results)

