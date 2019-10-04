# @Time    : 22/11/18 6:49 PM
# @Author  : Huaizheng Zhang
# @Site    : zhanghuaizheng.info
# @File    : test_product.py

import _init_paths
from search.product_search import ProductSearch
import cv2


VIDEO_DATA_PATH = "/data/disk2/hysia_data/Stanford_Online_Products/"

product_machine = ProductSearch()

results = product_machine.search(100, 'test_clip_1.mp4')

print(results)
for i in results:
    cv2.imshow('Image', cv2.imread(i['IMAGE']))
    cv2.waitKey(0)
