#!/usr/bin/env python
# @Time    : 17/8/18 3:53 PM
# @Author  : Huaizheng Zhang
# @Site    : zhanghuaizheng.info
# @File    : _init_paths.py


import os.path as osp
import sys


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


this_dir = osp.dirname(__file__)

# Add lib to PYTHONPATH
lib_path = osp.join(this_dir, '../')
third_path = osp.join(this_dir, '../third/')
ctpn_path = osp.join(this_dir, '../third/ctpn/')
models_path = osp.join(this_dir, '../hysia/')
test_path = osp.join(this_dir, '../tests/')
# mtcnn_path = osp.join(this_dir, '../models/face')

add_path(lib_path)
add_path(third_path)
add_path(ctpn_path)
add_path(models_path)
add_path(test_path)
# add_path(mtcnn_path)