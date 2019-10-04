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
lib_path = osp.join(this_dir, '../../hysia/')
add_path(lib_path)

third_path = osp.join(this_dir, '../../third/')
add_path(third_path)