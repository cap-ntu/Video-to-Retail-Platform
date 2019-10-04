# @Time    : 5/11/18 3:43 PM
# @Author  : Huaizheng Zhang
# @Site    : zhanghuaizheng.info
# @File    : test_scene_split.py

import _init_paths
from models.scene.shot_detecor import Shot_Detector


if __name__ == "__main__":
    shot_detect = Shot_Detector()
    result = shot_detect.detect('test_generate.mp4')
    print(result)