# @Time    : 28/11/18 4:10 PM
# @Author  : Huaizheng Zhang
# @Site    : zhanghuaizheng.info
# @File    : test_db.py

import _init_paths
from database.video_post import VideoPostDB
from search.search import DatabasePklSearch
import os


os.environ["CUDA_VISIBLE_DEVICES"] = "3"

if __name__ == '__main__':
    # database = 'test_DB/hysia.db'
    # video_path = 'test_DB/BBT0624.mp4'
    #
    # video_db = VideoPostDB(database)
    # video_db.process(video_path)

    video_path = '/home/zhz/Hysia/tests/test_DB/multi_features'

    search_machine = DatabasePklSearch(video_path)

    results = search_machine.search(image_query=None, subtitle_query='The sofa is so comfortable.', face_query=None)

    print(results)
