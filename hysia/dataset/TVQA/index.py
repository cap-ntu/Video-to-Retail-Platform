# @Time    : 13/11/18 6:02 PM
# @Author  : Huaizheng Zhang
# @Site    : zhanghuaizheng.info
# @File    : TVQA_indexer.py

import os.path as osp
import pickle

import numpy as np

from .features import FeatureHandler
from .frames import FrameHandler
from .subtitles import SubtitleHandler


class TVQA_indexer(object):
    def __init__(self, path):
        print('Start to load the TVQA directory')
        self.path = path
        self.index_image = 'IMAGE'
        self.index_feature = 'FEATURE'
        self.index_subtitle = 'SUBTITLE'
        self.index_tv = 'TV_NAME'
        self.index_scene = 'SCENE'
        self.index_start = 'START_TIME'
        self.index_end = 'END_TIME'

        self.subtitle = SubtitleHandler(self.path)
        self.frame = FrameHandler(self.path)
        self.feature = FeatureHandler(self.path)

        self.shows = self.frame.get_shows()
        print('Finish load the TVQA directory')

    def index(self):
        for i in self.shows:

            tv_name = i.split('_')[0]
            filename = 'TVQA_' + tv_name + '_index.pkl'
            output = osp.join(self.path, filename)

            if osp.isfile(output):

                print('The {} has been indexed!'.format(i))

            else:
                print("Indexing show:", i)
                database = []
                clips = self.frame.get_clips(i)
                for j in clips:
                    '''
                    In this part we use subtitle as index.
                    The subtitle can be seen as a scene.
                    So we can use this function to process the video which processed by our system
                    '''
                    print("     Indexing clip: %s. Status:" % j, end='')
                    subtitle = self.subtitle.get_clip_data(j)
                    frame_size = self.frame.get_size(i, j)
                    if frame_size < 300:
                        feature = self.feature.get_clip_data(j)
                        if subtitle is not None and feature is not None:
                            print("Process.")
                            for k in subtitle:
                                mid_img_index, mid_img_path = self.frame.match_timeline(subtitle[k]['timeclip'][0],
                                                                                        subtitle[k]['timeclip'][1], i,
                                                                                        j)
                                sample = {self.index_tv: tv_name,
                                          self.index_scene: (j + '_' + str(k)),
                                          self.index_subtitle: subtitle[k]['content'],
                                          self.index_image: mid_img_path,
                                          self.index_feature: np.array(feature)[mid_img_index],
                                          self.index_start: subtitle[k]['timeclip'][0],
                                          self.index_end: subtitle[k]['timeclip'][1]}
                                database.append(sample)
                        else:
                            print("Pass.")
                    else:
                        print("Pass.")

                with open(output, 'wb') as f:
                    print('Dumping ...')
                    pickle.dump(database, f)
                    print('Finish indexing ', i)
                    del database

if __name__ == '__main__':
    path = '/data/disk2/hysia_data/UNC_TVQA_DATASET'
    indexer = TVQA_indexer(path)
    indexer.index()
