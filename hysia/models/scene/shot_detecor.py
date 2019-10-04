# @Time    : 5/11/18 4:00 PM
# @Author  : Huaizheng Zhang
# @Site    : zhanghuaizheng.info
# @File    : camera_view_detector.py

import os
from datetime import datetime, timedelta
import scenedetect
from scenedetect.video_manager import VideoManager
from scenedetect.scene_manager import SceneManager
from scenedetect.stats_manager import StatsManager
from scenedetect.detectors import ContentDetector

class Shot_Detector(object):
    '''
    This is for detect shot in videos. In other words, the frames in each shot are very similar.
    '''
    def __init__(self):
        '''
        Init the detection manager.
        '''

        print("Init the shot detector")
        print("PySceneDetect version being used: %s" % str(scenedetect.__version__))
        self.stats_manager = StatsManager()
        self.scene_manager = SceneManager(self.stats_manager)
        self.scene_manager.add_detector(ContentDetector())

    def detect(self, video_path):
        '''

        :param video_path:

        :return: the boundary of each shot:
        '''
        if not os.path.exists(video_path):
            print('The video does not exist!')

        self.video_manager = VideoManager([video_path])
        self.base_timecode = self.video_manager.get_base_timecode()
        self.video_manager.set_downscale_factor()
        self.video_manager.start()
        self.scene_manager.detect_scenes(frame_source=self.video_manager)

        scene_list = self.scene_manager.get_scene_list(self.base_timecode)

        split_frames = [0]
        split_time = [0]
        print('List of scenes obtained:')
        for i, scene in enumerate(scene_list):
            print('    Scene %2d: Start %s / Frame %d, End %s / Frame %d' % (
                i + 1,
                scene[0].get_timecode(), scene[0].get_frames(),
                scene[1].get_timecode(), scene[1].get_frames(),))
            split_frames.append(scene[1].get_frames())
            split_time.append(self.__time_trans(scene[1].get_timecode()))
        self.video_manager.release()

        return split_frames, split_time

    def __time_trans(self, second_time):
        # TODO better time transform
        temp = datetime.strptime(second_time, '%H:%M:%S.%f')
        milliseconds = (temp - datetime(1900, 1, 1)) // timedelta(milliseconds=1)
        return milliseconds
