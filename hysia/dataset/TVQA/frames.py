import os
import os.path as osp
import cv2
import math


class FrameHandler(object):
    def __init__(self, path):
        """
        Initialize the frame handler.
        :param path: This should be the root path of UNC TVQA dataset.
        """
        self.path = osp.join(path, 'tvqa_video_frames_fps3')
        self.__shows = []
        self.__clips = {}
        self.__set_shows()
        self.__init_all_clips()

    def __set_shows(self):
        """
        List all the names of shows in this dataset for further traversal and
        processing.

        :return name_list: A list of show names.
        """
        dirs = os.listdir(self.path)
        for show in dirs:
            if osp.isdir(osp.join(self.path, show)):
                self.__shows.append(show)

    def get_shows(self):
        return self.__shows

    def __is_show(self, show):
        if show in self.__shows:
            return True
        return False

    def __is_clip(self, show, clip):
        if clip in self.__clips[show]:
            return True
        return False

    def __set_clips(self, show):
        clip_lists = []
        if self.__is_show(show) and show not in self.__clips.keys():
            show_path = osp.join(self.path, show)
            clips = os.listdir(show_path)
            for clip in clips:
                if osp.isdir(osp.join(show_path, clip)):
                    clip_lists.append(clip)
            self.__clips[show] = clip_lists

    def __init_all_clips(self):
        for show in self.__shows:
            self.__set_clips(show)

    def get_clips(self, show):
        if self.__is_show(show):
            show_path = osp.join(self.path, show)
            return self.__clips[show]

    def get_clip_data(self, show, clip):
        if self.__is_show(show):
            if self.__is_clip(show, clip):
                show_path = osp.join(self.path, show)
                clip_path = osp.join(show_path, clip)
                clip_data = []
                frames = os.listdir(clip_path)
                index = len(frames)
                for i in range(1, index + 1):
                    jpgpath = clip_path + '/' + str(i).zfill(5) + '.jpg'
                    frame = cv2.imread(jpgpath)
                    clip_data.append(frame)
                return clip_data

    def show_frames(self, show, clip):
        data = self.get_clip_data(show, clip)
        for item in data:
            cv2.imshow('show_img', item)
            cv2.waitKey(0)

    def img2video(self, show, clip):
        fps = 3
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_name = clip + ".mp4"
        videoWriter = cv2.VideoWriter(video_name, fourcc, fps, (640, 480))
        data = self.get_clip_data(show, clip)
        for item in data:
            videoWriter.write(item)
        videoWriter.release()

    def match_timeline(self, start, end, show, clip):
        if self.__is_show(show):
            if self.__is_clip(show, clip):
                frame_size = self.get_size(show, clip)
                show_path = osp.join(self.path, show)
                clip_path = osp.join(show_path, clip)
                start_img_index = math.floor(start / 333)
                end_img_index = math.floor(end / 333)
                if frame_size > end_img_index:
                    mid_img_index = int((start_img_index + end_img_index) / 2)
                else:
                    mid_img_index = int((start_img_index + frame_size) / 2)
                jpgpath = clip_path + '/' + str(mid_img_index).zfill(5) + '.jpg'
                return mid_img_index, jpgpath

    def get_size(self, show, clip):
        if self.__is_show(show):
            if self.__is_clip(show, clip):
                show_path = osp.join(self.path, show)
                clip_path = osp.join(show_path, clip)
                frames = os.listdir(clip_path)
                return frames.__len__()


if __name__ == '__main__':
    frame = FrameHandler("/data/disk2/hysia_data/UNC_TVQA_DATASET")
    print(frame.get_shows())
    print(frame.get_clips("friends_frames"))
    frame.show_frames("friends_frames", frame.get_clips("friends_frames")[0])
    frame.img2video("friends_frames", frame.get_clips("friends_frames")[0])