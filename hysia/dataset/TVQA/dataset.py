import cv2

from .features import FeatureHandler
from .frames import FrameHandler
from .qa import QAHandler
from .subtitles import SubtitleHandler


class DatasetHandler(object):
    """Handel the whole TVQA dataset except the feature dataset"""

    def __init__(self, path):
        # initialize the handlers
        self.path = path
        self.qa = QAHandler(self.path)
        self.subtitle = SubtitleHandler(self.path)
        self.frame = FrameHandler(self.path)
        self.feature = FeatureHandler(self.path)

    def shows(self):
        return self.frame.get_shows()

    def clips(self, show):
        return self.frame.get_clips(show)

    def match_data(self, show, clip):
        qa_result = self.qa.get_clip_data(clip)
        subtitle_result = self.subtitle.get_clip_data(clip)
        feature_result = self.feature.get_clip_data(clip)
        frame_result = self.frame.get_clip_data(show, clip)
        return qa_result, subtitle_result, feature_result, frame_result

    def match_subtitle(self, show, clip):
        """
        This method will return a dictionary which matches subtitle index and related frames.

        :param show: Name of the show folder.
        :param clip: The correct name of the show
        :return: A dictionary with subtitle index
        """
        subtitle_result = self.subtitle.get_clip_data(clip)
        if subtitle_result is None:
            print("No subtitle file related.")
            return None
        else:
            subtitle_index = {}
            for key in subtitle_result:
                timeclip = subtitle_result[key]['timeclip']
                frame_result = self.frame.match_timeline(timeclip[0], timeclip[1], show, clip)
                subtitle_index[subtitle_result[key]['content']] = frame_result
            return subtitle_index

    def show_data(self, data):
        qa_result, subtitle_result, feature_result, frame_result = data
        print("The QA pairs are:", qa_result)
        print("The subtitle results are:", subtitle_result)
        print("The feature results are:", feature_result)
        print("The frame results are:", frame_result)


if __name__ == '__main__':
    data = DatasetHandler("/data/disk2/hysia_data/UNC_TVQA_DATASET")
    #
    # tic = time.time()
    # matched_data = data.match_data("castle_frames", "castle_s01e01_seg02_clip_00")
    # toc = time.time()
    # print("It takes " + str((toc - tic)) + " ms")
    # data.show_data(matched_data)

    index = data.match_subtitle("castle_frames", "castle_s01e01_seg02_clip_02")
    if index is not None:
        for key in index:
            print(key)
            for item in index[key]:
                cv2.imshow('Image', item)
                cv2.waitKey(0)
            break