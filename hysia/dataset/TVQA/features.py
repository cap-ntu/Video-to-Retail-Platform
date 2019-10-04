import h5py
import os.path as osp
import numpy as np


class FeatureHandler(object):
    def __init__(self, path):
        """
        Initialize the feature handler.

        :param path: This is should be the root path of UNC TVQA dataset.
        """
        self.dir_path = osp.join(path, "tvqa_imagenet_resnet101_pool5")
        self.file_path = osp.join(self.dir_path, "tvqa_imagenet_pool5.h5")
        self.__data = h5py.File(self.file_path)
        self.__keys = list(self.__data.keys())

    def get_keys(self):
        return self.__keys

    def get_data(self):
        return self.__data

    def __is_key(self, key):
        if key in self.__keys:
            return True
        return False

    def get_clip_data(self, clip):
        """
        This method will return the data of a certain key (clip id).

        :param clip: This should be an existing clip id.

        :return key_data: The record related to the legal input key.
        :return None: In case the input key is illegal.
        """
        if self.__is_key(clip):
            features_data = self.__data[clip]
            return features_data
        return None


if __name__ == '__main__':
    feature = FeatureHandler("/data/disk2/hysia_data/UNC_TVQA_DATASET")
    keys = feature.get_keys()
    print("Show keys of this h5py file:")
    print(len(keys))
    data = feature.get_clip_data("castle_s01e01_seg02_clip_00")
    print("Show an example of data output:")
    print("The feature data of clip->'castle_s01e01_seg02_clip_00' is:")
    print(np.array(data))