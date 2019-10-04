import os.path as osp
import json


class QAHandler(object):
    def __init__(self, path, name="tvqa_train.jsonl"):
        """
        Initialize the annotation handler.

        :param path: This should be the root path of UNC TVQA dataset
        :param name: This is the file to be used, using "tvqa_train.jsonl" by default.
        """
        self.path = osp.join(path, "tvqa_qa_release")
        self.name = name
        self.__data = {}
        self.__set_data()

    def __set_data(self):
        file = osp.join(self.path, self.name)
        with open(file, 'r') as data:
            lines = data.readlines()
            for line in lines:
                json_data = json.loads(line.strip())
                self.__data[json_data['qid']] = json_data

    def get_data(self):
        return self.__data

    def get_show_data(self, show):
        result = []
        for i in range(self.__data.__len__()):
            if self.__data[i]['show_name'] == show:
                result.append(self.__data[i])
        return result

    def get_clip_data(self, clip):
        result = []
        for i in range(self.__data.__len__()):
            if self.__data[i]['vid_name'] == clip:
                result.append(self.__data[i])
        return result


if __name__ == '__main__':
    qa = QAHandler("/data/disk2/hysia_data/UNC_TVQA_DATASET")
    result = qa.get_clip_data("grey_s03e20_seg02_clip_14")
    for item in result:
        print(item)