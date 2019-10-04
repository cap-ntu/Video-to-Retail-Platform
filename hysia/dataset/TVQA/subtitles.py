import os
import os.path as osp
import re


class SubtitleHandler(object):
    def __init__(self, path):
        self.path = osp.join(path, "tvqa_subtitles")
        self.__subtitles = []
        self.__data = {}
        self.__set_subtitles()
        self.__init_all_data()

    def __set_subtitles(self):
        files = os.listdir(self.path)
        for file in files:
            name, form = file.split('.')
            if form == 'srt':
                self.__subtitles.append(name)

    def get_subtitles(self):
        return self.__subtitles

    def __is_subtitle(self, name):
        if name in self.__subtitles:
            return True
        return False

    def __set_data(self, name):
        file_name = name + ".srt"
        file_path = osp.join(self.path, file_name)
        if self.__is_subtitle(name) and name not in self.__data.keys():
            with open(file_path, 'r') as file:
                data_buf = file.readlines()
                is_multi_line = False  # This is for combining two consecutive non-blank lines in a single subtitle.
                count_empty_line = 0  # This is for recognizing the consecutive empty lines in the end of the file.
                subtitle_buf = {}
                line_buf = {}
                for line in data_buf:

                    if re.match(r'^\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}', line):
                        line_buf['timeclip'] = self.trim_time_clip(line.strip())

                    # Every time an index is matched, we will consider as a new subtitle begins.
                    elif re.match(r'^\d+$', line):
                        line_buf = {}
                        index = int(line.strip('\n'))
                        is_multi_line = False
                        count_empty_line = 0

                    # Every time an '\n' is matched, we consider that this subtitle ends.
                    elif re.match("\n", line):
                        is_multi_line = False
                        if count_empty_line == 0:
                            subtitle_buf[index] = line_buf
                        else:
                            count_empty_line += 1

                    else:
                        line = self.delete_speaker(line).strip('-')
                        if is_multi_line is False:
                            line_buf['content'] = line.strip()
                        else:
                            line_buf['content'] = line_buf['content'] + ' ' + line.strip()
                        is_multi_line = True
                self.__data[name] = subtitle_buf

    def __init_all_data(self):
        for sub in self.__subtitles:
            self.__set_data(sub)

    def get_clip_data(self, clip):
        if self.__is_subtitle(clip) and clip in self.__data.keys():
            return self.__data[clip]
        else:

            # Please comment this print line out when indexing the whole dataset to reduce clutter.

            # print("From Subtitle Handler: Clip does not exist or has not been set yet, please check")

            return None

    @staticmethod
    def trim_time_clip(line):
        trim = re.match(r'^(\d{2}):(\d{2}):(\d{2}),(\d{3}) --> (\d{2}):(\d{2}):(\d{2}),(\d{3})', line)
        start_time = int(trim.group(1)) * 3600000 + int(trim.group(2)) * 60000 + int(trim.group(3)) * 1000 + int(trim.group(4))
        end_time = int(trim.group(5)) * 3600000 + int(trim.group(6)) * 60000 + int(trim.group(7)) * 1000 + int(trim.group(8))
        return start_time, end_time

    @staticmethod
    def delete_speaker(line):
        trim = re.match(r'(\([\s\S]*:\))([\s\S]*)', line)
        if trim:
            return trim.group(2).strip()
        else:
            return line


if __name__ == '__main__':
    subtitle = SubtitleHandler("/data/disk2/hysia_data/UNC_TVQA_DATASET")
    print(subtitle.get_subtitles())
    data = subtitle.get_clip_data("castle_s01e01_seg02_clip_14")
    if data is not None:
        for item in data:
            print(item, data[item])
