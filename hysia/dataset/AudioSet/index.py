import os
import os.path as osp
import tensorflow as tf
import pandas as pd
import pickle


class AudioSet_indexer(object):
    def __init__(self, path):
        self.path = path
        self.__labels = self.__load_labels()

        self.index_feature = 'FEATURE'
        self.index_video = 'VIDEO'
        self.index_start_time = 'START_TIME'
        self.index_end_time = 'END_TIME'
        self.index_label = 'LABEL'

    def __load_labels(self):
        csv_name = 'class_labels_indices.csv'
        csv_path = osp.join(self.path, csv_name)
        if osp.isfile(csv_path):
            file = pd.read_csv(csv_path)
            return file
        else:
            print("Couldn't find csv file, please check.")

    def read_labels(self):
        first_rows = self.__labels.head(n=15)
        print(first_rows)

    def locate_label(self, index):
        return self.__labels.loc[index].display_name

    def index(self):
        dir_path = osp.join(self.path, 'audioset_v1_embeddings')
        for _set in os.listdir(dir_path):
            filename = 'AudioSet_' + _set + '_index.pkl'
            output = osp.join(self.path, filename)
            if osp.isfile(output):
                print('The {} has already been indexed!'.format(_set))
            else:
                print('Start Indexing {} ... '.format(_set))
                database = []
                _dir = osp.join(dir_path, _set)
                for record in os.listdir(_dir):
                    record_name, record_type = record.split('.')
                    if record_type == 'tfrecord':

                        print('    Indexing {}.{} ...    '.format(_set, record_name), end='')
                        record_path = osp.join(_dir, record)
                        tf_record_iterator = tf.python_io.tf_record_iterator(path=record_path)
                        for tf_record in tf_record_iterator:
                            example = tf.train.SequenceExample()
                            example.ParseFromString(tf_record)
                            video_id = example.context.feature['video_id'].bytes_list.value[0]
                            start_time_seconds = example.context.feature['start_time_seconds'].float_list.value[0]
                            end_time_seconds = example.context.feature['end_time_seconds'].float_list.value[0]
                            labels = list(example.context.feature['labels'].int64_list.value)
                            audio_embeddings = example.feature_lists.feature_list['audio_embedding'].feature
                            for audio_embedding in audio_embeddings:
                                audio_feature = audio_embedding.bytes_list.value[0].hex()
                                sample = {
                                    self.index_feature: [int(audio_feature[i:i + 2], 16)
                                                         for i in range(0, len(audio_feature), 2)],
                                    self.index_video: video_id,
                                    self.index_start_time: start_time_seconds,
                                    self.index_end_time: end_time_seconds,
                                    self.index_label: [self.locate_label(index) for index in labels]
                                }
                                database.append(sample)
                        print('Done.')
                with open(output, 'wb') as f:
                    print('Dumping {}... '.format(_set))
                    pickle.dump(database, f)
                    print('Finish indexing {}!'.format(_set))
                    del database


if __name__ == '__main__':
    path = '/data/disk2/hysia_data/AudioSet'
    indexer = AudioSet_indexer(path)
    indexer.index()