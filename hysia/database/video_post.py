# @Time    : 29/11/18 4:47 PM
# @Author  : Huaizheng Zhang
# @Site    : zhanghuaizheng.info
# @File    : video_post.py

import os.path as osp
import imageio
import cv2
import numpy as np
import tqdm
import math
import subprocess
import tensorflow as tf
from database.sqlite import SqliteDatabase
from models.scene.shot_detecor import Shot_Detector
from dataset.srt_handler import extract_srt
from models.nlp.sentence import TF_Sentence
from models.scene.detector import scene_visual
from models.object.audioset_feature_extractor import AudiosetFeatureExtractor
from PIL import Image
import pickle


class VideoPostDB(object):
    '''
    This class is for video post_processing.
    We insert some key information of processed videos to a small database and index it for inserting product.
    '''

    def __init__(self, db_path, image_model=None, sentence_model=None, audio_model=None):
        '''

        :param db_path: the path of sqlite database
        :param image_model: get only one image feature to represent the scene
        :param sentence_model: get the subtitle feature in this scene shot
        :param audio_model: get one audio feature in this scene shot
        '''

        self.database = db_path

        # for store the processed feature into .pkl file
        self.sample_list = list()

        if image_model is None:
            self.image_model = scene_visual('resnet50', '../weights/places365/{}.pth',
                                            '../weights/places365/categories.txt',
                                            'cuda:0')
        if sentence_model is None:
            self.sentence_model = TF_Sentence('../weights/sentence/96e8f1d3d4d90ce86b2db128249eb8143a91db73')

        if audio_model is None:
            vgg_graph = tf.Graph()
            with vgg_graph.as_default():
                od_graph_def = tf.GraphDef()
                with tf.gfile.GFile('../weights/audioset/vggish_fr.pb', 'rb') as fid:
                    serialized_graph = fid.read()
                    od_graph_def.ParseFromString(serialized_graph)
                    tf.import_graph_def(od_graph_def, name='')
            self.audio_model = AudiosetFeatureExtractor(vgg_graph, '../weights/audioset/vggish_pca_params.npz')

        # detect the shot bound and get the split frame and millisecond
        self.shot_detector = Shot_Detector()

        # create a simple hysia database or connect to the exist database
        if not osp.isfile(self.database):
            statement = (
                'CREATE TABLE %s (tv_name TEXT, start_time TEXT, end_time TEXT, scene_name TEXT, image_path TEXT, image_feature BLOB, subtitle TEXT, subtitle_feature BLOB, audio_path TEXT, audio_feature BLOB, object_json_path TEXT, face_json_path TEXT, insert_product_path TEXT);')
            tables = ['video']
            statements = [statement % table for table in tables]
            self.db = SqliteDatabase(self.database, statements)
        else:
            self.db = SqliteDatabase(self.database)

    def process(self, video_path):
        '''

        :param video_path: a video path which can also help to find the subtitle and audio
        :param features_path: store the features into a pickle file
        :return:
        '''
        # read video and process video, subtitle and audio information
        # save features into the database
        # TODO This part can be optimized during video content analysis stage in the 0.2 version

        video_name = video_path.split('/')[-1]

        splits_frame, splits_ms = self.shot_detector.detect(video_path)
        # print(splits_ms)
        # get middle frame as the key frame in this shot
        # package the split time
        middle_frame = list()
        middle_time = {}

        for i in range(len(splits_ms) - 1):
            temp = math.floor((splits_frame[i] + splits_frame[i + 1]) / 2.0)
            middle_frame.append(temp)
            middle_time[temp] = [splits_ms[i], splits_ms[i + 1]]
        # print(middle_frame)
        # print(middle_time)

        vid = imageio.get_reader(video_path, 'ffmpeg')
        frame_cnt = vid.get_meta_data()["nframes"]

        # get key frame image features, subtitle features and audio features
        try:
            with tqdm.tqdm(total=frame_cnt, unit="frames") as pbar:
                for id, img in enumerate(vid):
                    if id in middle_frame:
                        # TODO Try to classify image_io, cv.read, PIL
                        # get image feature
                        try:
                            scene_name, scene_feature = self.__get_scene_feature(img)
                        except:
                            # print('Can not extract image feature and assume it is none')
                            scene_feature = 'unknown_feature'
                            scene_name = 'unknown_scene'

                        try:
                            subtitle, subtitle_feature = self.__get_subtitle_feature(middle_time[id][0], middle_time[id][1],
                                                                                 video_path)
                        except:
                            # print('Can not extract subtitle feature and assume it is none')
                            subtitle_feature = 'unknown_feature'
                            subtitle = 'unknown_subtitle'

                        try:
                            audio_path, audio_feature = self.__get_audio_feature(middle_time[id][0], middle_time[id][1],
                                                                                 video_path)

                        except:
                            # print('Can not extract audio feature and assume it is none')
                            audio_feature = 'unknown_feature'
                            audio_path = 'unknown_audio_path'

                        # TODO get object json file and face json file

                        sql = "INSERT INTO video (tv_name, start_time, end_time, scene_name, image_feature, subtitle, subtitle_feature, audio_path, audio_feature) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
                        sample = (
                        video_name, middle_time[id][0], middle_time[id][1], scene_name, scene_feature, subtitle,
                        subtitle_feature, audio_path, audio_feature)

                        self.db.add(sql, sample)

                        # pack
                        self.__pack_samples(sample)

                    pbar.update(1)
        except:
            print('Throw the last frames')

        self.__insert_index(video_path)

    def __get_scene_feature(self, img):
        img_pil = Image.fromarray(img)
        scene_feature = self.image_model.extract_vec(img_pil, True)
        scene_name = self.image_model.detect(img_pil, True)
        # only return top 1 scene name
        return scene_name['scene'][0], scene_feature

    def __get_subtitle_feature(self, start_time, end_time, video_path):
        srt_name = video_path.split("/")[-1].split(".")[0] + ".srt"
        srt_path = osp.join(osp.dirname(osp.abspath(video_path)), 'subtitles', srt_name)
        sentences = extract_srt(start_time, end_time, srt_path)
        if len(sentences) == 0:
            sentences_feature = 'unknown_feature'
            sentences = 'unknown_subtitle'
        else:
            # TODO TEXT support what data types (BLOB only support numpy)
            sentences = " ".join(sentences)
            sentences_feature = self.sentence_model.encode(sentences)

        return sentences, np.array(sentences_feature)

    def __get_audio_feature(self, start_time, end_time, video_path):
        audio_name = video_path.split("/")[-1].split(".")[0] + ".wav"
        audio_path = osp.join(osp.dirname(osp.abspath(video_path)), 'audios', audio_name)
        # command = "ffmpeg -i %s -ab 160k -ac 2 -ar 44100 -vn %s" % (video_path, audio_path)
        # subprocess.call(command, shell=True)

        audio_feature = self.audio_model.extract(audio_path, start_time, end_time)[0]
        return audio_path, audio_feature

    def __get_object_json(self):
        pass

    def __get_frame_path(self):
        pass

    def __get_face_json(self):
        pass

    def __pack_samples(self, sample):
        sample_dict = {'TV_NAME': sample[0],
                       'START_TIME': sample[1],
                       'END_TIME': sample[2],
                       'SCENE': sample[3],
                       'FEATURE': sample[4],
                       'SUBTITLE': sample[5],
                       'SUBTITLE_FEATURE': sample[6],
                       'AUDIO': sample[7],
                       'AUDIO_FEATURE': sample[8]
                       }

        self.sample_list.append(sample_dict)

    def __insert_index(self, video_path):
        pkl_name = video_path.split("/")[-1].split(".")[0] + "_index.pkl"
        pkl_path = osp.join(osp.dirname(osp.abspath(video_path)), 'multi_features', pkl_name)
        with open(pkl_path, 'wb') as f:
            pickle.dump(self.sample_list, f)



