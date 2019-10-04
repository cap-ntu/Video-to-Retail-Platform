# Author: wang yongjie
# Email : yongjie.wang@ntu.edu.sg
import os
import tensorflow as tf
import numpy as np
import cv2
import pickle as pk
import copy

class extractor(object):
    def __init__(self, pb_file, in_dir, save_name, cast):

        self.pb_file = pb_file
        self.in_dir = in_dir
        self.save_name = save_name
        self.cast = cast

    def init_tf_env(self):
        config = tf.ConfigProto()
        config.allow_soft_placement = True
        config.gpu_options.allow_growth = True
        with tf.gfile.GFile(self.pb_file, 'rb') as f: 
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name = '')

        self.sess = tf.Session(config = config, graph = graph)
        self.input_image = graph.get_tensor_by_name("img_inputs:0")
        self.phase_train_placeholder = graph.get_tensor_by_name("dropout_rate:0")
        self.embeddings = graph.get_tensor_by_name("resnet_v1_50/E_BN2/Identity:0")

    def load_cast(self):
        self.cast_dict = dict()
        f = open(self.cast, 'r')
        content = f.readlines()
        for i in range(len(content)):
            tmp = content[i].split(' ')
            name = tmp[0]
            role = tmp[1][0:-1]
            self.cast_dict[name] = role

    def extract_feature(self):
        name_feature = []
        for item in os.listdir(self.in_dir):
            subdir = os.path.join(self.in_dir, item)
            for subitem in os.listdir(subdir):
                tmp = {}
                filename = os.path.join(subdir, subitem)
                img = cv2.imread(filename)
                tmp['IMAGE'] = filename
                img = (img - 127.5) / 128

                if isinstance(img, type(None)):
                    print(filename + ' is not existing')
                    continue
                img = np.expand_dims(img, axis=0)
                feature = self.sess.run(self.embeddings, feed_dict = {self.input_image:img, self.phase_train_placeholder:1})
                feature = feature[0]

                if np.any(np.isnan(feature)):
                    print("feature is none")
                    continue
                tmp['FEATURE'] = feature
                tmp['CELEBRITY'] = item
                tmp['ROLE'] = self.cast_dict[item]
                name_feature.append(tmp)

        return name_feature

    def save_feature(self):
        name_feature = self.extract_feature()
        f = open(self.save_name, 'wb')
        pk.dump(name_feature, f)

