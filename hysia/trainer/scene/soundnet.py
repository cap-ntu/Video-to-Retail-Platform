# TensorFlow version of NIPS2016 soundnet
# Required package: librosa: A python package for music and audio analysis.
# $ pip install librosa

from trainer.scene.soundnet_layers import batch_norm, conv2d, relu, maxpool, dense, denseBN
from models.scene.audio.audio_util import preprocess, load_from_list, load_audio
from models.scene.audio.data_util import unison_shuffled_copies
from models.scene.audio.dataset_util import get_dataconfig
from tensorflow.python.tools import freeze_graph
from random import shuffle
from glob import glob

import tensorflow as tf
import numpy as np
import argparse
import librosa
import pickle
import json
import math
import time
import os
import csv


class SoundNet:
    def __init__(self, session, config=None, param_G=None):
        self.sess = session
        self.config = config
        self.param_G = param_G
        self.g_step = tf.Variable(0, trainable=False)
        self.counter = 0
        self.model()

    def model(self):
        """
        Initialization of the tensors and ops used by the model

        :return: None
        """
        # Placeholder
        self.sound_input_placeholder = tf.placeholder(tf.float32,
                                                      shape=[None, self.config['sample_size'], 1,
                                                             1])  # batch x h x w x channel
        self.labels = tf.placeholder(tf.float32, shape=[None, 10])
        # Generator
        self.add_generator(name_scope=self.config['name_scope'])
        self.last_layer_index = 35
        # Losses
        self.retrain_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=self.layers[self.last_layer_index]))

        # Metrics
        with tf.name_scope('train_acc'):
            self.accuracy, self.update_acc = tf.metrics.accuracy(labels=tf.argmax(self.labels, 1),
                                                                predictions=tf.argmax(self.layers[self.last_layer_index], 1))
        self.accuracy_vars = [v for v in tf.local_variables() if 'train_acc/' in v.name]

        # Summary
        self.loss_sum = tf.summary.scalar("g_loss", self.retrain_loss)
        self.g_sum = tf.summary.merge([self.loss_sum])
        # self.writer = tf.summary.FileWriter("./logs", self.sess.graph)

        # variable collection
        # self.g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
        #                                 scope='conv9') # only train last layer
        self.retrain_vars = [v for v in tf.global_variables() if '/retrain' in v.name]
        self.saver = tf.train.Saver(keep_checkpoint_every_n_hours=12,
                                    max_to_keep=5,
                                    restore_sequentially=True)
        # print(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
        # Optimizer and summary
        self.retrain_opt = tf.train.AdamOptimizer(self.config['learning_rate'], beta1=self.config['beta1']) \
            .minimize(self.retrain_loss, var_list=(self.retrain_vars), global_step=self.g_step)
        self.retrain_grad = tf.gradients(self.retrain_loss, self.retrain_vars)



        # Initialize
        init_op = [tf.global_variables_initializer(), tf.local_variables_initializer()]
        self.sess.run(init_op)

        # Load checkpoint
        if self.load(self.config['checkpoint_dir'], mode=self.config['load_mode']):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

    def add_generator(self, name_scope='SoundNet'):
        """
        Define network structure of the soundnet layer by layer
        :parameter:
            name_scope: indicate the scope of the the layers to be defined under
        :return: None
        """
        with tf.variable_scope(name_scope) as scope:
            self.layers = {}

            # Stream one: conv1 ~ conv7
            self.layers[1] = conv2d(self.sound_input_placeholder, 1, 16, k_h=64, d_h=2, p_h=32, name_scope='conv1')
            self.layers[2] = batch_norm(self.layers[1], 16, self.config['eps'], name_scope='conv1')
            self.layers[3] = relu(self.layers[2], name_scope='conv1')
            self.layers[4] = maxpool(self.layers[3], k_h=8, d_h=8, name_scope='conv1')

            self.layers[5] = conv2d(self.layers[4], 16, 32, k_h=32, d_h=2, p_h=16, name_scope='conv2')
            self.layers[6] = batch_norm(self.layers[5], 32, self.config['eps'], name_scope='conv2')
            self.layers[7] = relu(self.layers[6], name_scope='conv2')
            self.layers[8] = maxpool(self.layers[7], k_h=8, d_h=8, name_scope='conv2')

            self.layers[9] = conv2d(self.layers[8], 32, 64, k_h=16, d_h=2, p_h=8, name_scope='conv3')
            self.layers[10] = batch_norm(self.layers[9], 64, self.config['eps'], name_scope='conv3')
            self.layers[11] = relu(self.layers[10], name_scope='conv3')

            self.layers[12] = conv2d(self.layers[11], 64, 128, k_h=8, d_h=2, p_h=4, name_scope='conv4')
            self.layers[13] = batch_norm(self.layers[12], 128, self.config['eps'], name_scope='conv4')
            self.layers[14] = relu(self.layers[13], name_scope='conv4')

            self.layers[15] = conv2d(self.layers[14], 128, 256, k_h=4, d_h=2, p_h=2, name_scope='conv5')
            self.layers[16] = batch_norm(self.layers[15], 256, self.config['eps'], name_scope='conv5')
            self.layers[17] = relu(self.layers[16], name_scope='conv5')
            self.layers[18] = maxpool(self.layers[17], k_h=4, d_h=4, name_scope='conv5')

            self.layers[19] = conv2d(self.layers[18], 256, 512, k_h=4, d_h=2, p_h=2, name_scope='conv6')
            self.layers[20] = batch_norm(self.layers[19], 512, self.config['eps'], name_scope='conv6')
            self.layers[21] = relu(self.layers[20], name_scope='conv6')

            self.layers[22] = conv2d(self.layers[21], 512, 1024, k_h=4, d_h=2, p_h=2, name_scope='conv7')
            self.layers[23] = batch_norm(self.layers[22], 1024, self.config['eps'], name_scope='conv7')
            self.layers[24] = relu(self.layers[23], name_scope='conv7')

            # Split one: conv8, conv8_2
            # NOTE: here we use a padding of 2 to skip an unknown error
            # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/common_shape_fns.cc#L45
            # self.layers[25] = conv2d(self.layers[24], 1024, 1000, k_h=8, d_h=2, p_h=2, name_scope='conv8')
            # self.layers[26] = conv2d(self.layers[24], 1024, 401, k_h=8, d_h=2, p_h=2, name_scope='conv8_2')

            #New Layers for Retrain
            self.layers[25] = tf.layers.flatten(self.layers[24])
            self.layers[26] = dense(self.layers[25], units=500, activation=None, name_scope='retrain1')
            self.layers[27] = denseBN(self.layers[26], name_scope='retrain_BN1')
            self.layers[28] = tf.nn.leaky_relu(self.layers[27])
            self.layers[29] = dense(self.layers[28], units=500, activation=None, name_scope='retrain2')
            self.layers[30] = denseBN(self.layers[29], name_scope='retrain_BN2')
            self.layers[31] = tf.nn.leaky_relu(self.layers[30])
            self.layers[32] = dense(self.layers[31], activation=None, units=10, name_scope='retrain3')
            self.layers[33] = denseBN(self.layers[32], name_scope='retrain_BN3')
            self.layers[34] = tf.nn.leaky_relu(self.layers[33])
            self.layers[35] = dense(self.layers[34], activation=None, units=10, name_scope='retrain4')

    def retrain(self):
        """
        Retrain soundnet

        args: None

        :return: None
        """

        self.sess.run([tf.local_variables_initializer()])
        start_time = time.time()

        # Data info
        data = glob('/home/lzy/dcase2018/development/audio/*.{}'.format(self.config['subname']))
        label_dict = self.load_dcase_label_from_csv(self.config['label_csv'])
        shuffle(data)
        scene_label = []
        for fn in data:
            scene_label.append(label_dict[fn.split('/')[-1]])
        scene_label = np.array(scene_label)
        batch_idxs = min(len(data), self.config['train_size']) * self.config['augment_factor'] // self.config['batch_size']
        load_size = self.config['batch_size'] // self.config['augment_factor']
        for epoch in range(self.counter // batch_idxs, self.config['epoch']):
            self.sess.run(tf.variables_initializer(self.accuracy_vars))
            for idx in range(self.counter % batch_idxs, batch_idxs):

                # By default, librosa will resample the signal to 22050Hz. And range in (-1., 1.)
                batch_sound = load_from_list(data[idx * load_size:(idx + 1) * load_size],
                                             self.config)
                batch_label = scene_label[idx * load_size:(idx + 1) * load_size]
                batch_sound, batch_label = self.dcase_data_augmentation(batch_sound, batch_label, hop=self.config['sample_size'])
                batch_label = self.sess.run(tf.one_hot(batch_label, depth=10))
                # Update G network
                # NOTE: Here we still use dummy random distribution for scene and objects
                # out = self.sess.run(tf.nn.softmax(self.layers[29]), feed_dict={self.sound_input_placeholder: batch_sound,
                #                self.labels: batch_label})
                output, _, summary_str, l_scn, acc = self.sess.run(
                    [self.layers[32], self.retrain_opt, self.g_sum, self.retrain_loss, self.update_acc],
                    feed_dict={self.sound_input_placeholder: batch_sound,
                               self.labels: batch_label})
                self.writer.add_summary(summary_str, self.counter)
                print("[Epoch {}] {}/{} | Time: {} | scene_loss: {} | batch_acc: {}".format(epoch, idx, batch_idxs,
                                                                                               time.time() - start_time,
                                                                                               l_scn, acc))

                if np.mod(self.counter + 1, self.config['save_interval']) == 0:
                    self.save(self.config['checkpoint_dir'], self.counter)

                self.counter += 1

    def extract_feature(self, input, layer=25):
        """

        :parameter:
            input: input audio stream samples
            layer: indicator of the output layer
        :return:
            output of the indicated layer as feature array
        """
        feature = self.sess.run(self.layers[layer], feed_dict={self.sound_input_placeholder: input})
        return feature

    def extract_feature_all(self, dataset_dict, label_dict, layer=25, augment=True):
        s = self.config['sample_size'] / self.config['sample_rate']

        feature_prefix = '/'.join([dataset_dict['feature_path'] + dataset_dict['features']['soundnet']])
        for file_name in label_dict.keys():
            file_path = '/'.join([dataset_dict['prefix'], file_name])
            sound_sample, _ = load_audio(file_path)
            sound_sample = [preprocess(sound_sample, self.config)]
            augmented, _ = self.dcase_data_augmentation(sound_sample, labels=[label_dict[file_name]],
                                                        hop=self.config['augment_hop'])
            features = self.extract_feature(augmented, layer=layer)
            pickle.dump('/'.join([feature_prefix, file_name + '.s{}f'.format(s)]), features)

    def run_with_pb(self, tensor, feed_dict):
        with self.graph.as_default():
            with tf.Session() as sess:
                return sess.run(tensor, feed_dict=feed_dict)

    def predict(self, input_sample, sr, fr, is_mono):

        """
        Predict scene labels of input audio stream
        If you wang to use this function, don't pass 'param_G' parameter during initialization
        :parameter:
            input_sample: audio stream from uploaded video
            sr: sample rate of the audio stream
            fr: frame rate of the video
            is_mono: if the input audio stream is a mono channel stream

        :return:
            {
                labels: ['bus', 'train_station', ...] # list of labels linked with number
                confidences:[
                                [0.1, 0.1, 0.1, ... , 0.1] # Confidence of of a snippet consists of several frames
                                ...
                                [0.1, 0.1, 0.1, ... , 0.1]
                            ]
            }
        """

        if not is_mono:
            input_sample = librosa.to_mono(input_sample)
        if sr != self.config['sample_rate']:
            librosa.resample(input_sample, sr, self.config['sample_rate'])
        sound_sample = preprocess(input_sample, self.config, is_train=False)
        hop = int(math.floor(self.config['sample_rate'] / fr))
        data_len = sound_sample.shape[1]
        sample_size = self.config['sample_size']
        frame_count = int(math.ceil((data_len - sample_size) / hop))
        input = np.empty([frame_count, sample_size, 1, 1])
        count = 0
        values = np.empty([frame_count, 5])
        indices = np.empty([frame_count, 5], dtype=np.int32)
        batch_size = self.config['batch_size']

        for j in range(0, data_len - sample_size, hop):
            input[count] = sound_sample[:, j:j + sample_size, :, :]
            count += 1

        if self.graph is not None:
            input_tensor = self.graph.get_tensor_by_name(self.sound_input_placeholder.name)
            output_tensor = tf.nn.top_k(tf.nn.softmax(self.graph.get_tensor_by_name(self.layers[35].name)),
                                        k=5, sorted=True)
            for j in range(0, frame_count, batch_size):
                # print("Running:{}".format(j))
                batch_input = input[j : (j + batch_size)]
                values[j:j+batch_size], indices[j:j+batch_size] \
                    = self.run_with_pb(output_tensor, feed_dict={input_tensor: batch_input})
        else:
            for j in range(0, frame_count, batch_size):
                batch_input = input[j : (j + batch_size)]
                values[j:j+batch_size], indices[j:j+batch_size] \
                    = self.sess.run(tf.nn.top_k(tf.nn.softmax(self.layers[35]), k=5, sorted=True),
                                    feed_dict={self.sound_input_placeholder: batch_input})

        labels = np.array(get_dataconfig(self.config['dataset_name'])['labels'])
        values = values.tolist()
        labels = labels[indices].tolist()

        output = []
        for i in range(len(values)):
            item = {}
            item['labels'] = labels[i]
            item['scores'] = values[i]
            output.append(item)
        return output

    def predict_file(self, sample_fp, fr):
        """
        Predict scene labels from audio files, a wrapper API of predict(self, inpout_sample, sr, fr, is_mono)
        If you wang to use this function, don't pass 'param_G' parameter during initialization

        :parameter:
            sample_fp: file path of the input audio stream
            fr: frame rate of the video

        :return:
            {
                labels: ['bus', 'train_station', ...] # list of labels linked with number
                confidences:[
                                [0.1, 0.1, 0.1, ... , 0.1] # Confidence of of a snippet consists of several frames
                                ...
                                [0.1, 0.1, 0.1, ... , 0.1]
                            ]
            }
        """

        sound_sample, _ = load_audio(sample_fp)
        return self.predict(sound_sample, sr=self.config['sample_rate'], fr=fr, is_mono=True)

    #########################
    #          Loss         #
    #########################
    # Adapt the answer here: http://stackoverflow.com/questions/41863814/kl-divergence-in-tensorflow
    def KL_divergence(self, dist_a, dist_b, name_scope='KL_Div'):
        return tf.reduce_mean(-tf.nn.softmax_cross_entropy_with_logits(logits=dist_a, labels=dist_b))

    def crossentropy(self, p_approx, p_true):
        return -tf.reduce_sum(tf.multiply(p_true, tf.log(p_approx)), 1)
    #########################
    #       Save/Load       #
    #########################
    @property
    def get_model_dir(self):
        if self.config['model_dir'] is None:
            return "{}_{}".format(
                self.config['dataset_name'], self.config['batch_size'])
        else:
            return self.config['model_dir']

    def load(self, ckpt_dir='/home/lzy/Code/HysiaSound/SceneClassification/checkpoint', mode='ckpt'):
        if self.param_G is not None:
            return self.load_from_npy()
        elif mode == 'ckpt':
            return self.load_from_ckpt(ckpt_dir)
        else:
            return self.load_from_pb()

    def save(self, checkpoint_dir, step):
        """ Checkpoint saver """
        model_name = "SoundNet3dbn5s.model"
        checkpoint_dir = os.path.join(checkpoint_dir, self.get_model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def save_pb(self):
        save_dir = self.config['checkpoint_dir']
        graph_pb_name = 'soundnet_gr.pb'
        frozen_pb_name = 'soundnet_fr.pb'
        tf.train.write_graph(self.sess.graph_def, save_dir, graph_pb_name, as_text=False)
        freeze_graph.freeze_graph(input_graph=os.path.join(save_dir, graph_pb_name),
                                  input_saver='',
                                  input_binary=True,
                                  input_checkpoint= os.path.join(save_dir, self.get_model_dir, 'SoundNet3dbn5s.model-15999'),
                                  output_node_names='SoundNet/retrain4/dense/BiasAdd',
                                  restore_op_name='save/restore_all',
                                  filename_tensor_name='save/Const:0',
                                  output_graph=os.path.join(save_dir, frozen_pb_name),
                                  clear_devices=False,
                                  initializer_nodes="")

    def load_from_ckpt(self, checkpoint_dir='/home/lzy/Code/HysiaSound/SceneClassification/checkpoint'):
        """ Checkpoint loader """
        print(" [*] Reading checkpoints...")

        checkpoint_dir = os.path.join(checkpoint_dir, self.get_model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            print(" [*] Success to read {}".format(ckpt_name))
            self.counter = int(ckpt_name.rsplit('-', 1)[-1])
            print(" [*] Start counter from {}".format(self.counter))
            return True
        else:
            print(" [*] Failed to find a checkpoint under {}".format(checkpoint_dir))
            return False

    def load_from_npy(self):
        if self.param_G is None: return False
        data_dict = self.param_G
        for key in data_dict:
            with tf.variable_scope(self.config['name_scope'] + '/' + key, reuse=True):
                for subkey in data_dict[key]:
                    try:
                        var = tf.get_variable(subkey)
                        self.sess.run(var.assign(data_dict[key][subkey]))
                        print('Assign pretrain model {} to {}'.format(subkey, key))
                    except:
                        print('Ignore {}'.format(key))

        self.param_G.clear()
        return True

    def load_from_pb(self):
        pb_path = os.path.join(self.config['checkpoint_dir'], self.config['pb_name'])
        self.graph = tf.Graph()
        with self.sess.as_default():
            with self.graph.as_default():
                self.graph_def = tf.GraphDef()
                with tf.gfile.GFile(pb_path, 'rb') as fid:
                    serialized_graph = fid.read()
                    self.graph_def.ParseFromString(serialized_graph)
                    tf.import_graph_def(self.graph_def, name='')
                    return True


    def load_dcase_label_from_csv(self, csv_file):
        labels = ['airport', 'shopping_mall', 'metro_station', 'street_pedestrian', 'public_square', 'street_traffic',
                  'tram', 'bus', 'metro', 'park']
        with open(csv_file, 'rt') as f:
            reader = csv.reader(f)
            next(reader, None)
            lis = list(reader)
            label_dict = {}
            for li in lis:
                # load data
                [na, lb, identifier, source] = li[0].split('\t')
                na = na.split('/')[1]
                label_dict[na] = labels.index(lb)
        return label_dict

    def dcase_data_augmentation(self, sound_data, labels, hop = 1, shuffle=False):
        sample_size = self.config['sample_size']
        augmented_data = []
        augmented_labels = []
        for i, sound_input in enumerate(sound_data):
            label = labels[i]
            data_len = len(sound_data[i])
            for j in range(0, data_len - sample_size + 1, hop):
                augmented_data.append(sound_data[i][j:j + sample_size])
                augmented_labels.append(label)
        if shuffle:
            return unison_shuffled_copies(augmented_data, augmented_labels)
        else:
            return augmented_data, augmented_labels


def main():
    args = parse_args()
    local_config['phase'] = args.phase

    # Setup visible device
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    # Make path
    if not os.path.exists(args.outpath):
        os.mkdir(args.outpath)

    # Load pre-trained model
    param_G = np.load(local_config['param_g_dir'], encoding='latin1').item()

    # Init. Session
    sess_config = tf.ConfigProto()
    sess_config.allow_soft_placement = True
    sess_config.gpu_options.allow_growth = True

    with tf.Session(config=sess_config) as session:
        # Build model
        model = SoundNet(session, config=local_config, param_G=param_G)
        model.retrain()


def parse_args():
    """ Parse input arguments """
    parser = argparse.ArgumentParser(description='SoundNet')

    parser.add_argument('-o', '--outpath', dest='outpath', help='output feature path. e.g., [output]', default='PretrainedModel/')

    parser.add_argument('-p', '--phase', dest='phase', help='demo or extract feature. e.g., [train, finetune, extract]',
                        default='finetune')

    parser.add_argument('-m', '--layer', dest='layer_min', help='start from which feature layer. e.g., [1]', type=int,
                        default=1)

    parser.add_argument('-x', dest='layer_max', help='end at which feature layer. e.g., [24]', type=int, default=None)

    parser.add_argument('-c', '--cuda', dest='cuda_device', help='which cuda device to use. e.g., [0]', default='0')

    feature_parser = parser.add_mutually_exclusive_group(required=False)
    feature_parser.add_argument('-s', '--save', dest='is_save', help='Turn on save mode. [False(default), True]',
                                action='store_true')
    parser.set_defaults(is_save=False)

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    main()
