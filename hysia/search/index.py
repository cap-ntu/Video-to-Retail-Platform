# @Time    : 12/11/18 3:54 PM
# @Author  : Huaizheng Zhang
# @Site    : zhanghuaizheng.info
# @File    : index.py

import os.path as osp
import glob
import pickle
import faiss
import numpy as np
from models.nlp.sentence import TF_Sentence


THIS_DIR = osp.dirname(osp.abspath(__file__))


class BasicIndex(object):
    '''
    Mainly for the image search. Will revise this part.
    '''

    def __init__(self, dimension, dir_path):
        self.dimension = dimension
        self.dir_path = dir_path
        self.database = None
        self.image_feature = None

        # for the second search
        self.temp_database = None

        self.res = faiss.StandardGpuResources()
        self.index_flat = faiss.IndexFlatL2(self.dimension)
        self.gpu_index_flat = faiss.index_cpu_to_gpu(self.res, 0, self.index_flat)

        # We need more index to reset
        self.res_2 = faiss.StandardGpuResources()
        self.index_flat_2 = faiss.IndexFlatL2(self.dimension)
        self.gpu_index_flat_2 = faiss.index_cpu_to_gpu(self.res, 0, self.index_flat)

    def index(self, all=True, feature='FEATURE', tv_name=None):

        '''

        :param all: index all data in the dir, suit for when the second of the search is to 1
        :param feature: which feature will be indexed
        :param tv_name: if none, will index all TVs
        :return:
        '''

        # TODO all parameter may have some bugs
        self.database = []

        if tv_name is not None:
            tv_file = '{}/{}_index.pkl'.format(self.dir_path, tv_name)

        else:
            tv_file = glob.glob('{}/*index.pkl'.format(self.dir_path))

        if isinstance(tv_file, str):
            tv_file = [tv_file]

        for i in tv_file:

            print(i)

            with open(i, 'rb') as f:
                data = pickle.load(f)
            self.database.extend(data)
        print('Finish loading {} samples'.format(len(self.database)))

        # index image feature
        self.image_feature = list()
        for i in self.database:
            # print(i)
            # TODO process no feature, array can not compare to string
            if isinstance(i[feature], str) and i[feature] == 'unknown_feature':
                i[feature] = np.zeros((512, ))
            self.image_feature.append(i[feature])
        # print(self.image_feature)
        self.image_feature = np.array(self.image_feature).astype(np.float32)
        if all == True:
            self.gpu_index_flat.add(self.image_feature)
            print('Finish loading {} image features'.format(self.gpu_index_flat.ntotal))

    def search(self, query, k, second='1'):
        '''

        :param query: query vector
        :param k: return how many results
        :param second: the first search, and can search more rounds
        :return: results and index
        '''
        assert self.dimension == len(query)

        query = np.expand_dims(query, axis=0)
        # query = np.stack((query, query))

        if second == '1':
            D, I = self.gpu_index_flat.search(query, k)
            results = [self.database[x] for x in np.squeeze(I)]
        else:
            D, I = self.gpu_index_flat_2.search(query, k)
            results = [self.temp_database[x] for x in np.squeeze(I)]
            self.gpu_index_flat_2.reset()

        return results, I, D

    def reset_GPU(self):
        self.gpu_index_flat.reset()
        self.gpu_index_flat_2.reset()


    def re_index(self, new_index):
        '''
        We use subtiltle to search first. After we get the first 30 results, we use image feature to re-index
        :param new_index:
        :return:
        '''
        idx = np.squeeze(new_index)

        # print(idx)
        # print(self.image_feature)

        new_index_feature = self.image_feature[idx, ]
        self.temp_database = np.array(self.database)[idx, ]

        # We need to reset the index
        self.gpu_index_flat_2.add(new_index_feature)

        print('Finish loading {} features'.format(self.gpu_index_flat_2.ntotal))


    @classmethod
    def init_size(cls, dir_path):
        '''
        When you do not know the dimension of the features, use this methods.

        :param dir_path: p
        :return:
        '''
        if not osp.isdir(dir_path):
            raise Exception('{} does not exist'.format(dir_path))

        dimension = cls.stat(dir_path)

        return cls(dimension, dir_path)

    @staticmethod
    def stat(dir_path):
        tv_file = glob.glob('{}/*.pkl'.format(dir_path))

        if len(tv_file) < 1:
            print("You have not finished indexing, please run indexer first")
        else:
            with open(tv_file[0], 'rb') as f:
                sample = pickle.load(f)[0]

        dimension = len(sample['FEATURE'])

        return dimension


class SubtitleIndex(BasicIndex):
    '''
    For senentence index
    '''
    def __init__(self, dimension, dir_path):
        super(SubtitleIndex, self).__init__(dimension, dir_path)
        self.sentence_feature = None
        self.sentence_model = TF_Sentence(osp.join(THIS_DIR, '../../weights/sentence/96e8f1d3d4d90ce86b2db128249eb8143a91db73'))

    def index(self, tv_name=None):
        self.database = []

        if tv_name is not None:
            tv_file = '{}/{}_index.pkl'.format(self.dir_path, tv_name)
        else:
            tv_file = glob.glob('{}/*index.pkl'.format(self.dir_path))

        for i in tv_file:
            with open(i, 'rb') as f:
                data = pickle.load(f)
            self.database.extend(data)
        print('Finish loading {} samples'.format(len(self.database)))

        self.sentence_feature = list()
        if osp.isfile('{}/sentence.pkl'.format(self.dir_path)):
            print('The sentence has been encoded')
            with open(self.dir_path + '/sentence.pkl', 'rb') as f:
                self.sentence_feature = pickle.load(f)

        else:
            print('Start to encode sentence')
            for i in self.database:
                sentence_feature = self.sentence_model.encode(i['SUBTITLE'])
                self.sentence_feature.append(sentence_feature)
            self.sentence_feature = np.array(self.sentence_feature)
            with open(self.dir_path + '/sentence.pkl', 'wb') as f:
                pickle.dump(self.sentence_feature, f)

        # TypeError: ndarrays must be of numpy.float32, and not float64.
        self.gpu_index_flat.add(self.sentence_feature.astype(np.float32))
        print('Finish loading {} sentence features'.format(self.gpu_index_flat.ntotal))

