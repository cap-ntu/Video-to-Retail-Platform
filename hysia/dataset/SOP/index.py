#!/usr/bin/env python
import os
import os.path as osp
import pickle
import cv2
from models.scene.detector import scene_visual
from PIL import Image


class SOP_indexer(object):
    def __init__(self, path):
        self.path = path
        self.__products = self.__load_products()
        self.scene_model = scene_visual('resnet50', '../../../weights/places365/{}.pth',
                                        '../../../weights/places365/categories.txt', 'cuda:0')
        self.index_image = 'IMAGE'
        self.index_feature = 'FEATURE'
        self.index_product = 'PRODUCT'

    def get_feature(self, img):
        temp = cv2.imread(img)
        q_tensor = Image.fromarray(cv2.cvtColor(temp, cv2.COLOR_BGR2RGB))
        q_vec = self.scene_model.extract_vec(q_tensor, True)
        return q_vec

    def __load_products(self):
        products = []
        if osp.isdir(self.path):
            for file in os.listdir(self.path):
                if osp.isdir(osp.join(self.path, file)):
                    if file != '.idea':
                        products.append(file)
            if products.__len__() == 12:
                return products
            else:
                print("Wrong product number, please check your dataset.")
                del products
        else:
            print("Wrong path, please check.")

    def get_products(self):
        return self.__products

    def index(self):
        filename = 'SOP_index.pkl'
        output = osp.join(self.path, filename)
        if osp.isfile(output):
            print('The dataset has already been indexed!')
        else:
            print("Start Indexing ...")
            database = []
            for product in self.__products:
                print("    Indexing {} ...".format(product))
                product_path = osp.join(self.path, product)
                imgs = os.listdir(product_path)
                for img in imgs:
                    print("    Indexing image {} ...".format(img), end='')
                    img_path = osp.join(product_path, img)
                    img_feature = self.get_feature(img_path)
                    sample = {
                        self.index_image: img_path,
                        self.index_feature: img_feature,
                        self.index_product: product.split('_')[0]
                    }
                    database.append(sample)
                    print('  Done.')
            with open(output, 'wb') as f:
                print('Dumping ...')
                pickle.dump(database, f)
                print('Finish indexing ')
                del database
