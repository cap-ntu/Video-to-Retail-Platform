import pickle
import os
import os.path as osp
import cv2
from models.scene.detector import scene_visual
from PIL import Image


class SP_indexer(object):
    def __init__(self, path):
        self.path = path
        self.__products = self.__load_products()
        self.__scenes = self.__load_scenes()

        self.scene_model = scene_visual('resnet50', '../../../weights/places365/{}.pth',
                                        '../../../weights/places365/categories.txt', 'cuda:0')

        self.index_image = 'IMAGE'
        self.index_feature = 'FEATURE'
        self.index_description = 'DESCRIPTION'
        self.index_scene = 'SCENE'
        self.index_product = 'PRODUCT'

    def get_feature(self, img):
        temp = cv2.imread(img)
        q_tensor = Image.fromarray(cv2.cvtColor(temp, cv2.COLOR_BGR2RGB))
        q_vec = self.scene_model.extract_vec(q_tensor, True)
        return q_vec

    def __load_products(self):
        products = {}
        product_path = osp.join(self.path, 'Products')
        print('Loading Products ... ', end='')
        for file in os.listdir(product_path):
            products[file] = []
        print('Done.')
        return products

    def get_products(self):
        return self.__products

    def __load_scenes(self):
        scenes = []
        scene_path = osp.join(self.path, 'Scenes')
        print('Loading Scenes ... ', end='')
        for file in os.listdir(scene_path):
            scene_name, _ = file.split('.')
            scenes.append(scene_name)
        print('Done.')
        return scenes

    def get_scenes(self):
        return self.__scenes

    def relation(self):
        scene_path = osp.join(self.path, 'Scenes')
        files = os.listdir(scene_path)
        print('Building relationship ... ', end='')
        for file in files:
            scene_name, _ = file.split('.')
            file_path = osp.join(scene_path, file)
            with open(file_path, 'r') as scene_file:
                scene_data = scene_file.readlines()
                for line in scene_data:
                    product = line.strip()
                    if product in self.__products:
                        self.__products[product].append(scene_name)
                    else:
                        print('Error.')
                        print('Unknown record, please check your dataset.')
        print('Done')

    def index(self):
        filename = 'ScenesProducts_index.pkl'
        output = osp.join(self.path, filename)
        if osp.isfile(output):
            print('This dataset has been indexed!')
        else:
            print('Indexing ... ')
            self.relation()
            database = []
            products_path = osp.join(self.path, 'Products')
            products = os.listdir(products_path)
            for product in products:
                print('Indexing {} ...'.format(product), end='')
                product_path = osp.join(products_path, product)
                files = os.listdir(product_path)
                description = None
                image_list = []
                for file in files:
                    name, ftype = file.split('.')
                    if ftype == 'txt':
                        file_path = osp.join(product_path, file)
                        with open(file_path, 'r') as description_file:
                            description = description_file.readlines()[0].strip()
                    elif ftype == 'jpg':
                        img_path = osp.join(product_path, file)
                        image_list.append(img_path)
                for image in image_list:
                    feature = self.get_feature(image)
                    sample = {self.index_image: image,
                              self.index_feature: feature,
                              self.index_scene: self.__products[product],
                              self.index_product: product,
                              self.index_description: description}
                    database.append(sample)
                print(' Done.')
            with open(output, 'wb') as f:
                print('Dumping ...')
                pickle.dump(database, f)
                print('Finish indexing.')
                del database
