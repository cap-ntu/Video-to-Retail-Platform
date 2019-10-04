#!/usr/bin/env python
# @Time    : 9/10/18 10:20 AM
# @Author  : Huaizheng Zhang
# @Site    : zhanghuaizheng.info
# @File    : relationship.py


from PIL import Image
import os
import os.path as osp
import pickle
import numpy as np
import numpy.random as npr
import json
import cv2
import torch
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
import yaml


class pre_VRD(object):
    '''

    This is for preprocessing the vrd relationship dataset.
    '''

    def __init__(self, opts, image_set='train'):
        super(pre_VRD, self).__init__()
        self._name = image_set
        self.opts = opts
        self._image_set = image_set
        self._data_path = osp.join(self.opts['dir'], 'sg_dataset', 'sg_{}_images'.format(image_set))
        # load category names and annotations
        annotation_dir = osp.join(self.opts['dir'], 'json_dataset')

        ann_file_path = osp.join(annotation_dir, self.name + '.json')
        self.annotations = json.load(open(ann_file_path))

        # categories
        obj_cats = json.load(open(osp.join(annotation_dir, 'objects.json')))
        self._object_classes = tuple(['__background__'] + obj_cats)
        pred_cats = json.load(open(osp.join(annotation_dir, 'predicates.json')))
        self._predicate_classes = tuple(['__background__'] + pred_cats)
        self._object_class_to_ind = dict(zip(self.object_classes, range(self.num_object_classes)))
        self._predicate_class_to_ind = dict(zip(self.predicate_classes, range(self.num_predicate_classes)))

        self.cfg_key = image_set.split('_')[0]
        self._feat_stride = None
        self._rpn_opts = None

    def imdb(self, cache):
        items = list()
        for index in range(len(self.annotations)):
            item = {}
            target_scale = self.opts[self.cfg_key]['SCALES'][
                npr.randint(0, high=len(self.opts[self.cfg_key]['SCALES']))]
            item['target_scale'] = target_scale
            img = cv2.imread(osp.join(self._data_path, self.annotations[index]['path']))
            if img is None:
                continue

            item['path'] = self.annotations[index]['path']
            item['max_size'] = self.opts[self.cfg_key]['MAX_SIZE']

            _annotation = self.annotations[index]

            gt_boxes_object = np.zeros((len(_annotation['objects']), 5))
            gt_boxes_object[:, 0:4] = np.array([obj['bbox'] for obj in _annotation['objects']], dtype=np.float)
            gt_boxes_object[:, 4] = np.array([obj['class'] for obj in _annotation['objects']])
            item['gt_boxes'] = gt_boxes_object

            gt_relationships = np.zeros([len(_annotation['objects']), (len(_annotation['objects']))], dtype=np.long)
            for rel in _annotation['relationships']:
                gt_relationships[rel['sub_id'], rel['obj_id']] = rel['predicate']
            item['relations'] = gt_relationships
            items.append(item)
        with open(cache, 'wb') as f:
            pickle.dump(items, f)
        print("================================================")
        print("Finish get {} imdb".format(len(items)))
        return items

    @property
    def voc_size(self):
        return len(self.idx2word)

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(i)

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        # Example image path for index=119993:
        #   images/train2014/COCO_train2014_000000119993.jpg
        file_name = self.annotations[index]['path']
        image_path = osp.join(self._data_path, file_name)
        assert osp.exists(image_path), \
            'Path does not exist: {}'.format(image_path)
        return image_path

    @property
    def name(self):
        return self._name

    @property
    def num_object_classes(self):
        return len(self._object_classes)

    @property
    def num_predicate_classes(self):
        return len(self._predicate_classes)

    @property
    def object_classes(self):
        return self._object_classes

    @property
    def predicate_classes(self):
        return self._predicate_classes


class VRD(data.Dataset):
    '''
    This is a pytorch dataloader for vrd dataset.

    '''

    def __init__(self, opt, imdb, max_class, train_test):
        super(VRD, self).__init__()
        self.opts = opt
        self.imdb = imdb
        self.max_class = max_class

        self._image_set = train_test
        self._data_path = osp.join(self.opts['dir'], 'sg_dataset', 'sg_{}_images'.format(train_test))

        # image transformation
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

        annotation_dir = osp.join(self.opts['dir'], 'json_dataset')

        # categories
        obj_cats = json.load(open(osp.join(annotation_dir, 'objects.json')))
        self._object_classes = tuple(['__background__'] + obj_cats)
        pred_cats = json.load(open(osp.join(annotation_dir, 'predicates.json')))
        self._predicate_classes = tuple(['__background__'] + pred_cats)
        self._object_class_to_ind = dict(zip(self.object_classes, range(self.num_object_classes)))
        self._predicate_class_to_ind = dict(zip(self.predicate_classes, range(self.num_predicate_classes)))

    def __len__(self):
        return len(self.imdb)

    def __getitem__(self, index):

        target_scale = self.imdb[index]['target_scale']
        img = cv2.imread(osp.join(self._data_path, self.imdb[index]['path']))

        img, im_scale = self._image_resize(img, target_scale, self.imdb[index]['max_size'])

        im_info = np.array([img.shape[0], img.shape[1], im_scale], dtype=np.float)
        im_data = Image.fromarray(img)

        if self.transform is not None:
            im_data = self.transform(im_data)

        self.imdb[index]['gt_boxes'][:, 0:4] = self.imdb[index]['gt_boxes'][:, 0:4] * im_scale

        gt_boxes = self.imdb[index]['gt_boxes']
        num_boxes = len(gt_boxes)

        relations = self.imdb[index]['relations']

        return im_data, im_info, gt_boxes, num_boxes, relations

    @staticmethod
    def collate(batch):
        batch_item = {}

        return batch_item

    def _image_resize(self, im, target_size, max_size):
        """Builds an input blob from the images in the roidb at the specified
        scales.
        """
        im_shape = im.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > max_size:
            im_scale = float(max_size) / float(im_size_max)
        im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_LINEAR)

        return im, im_scale

    @property
    def num_object_classes(self):
        return len(self._object_classes)

    @property
    def num_predicate_classes(self):
        return len(self._predicate_classes)

    @property
    def object_classes(self):
        return self._object_classes

    @property
    def predicate_classes(self):
        return self._predicate_classes


if __name__ == '__main__':

    imdb_name = "vrd_train"
    imdbval_name = "vrd_test"

    with open('cfgs/vrd.yml', 'r') as f:
        data_opts = yaml.load(f)

    imdb_train = 'cache' + imdb_name + '.pkl'
    if os.path.isfile(imdb_train):
        print("===========================")
        print("imdb has prepared")
        with open(imdb_train, 'rb') as f:
            train_set = pickle.load(f)

    else:
        pre_train = pre_VRD(data_opts, 'train')
        train_set = pre_train.imdb(imdb_train)
    train_size = len(train_set)
    print("===========================")
    print("We have {} to train".format(train_size))

    train_dataset = VRD(data_opts, train_set, data_opts['MAX_NUM_GT_BOXES'], 'train')

    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=False)
