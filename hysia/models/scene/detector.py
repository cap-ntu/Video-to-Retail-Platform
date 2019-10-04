#!/usr/bin/env python
# @Time    : 19/10/18 5:13 PM
# @Author  : Huaizheng Zhang
# @Site    : zhanghuaizheng.info
# @File    : detector.py


import torch
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
from PIL import Image
from cirtorch.layers.pooling import MAC, SPoC, GeM, RMAC
from cirtorch.layers.normalization import L2N
from cirtorch.networks.imageretrievalnet import ImageRetrievalNet, extract_vectors


class scene_visual(object):
    '''
    This is a scene recognition model trained on place365 dataset.
    '''

    def __init__(self, network, model_file, category_file, device):
        '''
        This is for init a pytorch model

        :param network:
        :param model_file:
        :param category_file:

        '''
        self.network = network
        self.model = models.__dict__[self.network](num_classes=365)
        self.model_file = model_file.format(self.network)
        checkpoint = torch.load(self.model_file, map_location=lambda storage, loc: storage)
        state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
        self.model.load_state_dict(state_dict)
        self.device = device
        self.model.to(self.device)
        self.model.eval()

        self.file_name = category_file
        classes = list()
        with open(self.file_name) as class_file:
            for line in class_file:
                classes.append(line.strip().split(' ')[0][3:])

        self.classes = tuple(classes)

        self.centre_crop = trn.Compose([
            trn.Resize((256, 256)),
            trn.CenterCrop(224),
            trn.ToTensor(),
            trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        print('==============Finish loading {}==============='.format(network))

        # extract conv_feature
        # build sub_net
        # TODO the parameter can be adjusted
        self.features = list(self.model.children())[:-2]
        self.pooling = MAC()
        self.dim = 2048
        self.whiten = None
        meta = {'architecture': self.network, 'pooling': self.pooling,
                'whitening': self.whiten,
                'outputdim': self.dim}
        self.vector_model = ImageRetrievalNet(self.features, self.pooling, self.whiten, meta)

    def detect(self, image_path, tensor=False):
        '''

        :param image_path:
        :param tensor:
        :return: dict:
        '''
        output_dict = {}
        scenes = list()
        confidences = list()
        if not tensor:
            img = Image.open(image_path)
        else:
            img = image_path
        input_img = torch.FloatTensor(self.centre_crop(img).unsqueeze(0)).to(self.device)

        logit = self.model.forward(input_img)
        h_x = F.softmax(logit, 1).data.squeeze()
        probs, idx = h_x.sort(0, True)

        # print('{} prediction on {}'.format(self.network, image_path))
        # output the prediction
        for i in range(0, 5):
            # print('{:.3f} -> {}'.format(probs[i], self.classes[idx[i]]))
            scenes.append(self.classes[idx[i]])
            confidences.append(round(probs[i].item(), 4))
        output_dict['scene'] = scenes
        output_dict['score'] = confidences

        return output_dict

    def extract_vec(self, image_path, tensor=False):
        if not tensor:
            img = Image.open(image_path)
        else:
            img = image_path
        input_img = torch.FloatTensor(self.centre_crop(img).unsqueeze(0)).to(self.device)
        vec = self.vector_model.forward(input_img)
        return vec.cpu().data.squeeze().numpy()
