# @Time    : 22/11/18 3:16 PM
# @Author  : Huaizheng Zhang
# @Site    : zhanghuaizheng.info
# @File    : pretrained_imagenet.py

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms as trn
from PIL import Image
from cirtorch.layers.pooling import MAC, SPoC, GeM, RMAC
from cirtorch.layers.normalization import L2N
from cirtorch.networks.imageretrievalnet import ImageRetrievalNet, extract_vectors


class Img2Vec(object):
    '''
    refer https://github.com/christiansafka/img2vec/blob/master/img_to_vec.py
    '''

    def __init__(self, cuda=False, model='resnet50', layer='default', layer_output_size=512):
        """ Img2Vec
        :param cuda: If set to True, will run forward pass on GPU
        :param model: String name of requested model
        :param layer: String or Int depending on model.  See more docs: https://github.com/christiansafka/img2vec.git
        :param layer_output_size: Int depicting the output size of the requested layer
        """
        self.device = torch.device("cuda" if cuda else "cpu")
        self.layer_output_size = layer_output_size
        self.model, self.extraction_layer = self._get_model_and_layer(model, layer)

        self.model = self.model.to(self.device)

        self.model.eval()

        self.centre_crop = trn.Compose([
            trn.Resize((256, 256)),
            trn.CenterCrop(224),
            trn.ToTensor(),
            trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # TODO extract conv_feature
        # TODO all networks can support whitening or something
        '''
        build sub_net
        the parameter can be adjusted
        '''

        self.features = list(self.model.children())[:-2]
        self.pooling = MAC()
        self.dim = 2048
        self.whiten = None
        meta = {'architecture': model, 'pooling': self.pooling,
                'whitening': self.whiten,
                'outputdim': self.dim}
        self.vector_model = ImageRetrievalNet(self.features, self.pooling, self.whiten, meta)

    def get_vec(self, img, tensor=False):
        """ Get vector embedding from PIL image
        :param img: PIL Image
        :param tensor: If True, get_vec will return a FloatTensor instead of Numpy array
        :returns: Numpy ndarray
        """
        image = self.centre_crop(img).unsqueeze(0).to(self.device)

        my_embedding = torch.zeros(1, self.layer_output_size, 1, 1)

        def copy_data(m, i, o):
            my_embedding.copy_(o.data)

        h = self.extraction_layer.register_forward_hook(copy_data)
        h_x = self.model(image)
        h.remove()

        if tensor:
            return my_embedding
        else:
            return my_embedding.numpy()[0, :, 0, 0]

    def extract_vec(self, image_path, tensor=False):
        # TODO only support resnet50
        if not tensor:
            img = Image.open(image_path)
        else:
            img = image_path
        input_img = torch.FloatTensor(self.centre_crop(img).unsqueeze(0)).to(self.device)
        vec = self.vector_model.forward(input_img)
        return vec.cpu().data.squeeze().numpy()

    def _get_model_and_layer(self, model_name, layer):
        """ Internal method for getting layer from model
        :param model_name: model name such as 'resnet-18'
        :param layer: layer as a string for resnet-18 or int for alexnet
        :returns: pytorch model, selected layer
        """
        if model_name == 'resnet-18':
            model = models.resnet18(pretrained=True)
            if layer == 'default':
                layer = model._modules.get('avgpool')
                self.layer_output_size = 512
            else:
                layer = model._modules.get(layer)

            return model, layer

        elif model_name == 'alexnet':
            model = models.alexnet(pretrained=True)
            if layer == 'default':
                layer = model.classifier[-2]
                self.layer_output_size = 4096
            else:
                layer = model.classifier[-layer]

            return model, layer

        elif model_name == 'resnet50':
            # TODO only support resnet50
            model = models.resnet50(pretrained=True)
            # print(model)
            # if layer == 'default':
            #     layer = model[-2]
            #     self.layer_output_size = 2048
            # else:
            #     layer = model.classifier[-layer]
            layer = None
            return model, layer

        else:
            raise KeyError('Model %s was not found' % model_name)
