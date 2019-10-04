# Author: wang yongjie
# Email : yongjie.wang@ntu.edu.sg
import os
import torch
import numpy as np
import cv2
import pickle as pk
from models.face_pytorch.recognition.model_irse import *

class extractor(object):
    def __init__(self, pth_file, in_dir, save_name, cast):

        self.pth_file = pth_file
        self.in_dir = in_dir
        self.save_name = save_name
        self.cast = cast

    def init_pytorch_env(self):
        # load backbone from a checkpoint
        self.backbone = IR_50([112, 112])
        self.backbone.load_state_dict(torch.load(self.pth_file))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.backbone.to(self.device)

        # extract features
        self.backbone.eval() # set to evaluation mode

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
                if isinstance(img, type(None)):
                    print(filename + ' is not existing')
                    continue
                img = img[...,::-1] # BGR to RGB
                img = img.swapaxes(1, 2).swapaxes(0, 1)
                img = np.reshape(img, [1, 3, 112, 112])
                img = np.array(img, dtype = np.float32)
                tmp['IMAGE'] = filename
                img = (img - 127.5) / 128
                img = torch.from_numpy(img)

                feature = self.backbone(img.to(self.device)).cpu()
                feature = feature[0].cpu().detach().numpy()

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

