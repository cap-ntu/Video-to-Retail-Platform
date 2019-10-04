# Copyright 2018 Antoine Miech All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.



import torch as th
from torch.utils.data import Dataset
import numpy as np 
import os    
import math    
import random

class LSMDC(Dataset):
    """LSMDC dataset."""

    def __init__(self, clip_path, text_features, audio_features, flow_path, face_path, coco_visual_path='../X_train2014_resnet152.npy' ,coco_text_path='../w2v_coco_train2014_1.npy', coco=True, max_words=30, video_features_size=2048, text_features_size=300, audio_features_size=128, face_features_size=128, flow_features_size=1024,verbose=False):
        """
        Args:
        """

        self.visual_features = np.load(clip_path)
        self.flow_features = np.load(flow_path)
        self.face_features = np.load(face_path)
        self.audio_features = np.load(audio_features)
        self.text_features = np.load(text_features)
        

        audio_sizes = map(len,self.audio_features)
        self.audio_sizes = np.array(audio_sizes)

        self.video_features_size = video_features_size
        self.text_features_size = text_features_size
        self.audio_features_size = audio_features_size
        self.flow_features_size = flow_features_size
        self.face_features_size = face_features_size
        
        self.max_len_text = max_words
        
        text_sizes = map(len,self.text_features)
        self.text_sizes = np.array(text_sizes)
        self.text_sizes = self.text_sizes.astype(int)
        
        mask = self.text_sizes > 0

        self.text_features = self.text_features[mask]
        self.text_sizes = self.text_sizes[mask]
        self.visual_features = self.visual_features[mask]
        self.flow_features = self.flow_features[mask]
        self.face_features = self.face_features[mask]
        self.audio_features = self.audio_features[mask]
        self.audio_sizes = self.audio_sizes[mask]
        self.audio_sizes.astype(int) 
        
        self.max_len_audio = max(self.audio_sizes)
       
        audio_tensors = np.zeros((len(self.audio_features),
                        max(self.audio_sizes), self.audio_features[0].shape[1]))

        for j in range(len(self.audio_features)):
            audio_tensors[j,0:self.audio_sizes[j],:] = self.audio_features[j]
        

        if coco:
            # adding coco data
            coco_visual = np.load(coco_visual_path)
            coco_text = np.load(coco_text_path)
            

            self.n_lsmdc = len(self.visual_features)
            self.n_coco = len(coco_visual)
           
            self.visual_features = np.concatenate((self.visual_features, coco_visual), axis=0)
            self.text_features = np.concatenate((self.text_features, coco_text), axis=0)

            text_sizes = map(len,self.text_features)
            self.text_sizes = np.array(text_sizes)
            self.text_sizes = self.text_sizes.astype(int)
            self.coco_ind = np.zeros((self.n_lsmdc+self.n_coco))
            self.coco_ind[self.n_lsmdc:] = 1
        else:
            self.n_lsmdc = len(self.visual_features)
            self.coco_ind = np.zeros((self.n_lsmdc))

      
        text_tensors = np.zeros((len(self.text_features),
                        max_words, self.text_features[0].shape[1]))


        for j in range(len(self.text_features)):
            if self.text_sizes[j] > max_words:                
                text_tensors[j] = self.text_features[j][0:max_words,:]
            else:
                text_tensors[j,0:self.text_sizes[j],:] = self.text_features[j] 
                
        self.text_features = th.from_numpy(text_tensors)
        self.text_features = self.text_features.float()

        self.audio_features = th.from_numpy(audio_tensors)
        self.audio_features = self.audio_features.float()

        self.flow_features = th.from_numpy(self.flow_features)
        self.flow_features = self.flow_features.float()

        self.visual_features = th.from_numpy(self.visual_features)
        self.visual_features = self.visual_features.float()

        self.face_features = th.from_numpy(self.face_features)
        self.face_features = self.face_features.float()
        
    def __len__(self):
        return len(self.text_features)

    def __getitem__(self, idx):

        face_ind = 1

        if idx >= self.n_lsmdc:
            flow = th.zeros(self.flow_features_size)
            face = th.zeros(self.face_features_size)
            audio = th.zeros(self.audio_features.size()[1],self.audio_features_size)
            audio_size = 1
            face_ind = 0
        else:
            flow = self.flow_features[idx]
            face = self.face_features[idx]
            audio = self.audio_features[idx]
            audio_size = self.audio_sizes[idx]

            if th.sum(face) == 0:
                face_ind = 0
        return {'video': self.visual_features[idx], 
                'flow': flow,
                'face': face,
                'text': self.text_features[idx],
                'audio': audio,
                'audio_size': audio_size,
                'coco_ind': self.coco_ind[idx],
                'face_ind': face_ind,
                'text_size': self.text_sizes[idx]
                }


    def getVideoFeatureSize(self):
        return self.video_features_size
    def getTextFeatureSize(self):
        return self.text_features_size
    def getAudioFeatureSize(self):
        return self.audio_features_size
    def getFlowFeatureSize(self):
        return self.flow_features_size
    def getText(self):
        return self.text_features
 

    def shorteningTextTensor(self,text_features, text_sizes):
        m = int(max(text_sizes))
        return text_features[:,0:m,:]

class LSMDC_qcm(Dataset):
    """LSMDC dataset."""

    def __init__(self, clip_path, text_features, audio_features, flow_path, face_path, max_words=30, video_features_size=2048, text_features_size=300, audio_features_size=128, face_features_size=128, flow_features_size=1024):
        """
        Args:
        """
        self.visual_features = np.load(clip_path)
        self.flow_features = np.load(flow_path)
        self.face_features = np.load(face_path)
        self.audio_features = np.load(audio_features)
        self.text_features = np.load(text_features)
        print 'features loaded'

        audio_sizes = map(len,self.audio_features)
        self.audio_sizes = np.array(audio_sizes)

        self.video_features_size = video_features_size
        self.text_features_size = text_features_size
        self.audio_features_size = audio_features_size
        self.flow_features_size = flow_features_size
        self.face_features_size = face_features_size
        
        self.max_len_text = max_words
        
        text_sizes = map(len,self.text_features)
        self.text_sizes = np.array(text_sizes)
        self.text_sizes = self.text_sizes.astype(int)
        
       
        self.max_len_audio = max(self.audio_sizes)
        

        audio_tensors = np.zeros((len(self.audio_features),
                        max(self.audio_sizes), self.audio_features[0].shape[1]))

        for j in range(len(self.audio_features)):
            audio_tensors[j,0:self.audio_sizes[j],:] = self.audio_features[j]

        text_tensors = np.zeros((len(self.text_features),
                        max_words, self.text_features[0].shape[1]))


        for j in range(len(self.text_features)):
            if self.text_sizes[j] > max_words:                
                text_tensors[j] = self.text_features[j][0:max_words,:]
            else:
                text_tensors[j,0:self.text_sizes[j],:] = self.text_features[j] 
                
        self.text_features = th.from_numpy(text_tensors)
        self.text_features = self.text_features.float()

        self.audio_features = th.from_numpy(audio_tensors)
        self.audio_features = self.audio_features.float()

        self.flow_features = th.from_numpy(self.flow_features)
        self.flow_features = self.flow_features.float()

        self.visual_features = th.from_numpy(self.visual_features)
        self.visual_features = self.visual_features.float()

        self.face_features = th.from_numpy(self.face_features)
        self.face_features = self.face_features.float()


    def __len__(self):
        return len(self.visual_features)
    


    def __getitem__(self, tidx):
    
        idx, idx2 = tidx

        face_ind = 1

        flow = self.flow_features[idx]
        face = self.face_features[idx]
        audio = self.audio_features[idx]
        audio_size = self.audio_sizes[idx]

        if th.sum(face) == 0:
            face_ind = 0

        return {'video': self.visual_features[idx], 
                'flow': flow,
                'face': face,
                'text': self.text_features[idx2],
                'audio': audio,
                'face_ind': face_ind,
                'audio_size': audio_size,
                'text_size': self.text_sizes[idx2]
                }


    def getVideoFeatureSize(self):
        return self.video_features_size
    def getTextFeatureSize(self):
        return self.text_features_size
    def getAudioFeatureSize(self):
        return self.audio_features_size
    def getFlowFeatureSize(self):
        return self.flow_features_size


    def shorteningTextTensor(self,text_features, text_sizes):
        m = int(max(text_sizes))
        return text_features[:,0:m,:]

