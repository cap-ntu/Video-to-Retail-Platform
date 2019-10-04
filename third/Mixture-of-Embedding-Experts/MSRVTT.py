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
import pickle

class MSRVTT(Dataset):
    """LSMDC dataset."""

    def __init__(self, visual_features, flow_features, text_features, audio_features,
            face_features, train_list, test_list, coco_visual_path='data/X_train2014_resnet152.npy',
            coco_text_path='data/w2v_coco_train2014_1.npy',coco=True, max_words=30,verbose=False):
        """
        Args:
        """
        self.max_words = max_words
        print 'loading data ...'

        with open(train_list) as f:
            self.train_list = f.readlines()

        self.train_list = [x.strip() for x in self.train_list]

        with open(test_list) as f:
            self.test_list = f.readlines()

        self.test_list = [x.strip() for x in self.test_list]


        pickle_in = open(visual_features,'rb')
        self.visual_features = pickle.load(pickle_in)
  
        pickle_in = open(flow_features,'rb')
        self.flow_features = pickle.load(pickle_in)

        pickle_in = open(audio_features,'rb')
        self.audio_features = pickle.load(pickle_in)

        pickle_in = open(text_features,'rb')
        self.text_features = pickle.load(pickle_in)

        pickle_in = open(face_features,'rb')
        self.face_features = pickle.load(pickle_in)

        self.coco = coco

        if coco:
            # adding coco data
            self.coco_visual = np.load(coco_visual_path)
            self.coco_text = np.load(coco_text_path)
            
            self.n_MSR = len(self.train_list)
            self.n_coco = len(self.coco_visual)
            
            self.coco_ind = np.zeros((self.n_MSR+self.n_coco))
            self.coco_ind[self.n_MSR:] = 1

        else:
            self.n_MSR = len(self.train_list)
            self.coco_ind = np.zeros((self.n_MSR))
            self.n_coco = 0
 

        # computing retrieval

        self.video_retrieval = np.zeros((len(self.test_list),2048))
        self.flow_retrieval = np.zeros((len(self.test_list),1024))
        self.audio_retrieval = np.zeros((len(self.test_list), max_words, 128))
        self.face_retrieval = np.zeros((len(self.test_list), 128))
        self.text_retrieval = np.zeros((len(self.test_list), max_words, 300))
        self.face_ind_retrieval = np.ones((len(self.test_list)))
        
        for i in range(len(self.test_list)):
            self.video_retrieval[i] = self.visual_features[self.test_list[i]]
            self.flow_retrieval[i] = self.flow_features[self.test_list[i]]
            
            if len(self.face_features[self.test_list[i]]) > 0:
                self.face_retrieval[i] = self.face_features[self.test_list[i]]
       
            if np.sum(self.face_retrieval[i]) == 0:
                self.face_ind_retrieval[i] = 0

            la = len(self.audio_features[self.test_list[i]])
            self.audio_retrieval[i,:min(max_words,la),:] = self.audio_features[self.test_list[i]][:min(max_words,la)]
            
            lt = len(self.text_features[self.test_list[i]][0])
            self.text_retrieval[i,:min(max_words,lt),:] = self.text_features[self.test_list[i]][0][:min(max_words,lt)]

        
        self.video_retrieval = th.from_numpy(self.video_retrieval).float()
        self.flow_retrieval = th.from_numpy(self.flow_retrieval).float()
        self.audio_retrieval = th.from_numpy(self.audio_retrieval).float()
        self.face_retrieval = th.from_numpy(self.face_retrieval).float()
        self.text_retrieval = th.from_numpy(self.text_retrieval).float()
        
        print 'done'

    def collate_data(self, data):
        video_tensor = np.zeros((len(data), 2048))
        flow_tensor = np.zeros((len(data), 1024))
        face_tensor = np.zeros((len(data), 128))
        audio_tensor = np.zeros((len(data), self.max_words,128))
        text_tensor = np.zeros((len(data), self.max_words, 300))
        coco_ind = np.zeros((len(data)))
        face_ind = np.zeros((len(data)))

        for i in range(len(data)):

            coco_ind[i] = data[i]['coco_ind']
            face_ind[i] = data[i]['face_ind']
            video_tensor[i] = data[i]['video']
            flow_tensor[i] = data[i]['flow']

            if len(data[i]['face']) > 0:
                face_tensor[i] = data[i]['face']
            
            la = len(data[i]['audio'])
            audio_tensor[i,:min(la,self.max_words), :] = data[i]['audio'][:min(self.max_words,la)]

            lt = len(data[i]['text'])
            text_tensor[i,:min(lt,self.max_words), :] = data[i]['text'][:min(self.max_words,lt)]


        return {'video': th.from_numpy(video_tensor).float(),
                'flow': th.from_numpy(flow_tensor).float(),
                'face': th.from_numpy(face_tensor).float(),
                'coco_ind': coco_ind,
                'face_ind': face_ind,
                'text': th.from_numpy(text_tensor).float(),
                'audio': th.from_numpy(audio_tensor).float()}


    def __len__(self):
        return len(self.coco_ind)

    def __getitem__(self, idx):

        face_ind = 1
        if idx < self.n_MSR:
            vid = self.train_list[idx]
            text = self.text_features[vid]
            r = random.randint(0, len(text)-1)
            text = text[r]
            flow = self.flow_features[vid]
            audio = self.audio_features[vid]
            video = self.visual_features[vid]
            face = self.face_features[vid]

            if np.sum(face) == 0:
                face_ind = 0
        elif self.coco:
            video = self.coco_visual[idx-self.n_MSR]
            text = self.coco_text[idx-self.n_MSR]
            audio = th.zeros(1,128)
            flow = th.zeros(1024)
            face = th.zeros(128)
            face_ind = 0

        return {'video': video, 
                'flow': flow,
                'face': face,
                'text': text,
                'coco_ind': self.coco_ind[idx],
                'face_ind': face_ind,
                'audio': audio
                }

    def getRetrievalSamples(self):
        return {'video': self.video_retrieval, 
                'flow': self.flow_retrieval,
                'text': self.text_retrieval,
                'face': self.face_retrieval,
                'face_ind': self.face_ind_retrieval,
                'audio': self.audio_retrieval}

