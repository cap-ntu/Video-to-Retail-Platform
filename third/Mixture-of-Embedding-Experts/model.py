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


from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch as th
from loupe import NetVLAD
import numpy as np
from torch.autograd import Function

class Net(nn.Module):
    def __init__(self, video_modality_dim, text_dim, audio_cluster=8,  text_cluster=32):
        super(Net, self).__init__()
        
        self.audio_pooling = NetVLAD(feature_size=video_modality_dim['audio'][1],
                cluster_size=audio_cluster)
        self.text_pooling = NetVLAD(feature_size=text_dim,
                cluster_size=text_cluster)

        self.mee = MEE(video_modality_dim, self.text_pooling.out_dim)

    def forward(self, text, video, ind, conf=True):

        aggregated_video = {}
        
        aggregated_video['audio'] = self.audio_pooling(video['audio'])
        aggregated_video['face'] = video['face'] 
        aggregated_video['motion'] = video['motion']
        aggregated_video['visual'] = video['visual']
        
        text = self.text_pooling(text)

        return self.mee(text, aggregated_video, ind, conf)


class MEE(nn.Module):
    def __init__(self, video_modality_dim, text_dim):
        super(MEE, self).__init__()

        m = video_modality_dim.keys()

        self.m = m
        
        self.video_GU = nn.ModuleList([Gated_Embedding_Unit(video_modality_dim[m[i]][0],
            video_modality_dim[m[i]][1]) for i in range(len(m))])

        self.text_GU = nn.ModuleList([Gated_Embedding_Unit(text_dim,
            video_modality_dim[m[i]][1]) for i in range(len(m))])

        self.moe_fc = nn.Linear(text_dim, len(video_modality_dim))
    

    def forward(self, text, video, ind, conf=True):

        text_embd = {}

        for i, l in enumerate(self.video_GU):
            video[self.m[i]] = l(video[self.m[i]])

        for i, l in enumerate(self.text_GU):
            text_embd[self.m[i]] = l(text)


        #MOE weights computation + normalization ------------
        moe_weights = self.moe_fc(text)
        moe_weights = F.softmax(moe_weights, dim=1)

        available_m = np.zeros(moe_weights.size())

        i = 0
        for m in video:
            available_m[:,i] = ind[m]
            i += 1

        available_m = th.from_numpy(available_m).float()
        available_m = Variable(available_m.cuda())

        moe_weights = available_m*moe_weights

        norm_weights = th.sum(moe_weights, dim=1)
        norm_weights = norm_weights.unsqueeze(1)
        moe_weights = th.div(moe_weights, norm_weights)

        #MOE weights computation + normalization ------ DONE

        if conf:
            conf_matrix = Variable(th.zeros(len(text),len(text)).cuda())
            i = 0
            for m in video:
                video[m] = video[m].transpose(0,1)
                conf_matrix += moe_weights[:,i:i+1]*th.matmul(text_embd[m], video[m])
                i += 1

            return conf_matrix
        else:
            i = 0
            scores = Variable(th.zeros(len(text)).cuda())
            for m in video:
                text_embd[m] = moe_weights[:,i:i+1]*text_embd[m]*video[m]
                scores += th.sum(text_embd[m], dim=-1)
                i += 1
             
            return scores

class Gated_Embedding_Unit(nn.Module):
    def __init__(self, input_dimension, output_dimension):
        super(Gated_Embedding_Unit, self).__init__()

        self.fc = nn.Linear(input_dimension, output_dimension)
        self.cg = Context_Gating(output_dimension)
  
    def forward(self,x):
        
        x = self.fc(x)
        x = self.cg(x)
        x = F.normalize(x)

        return x


class Context_Gating(nn.Module):
    def __init__(self, dimension, add_batch_norm=True):
        super(Context_Gating, self).__init__()
        self.fc = nn.Linear(dimension, dimension)
        self.add_batch_norm = add_batch_norm
        self.batch_norm = nn.BatchNorm1d(dimension)
        
    def forward(self,x):
        x1 = self.fc(x)

        if self.add_batch_norm:
            x1 = self.batch_norm(x1) 

        x = th.cat((x, x1), 1)
        
        return F.glu(x,1)


