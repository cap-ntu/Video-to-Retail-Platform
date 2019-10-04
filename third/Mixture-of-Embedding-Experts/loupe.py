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
import math


class NetVLAD(nn.Module):
    def __init__(self, cluster_size, feature_size, add_batch_norm=True):
        super(NetVLAD, self).__init__()
        self.feature_size = feature_size
        self.cluster_size = cluster_size
        self.clusters = nn.Parameter((1/math.sqrt(feature_size))
                *th.randn(feature_size, cluster_size))
        self.clusters2 = nn.Parameter((1/math.sqrt(feature_size))
                *th.randn(1, feature_size, cluster_size))

        self.add_batch_norm = add_batch_norm
        self.batch_norm = nn.BatchNorm1d(cluster_size)
        self.out_dim = cluster_size*feature_size

    def forward(self,x):
        max_sample = x.size()[1]
        x = x.view(-1,self.feature_size)
        assignment = th.matmul(x,self.clusters)

        if self.add_batch_norm:
            assignment = self.batch_norm(assignment)

        assignment = F.softmax(assignment,dim=1)
        assignment = assignment.view(-1, max_sample, self.cluster_size)

        a_sum = th.sum(assignment,-2,keepdim=True)
        a = a_sum*self.clusters2

        assignment = assignment.transpose(1,2)

        x = x.view(-1, max_sample, self.feature_size)
        vlad = th.matmul(assignment, x)
        vlad = vlad.transpose(1,2)
        vlad = vlad - a

        # L2 intra norm
        vlad = F.normalize(vlad)
        
        # flattening + L2 norm
        vlad = vlad.view(-1, self.cluster_size*self.feature_size)
        vlad = F.normalize(vlad)

        return vlad

class NetRVLAD(nn.Module):
    def __init__(self, cluster_size, feature_size, add_batch_norm=True):
        super(NetRVLAD, self).__init__()
        self.feature_size = feature_size
        self.cluster_size = cluster_size
        self.clusters = nn.Parameter((1/math.sqrt(feature_size))
                *th.randn(feature_size, cluster_size))

        self.add_batch_norm = add_batch_norm
        self.batch_norm = nn.BatchNorm1d(cluster_size)
        self.out_dim = cluster_size*feature_size

    def forward(self,x):
        max_sample = x.size()[1]
        x = x.view(-1,self.feature_size)
        assignment = th.matmul(x,self.clusters)

        if self.add_batch_norm:
            assignment = self.batch_norm(assignment)

        assignment = F.softmax(assignment, dim=1)
        assignment = assignment.view(-1, max_sample, self.cluster_size)

        assignment = assignment.transpose(1,2)

        x = x.view(-1, max_sample, self.feature_size)
        rvlad = th.matmul(assignment, x)
        rvlad = rvlad.transpose(-1,1)

        # L2 intra norm
        rvlad = F.normalize(rvlad)
        
        # flattening + L2 norm
        rvlad = rvlad.view(-1, self.cluster_size*self.feature_size)
        rvlad = F.normalize(rvlad)

        return rvlad

