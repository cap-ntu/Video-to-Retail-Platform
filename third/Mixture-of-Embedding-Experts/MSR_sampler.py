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
from torch.utils.data.sampler import Sampler
import numpy as np

class MSRSampler(Sampler):

    def __init__(self, n_MSR, n_COCO, sampling_rate):
        self.n_MSR = n_MSR
        self.n_COCO = n_COCO
        self.sampling_rate = sampling_rate

    def __iter__(self):
        idx_MSR = np.arange(self.n_MSR)
        idx_coco = np.arange(self.n_MSR,self.n_MSR+self.n_COCO)

        np.random.shuffle(idx_coco)
        idx_coco = idx_coco[:min(self.n_COCO,int(self.sampling_rate*self.n_MSR))]

        idx = np.concatenate((idx_MSR,idx_coco), axis=0)
        np.random.shuffle(idx)

        return iter(idx)

    def __len__(self):
        return self.n_MSR+int(self.sampling_rate*self.n_COCO)
