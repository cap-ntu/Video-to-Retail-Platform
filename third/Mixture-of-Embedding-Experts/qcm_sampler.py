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

class QCMSampler(Sampler):

    def __init__(self, n):
        self.n = n

    def __iter__(self):
        idx = np.arange(self.n)
        idx = np.repeat(idx,5)
        
        idx2 = np.arange(self.n*5)

        return iter(zip(idx,idx2))

    def __len__(self):
        return self.n*5
