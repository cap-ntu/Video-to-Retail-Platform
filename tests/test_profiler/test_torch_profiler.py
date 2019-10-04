# Desc: PyTorch model profiling test.
# Author: Zhou Shengsheng
# Date: 19/04/19
# Node: Please install modifed torchstat: pip install git+git://github.com/ZhouShengsheng/torchstat.git@master

import torchstat as ts
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

import os
import sys
this_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(this_dir, ".."))
import _init_paths

from core.profiler import *


# Test torchstat
# print("===> Test torchstat")
# model = models.resnet18()
# stats = ts.stat_simple(model, (3, 224, 224))
# print("Stats for Resnet18:")
# print("total_parameters_quantity, total_memory, total_operation_quantity, total_flops, total_duration, total_mread, total_mwrite, total_memrw")
# print(stats)
# print()


# Test torch profiler
print("===> Test TorchProfiler")

# Create torch profiler via factory
profiler = ProfilerFactory.create_profiler(ProfilerEngine.TORCH)
# Create torch model (resnet18)
model = models.resnet18()
input_size = (3, 224, 224)
# Start profiling process
profile_result = profiler.profile(model, args=(input_size,))
# Show result
print("resnet18 profile_result:", profile_result)

# Profile resnet101
model = models.resnet101()
input_size = (3, 224, 224)
profile_result = profiler.profile(model, args=(input_size,))
print("resnet101 profile_result:", profile_result)

# Profile vgg16
model = models.vgg16()
input_size = (3, 224, 224)
profile_result = profiler.profile(model, args=(input_size,))
print("vgg16 profile_result:", profile_result)

# Profile self defined network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(56180, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 56180)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

model = Net()
input_size = (3, 224, 224)
profile_result = profiler.profile(model, args=(input_size,))
print("self defined network profile_result:", profile_result)

