# Desc: Test simple pytorch model with clipper. (http://docs.clipper.ai/en/latest/model_deployers.html#pytorch-models)
# Author: Zhou Shengsheng
# Date: 14-02-19

import json

import numpy as np
import requests
import torch
import yaml
from clipper_admin import ClipperConnection, DockerContainerManager
from clipper_admin.deployers.pytorch import deploy_pytorch_model
from torch import nn

# # Extract dependencies from conda config file
# with open('../environment.yml', 'r') as f:
#     conda_config = yaml.load(f)
# deps = conda_config['dependencies']
# pip_deps = deps[-1]['pip']
# Extract dependencies from pip config file
with open('../pip_environment.yml', 'r') as f:
    pip_deps = yaml.load(f).split(' ')
deleting_pip_deps = ['numpy', 'cloudpickle', 'faiss', 'mysqlclient', 'torch', 'numpy-base',
                     'pycocotools']  # To be deleted or replaced
dep_count = len(pip_deps)
i = 0
while i < dep_count:
    dep = pip_deps[i]
    for d in deleting_pip_deps:
        if dep.startswith(d + '==') or dep == d:
            pip_deps.remove(dep)
            i -= 1
            dep_count -= 1
    i += 1
pip_deps.append('numpy==1.16.1')
pip_deps.append('cloudpickle==0.5.3')
pip_deps.append('torch==0.4.1')
print("pip_deps:", pip_deps)

# Define a shift function to normalize prediction inputs
def shift(x):
    return x - np.mean(x)

# Build a simple torch model
model = nn.Linear(3, 1)

def predict(model, inputs):
    inputs = shift(inputs)
    inputs = torch.tensor(inputs).float()
    pred = model(inputs)
    pred = pred.data.numpy()
    return [str(x) for x in pred]

APP_NAME = "test-app"
MODEL_NAME = "test-pytorch-model"

# Setup clipper and deploy pytorch model
clipper_conn = ClipperConnection(DockerContainerManager(redis_port=6380))
try:
    clipper_conn.start_clipper()
    clipper_conn.register_application(name=APP_NAME, input_type="doubles", default_output="-1.0", slo_micros=1000000)
    deploy_pytorch_model(
        clipper_conn,
        name=MODEL_NAME,
        version="1",
        input_type="doubles",
        func=predict,
        pytorch_model=model,
        pkgs_to_install=pip_deps)
    clipper_conn.link_model_to_app(app_name=APP_NAME, model_name=MODEL_NAME)
except:
    clipper_conn.connect()

# Check all apps
print(clipper_conn.get_all_apps())

# Test inference
# inputs = np.array([[1., 2., 3.], [2., 3., 4.], [3., 4., 5.]])
# print(predict(model, inputs))
inputs = np.array([1., 2., 3.]).tolist()  # Inputs can only be one-dimensional or there will be json serialization error
headers = {"Content-type": "aplication/json"}
result = requests.post("http://localhost:1337/" + APP_NAME + "/predict", headers=headers,
                       data=json.dumps({"input": inputs})).json()
print(result)
