# Desc: Test clipper mmdet inference.
# Author: Zhou Shengsheng
# Date: 16-02-19

import json
import requests
import cv2
import pickle
import base64
from PIL import Image

APP_NAME = 'clipper-imagenet'

img = Image.open('../../test_sofa1.jpg')
serialized_img = pickle.dumps(img)
base64_img = base64.b64encode(serialized_img).decode()  # Bytes to unicode
inputs = [base64_img]
# data = {"input": inputs}
data = {"input": base64_img}

headers = {"content-type": "aplication/json"}
response = requests.post("http://localhost:1337/" + APP_NAME + "/predict", headers=headers, json=data)
data = response.json()
result_bytes = base64.b64decode(data['output'])
result = pickle.loads(result_bytes)
print('result:', result)
