# Desc: Test clipper mmdet inference.
# Author: Zhou Shengsheng
# Date: 16-02-19

import base64
import pickle
import requests


APP_NAME = 'clipper-sentence-embedding'

# Test sentence
sentence = "I am a sentence for which I would like to get its embedding."

raw_input = sentence
serialized_input = pickle.dumps(raw_input)
base64_input = base64.b64encode(serialized_input).decode()  # Bytes to unicode
data = {"input": base64_input}

headers = {"content-type": "aplication/json"}
response = requests.post("http://localhost:1337/" + APP_NAME + "/predict", headers=headers, json=data)
data = response.json()
result_bytes = base64.b64decode(data['output'])
result = pickle.loads(result_bytes)
print('result:', result)
