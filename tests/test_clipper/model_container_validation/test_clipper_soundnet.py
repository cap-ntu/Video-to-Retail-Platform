# Desc: Test clipper mmdet inference.
# Author: Zhou Shengsheng
# Date: 16-02-19

import _init_paths
import json
import requests
import cv2
import pickle
import base64
from PIL import Image
from models.scene.audio.audio_util import load_audio

APP_NAME = 'clipper-soundnet'

# Load text module graph
PATH_TO_TEST_AUDIO = '../../audio/test_airport.wav'

# Scene start&end timestamp in milliseconds
task_type = 'classify_scene'
# task_type = 'classify_frame'
sc_start = 0
sc_end = 2000
frame = 10

wav_data, sample_rate = load_audio(PATH_TO_TEST_AUDIO, sr=22050, mono=True)
raw_input = (task_type, wav_data, sample_rate, sc_start, sc_end, frame)
serialized_input = pickle.dumps(raw_input)
base64_input = base64.b64encode(serialized_input).decode()  # Bytes to unicode
# inputs = [base64_input]
# data = {"input": inputs}
data = {"input": base64_input}

headers = {"content-type": "aplication/json"}
response = requests.post("http://localhost:1337/" + APP_NAME + "/predict", headers=headers, json=data)
data = response.json()
result_bytes = base64.b64decode(data['output'])
result = pickle.loads(result_bytes)
print('result:', result)
