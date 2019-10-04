import _init_paths
import requests
import cv2
import mmcv
from models.object.pt_detector import PT_MMdetector
from clipper_admin import ClipperConnection, DockerContainerManager
from clipper_admin.deployers.python import deploy_python_closure

APP_NAME = "mmdet_clipper"
MODEL_NAME = "mmdet"

cfg_path = '../hysia/models/object/mmdetection/configs/faster_rcnn_features.py'
model_path = '../weights/mmdetection/faster_rcnn_r50_fpn_1x_20181010-3d1b3351.pth'
model = PT_MMdetector(model_path, 'cuda:0', cfg_path)


def predict(image_array):
    global model
    pred = model.detect(image_array)
    return pred


clipper_conn = ClipperConnection(DockerContainerManager(redis_port=6380))
try:
    clipper_conn.start_clipper()
except:
    clipper_conn.connect()

clipper_conn.register_application(name=APP_NAME, input_type="integers", default_output="-1.0", slo_micros=100000)
# Check all apps
print(clipper_conn.get_all_apps())

deploy_python_closure(clipper_conn, name=MODEL_NAME, version="1", input_type="integers", func=predict)
clipper_conn.link_model_to_app(app_name=APP_NAME, model_name=MODEL_NAME)

import json
inputs = cv2.imread('test1.jpg')
headers = {"Content-type": "aplication/json"}
result = requests.post("http://localhost:1337/" + APP_NAME + "/predict", headers = headers, data = json.dumps({"input" : list(inputs)})).json()
print(result)
