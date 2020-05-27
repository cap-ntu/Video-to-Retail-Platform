"""
gRPC server which hosts varies visual models
Author: Jiang Wenbo
Email: jian0085@e.ntu.edu.sg
"""
import json
import os
import time
from concurrent import futures

import cv2
import grpc
import numpy as np
from PIL import Image

# face imports
from hysia.models.face.recognition.recognition import recog
# object_detection imports
from hysia.models.object.tf_detector import TF_SSD as DetectorEngine
# place 365 imports
from hysia.models.scene import detector as places365_detector
# Import ctpn models
from hysia.models.text.tf_detector import TF_CTPN as CtpnEngine
from hysia.utils.logger import Logger
from hysia.utils.perf import StreamSuppressor
from model_server import config, WEIGHT_DIR
from model_server.misc import load_tf_graph
from protos import api2msl_pb2, api2msl_pb2_grpc

# Time constant
_ONE_DAY_IN_SECONDS = 24 * 60 * 60
# List of the strings that is used to add correct label for each box.
NUM_CLASS = 90

logger = Logger(
    name='visual_model_server',
    severity_levels={'StreamHandler': 'INFO'}
)

# Locate faces frozen graph and face recognition module
factor = 0.7
threshold = [0.7, 0.7, 0.9]
minisize = 25


def load_detector_engine():

    ssd_config = config.object_detection.ssd
    backbone = ssd_config.backbone
    model_path = WEIGHT_DIR / getattr(ssd_config, backbone)
    label_path = str(WEIGHT_DIR / ssd_config.label)

    # Load the TensorFlow graph
    detection_graph = load_tf_graph(model_path)

    # Instantiate a DetectorEngine
    with StreamSuppressor():
        detector_engine = DetectorEngine(detection_graph, label_path, NUM_CLASS)
    logger.info('Finished loading {} detector'.format(backbone))

    return detector_engine


def load_ctpn_engine():
    text_detection_graph = load_tf_graph(WEIGHT_DIR / config.text.ctpn)

    # Instantiate a CtpnEngine as global variable
    with StreamSuppressor():
        ctpn_engine = CtpnEngine(text_detection_graph)
    logger.info('Finished loading text detector')
    return ctpn_engine


def load_face_engine():

    face_config = config.face
    model_path = str(WEIGHT_DIR / face_config.model)
    dataset_path = WEIGHT_DIR / face_config.saved_dataset
    mtcnn_path = str(WEIGHT_DIR / face_config.mtcnn_model)

    # Instantiate a face detection and recognition engine
    with StreamSuppressor():
        face_engine = recog(model_path, dataset_path)
        face_engine.init_tf_env()
        face_engine.load_feature()
        face_engine.init_mtcnn_detector(mtcnn_path, threshold, minisize, factor)
    logger.info('Finished loading face models')
    return face_engine


# Custom request servicer
class Api2MslServicer(api2msl_pb2_grpc.Api2MslServicer):
    def __init__(self):
        super().__init__()
        # todo(zhz): use device_placement.yml device
        os.environ['CUDA_VISIBLE_DEVICES'] = '1'
        logger.info('Using GPU:' + os.environ['CUDA_VISIBLE_DEVICES'])
        self.detector_engine = load_detector_engine()
        self.ctpn_engine = load_ctpn_engine()
        self.face_engine = load_face_engine()

        # load scene detection model
        places365_config = config.scene.places365
        self.backbone = places365_config.backbone
        model = str(WEIGHT_DIR / places365_config.model)
        label = WEIGHT_DIR / places365_config.label
        self.places365_engine = places365_detector.scene_visual(self.backbone, model, label, 'cuda:0')

    def GetJson(self, request, context):
        img = cv2.imdecode(np.fromstring(request.buf, dtype=np.uint8), -1)
        logger.info('Receiving image of shape ' + str(img.shape))
        models = request.meta
        models = models.split(',')
        det = {}
        if 'SSD-mobilenet' in models or 'SSD-inception' in models:
            logger.info('Processing with SSD_inception')
            det.update(self.detector_engine.detect(img))
        if 'ctpn' in models:
            logger.info('Processing with CTPN')
            det.update(self.ctpn_engine.detect(img))
        if 'mtcnn' in models:
            logger.info('Processing with mtcnn')
            boxes, name_list = self.face_engine.get_indentity(img, role=True)
            boxes = np.array(boxes)
            if boxes.size != 0:
                # normalize box dimensions
                img_shape = img.shape
                boxes[:, (0, 2)] /= img_shape[1]
                boxes[:, (1, 3)] /= img_shape[0]
                # Swap x and y to make it consistent with other DNN APIs
                boxes[:, (0, 1, 2, 3)] = boxes[:, (1, 0, 3, 2)]
            det.update({
                'face_bboxes': boxes.tolist(),
                'face_names': name_list
            })
        if f'{self.backbone}-places365' in models:
            logger.info(f'Processing with {self.backbone}-places365')
            det.update(self.places365_engine.detect(Image.fromarray(img), tensor=True))

        logger.info('Processing done')
        return api2msl_pb2.JsonReply(json=json.dumps(det), meta='')


def main():
    # gRPC server configurations
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=8))
    api2msl_pb2_grpc.add_Api2MslServicer_to_server(Api2MslServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    logger.info('Listening on port 50051')
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        logger.info('Shutting down visual model server')
        server.stop(0)


if __name__ == '__main__':
    main()
