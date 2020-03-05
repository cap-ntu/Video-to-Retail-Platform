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
# gRPC imports
import grpc
# Basic imports
import numpy as np
import tensorflow as tf
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
from model_server.utils import StreamSuppressor
from protos import api2msl_pb2, api2msl_pb2_grpc
from hysia.utils.perf import timeit

# Load detector graph

# Time constant
_ONE_DAY_IN_SECONDS = 24 * 60 * 60
# List of the strings that is used to add correct label for each box.
NUM_CLASS = 90
# What model to download.
SERVER_ROOT = os.path.dirname(os.path.abspath(__file__)) + '/'

logger = Logger(
    name='visual_model_server',
    severity_levels={'StreamHandler': 'INFO'}
)

# Path to frozen detection graph. This is the actual model that is used for the object detection.
SSD_mobile_path = SERVER_ROOT + \
                  '../../weights/ssd_mobilenet_v1_coco_2017_11_17/frozen_inference_graph.pb'
SSD_inception_path = SERVER_ROOT + \
                     '../../weights/ssd_inception_v2_coco_2018_01_28/frozen_inference_graph.pb'

FASTERRCNN_resnet101_path = SERVER_ROOT + \
                     '../../weights/faster_rcnn_resnet101_coco_2018_01_28/frozen_inference_graph.pb'

PATH_TO_LABELS = SERVER_ROOT + '../../third/object_detection/data/mscoco_label_map.pbtxt'

# Load text detector graph
PATH_TO_FROZEN_GRAPH = SERVER_ROOT + '../../weights/ctpn/ctpn.pb'

# Locate faces frozen graph and face recognition module
mtcnn_model = SERVER_ROOT + '../../weights/mtcnn/mtcnn.pb'
model = SERVER_ROOT + '../../weights/face_recog/InsightFace_TF.pb'
saved_dataset = SERVER_ROOT + '../../weights/face_recog/dataset48.pkl'
factor = 0.7
threshold = [0.7, 0.7, 0.9]
minisize = 25

# Locate place365 model
PLACES365_MODEL_PATH = SERVER_ROOT + '../../weights/places365/{}.pth'
PLACES365_LABEL_PATH = SERVER_ROOT + '../../weights/places365/categories.txt'

@timeit
def load_detector_engine(model_path=SSD_inception_path):
    # Load the TensorFlow graph
    with StreamSuppressor():
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(model_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

    # Instantiate a DetectorEngine
    with StreamSuppressor():
        detector_engine = DetectorEngine(detection_graph, PATH_TO_LABELS, NUM_CLASS)
    logger.info('Finished loading {} detector'.format(model_path))

    return detector_engine

@timeit
def load_ctpn_engine():
    with StreamSuppressor():
        text_detection_graph = tf.Graph()
        with text_detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

    # Instantiate a CtpnEngine as global variable
    with StreamSuppressor():
        ctpn_engine = CtpnEngine(text_detection_graph)
    logger.info('Finished loading text detector')
    return ctpn_engine

@timeit
def load_face_engine():
    # Instantiate a face detection and recognition engine
    with StreamSuppressor():
        face_engine = recog(model, saved_dataset)
        face_engine.init_tf_env()
        face_engine.load_feature()
        face_engine.init_mtcnn_detector(mtcnn_model, threshold, minisize, factor)
    logger.info('Finished loading face models')
    return face_engine

@timeit
def load_places365_engine():
    # Instantiate a place365 model
    with StreamSuppressor():
        places365_engine = places365_detector.scene_visual('densenet161', PLACES365_MODEL_PATH, PLACES365_LABEL_PATH,
                                                           'cuda:0')
    logger.info('Finished loading scene classifier')
    return places365_engine


# Custom request servicer
class Api2MslServicer(api2msl_pb2_grpc.Api2MslServicer):
    def __init__(self):
        super().__init__()
        # todo(zhz): use device_placement.yml device
        os.environ['CUDA_VISIBLE_DEVICES'] = '1'
        logger.info('Using GPU:' + os.environ['CUDA_VISIBLE_DEVICES'])
        self.detector_engine = load_detector_engine(SSD_mobile_path)
        self.ctpn_engine = load_ctpn_engine()
        self.face_engine = load_face_engine()
        self.places365_engine = load_places365_engine()

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
        if 'res18-places365' in models:
            logger.info('Processing with res18-places365')
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
