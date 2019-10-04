import json
import os
import time
from concurrent import futures

import cv2
# rpc imports
import grpc
import numpy as np
import tensorflow as tf
from PIL import Image

from hysia.dataset.srt_handler import extract_srt
from hysia.models.nlp.sentence import TF_Sentence
from hysia.models.object.audioset_feature_extractor import AudiosetFeatureExtractor
from hysia.models.scene.detector import scene_visual
from hysia.utils.logger import Logger
from model_server.utils import StreamSuppressor
from protos import api2msl_pb2, api2msl_pb2_grpc

# Time constant
_ONE_DAY_IN_SECONDS = 24 * 60 * 60

SERVER_ROOT = os.path.dirname(os.path.abspath(__file__)) + '/'

logger = Logger(
    name='feature_extract_model_server',
    severity_levels={'StreamHandler': 'INFO'}
)

sentence_model_path = os.path.join(SERVER_ROOT,
                                   '../../weights/sentence/96e8f1d3d4d90ce86b2db128249eb8143a91db73')
vggish_fr_path = os.path.join(SERVER_ROOT, '../../weights/audioset/vggish_fr.pb')

vggish_pca_path = os.path.join(SERVER_ROOT, '../../weights/audioset/vggish_pca_params.npz')

resnet_places365_path = os.path.join(SERVER_ROOT, '../../weights/places365/{}.pth')

place365_category_path = os.path.join(SERVER_ROOT, '../../weights/places365/categories.txt')


def load_sentence_model():
    # Instantiate sentence feature extractor
    return TF_Sentence(sentence_model_path)


def load_audio_model():
    # Instantiate audio feature extractor
    with StreamSuppressor():
        vgg_graph = tf.Graph()
        with vgg_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(vggish_fr_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        audio_model = AudiosetFeatureExtractor(vgg_graph, vggish_pca_path)
    return audio_model


def load_image_model():
    # Instantiate scene feature extractor
    return scene_visual('resnet50', resnet_places365_path, place365_category_path, 'cuda:0')


# Custom request servicer
class Api2MslServicer(api2msl_pb2_grpc.Api2MslServicer):
    def __init__(self):
        super().__init__()
        os.environ['CUDA_VISIBLE_DEVICES'] = '3'
        logger.info('Using GPU:' + os.environ['CUDA_VISIBLE_DEVICES'])
        self.sentence_model = load_sentence_model()
        self.audio_model = load_audio_model()
        self.image_model = load_image_model()

    def GetJson(self, request, context):
        res = {}
        meta = request.meta
        meta = meta.split(',')
        # Process entire audio file
        # Extract nlp feature from subtitle
        if 'subtitle' in meta:
            subtitle_path = request.buf.decode()
            logger.info('Extracting from subtitle: ' + subtitle_path)
            start_time = int(meta[1])
            end_time = int(meta[2])
            sentences = extract_srt(start_time, end_time, subtitle_path)
            if len(sentences) == 0:
                sentences_feature = 'unknown_feature'
                sentences = 'unknown_subtitle'
            else:
                # TODO TEXT support what data types (BLOB only support numpy)
                sentences = ' '.join(sentences)
                sentences_feature = self.sentence_model.encode(sentences)
            res['features'] = sentences_feature
            return api2msl_pb2.JsonReply(json=json.dumps(res), meta=sentences)

        # Extract audio feature
        if 'audio' in meta:
            audio_path = request.buf.decode()
            logger.info('Extracting from audio: ' + audio_path)
            start_time = int(meta[1])
            end_time = int(meta[2])
            audio_feature = self.audio_model.extract(audio_path, start_time, end_time)[0]
            res['features'] = audio_feature.tolist()
            return api2msl_pb2.JsonReply(json=json.dumps(res), meta='')
        if 'scene' in meta:
            img = cv2.imdecode(np.fromstring(request.buf, dtype=np.uint8), -1)
            logger.info('Extracting from image of shape ' + str(img.shape))
            img_pil = Image.fromarray(img)
            scene_feature = self.image_model.extract_vec(img_pil, True)
            scene_name = self.image_model.detect(img_pil, True)
            res['features'] = scene_feature.tolist()
            return api2msl_pb2.JsonReply(json=json.dumps(res), meta=scene_name['scene'][0])

        return api2msl_pb2.JsonReply(json=json.dumps(res), meta='')


def main():
    # gRPC server configurations
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=8))
    api2msl_pb2_grpc.add_Api2MslServicer_to_server(Api2MslServicer(), server)
    server.add_insecure_port('[::]:50055')
    server.start()
    logger.info('Listening on port 50055')
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        logger.info('Shutting down feature extract model server')
        server.stop(0)


if __name__ == '__main__':
    main()
