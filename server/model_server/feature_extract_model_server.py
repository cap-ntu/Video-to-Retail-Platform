import json
import os
import time
from concurrent import futures

import cv2
import grpc
import numpy as np
from PIL import Image

from hysia.dataset.srt_handler import extract_srt
from hysia.models.nlp.sentence import TF_Sentence
from hysia.models.object.audioset_feature_extractor import AudiosetFeatureExtractor
from hysia.models.scene.detector import scene_visual
from hysia.utils.logger import Logger
from model_server import config, WEIGHT_DIR
from model_server.misc import load_tf_graph
from protos import api2msl_pb2, api2msl_pb2_grpc

# Time constant
_ONE_DAY_IN_SECONDS = 24 * 60 * 60

logger = Logger(
    name='feature_extract_model_server',
    severity_levels={'StreamHandler': 'INFO'}
)


def load_sentence_model():
    model_path = str(WEIGHT_DIR / config.feature_extraction.sentence_encoder)
    # Instantiate sentence feature extractor
    return TF_Sentence(model_path)


def load_audio_model():
    # Instantiate audio feature extractor
    vgg_graph = load_tf_graph(WEIGHT_DIR / config.feature_extraction.vggish.fr)
    vgg_pca_path = WEIGHT_DIR / config.feature_extraction.vggish.pca

    audio_model = AudiosetFeatureExtractor(vgg_graph, vgg_pca_path)
    return audio_model


# Custom request servicer
class Api2MslServicer(api2msl_pb2_grpc.Api2MslServicer):
    def __init__(self):
        super().__init__()
        os.environ['CUDA_VISIBLE_DEVICES'] = '3'
        logger.info('Using GPU:' + os.environ['CUDA_VISIBLE_DEVICES'])
        self.sentence_model = load_sentence_model()
        self.audio_model = load_audio_model()

        # load object detection model
        places365_config = config.scene.places365
        self.backbone = places365_config.backbone
        model_path = str(WEIGHT_DIR / places365_config.model)
        label_path = WEIGHT_DIR / places365_config.label
        self.scene_recognition_model = scene_visual(self.backbone, model_path, label_path, 'cuda:0')

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
            scene_feature = self.scene_recognition_model.extract_vec(img_pil, True)
            scene_name = self.scene_recognition_model.detect(img_pil, True)
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
