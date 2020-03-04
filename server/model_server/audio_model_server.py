import json
import os
import time
from concurrent import futures

# rpc imports
import grpc
import tensorflow as tf

from hysia.utils.logger import Logger
from protos import api2msl_pb2, api2msl_pb2_grpc
from .utils import StreamSuppressor

with StreamSuppressor():
    from hysia.models.scene.soundnet_classifier import SoundNetClassifier

# Time constant
_ONE_DAY_IN_SECONDS = 24 * 60 * 60

FRAME_RATE = 10

SERVER_ROOT = os.path.dirname(os.path.abspath(__file__)) + '/'

PATH_TO_PB = SERVER_ROOT + '../../weights/soundnet/soundnet_fr.pb'

logger = Logger(
    name='audio_model_server',
    severity_levels={'StreamHandler': 'INFO'}
)


def load_soundnet():
    with StreamSuppressor():
        soundnet_graph = tf.Graph()
        with soundnet_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_PB, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        soundnet = SoundNetClassifier(soundnet_graph)

    logger.info('Soundnet loaded')
    return soundnet


# Custom request servicer
class Api2MslServicer(api2msl_pb2_grpc.Api2MslServicer):
    def __init__(self):
        super().__init__()
        os.environ['CUDA_VISIBLE_DEVICES'] = '2'
        logger.info('Using GPU:' + os.environ['CUDA_VISIBLE_DEVICES'])
        self.soundnet = load_soundnet()

    def GetJson(self, request, context):
        res = {}
        models = request.meta
        models = models.split(',')
        if 'soundnet' in models:
            audio_path = request.buf.decode()
            logger.info('Processing audio ' + audio_path)
            with StreamSuppressor():
                res = self.soundnet.classify_frame(audio_path, fr=FRAME_RATE)
            logger.info('Finished Processing audio ' + audio_path)
        return api2msl_pb2.JsonReply(json=json.dumps(res), meta=str(FRAME_RATE))


def main():
    # gRPC server configurations
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=8))
    api2msl_pb2_grpc.add_Api2MslServicer_to_server(Api2MslServicer(), server)
    server.add_insecure_port('[::]:50052')
    server.start()
    logger.info('Listening on port 50052')
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        logger.info('Shutting down audio model server')
        server.stop(0)


if __name__ == '__main__':
    main()
