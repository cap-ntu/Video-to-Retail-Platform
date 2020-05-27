import json
import os
import time
from concurrent import futures

import grpc

from hysia.utils.logger import Logger
from hysia.utils.perf import StreamSuppressor
from model_server import config, WEIGHT_DIR, device_config
from model_server.misc import load_tf_graph, obtain_device
from protos import api2msl_pb2, api2msl_pb2_grpc

with StreamSuppressor():
    from hysia.models.scene.soundnet_classifier import SoundNetClassifier

# Time constant
_ONE_DAY_IN_SECONDS = 24 * 60 * 60

FRAME_RATE = 10

logger = Logger(
    name='audio_model_server',
    severity_levels={'StreamHandler': 'INFO'}
)


def load_sound_net(sound_net_model_path):
    soundnet_graph = load_tf_graph(sound_net_model_path)
    soundnet = SoundNetClassifier(soundnet_graph)

    logger.info('SoundNet loaded')
    return soundnet


# Custom request servicer
class Api2MslServicer(api2msl_pb2_grpc.Api2MslServicer):
    def __init__(self):
        super().__init__()

        cuda, device_num = obtain_device(device_config.audio_model_server)

        if cuda:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(device_num)
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = ''

        logger.info(f'Using {"CUDA:" if cuda else "CPU"}{os.environ["CUDA_VISIBLE_DEVICES"]}')

        self.sound_net = load_sound_net(WEIGHT_DIR / config.scene.sound_net)

    def GetJson(self, request, context):
        res = {}
        models = request.meta
        models = models.split(',')
        if 'soundnet' in models:
            audio_path = request.buf.decode()
            logger.info('Processing audio ' + audio_path)
            with StreamSuppressor():
                res = self.sound_net.classify_frame(audio_path, fr=FRAME_RATE)
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
