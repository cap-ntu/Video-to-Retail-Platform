import json
import os
import os.path as osp
import time
from concurrent import futures

# rpc imports
import grpc

from hysia.search.search import DatabasePklSearch
from hysia.utils.logger import Logger
from hysia.utils.perf import StreamSuppressor
from protos import api2msl_pb2, api2msl_pb2_grpc

SERVER_ROOT = os.path.dirname(os.path.abspath(__file__)) + '/'

# Time constant
_ONE_DAY_IN_SECONDS = 24 * 60 * 60

# TVQA dataset for efficient test
# VIDEO_DATA_PATH = '/data/disk2/hysia_data/UNC_TVQA_DATASET'
# search_machine = BasicSearch(VIDEO_DATA_PATH)

logger = Logger(
    name='scene_search_model_server',
    severity_levels={'StreamHandler': 'INFO'}
)

video_path = osp.join(SERVER_ROOT, '../output/multi_features')


def load_search_machine():
    with StreamSuppressor():
        search_machine = DatabasePklSearch(video_path)
    return search_machine


# Custom request servicer
class Api2MslServicer(api2msl_pb2_grpc.Api2MslServicer):
    def __init__(self):
        super().__init__()
        os.environ['CUDA_VISIBLE_DEVICES'] = '3'
        logger.info('Using GPU:' + os.environ['CUDA_VISIBLE_DEVICES'])
        self.search_machine = load_search_machine()

    def GetJson(self, request, context):
        meta = json.loads(request.meta)
        img_path = request.buf.decode()
        logger.info('Searching by ' + img_path)
        # Decode image from buf
        with StreamSuppressor():
            results = self.search_machine.search(
                image_query=img_path,
                subtitle_query=meta['text'],
                face_query=None,
                topK=5,
                tv_name=meta['target_videos'][0] if len(meta['target_videos']) else None
                # TODO Currently only support one target video
            )
        # Convert tensor to list to make it serializable
        for res in results:

            # TODO Here has some bugs, json can not accept numpy results
            if not type(res['FEATURE']) == list:
                res['FEATURE'] = res['FEATURE'].tolist()
            try:
                if not type(res['AUDIO_FEATURE']) == list:
                    res['AUDIO_FEATURE'] = res['AUDIO_FEATURE'].tolist()
            except:
                pass

            try:
                if not type(res['SUBTITLE_FEATURE']) == list and not type(res['SUBTITLE_FEATURE']) == str:
                    res['SUBTITLE_FEATURE'] = res['SUBTITLE_FEATURE'].tolist()
            except:
                pass

        return api2msl_pb2.JsonReply(json=json.dumps(results), meta='')


def main():
    # gRPC server configurations
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=8))
    api2msl_pb2_grpc.add_Api2MslServicer_to_server(Api2MslServicer(), server)
    server.add_insecure_port('[::]:50053')
    server.start()
    logger.info('Listening on port 50053')
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        logger.info('Shutting down scene search model server')
        server.stop(0)


if __name__ == '__main__':
    main()
