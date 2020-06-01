import concurrent.futures
import logging

import grpc
import uvicorn
import yaml
from grpc._cython import cygrpc

from engine import Engine
from predictor import PredictorServicer, PredictEndPoints
from protos import api2msl_pb2_grpc
from utils import dict_to_object


def load_config():
    with open('config.yml', 'r') as stream:
        try:
            config = yaml.safe_load(stream)
            config = dict_to_object(config)
        except yaml.YAMLError as e:
            print(e)
            exit(1)

    return config


def service_starter(config):
    # configuration
    grpc_config = config.grpc
    max_workers = grpc_config.max_workers
    grpc_port = grpc_config.port

    http_port = config.http.port

    # engine
    engine = Engine(config.engine)

    # servicer
    servicer = PredictorServicer(engine)

    # gRPC server
    server = grpc.server(
        concurrent.futures.ThreadPoolExecutor(max_workers=max_workers),
        options=[
            (cygrpc.ChannelArgKey.max_send_message_length, -1),
            (cygrpc.ChannelArgKey.max_receive_message_length, -1),
        ],
    )

    api2msl_pb2_grpc.add_Api2MslServicer_to_server(servicer, server)
    server.add_insecure_port(f'[::]:{grpc_port}')

    server.start()
    logging.info(f'gRPC listening on port {grpc_port}')

    # HTTP server
    fast_api_app = PredictEndPoints.app
    uvicorn.run(fast_api_app, host='0.0.0.0', port=http_port)


if __name__ == "__main__":
    service_starter(load_config())
