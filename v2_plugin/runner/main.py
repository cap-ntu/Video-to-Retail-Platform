import concurrent.futures
import logging
import multiprocessing as mp

import grpc
import uvicorn
from fastapi import FastAPI
from grpc._cython import cygrpc
from starlette.middleware.cors import CORSMiddleware

from engine import Engine
from predictor import PredictorServicer, PredictorEndPoints
from protos import api2msl_pb2_grpc
from utils import load_config


def load_engine(config):
    engine = Engine(config.engine)
    return engine


def grpc_service_starter(engine: Engine, config):
    # configuration
    grpc_config = config.grpc
    max_workers = grpc_config.max_workers
    port = grpc_config.port

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
    server.add_insecure_port(f'[::]:{port}')

    server.start()
    logging.info(f'gRPC server listening on port {port}')
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logging.info('gRPC server shutdown.')


def http_service_starter(engine: Engine, config):
    name: str = config.name
    host: str = config.http.host
    port: int = config.http.port

    # HTTP server
    app = FastAPI(title=name, openapi_url='/openapi.json')
    app.add_middleware(
        CORSMiddleware,
        allow_origins=['*'],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    endpoints = PredictorEndPoints(engine)

    app.include_router(endpoints.router)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    config_ = load_config()

    engine_ = load_engine(config_)

    # start services
    grpc_proc = mp.Process(target=grpc_service_starter, args=(engine_, config_,))
    grpc_proc.start()

    # blocking call
    http_service_starter(engine_, config_)

    grpc_proc.terminate()
