from typing import Union, Generator, Iterator

import grpc
from grpc._cython import cygrpc

from v2_plugin.protos import infer_pb2_grpc
from v2_plugin.protos.infer_pb2 import InferRequest, Empty, InferResponse


class RPCClient(object):
    def __init__(self, port: int):
        self.channel = grpc.insecure_channel(
            f'localhost:{port}',
            options=[
                (cygrpc.ChannelArgKey.max_send_message_length, -1),
                (cygrpc.ChannelArgKey.max_receive_message_length, -1),
            ]
        )
        self.stub = infer_pb2_grpc.InferProtoStub(self.channel)

    def service_request(
            self,
            request: Union[InferRequest, Iterator[InferRequest]],
    ):
        if isinstance(request, Iterator):
            return self._service_request_stream(request)
        else:
            return self._service_request(request)

    def _service_request(self, request: InferRequest) -> InferResponse:
        response = self.stub.Infer(request)
        return response

    def _service_request_stream(
            self,
            request_generator: Iterator[InferRequest]
    ) -> Generator[None, InferResponse, None]:
        responses = self.stub.StreamInfer(request_generator)
        return responses

    def stop_model_instance(self):
        response = self.stub.Stop(Empty())
        return response.status

    def close(self):
        self.channel.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
