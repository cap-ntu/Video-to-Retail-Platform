import grpc

# gRPC imports
from protos import api2msl_pb2, api2msl_pb2_grpc


class RpcClient:
    def __init__(self, port):
        self.channel = grpc.insecure_channel("localhost:" + str(port))
        self.stub = api2msl_pb2_grpc.Api2MslStub(self.channel)

    def service_request(self, buf, meta):
        response = self.stub.GetJson(api2msl_pb2.BundleRequest(buf=buf, meta=meta))
        return response.json, response.meta

    def close(self):
        self.channel.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
