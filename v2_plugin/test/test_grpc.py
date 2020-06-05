import argparse
import json

import cv2

from v2_plugin.client import RPCClient
from v2_plugin.client.make_request_util import make_request


def grpc_request_test(port: int):

    image = cv2.imread('../../tests/test1.jpg')

    request = make_request(model_name='mask_rcnn', inputs=[image])

    with RPCClient(port=port) as rpc_client:
        response = rpc_client.service_request(request)

    print(json.loads(response.json))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--port', type=int, default=8000, help='Port for HTTP service.')
    args = parser.parse_args()

    grpc_request_test(args.port)
