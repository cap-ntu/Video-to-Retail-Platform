import argparse

import cv2
import numpy as np
import requests

from v2_plugin.runner.utils import type_serializer


def http_request_test(port: int):
    image: np.ndarray = cv2.imread('../../tests/test1.jpg')

    data = {'raw_input': image.tolist(), 'dtype': type_serializer(image.dtype)}

    response = requests.post(f'http://localhost:{port}/predict', json=data)

    print(response.text)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--port', type=int, default=8001, help='Port for HTTP service.')
    args = parser.parse_args()

    http_request_test(args.port)
