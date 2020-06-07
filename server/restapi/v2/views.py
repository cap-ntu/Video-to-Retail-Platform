import base64
import json
import time

import cv2
import numpy as np
from rest_framework.parsers import MultiPartParser
from rest_framework.renderers import JSONRenderer
from rest_framework.response import Response
from rest_framework.views import APIView

from server.restapi.v2.coco_demo import COCODemoHelper
from v2_plugin.client import RPCClient
from v2_plugin.client.make_request_util import make_request


class Predict(APIView):
    renderer_classes = [JSONRenderer]
    parser_classes = [MultiPartParser]

    def post(self, request):
        image_file = request.data['file']

        img_array = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
        np_array: np.ndarray = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        request = make_request(model_name='mask-rcnn', inputs=[np_array])

        with RPCClient(port=50056) as rpc_client:
            response = rpc_client.service_request(request)

        predictions = json.loads(response.json)

        result = np_array.copy()

        result = COCODemoHelper.overlay_boxes(result, predictions)
        result = COCODemoHelper.overlay_class_names(result, predictions)
        if predictions['mask'] is not None:
            result = COCODemoHelper.overlay_mask(result, predictions)

        _, buffer = cv2.imencode('.jpg', result)

        encoded_string = base64.b64encode(buffer)

        return Response({
            'timestamp': time.time(),
            'data': {'raw': encoded_string.decode()},
            'message': 'ok',
            'statusCode': 200
        })


__all__ = ['Predict']
