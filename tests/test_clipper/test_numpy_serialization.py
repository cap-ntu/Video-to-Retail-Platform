# Desc: test numpy serialization.
# Author: Zhou Shengsheng
# Date: 16-02-19
# Reference:
#   https://stackoverflow.com/questions/30698004/how-can-i-serialize-a-numpy-array-while-preserving-matrix-dimensions
#   https://medium.com/datadriveninvestor/deploy-your-pytorch-model-to-production-f69460192217

import numpy as np
import pickle
import cv2
import base64


def test_numpy_serialization(a):
    # Serialize
    serialized = pickle.dumps(a)
    print('type(serialized):', type(serialized))  # <class 'bytes'>
    print('serialized:', serialized)

    # Deserialize
    deserialized_a = pickle.loads(serialized)
    print('deserialized_a:', deserialized_a)


# Some numpy array
a = np.array([[1., 2., 3.], [2., 3., 4.]])
test_numpy_serialization(a)

# List of numpy array
a = [np.array([1, 2, 3], dtype=np.int32), np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)]
test_numpy_serialization(a)

# More complex situation
a = {'model': 'test-model',
     'head': np.array([1, 2, 3]),
     'body': [np.array([1, 2, 3], dtype=np.int32), np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)]}
test_numpy_serialization(a)

# Test base64
image_path = '../test1.jpg'
img = cv2.imread(image_path)
serialized_img = pickle.dumps(img)
# print('serialized_img:', serialized_img)
base64_img = base64.b64encode(serialized_img).decode()
# print('base64_img:', base64_img)
inputs = [base64_img]
data = {"input": inputs}
# print('data:', data)
original_serialized_img = base64.b64decode(base64_img)
# print(original_serialized_img)
