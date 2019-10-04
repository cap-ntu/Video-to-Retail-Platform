# Author: Wang Yongjie
# Email:  yongjie.wang@ntu.edu.sg
# Description:  Hysia unit test

# import preinstalled libraries required
import unittest
import cv2
import sys
import _init_paths
import os
import tensorflow as np
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# import hysia models
# import your personal models based on the following example
from  models.face.mtcnn.detector import mtcnn_detector
from models.face.alignment.alignment import transform
from models.face.recognition.recognition import recog
from models.object.pretrained_imagenet import Img2Vec
from models.scene.detector import scene_visual
from search.product_search import ProductSearch

# set tensorflow log level
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

## test files listed here
test1 = "test1.jpg"
test3 = "test3.jpg"
test_sofa = "test_sofa1.jpg"
test_beach = "test_beach.jpg"
VIDEO_DATA_PATH = "/data/disk2/hysia_data/Stanford_Online_Products/"

## pretrained weights
mtcnn_model = '../weights/mtcnn/mtcnn.pb'
face_model = '../weights/face_recog/InsightFace_TF.pb'
saved_dataset = '../weights/face_recog/dataset48.pkl'


class TestHysia(unittest.TestCase):

    def test_mtcnn(self):
        threshold = [0.6, 0.7, 0.9]
        factor = 0.7
        minisize = 20
        test = mtcnn_detector(mtcnn_model, threshold, minisize, factor)
        img = cv2.imread(test1)
        rectangles, points, duration = test.detect(img)
        for rec in rectangles:
            cv2.rectangle(img, (int(rec[0]), int(rec[1])), (int(rec[2]), int(rec[3])), (0, 0, 255), 1)
        cv2.imwrite('mtcnn_test1.jpg', img)


    def test_alignment(self):
        threshold = [0.6, 0.7, 0.9]
        factor = 0.7
        minisize = 20
        test = mtcnn_detector(mtcnn_model, threshold, minisize, factor)
        img = cv2.imread(test3)
        cols, rows, channel = img.shape
        rectangles, points, duration = test.detect(img)
        wrapped = transform(img, points, 112, 112)
        cv2.imwrite("align_test3.jpg", wrapped)

    def test_face_recognition(self):
        factor = 0.7
        threshold = [0.7, 0.7, 0.9]
        minisize = 25
        test = recog(face_model, saved_dataset)
        test.init_tf_env()
        test.load_feature()
        test.init_mtcnn_detector(mtcnn_model, threshold, minisize, factor)
        image = cv2.imread(test1)
        rectangles, name_lists, features = test.get_indentity(image, role = True)
        for i in range(len(rectangles)):
            rec = rectangles[i,:]
            cv2.rectangle(image, (int(rec[0]), int(rec[1])), (int(rec[2]), int(rec[3])), (0, 0, 255), 1)
            cv2.putText(image, name_lists[i], (int(rec[0]), int(rec[1])), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.75,color=(152, 255, 204), thickness=2)

        cv2.imwrite('recognition_test1.jpg', image)

    def test_imagenet(self):
        img = Image.open(test_sofa)
        model = Img2Vec(cuda=True)
        q_vec = model.extract_vec(img, True)
        temp = Image.open(test_beach)
        search_vec = model.extract_vec(temp, True)
        scores = np.dot(search_vec.T, q_vec)
        print(scores)
        cos = cosine_similarity(np.reshape(q_vec, (1,-1)), np.reshape(search_vec,(1, -1)))
        print(cos)


    def test_place365(self):
        scene_model = scene_visual('resnet50', '../weights/places365/{}.pth', '../weights/places365/categories.txt', 'cuda:0')
        for i in ['test1.jpg', 'test2.jpg']:
            temp = scene_model.detect(i)
            print(temp)
        temp = cv2.imread('test1.jpg')
        temp = Image.fromarray(cv2.cvtColor(temp, cv2.COLOR_BGR2RGB))
        temp = scene_model.detect(temp, True)
        print(temp)
        # Test vector extraction and cosine similarity
        # TODO The accuracy is decreasing when transforming
        temp = cv2.imread('test_sofa1.jpg')
        q_tensor = Image.fromarray(cv2.cvtColor(temp, cv2.COLOR_BGR2RGB))
        q_vec = scene_model.extract_vec(q_tensor, True)
        print(type(q_vec))
        temp = cv2.imread('test_beach.jpg')
        search_tensor = Image.fromarray(cv2.cvtColor(temp, cv2.COLOR_BGR2RGB))
        search_vec = scene_model.extract_vec(search_tensor, True)
        scores = np.dot(search_vec.T, q_vec)
        print(scores)

    def test_product(self):
        product_machine = ProductSearch()
        results = product_machine.search(100, 'test_clip_1.mp4')
        print(results)


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestHysia)
    unittest.TextTestRunner(verbosity = 2).run(suite)
