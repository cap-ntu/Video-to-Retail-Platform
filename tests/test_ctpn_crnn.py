import tensorflow as tf
import cv2
import _init_paths
from models.text.tf_crnn import TF_CRNN
from models.text.tf_detector import TF_CTPN

if __name__ == '__main__':
    # Load text module graph
    PATH_TO_TEST_IMAGE = 'test_restaurant.jpg'
    PATH_TO_CTPN_GRAPH = '../weights/ctpn/ctpn.pb'
    PATH_TO_CRNN_GRAPH = '../weights/crnn/crnn_fr.pb'

    text_detection_graph = tf.Graph()
    with text_detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CTPN_GRAPH, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    text_recognition_graph = tf.Graph()
    with text_recognition_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CRNN_GRAPH, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    img = cv2.imread(PATH_TO_TEST_IMAGE)
    w = img.shape[1]
    h = img.shape[0]
    text_detector = TF_CTPN(text_detection_graph)
    text_recognizer = TF_CRNN(text_recognition_graph)

    bboxes = text_detector.detect(img)['text_bboxes']
    result, cropped = text_recognizer.detect(img, bboxes)
    for i, img in enumerate(cropped):
        cv2.imwrite('test{}.jpg'.format(i), img)
    print(result)