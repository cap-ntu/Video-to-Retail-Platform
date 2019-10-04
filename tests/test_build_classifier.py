#Author: Wang Yongjie
#Email:  yongjie.wang@ntu.edu.sg

import _init_paths
from models.face.recognition.build_classifier import classifier

if __name__ == "__main__":
    model = '../weights/insightface.pb'
    input_dir = '/home/wyj/dataset/face_recg/bigbang-mini'
    output_dir = '/home/wyj/dataset/face_recg/bigbang-mini-output'
    test = classifier(model, input_dir, output_dir)
    test.build()
    test.test()
