# Author: Wang Yongjie
# Email : yongjie.wang@ntu.edu.sg


import _init_paths
from models.face.recognition.threshold import *

if __name__ == "__main__":
    directory = '/data/disk2/hysia_data/Face_Recog/dataset-clean-crop-112-112'
    weights = '../weights/InsightFace_TF.pb'
    save_name = "PR.png"
    get_threshold(directory, weights, save_name)
