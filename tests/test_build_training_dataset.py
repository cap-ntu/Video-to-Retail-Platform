#Author: wang yongjie
#Email:  yongjie.wang@ntu.edu.sg

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2, 3'
import _init_paths
from models.face.recognition.build_training_dataset import crop_align


if __name__  == "__main__":
    model = '../weights/mtcnn/mtcnn.pb'
    threshold = [0.7, 0.7, 0.9]
    minisize = 20
    factor = 0.719
    input_dir = '/data/disk2/hysia_data/Face_Recog/dataset-clean'
    output_dir = '/data/disk2/hysia_data/Face_Recog/dataset-clean-crop-112-112'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    crop_align(input_dir, output_dir, model, threshold, minisize, factor)
