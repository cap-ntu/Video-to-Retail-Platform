# Author: wang yongjie
# Email : yongjie.wang@ntu.edu.sg


import _init_paths
from models.face.recognition.build_feature_lib import extractor

if __name__ == "__main__":
    model = '../weights/InsightFace_TF.pb'
    in_dir = '/data/disk2/hysia_data/Face_Recog/dataset-clean-crop-112-112'
    cast = '/data/disk2/hysia_data/Face_Recog/cast.txt'

    save_name = 'dataset48.pkl'
    test = extractor(model, in_dir, save_name, cast)
    test.init_tf_env()
    test.load_cast()
    test.save_feature()
