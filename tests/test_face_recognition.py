# Author: Wang Yongjie
# Email : yongjie.wang@ntu.edu.sg


import _init_paths
import cv2
import sys

from models.face.recognition.recognition import recog


if __name__ == "__main__":
    mtcnn_model = '../weights/mtcnn/mtcnn.pb'
    model = '../weights/face_recog/InsightFace_TF.pb'
    saved_dataset = '../weights/face_recog/dataset48.pkl'
    factor = 0.7
    threshold = [0.7, 0.7, 0.9]
    minisize = 25

    test = recog(model, saved_dataset)


    test.init_tf_env()
    test.load_feature()
    test.init_mtcnn_detector(mtcnn_model, threshold, minisize, factor)
    image = cv2.imread('./test1.jpg')
    rectangles, name_lists = test.get_indentity(image, role = True)

    for i in range(len(rectangles)):
        rec = rectangles[i,:]
        cv2.rectangle(image, (int(rec[0]), int(rec[1])), (int(rec[2]), int(rec[3])), (0, 0, 255), 1)
        cv2.putText(image, name_lists[i], (int(rec[0]), int(rec[1])), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.75,color=(152, 255, 204), thickness=2)

    cv2.imwrite('test1-recognition-sphereface.jpg', image)

